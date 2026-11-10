# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Capture / track / save loop running off the UI thread."""

from __future__ import annotations

import queue
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from tracker_app.detector import DetectorCandidateSource, load_detector
from tracker_app.engine import DETECTOR_SOURCE, TEMPLATE_SOURCE, TrackingEngine, TrackResult, TrackStatus, annotate


@dataclass(frozen=True)
class SourceSpec:
    """Video source description.

    Attributes:
        kind: ``"camera"`` or ``"file"``.
        camera_index: OpenCV camera index for ``kind == "camera"``.
        path: Media file path for ``kind == "file"``.
    """

    kind: str
    camera_index: int = 0
    path: str = ""

    def describe(self) -> str:
        """Human-readable label."""
        return f"Camera {self.camera_index}" if self.kind == "camera" else Path(self.path).name


@dataclass(frozen=True)
class EngineSettings:
    """Initial engine configuration."""

    tracker_name: str
    gate_size: tuple[int, int]
    use_masks: bool
    show_candidates: bool = False
    candidate_source: str = TEMPLATE_SOURCE
    detector_model: str = ""


def to_qimage(frame_bgr: np.ndarray) -> QImage:
    """Convert a BGR ``ndarray`` into a detached ``QImage``."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


Command = Callable[["VideoWorker"], None]

# Consecutive failed camera reads tolerated before the live feed is reported as lost.
MAX_CAMERA_READ_FAILURES = 30


def open_camera(index: int) -> cv2.VideoCapture | None:
    """Open a camera, trying the backends that work best on this platform first.

    On Windows the DirectShow backend is tried before Media Foundation (MSMF), whose
    asynchronous reader fails on some webcams with ``can't grab frame`` errors.
    A backend counts as working only once it has delivered a frame.

    Args:
        index: Camera index.

    Returns:
        An opened capture, or ``None`` when no backend delivers frames.
    """
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    for backend in backends:
        capture = cv2.VideoCapture(index, backend)
        if capture.isOpened():
            for _ in range(10):  # some cameras need a few reads to warm up
                ok, _frame = capture.read()
                if ok:
                    return capture
                time.sleep(0.05)
        capture.release()
    return None


class VideoWorker(QThread):
    """Read frames, run the tracking engine, write the processed video and publish frames."""

    frameReady = Signal(QImage)
    trackUpdated = Signal(str, int, float)  # status, target id (-1 when none), score
    statsUpdated = Signal(float, int, int)  # processing fps, frame index, frame count (0 = live)
    sourceOpened = Signal(str, float, int, int)  # description, fps, width, height
    sourceFailed = Signal(str)
    sourceEnded = Signal()
    savingChanged = Signal(bool, str)
    engineError = Signal(str)
    candidateSourceChanged = Signal(str)  # source actually in use after a switch

    def __init__(self, source: SourceSpec, settings: EngineSettings, parent=None) -> None:
        super().__init__(parent)
        self._source = source
        self._settings = settings
        self._commands: queue.Queue[Command] = queue.Queue()
        self._stop = False
        self.paused = False
        self.engine: TrackingEngine | None = None
        self.show_candidates = settings.show_candidates
        self.frame: np.ndarray | None = None
        self.result = TrackResult(TrackStatus.IDLE)
        self.fps = 30.0
        self._writer: cv2.VideoWriter | None = None
        self._needs_redraw = False

    # ----------------------------------------------------------- thread-safe API
    def post(self, command: Command) -> None:
        """Queue ``command`` to run on the worker thread before the next frame."""
        self._commands.put(command)

    def request_stop(self) -> None:
        """Ask the loop to finish."""
        self._stop = True

    # ----------------------------------------------------------- used by commands
    def redraw(self) -> None:
        """Re-render the current frame (used while paused)."""
        self._needs_redraw = True

    def set_result(self, result: TrackResult) -> None:
        """Replace the current result and refresh the view."""
        self.result = result
        self._emit_track(result)
        self.redraw()

    def set_candidate_source(self, name: str, model_id: str) -> None:
        """Switch between template matching and the detector, loading the model if needed."""
        if self.engine is None:
            return
        source = None
        if name == DETECTOR_SOURCE:
            self.engineError.emit(f"Loading detector {model_id}…")
            try:
                source = DetectorCandidateSource(load_detector(model_id))
            except Exception as exc:
                self.engineError.emit(f"Detector unavailable, using template matching: {exc}")
                name = TEMPLATE_SOURCE
        result = self.engine.set_candidate_source(source)
        self.candidateSourceChanged.emit(name)
        if result is not None:
            self.set_result(result)

    def start_saving(self, path: str) -> None:
        """Start writing processed frames to ``path``."""
        self.stop_saving()
        if self.frame is None:
            self.savingChanged.emit(False, "No frame available yet")
            return
        h, w = self.frame.shape[:2]
        fourcc = cv2.VideoWriter.fourcc(*("XVID" if path.lower().endswith(".avi") else "mp4v"))
        writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
        if not writer.isOpened():
            self.savingChanged.emit(False, f"Cannot write {path}")
            return
        self._writer = writer
        self.savingChanged.emit(True, path)

    def stop_saving(self) -> None:
        """Finish the output file, if any."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self.savingChanged.emit(False, "")

    # ----------------------------------------------------------- loop
    def run(self) -> None:
        capture = self._open_capture()
        if capture is None:
            return
        try:
            self._loop(capture)
        finally:
            capture.release()
            self.stop_saving()

    def _open_capture(self) -> cv2.VideoCapture | None:
        if self._source.kind == "camera":
            capture = open_camera(self._source.camera_index)
        else:
            capture = cv2.VideoCapture(self._source.path)
        if capture is None or not capture.isOpened():
            self.sourceFailed.emit(f"Cannot open {self._source.describe()}")
            return None
        fps = capture.get(cv2.CAP_PROP_FPS)
        self.fps = fps if np.isfinite(fps) and 1.0 <= fps <= 240.0 else 30.0
        try:
            self.engine = TrackingEngine(
                tracker_name=self._settings.tracker_name,
                gate_size=self._settings.gate_size,
                frame_rate=self.fps,
                use_masks=self._settings.use_masks,
            )
        except Exception as exc:  # optional mask dependencies may fail to load
            self.engineError.emit(str(exc))
            capture.release()
            return None
        if self._settings.candidate_source == DETECTOR_SOURCE:
            self.set_candidate_source(DETECTOR_SOURCE, self._settings.detector_model)
        w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.sourceOpened.emit(self._source.describe(), self.fps, w, h)
        return capture

    def _loop(self, capture: cv2.VideoCapture) -> None:
        is_file = self._source.kind == "file"
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0
        frame_index = 0
        frame_period = 1.0 / self.fps
        next_due = time.perf_counter()
        fps_window: list[float] = []
        read_failures = 0
        while not self._stop:
            self._run_commands()
            if self.paused and self.frame is not None:
                if self._needs_redraw:
                    self._publish(self.frame, self.result, write=False)
                time.sleep(0.01)
                next_due = time.perf_counter()
                continue
            ok, frame = capture.read()
            if not ok or frame is None:
                if is_file:
                    self.sourceEnded.emit()
                    break
                read_failures += 1
                if read_failures < MAX_CAMERA_READ_FAILURES:
                    time.sleep(0.03)
                    continue
                self.sourceFailed.emit("Camera stopped delivering frames. Is another app using it?")
                break
            read_failures = 0
            frame_index += 1
            started = time.perf_counter()
            self.frame = frame
            self._step(frame)
            fps_window.append(time.perf_counter() - started)
            if len(fps_window) >= 10:
                mean = sum(fps_window) / len(fps_window)
                self.statsUpdated.emit(1.0 / mean if mean > 0 else 0.0, frame_index, frame_count)
                fps_window.clear()
            if is_file:
                next_due += frame_period
                delay = next_due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_due = time.perf_counter()

    def _step(self, frame: np.ndarray) -> None:
        if self.engine is None:
            return
        try:
            result = self.engine.process(frame)
        except Exception as exc:
            self.engineError.emit(f"Tracking failed: {exc}")
            self.engine.stop_tracking()
            result = TrackResult(TrackStatus.IDLE)
        self.result = result
        self._emit_track(result)
        self._publish(frame, result, write=True)

    def _publish(self, frame: np.ndarray, result: TrackResult, write: bool) -> None:
        self._needs_redraw = False
        annotated = annotate(frame, result, show_candidates=self.show_candidates)
        if write and self._writer is not None:
            self._writer.write(annotated)
        self.frameReady.emit(to_qimage(annotated))

    def _emit_track(self, result: TrackResult) -> None:
        if result.note:
            self.engineError.emit(result.note)
        target = -1 if result.target_id is None else result.target_id
        self.trackUpdated.emit(result.status.value, target, result.score)

    def _run_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                command(self)
            except Exception as exc:
                self.engineError.emit(str(exc))
