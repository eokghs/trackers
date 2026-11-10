# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""QML-facing controller: owns the video worker and exposes app state."""

from __future__ import annotations

from PySide6.QtCore import Property, QObject, QUrl, Signal, Slot
from PySide6.QtMultimedia import QMediaDevices

from tracker_app.detector import (
    DEFAULT_DETECTOR_MODEL,
    default_detector_model,
    detector_available,
    detector_model_presets,
)
from tracker_app.engine import (
    CANDIDATE_SOURCES,
    DEFAULT_GATE_SIZE,
    MCBYTE,
    TEMPLATE_SOURCE,
    TRACKER_NAMES,
    TrackResult,
    TrackStatus,
    masks_available,
)
from tracker_app.video_item import VideoFrameItem
from tracker_app.worker import EngineSettings, SourceSpec, VideoWorker


def _local_path(url_or_path: str) -> str:
    url = QUrl(url_or_path)
    return url.toLocalFile() if url.isLocalFile() else url_or_path


class AppController(QObject):
    """State and actions for ``Main.qml``."""

    stateChanged = Signal()
    trackChanged = Signal()
    statsChanged = Signal()
    settingsChanged = Signal()
    camerasChanged = Signal()
    message = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._worker: VideoWorker | None = None
        self._view: VideoFrameItem | None = None
        self._source_label = ""
        self._running = False
        self._paused = False
        self._saving = False
        self._save_path = ""
        self._status = TrackStatus.IDLE.value
        self._target_id = -1
        self._score = 0.0
        self._fps = 0.0
        self._frame_index = 0
        self._frame_count = 0
        self._resolution = ""
        self._tracker_name = TRACKER_NAMES[0]
        self._gate_w, self._gate_h = DEFAULT_GATE_SIZE
        self._use_masks = False
        self._show_candidates = False
        self._masks_available = masks_available()
        self._detector_available = detector_available()
        self._candidate_source = TEMPLATE_SOURCE
        self._detector_models = detector_model_presets()
        self._detector_model = default_detector_model()
        self._media_devices = QMediaDevices(self)
        self._media_devices.videoInputsChanged.connect(self.camerasChanged)

    # ================================================================ properties
    def _get_cameras(self) -> list[str]:
        names = [device.description() for device in QMediaDevices.videoInputs()]
        return names if names else ["Camera 0"]

    cameras = Property(list, _get_cameras, notify=camerasChanged)

    def _get_tracker_names(self) -> list:
        return list(TRACKER_NAMES)

    trackerNames = Property(list, _get_tracker_names, constant=True)

    def _get_masks_available(self) -> bool:
        return self._masks_available

    masksAvailable = Property(bool, _get_masks_available, constant=True)

    def _get_candidate_sources(self) -> list:
        return list(CANDIDATE_SOURCES)

    candidateSources = Property(list, _get_candidate_sources, constant=True)

    def _get_detector_models(self) -> list:
        return list(self._detector_models)

    detectorModels = Property(list, _get_detector_models, notify=settingsChanged)

    def _get_detector_available(self) -> bool:
        return self._detector_available

    detectorAvailable = Property(bool, _get_detector_available, constant=True)

    def _get_running(self) -> bool:
        return self._running

    running = Property(bool, _get_running, notify=stateChanged)

    def _get_paused(self) -> bool:
        return self._paused

    paused = Property(bool, _get_paused, notify=stateChanged)

    def _get_saving(self) -> bool:
        return self._saving

    saving = Property(bool, _get_saving, notify=stateChanged)

    def _get_save_path(self) -> str:
        return self._save_path

    savePath = Property(str, _get_save_path, notify=stateChanged)

    def _get_source_label(self) -> str:
        return self._source_label

    sourceLabel = Property(str, _get_source_label, notify=stateChanged)

    def _get_resolution(self) -> str:
        return self._resolution

    resolution = Property(str, _get_resolution, notify=stateChanged)

    def _get_status(self) -> str:
        return self._status

    status = Property(str, _get_status, notify=trackChanged)

    def _get_target_id(self) -> int:
        return self._target_id

    targetId = Property(int, _get_target_id, notify=trackChanged)

    def _get_score(self) -> float:
        return self._score

    score = Property(float, _get_score, notify=trackChanged)

    def _get_fps(self) -> float:
        return self._fps

    fps = Property(float, _get_fps, notify=statsChanged)

    def _get_frame_index(self) -> int:
        return self._frame_index

    frameIndex = Property(int, _get_frame_index, notify=statsChanged)

    def _get_frame_count(self) -> int:
        return self._frame_count

    frameCount = Property(int, _get_frame_count, notify=statsChanged)

    def _get_tracker_name(self) -> str:
        return self._tracker_name

    trackerName = Property(str, _get_tracker_name, notify=settingsChanged)

    def _get_gate_width(self) -> int:
        return self._gate_w

    gateWidth = Property(int, _get_gate_width, notify=settingsChanged)

    def _get_gate_height(self) -> int:
        return self._gate_h

    gateHeight = Property(int, _get_gate_height, notify=settingsChanged)

    def _get_use_masks(self) -> bool:
        return self._use_masks

    useMasks = Property(bool, _get_use_masks, notify=settingsChanged)

    def _get_show_candidates(self) -> bool:
        return self._show_candidates

    showCandidates = Property(bool, _get_show_candidates, notify=settingsChanged)

    def _get_candidate_source(self) -> str:
        return self._candidate_source

    candidateSource = Property(str, _get_candidate_source, notify=settingsChanged)

    def _get_detector_model(self) -> str:
        return self._detector_model

    detectorModel = Property(str, _get_detector_model, notify=settingsChanged)

    # ================================================================ view
    @Slot(QObject)
    def attachView(self, view: QObject) -> None:
        """Register the ``VideoFrameItem`` that displays processed frames."""
        if isinstance(view, VideoFrameItem):
            self._view = view

    # ================================================================ sources
    @Slot(int)
    def openCamera(self, index: int) -> None:
        """Start the live feed from camera ``index``."""
        self._start(SourceSpec(kind="camera", camera_index=max(0, index)))

    @Slot(str)
    def openFile(self, url: str) -> None:
        """Start playback and tracking of a media file."""
        path = _local_path(url)
        if path:
            self._start(SourceSpec(kind="file", path=path))

    @Slot()
    def togglePause(self) -> None:
        """Pause or resume the feed."""
        if self._worker is None:
            return
        self._paused = not self._paused
        self._worker.paused = self._paused
        self.stateChanged.emit()

    @Slot()
    def closeSource(self) -> None:
        """Stop the feed and release the source."""
        self._shutdown_worker()
        if self._view is not None:
            self._view.clear()
        self._set_track(TrackStatus.IDLE.value, -1, 0.0)

    # ================================================================ tracking
    @Slot(float, float)
    def selectTarget(self, x: float, y: float) -> None:
        """Start or switch tracking with a gate centred on image pixel ``(x, y)``."""
        if self._worker is None or x < 0 or y < 0:
            return

        def command(worker: VideoWorker) -> None:
            if worker.engine is None or worker.frame is None:
                return
            worker.set_result(worker.engine.select_target(worker.frame, x, y))

        self._worker.post(command)

    @Slot()
    def stopTracking(self) -> None:
        """Break the current track."""
        if self._worker is None:
            return

        def command(worker: VideoWorker) -> None:
            if worker.engine is None:
                return
            worker.engine.stop_tracking()
            worker.set_result(TrackResult(TrackStatus.IDLE))

        self._worker.post(command)

    @Slot(int, int)
    def setGateSize(self, width: int, height: int) -> None:
        """Change the gate size; applies to the active target immediately."""
        if (width, height) == (self._gate_w, self._gate_h):
            return
        self._gate_w, self._gate_h = width, height
        self.settingsChanged.emit()
        if self._worker is None:
            return

        def command(worker: VideoWorker) -> None:
            if worker.engine is None:
                return
            result = worker.engine.set_gate_size(width, height)
            if result is not None:
                worker.set_result(result)

        self._worker.post(command)

    @Slot(str)
    def setTracker(self, name: str) -> None:
        """Select the tracker by name."""
        if name not in TRACKER_NAMES or name == self._tracker_name:
            return
        self._tracker_name = name
        self.settingsChanged.emit()
        self._reconfigure_tracker()

    @Slot(bool)
    def setUseMasks(self, enabled: bool) -> None:
        """Toggle McByte's SAM + Cutie mask-conditioned association."""
        enabled = enabled and self._masks_available
        if enabled == self._use_masks:
            return
        self._use_masks = enabled
        self.settingsChanged.emit()
        if self._tracker_name == MCBYTE:
            self._reconfigure_tracker()

    @Slot(str, str)
    def setCandidateSource(self, name: str, model_id: str) -> None:
        """Choose template matching or the detector (with model ``model_id``)."""
        model_id = _local_path(model_id.strip()) or DEFAULT_DETECTOR_MODEL
        if model_id not in self._detector_models:
            self._detector_models.insert(0, model_id)
        if name not in CANDIDATE_SOURCES or (name, model_id) == (self._candidate_source, self._detector_model):
            return
        if name != TEMPLATE_SOURCE and not self._detector_available:
            self.message.emit(
                "No YOLO backend found: install ultralytics, put it in C:\\Ultralytics\\Yolo26, "
                "or pip install 'trackers[detection]'"
            )
            self.settingsChanged.emit()
            return
        self._candidate_source, self._detector_model = name, model_id
        self.settingsChanged.emit()
        if self._worker is not None:
            self._worker.post(lambda worker: worker.set_candidate_source(name, model_id))

    @Slot(bool)
    def setShowCandidates(self, enabled: bool) -> None:
        """Toggle drawing of template-matching candidates."""
        self._show_candidates = enabled
        self.settingsChanged.emit()
        if self._worker is not None:

            def command(worker: VideoWorker) -> None:
                worker.show_candidates = enabled
                worker.redraw()

            self._worker.post(command)

    # ================================================================ saving
    @Slot(str)
    def startSaving(self, url: str) -> None:
        """Write the processed video to ``url`` from now on."""
        if self._worker is None:
            self.message.emit("Open a video source before saving.")
            return
        path = _local_path(url)
        self._worker.post(lambda worker: worker.start_saving(path))

    @Slot()
    def stopSaving(self) -> None:
        """Finish the processed-video file."""
        if self._worker is not None:
            self._worker.post(lambda worker: worker.stop_saving())

    # ================================================================ internals
    def _reconfigure_tracker(self) -> None:
        if self._worker is None:
            return
        name, use_masks = self._tracker_name, self._use_masks

        def command(worker: VideoWorker) -> None:
            if worker.engine is None:
                return
            result = worker.engine.set_tracker(name, use_masks=use_masks)
            if result is not None:
                worker.set_result(result)

        self._worker.post(command)

    def _start(self, source: SourceSpec) -> None:
        self._shutdown_worker()
        worker = VideoWorker(
            source,
            EngineSettings(
                tracker_name=self._tracker_name,
                gate_size=(self._gate_w, self._gate_h),
                use_masks=self._use_masks,
                show_candidates=self._show_candidates,
                candidate_source=self._candidate_source,
                detector_model=self._detector_model,
            ),
        )
        if self._view is not None:
            worker.frameReady.connect(self._view.setFrame)
        worker.trackUpdated.connect(self._set_track)
        worker.statsUpdated.connect(self._set_stats)
        worker.sourceOpened.connect(self._on_opened)
        worker.sourceFailed.connect(self._on_failed)
        worker.sourceEnded.connect(self._on_ended)
        worker.savingChanged.connect(self._on_saving)
        worker.engineError.connect(self.message)
        worker.candidateSourceChanged.connect(self._on_source_changed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        self._paused = False
        self._source_label = source.describe()
        self._running = True
        self.stateChanged.emit()
        worker.start()

    def _shutdown_worker(self) -> None:
        worker = self._worker
        if worker is None:
            return
        self._worker = None
        worker.request_stop()
        worker.wait(5000)
        worker.deleteLater()
        self._running = False
        self._paused = False
        self._saving = False
        self.stateChanged.emit()

    @Slot(str, int, float)
    def _set_track(self, status: str, target_id: int, score: float) -> None:
        self._status, self._target_id, self._score = status, target_id, score
        self.trackChanged.emit()

    @Slot(float, int, int)
    def _set_stats(self, fps: float, frame_index: int, frame_count: int) -> None:
        self._fps, self._frame_index, self._frame_count = fps, frame_index, frame_count
        self.statsChanged.emit()

    @Slot(str, float, int, int)
    def _on_opened(self, label: str, fps: float, width: int, height: int) -> None:
        self._source_label = label
        self._resolution = f"{width}x{height} @ {fps:.1f} fps"
        self.stateChanged.emit()

    @Slot(str)
    def _on_failed(self, text: str) -> None:
        self.message.emit(text)

    @Slot()
    def _on_ended(self) -> None:
        self.message.emit("End of video. Processed output, if any, has been saved.")

    @Slot(bool, str)
    def _on_saving(self, saving: bool, detail: str) -> None:
        self._saving = saving
        if saving:
            self._save_path = detail
        elif detail:
            self.message.emit(detail)
        self.stateChanged.emit()

    @Slot(str)
    def _on_source_changed(self, name: str) -> None:
        if name != self._candidate_source:
            self._candidate_source = name
            self.settingsChanged.emit()

    @Slot()
    def _on_worker_finished(self) -> None:
        sender = self.sender()
        if isinstance(sender, VideoWorker):
            # ``finished`` fires just before the thread exits: wait for it, then let Qt delete
            # the object. Dropping the last Python reference to a still-running QThread aborts the app.
            sender.wait(2000)
            sender.deleteLater()
        if sender is not None and self._worker is not None and sender is not self._worker:
            return
        self._worker = None
        self._running = False
        self._paused = False
        self._saving = False
        self.stateChanged.emit()

    def shutdown(self) -> None:
        """Stop the worker; call before the application exits."""
        self._shutdown_worker()
