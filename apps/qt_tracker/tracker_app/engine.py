# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Single-target tracking engine on top of the C-BIoU and McByte trackers.

The engine turns a user click into a gate, feeds template-matching candidates to
the selected multi-object tracker every frame, and follows the track ID that was
born from the gate. It has no Qt dependency so it can be unit tested and reused.
"""

from __future__ import annotations

import enum
import importlib.util
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import supervision as sv

from tracker_app.candidates import CandidateSource, TemplateCandidateGenerator, box_from_center, clip_box
from trackers import CBIoUTracker, McByteTracker

CBIOU = "C-BIoU"
MCBYTE = "McByte"
TRACKER_NAMES: tuple[str, ...] = (CBIOU, MCBYTE)

TEMPLATE_SOURCE = "Template matching"
DETECTOR_SOURCE = "YOLO detector"
CANDIDATE_SOURCES: tuple[str, ...] = (TEMPLATE_SOURCE, DETECTOR_SOURCE)

DEFAULT_GATE_SIZE = (30, 30)
MIN_GATE_SIZE = 8
MAX_GATE_SIZE = 1024


class TrackStatus(enum.Enum):
    """State of the selected target."""

    IDLE = "Idle"
    TRACKING = "Tracking"
    LOST = "Lost"


@dataclass(frozen=True)
class TrackResult:
    """Outcome of processing one frame.

    Attributes:
        status: Current target state.
        box: Target box ``x1, y1, x2, y2``; while lost this is the last known box.
        score: Template correlation score of the tracked box (0 while lost).
        target_id: Tracker ID currently followed, or ``None`` when idle.
        candidates: Candidate boxes that were fed to the tracker this frame.
        note: Optional message for the user, e.g. when the detector found nothing under the click.
    """

    status: TrackStatus
    box: np.ndarray | None = None
    score: float = 0.0
    target_id: int | None = None
    candidates: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=np.float32))
    note: str = ""


def masks_available() -> bool:
    """Return whether McByte's SAM + Cutie mask pipeline can be imported."""
    return all(importlib.util.find_spec(name) is not None for name in ("torch", "segment_anything", "cutie"))


def build_tracker(name: str, frame_rate: float = 30.0, use_masks: bool = False) -> CBIoUTracker | McByteTracker:
    """Create a tracker configured for click-initialised single-target tracking.

    Args:
        name: One of :data:`TRACKER_NAMES`.
        frame_rate: Source frame rate, used to scale the lost-track buffer.
        use_masks: McByte only. Enable the SAM + Cutie mask-conditioned association.

    Returns:
        The tracker instance.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    frame_rate = frame_rate if np.isfinite(frame_rate) and frame_rate > 0 else 30.0
    if name == CBIOU:
        return CBIoUTracker(frame_rate=frame_rate, lost_track_buffer=60, minimum_consecutive_frames=1)
    if name == MCBYTE:
        return McByteTracker(
            frame_rate=frame_rate,
            lost_track_buffer=60,
            minimum_consecutive_frames=1,
            enable_mask_manager=use_masks,
        )
    raise ValueError(f"Unknown tracker {name!r}; expected one of {TRACKER_NAMES}.")


class TrackingEngine:
    """Follow one user-selected target through a video.

    Args:
        tracker_name: One of :data:`TRACKER_NAMES`.
        gate_size: Gate ``(width, height)`` in pixels drawn on click.
        frame_rate: Source frame rate.
        use_masks: Enable McByte's mask pipeline (ignored for C-BIoU).
        source: Candidate source; a template matcher is used when omitted. When
            the source finds nothing under a click (for example a detector on an
            unknown object), that target falls back to template matching.
        reacquire_after: Lost frames before another confident track may be adopted.
        reacquire_score: Minimum candidate score for adopting a new track; the
            source's own ``reacquire_score`` is used when omitted.
    """

    def __init__(
        self,
        tracker_name: str = CBIOU,
        gate_size: tuple[int, int] = DEFAULT_GATE_SIZE,
        frame_rate: float = 30.0,
        use_masks: bool = False,
        source: CandidateSource | None = None,
        reacquire_after: int = 5,
        reacquire_score: float | None = None,
    ) -> None:
        self._frame_rate = frame_rate
        self._use_masks = use_masks
        self._tracker_name = tracker_name
        self._tracker = build_tracker(tracker_name, frame_rate, use_masks)
        self._template = TemplateCandidateGenerator()
        self._source: CandidateSource = source if source is not None else self._template
        self._generator: CandidateSource = self._source
        self._gate_size = self._validate_gate(gate_size)
        self.reacquire_after = reacquire_after
        self.reacquire_score = reacquire_score
        self._target_id: int | None = None
        self._last_box: np.ndarray | None = None
        self._last_frame: np.ndarray | None = None
        self._lost_frames = 0

    # ------------------------------------------------------------------ properties
    @property
    def tracker_name(self) -> str:
        """Name of the active tracker."""
        return self._tracker_name

    @property
    def gate_size(self) -> tuple[int, int]:
        """Gate ``(width, height)`` in pixels."""
        return self._gate_size

    @property
    def active(self) -> bool:
        """Whether a target is selected."""
        return self._target_id is not None

    @property
    def target_id(self) -> int | None:
        """Tracker ID of the followed target."""
        return self._target_id

    @property
    def using_fallback(self) -> bool:
        """Whether the current target is followed by template matching instead of the chosen source."""
        return self._generator is not self._source

    # ------------------------------------------------------------------ control
    def set_candidate_source(self, source: CandidateSource | None) -> TrackResult | None:
        """Switch how candidate boxes are found; an active target is re-selected at its position.

        Args:
            source: New candidate source, or ``None`` for template matching.

        Returns:
            The re-initialised result when a target is active, else ``None``.
        """
        self._generator.reset()
        self._source = source if source is not None else self._template
        self._generator = self._source
        return self._reinitialize()

    def set_tracker(
        self, name: str, use_masks: bool | None = None, frame_rate: float | None = None
    ) -> TrackResult | None:
        """Switch tracker; an active target is re-initialised on its last box.

        Args:
            name: One of :data:`TRACKER_NAMES`.
            use_masks: McByte mask pipeline toggle; unchanged when ``None``.
            frame_rate: New source frame rate; unchanged when ``None``.

        Returns:
            The re-initialised result when a target is active, else ``None``.
        """
        if use_masks is not None:
            self._use_masks = use_masks
        if frame_rate is not None:
            self._frame_rate = frame_rate
        self._tracker = build_tracker(name, self._frame_rate, self._use_masks)
        self._tracker_name = name
        return self._reinitialize()

    def set_gate_size(self, width: int, height: int) -> TrackResult | None:
        """Change the gate size; an active target is re-initialised at the new size.

        Args:
            width: Gate width in pixels.
            height: Gate height in pixels.

        Returns:
            The re-initialised result when a target is active, else ``None``.
        """
        self._gate_size = self._validate_gate((width, height))
        return self._reinitialize()

    def select_target(self, frame: np.ndarray, x: float, y: float) -> TrackResult:
        """Start (or switch) tracking on a gate centred on ``(x, y)``.

        Args:
            frame: BGR frame the user clicked on.
            x: Click x in image pixels.
            y: Click y in image pixels.

        Returns:
            Tracking result for ``frame``.
        """
        self._last_frame = frame
        return self._start(frame, box_from_center(x, y, *self._gate_size))

    def stop_tracking(self) -> None:
        """Break the track and return to idle."""
        self._target_id = None
        self._last_box = None
        self._lost_frames = 0
        self._generator.reset()
        self._generator = self._source
        self._tracker.reset()

    def process(self, frame: np.ndarray) -> TrackResult:
        """Advance tracking by one frame.

        Args:
            frame: BGR frame of shape ``(H, W, 3)``.

        Returns:
            Tracking result for ``frame``.
        """
        self._last_frame = frame
        if self._target_id is None or self._last_box is None:
            return TrackResult(TrackStatus.IDLE)

        center = self._search_center()
        candidates = self._generator.generate(frame, center, self._lost_frames)
        detections = sv.Detections(xyxy=candidates.xyxy.astype(np.float32), confidence=candidates.scores)
        output = self._update_tracker(detections, frame)

        row = self._find_row(output, self._target_id)
        if row is None and self._lost_frames + 1 >= self.reacquire_after:
            row = self._reacquire_row(output)
            if row is not None and output.tracker_id is not None:
                self._target_id = int(output.tracker_id[row])

        if row is None:
            self._lost_frames += 1
            return TrackResult(
                TrackStatus.LOST, box=self._last_box, target_id=self._target_id, candidates=candidates.xyxy
            )

        box = clip_box(output.xyxy[row], frame.shape[1], frame.shape[0])
        score = float(output.confidence[row]) if output.confidence is not None else 0.0
        self._lost_frames = 0
        self._last_box = box
        self._generator.update_template(frame, box, score)
        return TrackResult(
            TrackStatus.TRACKING, box=box, score=score, target_id=self._target_id, candidates=candidates.xyxy
        )

    # ------------------------------------------------------------------ internals
    def _start(self, frame: np.ndarray, gate: np.ndarray) -> TrackResult:
        self._tracker.reset()
        self._lost_frames = 0
        self._generator.reset()
        self._generator = self._source
        note = ""
        box = self._source.initialize(frame, gate)
        if box is None:
            self._generator = self._template
            box = self._template.initialize(frame, gate)
            note = "No detected object under the click; tracking this target with template matching."
        if box is None:
            return TrackResult(TrackStatus.IDLE, note=note)
        self._last_box = box
        detections = sv.Detections(xyxy=box[None, :].astype(np.float32), confidence=np.array([1.0], dtype=np.float32))
        output = self._update_tracker(detections, frame)
        ids = [int(t) for t in output.tracker_id if t >= 0] if output.tracker_id is not None else []
        self._target_id = ids[0] if ids else None
        if self._target_id is None:
            return TrackResult(TrackStatus.IDLE, note=note)
        return TrackResult(TrackStatus.TRACKING, box=box, score=1.0, target_id=self._target_id, note=note)

    def _reinitialize(self) -> TrackResult | None:
        if self._target_id is not None and self._last_box is not None and self._last_frame is not None:
            cx, cy = self._center(self._last_box)
            return self._start(self._last_frame, box_from_center(cx, cy, *self._gate_size))
        self._tracker.reset()
        return None

    def _update_tracker(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        if isinstance(self._tracker, McByteTracker):
            # McByte's CMC and mask backends expect RGB frames.
            return self._tracker.update(detections, frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return self._tracker.update(detections)

    def _search_center(self) -> tuple[float, float]:
        """Use the Kalman state of the target track when available, else the last box."""
        for tracklet in getattr(self._tracker, "tracks", []):
            if getattr(tracklet, "tracker_id", -1) == self._target_id:
                state = np.asarray(tracklet.get_state_bbox(), dtype=np.float32).reshape(-1)
                if state.size == 4 and np.all(np.isfinite(state)):
                    return self._center(state)
        return self._center(self._last_box if self._last_box is not None else np.zeros(4))

    def _reacquire_row(self, output: sv.Detections) -> int | None:
        if output.tracker_id is None or output.confidence is None:
            return None
        threshold = self.reacquire_score if self.reacquire_score is not None else self._generator.reacquire_score
        best: int | None = None
        for row, (tid, conf) in enumerate(zip(output.tracker_id, output.confidence, strict=True)):
            if tid >= 0 and conf >= threshold and (best is None or conf > output.confidence[best]):
                best = row
        return best

    @staticmethod
    def _find_row(output: sv.Detections, target_id: int) -> int | None:
        if output.tracker_id is None:
            return None
        rows = np.where(output.tracker_id == target_id)[0]
        return int(rows[0]) if len(rows) else None

    @staticmethod
    def _center(box: Any) -> tuple[float, float]:
        return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)

    @staticmethod
    def _validate_gate(size: tuple[int, int]) -> tuple[int, int]:
        w, h = (int(min(max(v, MIN_GATE_SIZE), MAX_GATE_SIZE)) for v in size)
        return (w, h)


def annotate(frame: np.ndarray, result: TrackResult, show_candidates: bool = False) -> np.ndarray:
    """Draw the tracking result on a copy of ``frame``.

    Args:
        frame: BGR frame.
        result: Result returned by :class:`TrackingEngine`.
        show_candidates: Also draw the template-matching candidates.

    Returns:
        Annotated BGR frame.
    """
    out = frame.copy()
    if show_candidates:
        for c in result.candidates:
            x1, y1, x2, y2 = (round(v) for v in c)
            cv2.rectangle(out, (x1, y1), (x2, y2), (160, 160, 160), 1)
    if result.box is None or result.status is TrackStatus.IDLE:
        return out
    x1, y1, x2, y2 = (round(v) for v in result.box)
    if result.status is TrackStatus.TRACKING:
        color = (0, 220, 0)
        label = f"ID {result.target_id}  {result.score:.2f}"
    else:
        color = (0, 0, 255)
        label = "LOST"
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 8, 1)
    cv2.putText(out, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out
