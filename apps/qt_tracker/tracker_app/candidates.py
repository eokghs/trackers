# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Template-matching candidate generator.

The trackers in this repository (C-BIoU, McByte) are tracking-by-detection
multi-object trackers: they associate per-frame detections over time but do not
locate objects themselves. For click-to-track on an arbitrary object there is no
class-specific detector, so this module produces the per-frame detections from
the user's gate: the gate's pixels form an appearance template, and each new
frame is searched around the last known position with normalized
cross-correlation. The strongest peaks are returned as candidate boxes whose
confidence is the correlation score, which plugs directly into the trackers'
high/low confidence cascades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Candidates:
    """Candidate boxes for one frame.

    Attributes:
        xyxy: Array of shape ``(N, 4)`` with boxes in pixel ``x1, y1, x2, y2``.
        scores: Array of shape ``(N,)`` with correlation scores in ``[0, 1]``.
    """

    xyxy: np.ndarray
    scores: np.ndarray

    @staticmethod
    def empty() -> Candidates:
        """Return an empty candidate set."""
        return Candidates(xyxy=np.empty((0, 4), dtype=np.float32), scores=np.empty((0,), dtype=np.float32))

    def __len__(self) -> int:
        return len(self.scores)


class CandidateSource(Protocol):
    """Anything that turns a gate into a target and frames into candidate boxes."""

    reacquire_score: float

    def initialize(self, frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        """Select the target from the gate ``box``; return its box, or ``None`` if nothing is there."""

    def reset(self) -> None:
        """Forget the target."""

    def generate(self, frame: np.ndarray, center: tuple[float, float], lost_frames: int = 0) -> Candidates:
        """Return candidate boxes for ``frame``."""

    def update_template(self, frame: np.ndarray, box: np.ndarray, score: float) -> None:
        """Adapt to the target's appearance in the tracked ``box``."""


def clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    """Shift a box so it lies inside an image, keeping its size when possible.

    Args:
        box: Box ``x1, y1, x2, y2``.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Box of the same size (or clamped to the image size) inside the image.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    w = min(x2 - x1, float(width))
    h = min(y2 - y1, float(height))
    x1 = min(max(x1, 0.0), width - w)
    y1 = min(max(y1, 0.0), height - h)
    return np.array([x1, y1, x1 + w, y1 + h], dtype=np.float32)


def box_from_center(cx: float, cy: float, width: int, height: int) -> np.ndarray:
    """Build an ``xyxy`` box of the given size centred on ``(cx, cy)``."""
    return np.array([cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0], dtype=np.float32)


class TemplateCandidateGenerator:
    """Produce candidate detections by template matching around a search centre.

    Args:
        search_scale: Search window size as a multiple of the gate size on each
            side of the gate while the target is visible.
        max_search_scale: Upper bound for the search window multiple; the window
            grows towards it while the target is lost.
        max_candidates: Maximum number of peaks returned per frame. Keeping a few
            distractor peaks lets the tracker's association reject them rather
            than blindly following the single best score.
        min_score: Peaks scoring below this are discarded.
        template_learning_rate: Blend factor used to adapt the template to the
            target's current appearance.
        template_update_score: Minimum score of a tracked box before the
            template is adapted, which prevents learning occluders.
    """

    reacquire_score = 0.8

    def __init__(
        self,
        search_scale: float = 2.0,
        max_search_scale: float = 6.0,
        max_candidates: int = 5,
        min_score: float = 0.2,
        template_learning_rate: float = 0.05,
        template_update_score: float = 0.75,
    ) -> None:
        self.search_scale = search_scale
        self.max_search_scale = max_search_scale
        self.max_candidates = max_candidates
        self.min_score = min_score
        self.template_learning_rate = template_learning_rate
        self.template_update_score = template_update_score
        self._template: np.ndarray | None = None

    @property
    def initialized(self) -> bool:
        """Whether a template is set."""
        return self._template is not None

    @property
    def template_size(self) -> tuple[int, int]:
        """Template ``(width, height)`` in pixels, or ``(0, 0)`` when unset."""
        if self._template is None:
            return (0, 0)
        return (int(self._template.shape[1]), int(self._template.shape[0]))

    def initialize(self, frame: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Set the template from ``box`` in ``frame``.

        Args:
            frame: BGR image of shape ``(H, W, 3)``.
            box: Gate ``x1, y1, x2, y2``. It is shifted inside the frame if needed.

        Returns:
            The gate box actually used, clipped to the frame.
        """
        height, width = frame.shape[:2]
        box = clip_box(box, width, height)
        self._template = self._crop(frame, box).astype(np.float32)
        return box

    def reset(self) -> None:
        """Forget the template."""
        self._template = None

    def update_template(self, frame: np.ndarray, box: np.ndarray, score: float) -> None:
        """Blend the patch under ``box`` into the template if ``score`` is high.

        Args:
            frame: BGR image.
            box: Tracked box of the current frame.
            score: Correlation score of the tracked box.
        """
        if self._template is None or score < self.template_update_score:
            return
        height, width = frame.shape[:2]
        tw, th = self.template_size
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        patch_box = clip_box(box_from_center(cx, cy, tw, th), width, height)
        patch = self._crop(frame, patch_box).astype(np.float32)
        if patch.shape != self._template.shape:
            return
        rate = self.template_learning_rate
        self._template = (1.0 - rate) * self._template + rate * patch

    def generate(self, frame: np.ndarray, center: tuple[float, float], lost_frames: int = 0) -> Candidates:
        """Search ``frame`` for the template around ``center``.

        Args:
            frame: BGR image of shape ``(H, W, 3)``.
            center: Expected target centre ``(cx, cy)``.
            lost_frames: Number of consecutive frames the target has been lost;
                the search window grows with it to allow re-acquisition.

        Returns:
            Up to ``max_candidates`` non-overlapping peaks as candidate boxes.
        """
        if self._template is None:
            return Candidates.empty()
        height, width = frame.shape[:2]
        tw, th = self.template_size
        if tw > width or th > height:
            return Candidates.empty()

        scale = min(self.search_scale * (1.0 + 0.25 * lost_frames), self.max_search_scale)
        cx, cy = center
        x0 = int(max(0, round(cx - tw / 2.0 - scale * tw)))
        y0 = int(max(0, round(cy - th / 2.0 - scale * th)))
        x1 = int(min(width, round(cx + tw / 2.0 + scale * tw)))
        y1 = int(min(height, round(cy + th / 2.0 + scale * th)))
        if x1 - x0 < tw or y1 - y0 < th:
            return Candidates.empty()

        window = frame[y0:y1, x0:x1].astype(np.float32)
        response = cv2.matchTemplate(window, self._template, cv2.TM_CCOEFF_NORMED)
        response = np.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)
        return self._peaks(response, offset=(x0, y0))

    def _peaks(self, response: np.ndarray, offset: tuple[int, int]) -> Candidates:
        """Extract the strongest non-overlapping peaks from a response map."""
        tw, th = self.template_size
        boxes: list[list[float]] = []
        scores: list[float] = []
        work = response.copy()
        suppress_w, suppress_h = max(1, (3 * tw) // 4), max(1, (3 * th) // 4)
        for _ in range(self.max_candidates):
            _, max_val, _, max_loc = cv2.minMaxLoc(work)
            if max_val < self.min_score:
                break
            px, py = max_loc
            boxes.append([offset[0] + px, offset[1] + py, offset[0] + px + tw, offset[1] + py + th])
            scores.append(float(min(max_val, 1.0)))
            work[
                max(0, py - suppress_h) : py + suppress_h + 1,
                max(0, px - suppress_w) : px + suppress_w + 1,
            ] = -1.0
        if not boxes:
            return Candidates.empty()
        return Candidates(xyxy=np.array(boxes, dtype=np.float32), scores=np.array(scores, dtype=np.float32))

    @staticmethod
    def _crop(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = (round(v) for v in box)
        return frame[y1:y2, x1:x2]
