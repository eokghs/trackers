# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Object-detector candidate source (YOLO via Ultralytics, or YOLO / RF-DETR via ``inference-models``).

Instead of template matching, every frame is run through a pretrained detector
and its boxes become the tracker's detections. On click, the detection under the
user's gate is chosen as the target; later frames only keep detections of the
same class, which removes most distractors before association.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import supervision as sv

from tracker_app.candidates import Candidates

ULTRALYTICS_DIR_ENV = "TRACKER_ULTRALYTICS_DIR"
DEFAULT_ULTRALYTICS_DIR = r"C:\Ultralytics\Yolo26"
ULTRALYTICS_SUFFIXES = (".pt", ".onnx", ".engine", ".torchscript", ".mlpackage", ".openvino")

DEFAULT_DETECTOR_MODEL = "yolov8n-640"
INFERENCE_MODEL_PRESETS: tuple[str, ...] = ("yolov8n-640", "yolov8s-640", "yolov8m-640", "rfdetr-nano")

Detector = Callable[[np.ndarray], sv.Detections]


def ultralytics_dir() -> Path:
    """Folder holding a local Ultralytics install and its ``weights`` (``TRACKER_ULTRALYTICS_DIR`` overrides)."""
    return Path(os.environ.get(ULTRALYTICS_DIR_ENV, DEFAULT_ULTRALYTICS_DIR))


def _ensure_ultralytics_importable() -> bool:
    """Make ``ultralytics`` importable, adding the local Ultralytics folder to ``sys.path`` if needed.

    The folder is appended (not prepended) so the app's own NumPy / OpenCV keep precedence.
    """
    if importlib.util.find_spec("ultralytics") is not None:
        return True
    folder = ultralytics_dir()
    if (folder / "ultralytics").is_dir() and str(folder) not in sys.path:
        sys.path.append(str(folder))
        importlib.invalidate_caches()
    return importlib.util.find_spec("ultralytics") is not None


def ultralytics_available() -> bool:
    """Return whether Ultralytics is installed or found in the local Ultralytics folder."""
    return _ensure_ultralytics_importable()


def inference_models_available() -> bool:
    """Return whether ``inference-models`` is installed."""
    return importlib.util.find_spec("inference_models") is not None


def detector_available() -> bool:
    """Return whether any detector backend (Ultralytics or ``inference-models``) is available."""
    return ultralytics_available() or inference_models_available()


def local_weights() -> list[str]:
    """List YOLO weight files in the local Ultralytics folder's ``weights`` directory."""
    folder = ultralytics_dir() / "weights"
    if not folder.is_dir():
        return []
    return sorted(str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ULTRALYTICS_SUFFIXES)


def detector_model_presets() -> list[str]:
    """Models offered in the UI: local Ultralytics weights first, then ``inference-models`` aliases."""
    presets = local_weights()
    if inference_models_available():
        presets += list(INFERENCE_MODEL_PRESETS)
    return presets


def default_detector_model() -> str:
    """Model selected by default: the first local weight file, else ``yolov8n-640``."""
    presets = detector_model_presets()
    return presets[0] if presets else DEFAULT_DETECTOR_MODEL


def is_ultralytics_model(model_id: str) -> bool:
    """Whether ``model_id`` is an Ultralytics weights file or name (``*.pt``, ``*.onnx``, ...)."""
    return model_id.lower().endswith(ULTRALYTICS_SUFFIXES)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_detector(model_id: str = DEFAULT_DETECTOR_MODEL, device: str = "auto") -> Detector:
    """Load a pretrained detector.

    Ultralytics weights (a local ``.pt`` / ``.onnx`` / ``.engine`` file, or a name such as
    ``yolo26n.pt``) are loaded with Ultralytics; other IDs such as ``yolov8n-640`` or
    ``rfdetr-nano`` with ``inference-models``.

    Args:
        model_id: Weights path / name or ``inference-models`` alias.
        device: ``auto``, ``cpu``, ``cuda`` or ``mps``. ``auto`` picks CUDA when available.

    Returns:
        A callable mapping a BGR frame to ``sv.Detections``.

    Raises:
        RuntimeError: If the needed backend is missing or the model is not a detector.
    """
    if is_ultralytics_model(model_id):
        return _load_ultralytics(model_id, device)
    if not inference_models_available():
        raise RuntimeError(
            f"{model_id!r} needs inference-models (pip install 'trackers[detection]'); "
            "or pick a local Ultralytics .pt file"
        )
    from inference_models import AutoModel

    model = AutoModel.from_pretrained(model_id, device=_resolve_device(device))

    def detect(frame: np.ndarray) -> sv.Detections:
        predictions = model(frame)
        if not predictions:
            return sv.Detections.empty()
        return predictions[0].to_supervision()

    return detect


def _load_ultralytics(model_id: str, device: str) -> Detector:
    if not _ensure_ultralytics_importable():
        raise RuntimeError(
            f"Ultralytics not found. Install it (pip install ultralytics) or set {ULTRALYTICS_DIR_ENV} "
            f"to the folder that contains it (default {DEFAULT_ULTRALYTICS_DIR})."
        )
    from ultralytics import YOLO

    model = YOLO(model_id)
    if model.task == "classify":
        raise RuntimeError(f"{Path(model_id).name} is a classification model; pick a detection model")
    resolved = _resolve_device(device)

    def detect(frame: np.ndarray) -> sv.Detections:
        results = list(model.predict(frame, verbose=False, device=resolved))
        if not results:
            return sv.Detections.empty()
        return sv.Detections.from_ultralytics(results[0])

    return detect


class DetectorCandidateSource:
    """Produce candidate detections from an object detector.

    Args:
        detector: Callable mapping a BGR frame to ``sv.Detections``.
        min_confidence: Detections below this confidence are dropped.
        same_class_only: Keep only detections of the target's class after selection.
    """

    reacquire_score = 0.5

    def __init__(self, detector: Detector, min_confidence: float = 0.15, same_class_only: bool = True) -> None:
        self._detector = detector
        self.min_confidence = min_confidence
        self.same_class_only = same_class_only
        self._class_id: int | None = None
        self._initialized = False

    @property
    def initialized(self) -> bool:
        """Whether a target was selected."""
        return self._initialized

    @property
    def target_class(self) -> int | None:
        """Class ID of the selected target, when the detector reports classes."""
        return self._class_id

    def initialize(self, frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        """Pick the detection under the gate ``box`` as the target.

        A detection containing the gate centre wins (the smallest such box, so a
        click on a person in front of a bus selects the person); otherwise the
        detection overlapping the gate the most is used.

        Args:
            frame: BGR frame the user clicked on.
            box: Gate ``x1, y1, x2, y2``.

        Returns:
            The selected detection box, or ``None`` when no detection touches the gate.
        """
        detections = self._detect(frame)
        if len(detections) == 0:
            return None
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        xyxy = detections.xyxy
        contains = (xyxy[:, 0] <= cx) & (cx <= xyxy[:, 2]) & (xyxy[:, 1] <= cy) & (cy <= xyxy[:, 3])
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        if contains.any():
            row = int(np.flatnonzero(contains)[np.argmin(areas[contains])])
        else:
            overlap = _intersection(xyxy, box)
            if overlap.max() <= 0:
                return None
            row = int(np.argmax(overlap))
        self._class_id = int(detections.class_id[row]) if detections.class_id is not None else None
        self._initialized = True
        return xyxy[row].astype(np.float32)

    def reset(self) -> None:
        """Forget the selected target."""
        self._class_id = None
        self._initialized = False

    def update_template(self, frame: np.ndarray, box: np.ndarray, score: float) -> None:
        """No appearance model to adapt; present for interface parity."""

    def generate(self, frame: np.ndarray, center: tuple[float, float], lost_frames: int = 0) -> Candidates:
        """Run the detector on ``frame`` and return target-class boxes.

        Args:
            frame: BGR frame.
            center: Unused; the detector sees the whole frame.
            lost_frames: Unused.

        Returns:
            Candidate boxes with detector confidences.
        """
        detections = self._detect(frame)
        if self.same_class_only and self._class_id is not None and detections.class_id is not None:
            detections = cast(sv.Detections, detections[detections.class_id == self._class_id])
        if len(detections) == 0:
            return Candidates.empty()
        confidence = detections.confidence if detections.confidence is not None else np.ones(len(detections))
        return Candidates(xyxy=detections.xyxy.astype(np.float32), scores=confidence.astype(np.float32))

    def _detect(self, frame: np.ndarray) -> sv.Detections:
        detections = self._detector(frame)
        if len(detections) and detections.confidence is not None:
            detections = cast(sv.Detections, detections[detections.confidence >= self.min_confidence])
        return detections


def _intersection(xyxy: np.ndarray, box: np.ndarray) -> np.ndarray:
    w = np.clip(np.minimum(xyxy[:, 2], box[2]) - np.maximum(xyxy[:, 0], box[0]), 0, None)
    h = np.clip(np.minimum(xyxy[:, 3], box[3]) - np.maximum(xyxy[:, 1], box[1]), 0, None)
    return w * h
