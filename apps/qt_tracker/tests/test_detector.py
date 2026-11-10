# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np
import pytest
import supervision as sv
from tracker_app.detector import DetectorCandidateSource
from tracker_app.engine import CBIOU, MCBYTE, TrackingEngine, TrackStatus

SIZE = 40
# (x0, y0, vx, vy, class_id): objects 0 and 2 share a class, object 1 differs.
OBJECTS = [(40.0, 60.0, 3.0, 1.0, 0), (200.0, 150.0, -2.0, -1.0, 2), (230.0, 30.0, -1.0, 2.0, 0)]


def boxes_at(t: int) -> np.ndarray:
    return np.array([[x + vx * t, y + vy * t, x + vx * t + SIZE, y + vy * t + SIZE] for x, y, vx, vy, _ in OBJECTS])


class FakeDetector:
    """Detector returning jittered ground-truth boxes for the frame index it is called on."""

    def __init__(self) -> None:
        self.t = 0
        self.rng = np.random.default_rng(1)

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        xyxy = boxes_at(self.t) + self.rng.normal(0.0, 1.0, size=(len(OBJECTS), 4))
        return sv.Detections(
            xyxy=xyxy.astype(np.float32),
            confidence=np.full(len(OBJECTS), 0.85, dtype=np.float32),
            class_id=np.array([o[4] for o in OBJECTS]),
        )


def frame_at(t: int) -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


def centre(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


@pytest.mark.parametrize("name", [CBIOU, MCBYTE])
def test_detector_source_follows_clicked_object(name: str) -> None:
    detector = FakeDetector()
    source = DetectorCandidateSource(detector)
    engine = TrackingEngine(tracker_name=name, source=source)
    click = centre(boxes_at(0)[0]) + 5.0
    first = engine.select_target(frame_at(0), *click)
    assert first.status is TrackStatus.TRACKING
    assert first.box is not None
    np.testing.assert_allclose(first.box[2:] - first.box[:2], [SIZE, SIZE], atol=4.0)
    assert source.target_class == 0
    for t in range(1, 30):
        detector.t = t
        result = engine.process(frame_at(t))
        assert result.status is TrackStatus.TRACKING, t
        assert result.target_id == first.target_id
        assert result.box is not None
        np.testing.assert_allclose(centre(result.box), centre(boxes_at(t)[0]), atol=3.0)
        # Only same-class boxes reach the tracker.
        assert len(result.candidates) == 2


def test_click_prefers_smallest_containing_box() -> None:
    def detector(frame: np.ndarray) -> sv.Detections:
        return sv.Detections(
            xyxy=np.array([[0, 0, 200, 200], [50, 50, 90, 90]], dtype=np.float32),
            confidence=np.array([0.9, 0.9], dtype=np.float32),
            class_id=np.array([5, 0]),
        )

    source = DetectorCandidateSource(detector)
    box = source.initialize(frame_at(0), np.array([55.0, 55.0, 85.0, 85.0]))
    assert box is not None
    np.testing.assert_allclose(box, [50, 50, 90, 90])
    assert source.target_class == 0


def test_click_on_nothing_falls_back_to_template() -> None:
    detector = FakeDetector()
    engine = TrackingEngine(source=DetectorCandidateSource(detector))
    frame = np.random.default_rng(3).integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    result = engine.select_target(frame, 20.0, 220.0)
    assert result.status is TrackStatus.TRACKING
    assert result.note
    assert engine.using_fallback
    engine.stop_tracking()
    assert not engine.using_fallback


def test_switch_source_keeps_target_position() -> None:
    detector = FakeDetector()
    engine = TrackingEngine()
    engine.select_target(frame_at(0), *centre(boxes_at(0)[0]))
    switched = engine.set_candidate_source(DetectorCandidateSource(detector))
    assert switched is not None
    assert switched.status is TrackStatus.TRACKING
    assert switched.box is not None
    np.testing.assert_allclose(centre(switched.box), centre(boxes_at(0)[0]), atol=3.0)
    back = engine.set_candidate_source(None)
    assert back is not None
    assert back.status is TrackStatus.TRACKING
