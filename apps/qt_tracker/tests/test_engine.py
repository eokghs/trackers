# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np
import pytest
from tracker_app.candidates import TemplateCandidateGenerator, box_from_center, clip_box
from tracker_app.engine import CBIOU, MCBYTE, TrackingEngine, TrackStatus, annotate, build_tracker

OBJECTS = [(40.0, 60.0, 3.0, 1.0), (200.0, 150.0, -2.0, -1.0)]


def center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def truth(t: int, k: int, size: int = 40) -> np.ndarray:
    x0, y0, vx, vy = OBJECTS[k]
    return np.array([x0 + vx * t + size / 2.0, y0 + vy * t + size / 2.0])


def test_clip_box_keeps_size_inside_image() -> None:
    box = clip_box(np.array([-10.0, 230.0, 20.0, 260.0]), 320, 240)
    np.testing.assert_allclose(box, [0.0, 210.0, 30.0, 240.0])


def test_generator_finds_template_peak(scene) -> None:
    generator = TemplateCandidateGenerator()
    frame0 = scene(0, OBJECTS)
    generator.initialize(frame0, box_from_center(*truth(0, 0), 30, 30))
    frame5 = scene(5, OBJECTS)
    candidates = generator.generate(frame5, tuple(truth(0, 0)))
    assert len(candidates) >= 1
    assert candidates.scores[0] > 0.9
    np.testing.assert_allclose(center(candidates.xyxy[0]), truth(5, 0), atol=1.5)


@pytest.mark.parametrize("name", [CBIOU, MCBYTE])
def test_engine_follows_clicked_target(scene, name: str) -> None:
    engine = TrackingEngine(tracker_name=name)
    first = engine.select_target(scene(0, OBJECTS), *truth(0, 0))
    assert first.status is TrackStatus.TRACKING
    assert first.box is not None
    np.testing.assert_allclose(first.box[2:] - first.box[:2], [30, 30])
    target = first.target_id
    for t in range(1, 40):
        result = engine.process(scene(t, OBJECTS))
        assert result.status is TrackStatus.TRACKING, t
        assert result.target_id == target
        assert result.box is not None
        np.testing.assert_allclose(center(result.box), truth(t, 0), atol=2.0)


def test_switch_target_and_stop(scene) -> None:
    engine = TrackingEngine()
    engine.select_target(scene(0, OBJECTS), *truth(0, 0))
    for t in range(1, 5):
        engine.process(scene(t, OBJECTS))
    switched = engine.select_target(scene(5, OBJECTS), *truth(5, 1))
    assert switched.status is TrackStatus.TRACKING
    for t in range(6, 20):
        result = engine.process(scene(t, OBJECTS))
    assert result.box is not None
    np.testing.assert_allclose(center(result.box), truth(19, 1), atol=2.0)

    engine.stop_tracking()
    assert not engine.active
    assert engine.process(scene(20, OBJECTS)).status is TrackStatus.IDLE


def test_gate_resize_applies_to_active_target(scene) -> None:
    engine = TrackingEngine()
    engine.select_target(scene(0, OBJECTS), *truth(0, 0))
    engine.process(scene(1, OBJECTS))
    resized = engine.set_gate_size(20, 36)
    assert resized is not None
    assert resized.box is not None
    np.testing.assert_allclose(resized.box[2:] - resized.box[:2], [20, 36])
    result = engine.process(scene(2, OBJECTS))
    assert result.status is TrackStatus.TRACKING
    assert result.box is not None
    np.testing.assert_allclose(result.box[2:] - result.box[:2], [20, 36])
    np.testing.assert_allclose(center(result.box), truth(2, 0), atol=2.0)


def test_target_lost_when_it_disappears(scene) -> None:
    engine = TrackingEngine()
    engine.select_target(scene(0, OBJECTS), *truth(0, 0))
    for t in range(1, 5):
        engine.process(scene(t, OBJECTS))
    statuses = [engine.process(scene(5, OBJECTS[1:])).status for _ in range(3)]
    assert TrackStatus.LOST in statuses


def test_switch_tracker_keeps_target(scene) -> None:
    engine = TrackingEngine()
    engine.select_target(scene(0, OBJECTS), *truth(0, 0))
    engine.process(scene(1, OBJECTS))
    assert engine.set_tracker(MCBYTE) is not None
    result = engine.process(scene(2, OBJECTS))
    assert result.status is TrackStatus.TRACKING
    assert result.box is not None
    np.testing.assert_allclose(center(result.box), truth(2, 0), atol=2.0)


def test_annotate_draws_on_copy(scene) -> None:
    engine = TrackingEngine()
    frame = scene(0, OBJECTS)
    result = engine.select_target(frame, *truth(0, 0))
    out = annotate(frame, result, show_candidates=True)
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)


def test_unknown_tracker_name() -> None:
    with pytest.raises(ValueError):
        build_tracker("SORT")
