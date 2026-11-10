# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_scene(t: int, objects: list[tuple[float, float, float, float]], size: int = 40) -> np.ndarray:
    """Render a frame with textured squares moving linearly over a static textured background.

    Args:
        t: Frame index.
        objects: ``(x0, y0, vx, vy)`` per object, top-left start and per-frame velocity.
        size: Object side length in pixels.
    """
    rng = np.random.default_rng(0)
    frame = rng.integers(60, 120, size=(240, 320, 3), dtype=np.uint8)
    for k, (x0, y0, vx, vy) in enumerate(objects):
        obj_rng = np.random.default_rng(100 + k)
        patch = obj_rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
        x, y = round(x0 + vx * t), round(y0 + vy * t)
        frame[y : y + size, x : x + size] = patch
    return frame


@pytest.fixture
def scene():
    return make_scene
