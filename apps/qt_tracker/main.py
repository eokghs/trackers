# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Entry point of the Video Object Tracker desktop app."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import tracker_app.video_item  # noqa: E402,F401  (registers the VideoFrameItem QML type)
from tracker_app.controller import AppController  # noqa: E402


def main() -> int:
    """Start the application."""
    app = QGuiApplication(sys.argv)
    app.setApplicationName("Video Object Tracker")
    app.setOrganizationName("trackers")
    QQuickStyle.setStyle("Fusion")

    controller = AppController()
    app.aboutToQuit.connect(controller.shutdown)

    engine = QQmlApplicationEngine()
    engine.setInitialProperties({"controller": controller})
    engine.load(APP_DIR / "qml" / "Main.qml")
    if not engine.rootObjects():
        return 1
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
