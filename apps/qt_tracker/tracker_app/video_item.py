# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""QML item that paints the latest processed frame with aspect-ratio fit."""

from __future__ import annotations

from PySide6.QtCore import Property, QPointF, QRect, QRectF, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtQml import QmlElement
from PySide6.QtQuick import QQuickPaintedItem

QML_IMPORT_NAME = "TrackerApp"
QML_IMPORT_MAJOR_VERSION = 1


@QmlElement
class VideoFrameItem(QQuickPaintedItem):
    """Paint frames pushed from the controller and map view points to image pixels."""

    hasFrameChanged = Signal()
    frameSizeChanged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._image = QImage()
        self.setFillColor(QColor("#101114"))
        self.setRenderTarget(QQuickPaintedItem.RenderTarget.FramebufferObject)

    @Slot(QImage)
    def setFrame(self, image: QImage) -> None:
        """Display ``image``."""
        had_frame = not self._image.isNull()
        size_changed = image.size() != self._image.size()
        self._image = image
        if not had_frame:
            self.hasFrameChanged.emit()
        if size_changed:
            self.frameSizeChanged.emit()
        self.update()

    @Slot()
    def clear(self) -> None:
        """Remove the displayed frame."""
        self._image = QImage()
        self.hasFrameChanged.emit()
        self.update()

    def _get_has_frame(self) -> bool:
        return not self._image.isNull()

    hasFrame = Property(bool, _get_has_frame, notify=hasFrameChanged)

    def _get_frame_width(self) -> int:
        return self._image.width()

    def _get_frame_height(self) -> int:
        return self._image.height()

    frameWidth = Property(int, _get_frame_width, notify=frameSizeChanged)
    frameHeight = Property(int, _get_frame_height, notify=frameSizeChanged)

    def _target_rect(self) -> QRectF:
        if self._image.isNull() or self.width() <= 0 or self.height() <= 0:
            return QRectF()
        scale = min(self.width() / self._image.width(), self.height() / self._image.height())
        w, h = self._image.width() * scale, self._image.height() * scale
        return QRectF((self.width() - w) / 2.0, (self.height() - h) / 2.0, w, h)

    @Slot(float, float, result=QPointF)
    def mapToImage(self, x: float, y: float) -> QPointF:
        """Map an item coordinate to image pixels; returns ``(-1, -1)`` outside the image."""
        rect = self._target_rect()
        if rect.isEmpty() or not rect.contains(QPointF(x, y)):
            return QPointF(-1.0, -1.0)
        scale = rect.width() / self._image.width()
        return QPointF((x - rect.x()) / scale, (y - rect.y()) / scale)

    def paint(self, painter: QPainter) -> None:
        """Draw the current image letterboxed in the item."""
        if self._image.isNull():
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(self._target_rect(), self._image)

    def geometryChange(self, new_geometry: QRectF | QRect, old_geometry: QRectF | QRect) -> None:
        """Repaint on resize."""
        super().geometryChange(new_geometry, old_geometry)
        self.update()
