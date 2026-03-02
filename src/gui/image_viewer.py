"""Zoomable, pannable image viewer using QGraphicsView."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)


class ImageViewer(QGraphicsView):
    """Image display widget with zoom/pan support."""

    zoom_changed = Signal(float)

    ZOOM_IN_FACTOR = 1.25
    ZOOM_OUT_FACTOR = 0.8
    MIN_ZOOM = 0.1
    MAX_ZOOM = 20.0

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._zoom_level = 1.0

        self.setScene(self._scene)
        self.setRenderHints(
            self.renderHints()
            | self.renderHints().__class__.Antialiasing
            | self.renderHints().__class__.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(Qt.GlobalColor.black)

    def set_image(self, pixmap: QPixmap) -> None:
        """Display an image, fitting it to the viewport."""
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(pixmap.rect().toRectF())
        self._zoom_level = 1.0
        self.fit_to_view()

    def clear(self) -> None:
        self._scene.clear()
        self._pixmap_item = None
        self._zoom_level = 1.0

    def has_image(self) -> bool:
        return self._pixmap_item is not None

    def get_current_pixmap(self) -> QPixmap | None:
        if self._pixmap_item:
            return self._pixmap_item.pixmap()
        return None

    def fit_to_view(self) -> None:
        if self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom_level = 1.0
            self.zoom_changed.emit(self._zoom_level * 100)

    def zoom_in(self) -> None:
        self._apply_zoom(self.ZOOM_IN_FACTOR)

    def zoom_out(self) -> None:
        self._apply_zoom(self.ZOOM_OUT_FACTOR)

    def _apply_zoom(self, factor: float) -> None:
        new_zoom = self._zoom_level * factor
        if self.MIN_ZOOM <= new_zoom <= self.MAX_ZOOM:
            self.scale(factor, factor)
            self._zoom_level = new_zoom
            self.zoom_changed.emit(self._zoom_level * 100)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self._apply_zoom(self.ZOOM_IN_FACTOR)
            else:
                self._apply_zoom(self.ZOOM_OUT_FACTOR)
        else:
            super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.fit_to_view()
        super().mouseDoubleClickEvent(event)
