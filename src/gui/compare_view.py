"""Before/after image comparison widget with draggable slider."""

from __future__ import annotations

from enum import Enum, auto

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget


class CompareMode(Enum):
    SLIDER = auto()
    SIDE_BY_SIDE = auto()
    TOGGLE = auto()


class CompareView(QWidget):
    """Before/after image comparison with multiple view modes.

    SLIDER: single view with a vertical divider — left=before, right=after.
    SIDE_BY_SIDE: two images rendered next to each other.
    TOGGLE: click to toggle between before and after.
    """

    mode_changed = Signal(CompareMode)

    HANDLE_WIDTH = 4
    HANDLE_COLOR = QColor("#7c3aed")
    LABEL_BG = QColor(0, 0, 0, 160)
    LABEL_FG = QColor(255, 255, 255)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._before: QPixmap | None = None
        self._after: QPixmap | None = None
        self._mode = CompareMode.SLIDER
        self._slider_pos = 0.5  # 0.0 = all after, 1.0 = all before
        self._dragging = False
        self._showing_before = True  # For TOGGLE mode

        self.setMouseTracking(True)
        self.setMinimumSize(200, 150)

    def set_images(self, before: QPixmap, after: QPixmap) -> None:
        self._before = before
        self._after = after
        self._slider_pos = 0.5
        self._showing_before = True
        self.update()

    def set_mode(self, mode: CompareMode) -> None:
        self._mode = mode
        self.setCursor(
            Qt.CursorShape.SplitHCursor
            if mode == CompareMode.SLIDER
            else Qt.CursorShape.ArrowCursor
        )
        self.mode_changed.emit(mode)
        self.update()

    def clear(self) -> None:
        self._before = None
        self._after = None
        self.update()

    def has_images(self) -> bool:
        return self._before is not None and self._after is not None

    # --- Painting ---

    def paintEvent(self, event) -> None:
        if not self.has_images():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if self._mode == CompareMode.SLIDER:
            self._paint_slider(painter)
        elif self._mode == CompareMode.SIDE_BY_SIDE:
            self._paint_side_by_side(painter)
        elif self._mode == CompareMode.TOGGLE:
            self._paint_toggle(painter)

        painter.end()

    def _paint_slider(self, painter: QPainter) -> None:
        w, h = self.width(), self.height()
        split_x = int(w * self._slider_pos)

        # Scale pixmaps to fill widget
        before_scaled = self._before.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio)
        after_scaled = self._after.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio)

        # Center the scaled images
        bx = (w - before_scaled.width()) // 2
        by = (h - before_scaled.height()) // 2
        ax = (w - after_scaled.width()) // 2
        ay = (h - after_scaled.height()) // 2

        # Draw after (full), then before clipped to left of slider
        painter.drawPixmap(ax, ay, after_scaled)

        painter.setClipRect(QRect(0, 0, split_x, h))
        painter.drawPixmap(bx, by, before_scaled)
        painter.setClipping(False)

        # Draw slider handle
        pen = QPen(self.HANDLE_COLOR, self.HANDLE_WIDTH)
        painter.setPen(pen)
        painter.drawLine(split_x, 0, split_x, h)

        # Draw handle grip (small circle)
        painter.setBrush(self.HANDLE_COLOR)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPoint(split_x, h // 2), 10, 10)

        # Draw labels
        self._draw_label(painter, "Before", 10, 10)
        self._draw_label(painter, "After", w - 60, 10)

    def _paint_side_by_side(self, painter: QPainter) -> None:
        w, h = self.width(), self.height()
        half_w = w // 2 - 2

        before_scaled = self._before.scaled(
            half_w, h, Qt.AspectRatioMode.KeepAspectRatio
        )
        after_scaled = self._after.scaled(
            half_w, h, Qt.AspectRatioMode.KeepAspectRatio
        )

        bx = (half_w - before_scaled.width()) // 2
        by = (h - before_scaled.height()) // 2
        ax = half_w + 4 + (half_w - after_scaled.width()) // 2
        ay = (h - after_scaled.height()) // 2

        painter.drawPixmap(bx, by, before_scaled)
        painter.drawPixmap(ax, ay, after_scaled)

        # Divider line
        pen = QPen(QColor("#45475a"), 2)
        painter.setPen(pen)
        painter.drawLine(half_w + 1, 0, half_w + 1, h)

        self._draw_label(painter, "Before", 10, 10)
        self._draw_label(painter, "After", half_w + 14, 10)

    def _paint_toggle(self, painter: QPainter) -> None:
        w, h = self.width(), self.height()
        pixmap = self._before if self._showing_before else self._after
        scaled = pixmap.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio)
        x = (w - scaled.width()) // 2
        y = (h - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)

        label = "Before (click to toggle)" if self._showing_before else "After (click to toggle)"
        self._draw_label(painter, label, 10, 10)

    def _draw_label(self, painter: QPainter, text: str, x: int, y: int) -> None:
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)

        metrics = painter.fontMetrics()
        rect = metrics.boundingRect(text)
        bg_rect = QRect(x - 4, y - 2, rect.width() + 8, rect.height() + 4)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.LABEL_BG)
        painter.drawRoundedRect(bg_rect, 4, 4)

        painter.setPen(self.LABEL_FG)
        painter.drawText(x, y + rect.height() - 2, text)

    # --- Mouse Events ---

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._mode == CompareMode.SLIDER:
            self._dragging = True
            self._update_slider_pos(event.position().x())
        elif self._mode == CompareMode.TOGGLE:
            self._showing_before = not self._showing_before
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._mode == CompareMode.SLIDER and self._dragging:
            self._update_slider_pos(event.position().x())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging = False

    def _update_slider_pos(self, x: float) -> None:
        self._slider_pos = max(0.0, min(1.0, x / self.width()))
        self.update()
