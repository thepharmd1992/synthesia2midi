"""Prevent touchpad and mouse-wheel scrolling from changing form values."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QSlider,
    QWidget,
)


class WheelSafeControlsFilter(QObject):
    """Route wheel gestures over value controls to their nearest scroller."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() != QEvent.Wheel:
            return super().eventFilter(obj, event)
        if not isinstance(obj, (QSlider, QAbstractSpinBox, QComboBox)):
            return super().eventFilter(obj, event)
        if isinstance(obj, QComboBox) and obj.view().isVisible():
            return super().eventFilter(obj, event)

        scroll_area = self._nearest_scroll_area(obj)
        if scroll_area is not None:
            QApplication.sendEvent(scroll_area.viewport(), QWheelEvent(event))
        event.accept()
        return True

    @staticmethod
    def _nearest_scroll_area(widget: QWidget) -> QAbstractScrollArea | None:
        ancestor = widget.parentWidget()
        while ancestor is not None:
            if isinstance(ancestor, QAbstractScrollArea):
                return ancestor
            ancestor = ancestor.parentWidget()
        return None


def install_wheel_safe_controls(app: QApplication) -> WheelSafeControlsFilter:
    """Install the application-wide wheel guard once and return it."""
    existing = getattr(app, "_wheel_safe_controls_filter", None)
    if existing is not None:
        return existing

    wheel_filter = WheelSafeControlsFilter(app)
    app.installEventFilter(wheel_filter)
    app._wheel_safe_controls_filter = wheel_filter
    return wheel_filter
