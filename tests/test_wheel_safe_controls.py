from collections.abc import Callable

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from synthesia2midi.gui.wheel_safe_controls import install_wheel_safe_controls


def _slider() -> QSlider:
    widget = QSlider(Qt.Horizontal)
    widget.setRange(0, 10)
    widget.setValue(5)
    return widget


def _spinbox() -> QSpinBox:
    widget = QSpinBox()
    widget.setRange(0, 10)
    widget.setValue(5)
    return widget


def _double_spinbox() -> QDoubleSpinBox:
    widget = QDoubleSpinBox()
    widget.setRange(0.0, 10.0)
    widget.setValue(5.0)
    return widget


def _combo() -> QComboBox:
    widget = QComboBox()
    widget.addItems(["First", "Second", "Third"])
    widget.setCurrentIndex(1)
    return widget


@pytest.mark.parametrize(
    ("factory", "read_value"),
    [
        (_slider, lambda widget: widget.value()),
        (_spinbox, lambda widget: widget.value()),
        (_double_spinbox, lambda widget: widget.value()),
        (_combo, lambda widget: widget.currentIndex()),
    ],
)
def test_wheel_over_value_control_scrolls_page_without_changing_value(
    factory: Callable[[], QWidget],
    read_value: Callable[[QWidget], object],
):
    app = QApplication.instance() or QApplication([])
    install_wheel_safe_controls(app)

    scroll_area = QScrollArea()
    scroll_area.resize(320, 120)
    scroll_area.setWidgetResizable(True)
    content = QWidget()
    content.setMinimumHeight(600)
    content_layout = QVBoxLayout(content)
    control = factory()
    content_layout.addWidget(control)
    content_layout.addStretch(1)
    scroll_area.setWidget(content)
    scroll_area.show()
    QApplication.processEvents()

    control.setFocus()
    before = read_value(control)
    global_position = control.mapToGlobal(control.rect().center())
    wheel_event = QWheelEvent(
        QPointF(control.rect().center()),
        QPointF(global_position),
        QPoint(),
        QPoint(0, -120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.ScrollUpdate,
        False,
    )

    QApplication.sendEvent(control, wheel_event)
    QApplication.processEvents()

    assert read_value(control) == before
    assert scroll_area.verticalScrollBar().value() > 0
    scroll_area.close()
