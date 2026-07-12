import numpy as np
import pytest
from PySide6.QtCore import QSignalBlocker
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QSpinBox

from synthesia2midi.core.app_state import AppState
from synthesia2midi.gui.auto_detect_tuning_dialog import AutoDetectTuningDialog
from synthesia2midi.gui.controls_qt import ControlPanelQt
from synthesia2midi.gui.manual_keyboard_fit_dialog import ManualKeyboardFitDialog
from synthesia2midi.gui.wizard import CalibrationWizard
from synthesia2midi.localization import install_translator


def _surfaces():
    return [
        ControlPanelQt(app_state=AppState()),
        ManualKeyboardFitDialog(),
        CalibrationWizard(None, AppState()),
        AutoDetectTuningDialog(
            None,
            AppState(),
            np.zeros((16, 32, 3), dtype=np.uint8),
            (0, 0, 32, 16),
            initial_detection_results={"total_keys": 88},
            fallback_used=False,
            apply_detection_callback=lambda _result: True,
        ),
    ]


@pytest.mark.parametrize(("locale_name", "font_scale"), [("en", 1.0), ("qps", 1.5)])
def test_all_numeric_fields_fit_minimum_and_maximum(locale_name, font_scale):
    app = QApplication.instance() or QApplication([])
    original_font = QFont(app.font())
    scaled_font = QFont(original_font)
    base_size = original_font.pointSizeF() if original_font.pointSizeF() > 0 else 13.0
    scaled_font.setPointSizeF(base_size * font_scale)
    install_translator(app, locale_name)
    app.setFont(scaled_font)
    surfaces = _surfaces()
    try:
        for surface in surfaces:
            surface.show()
        QApplication.processEvents()

        for surface in surfaces:
            spinboxes = [
                *surface.findChildren(QSpinBox),
                *surface.findChildren(QDoubleSpinBox),
            ]
            assert spinboxes
            for spinbox in spinboxes:
                for value in (spinbox.minimum(), spinbox.maximum()):
                    blocker = QSignalBlocker(spinbox)
                    spinbox.setValue(value)
                    QApplication.processEvents()
                    required = spinbox.lineEdit().fontMetrics().horizontalAdvance(
                        spinbox.text()
                    )
                    available = spinbox.lineEdit().contentsRect().width()
                    assert available >= required, (
                        f"{surface.__class__.__name__} {spinbox.objectName() or spinbox.__class__.__name__} "
                        f"clips {spinbox.text()!r}: {available} < {required}"
                    )
                    del blocker
    finally:
        for surface in surfaces:
            surface.close()
            surface.deleteLater()
        install_translator(app, "en")
        app.setFont(original_font)


def test_manual_fit_numeric_fields_expand_for_large_font_metrics():
    app = QApplication.instance() or QApplication([])
    original_font = QFont(app.font())
    large_font = QFont(original_font)
    large_font.setPointSizeF(24.0)
    app.setFont(large_font)
    dialog = ManualKeyboardFitDialog()
    try:
        dialog.show()
        app.processEvents()
        spinboxes = [
            dialog.octave_spinbox,
            *dialog.param_spinboxes.values(),
            *dialog.local_param_spinboxes.values(),
        ]
        for spinbox in spinboxes:
            for value in (spinbox.minimum(), spinbox.maximum()):
                spinbox.setValue(value)
                app.processEvents()
                required = spinbox.lineEdit().fontMetrics().horizontalAdvance(spinbox.text())
                available = spinbox.lineEdit().contentsRect().width()
                assert available >= required
    finally:
        dialog.close()
        dialog.deleteLater()
        app.setFont(original_font)
        app.processEvents()
