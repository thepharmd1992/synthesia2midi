from PySide6.QtWidgets import QApplication, QLabel

from synthesia2midi.core.app_state import AppState
from synthesia2midi.gui.wizard import CalibrationWizard


def test_calibration_wizard_uses_plain_keyboard_box_language():
    QApplication.instance() or QApplication([])
    dialog = CalibrationWizard(None, AppState())
    try:
        dialog.show()
        QApplication.processEvents()
        button_texts = [button.text() for button in dialog.findChildren(type(dialog.edit_current_calibration_button))]
        label_texts = [label.text() for label in dialog.findChildren(QLabel)]

        assert "Draw Keyboard Box and Find Keys" in button_texts
        assert "Select Keyboard Region With Autodetector" not in button_texts
        assert "Pause on a clear frame where the full keyboard is visible." in label_texts
        assert dialog.edit_current_reason_label.text() == (
            "Edit becomes available after you create key overlays."
        )
        assert dialog.edit_current_reason_label.isVisible()

        dialog.set_edit_current_calibration_enabled(True)

        assert dialog.edit_current_calibration_button.isEnabled()
        assert not dialog.edit_current_reason_label.isVisible()
    finally:
        dialog.close()
        dialog.deleteLater()
