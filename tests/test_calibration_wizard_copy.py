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


def test_edit_current_reason_label_keeps_base_copy_when_controller_sets_tooltip():
    QApplication.instance() or QApplication([])
    dialog = CalibrationWizard(None, AppState())
    try:
        dialog.show()
        QApplication.processEvents()
        dialog.set_edit_current_calibration_enabled(False, "Some controller tooltip")

        assert dialog.edit_current_reason_label.text() == (
            "Edit becomes available after you create key overlays."
        )
        assert dialog.edit_current_reason_label.isVisible()
        assert dialog.edit_current_calibration_button.toolTip() == "Some controller tooltip"
    finally:
        dialog.close()
        dialog.deleteLater()


def test_edit_current_calibration_action_uses_the_full_dialog_width():
    QApplication.instance() or QApplication([])
    dialog = CalibrationWizard(None, AppState())
    try:
        layout = dialog.layout()
        index = layout.indexOf(dialog.edit_current_calibration_button)

        assert layout.getItemPosition(index) == (2, 0, 1, 3)
        assert dialog.edit_current_calibration_button.maximumWidth() > 1000
    finally:
        dialog.close()
        dialog.deleteLater()
