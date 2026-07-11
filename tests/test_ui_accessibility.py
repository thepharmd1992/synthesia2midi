from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication

from synthesia2midi.core.app_state import AppState
from synthesia2midi.gui.controls_qt import ControlPanelQt
from synthesia2midi.gui.startup_dialog import StartupDialog
from synthesia2midi.gui.wizard import CalibrationWizard
from synthesia2midi.gui.youtube_download_dialog import YouTubeDownloadDialog
from synthesia2midi.main import Video2MidiApp


def _next(widget):
    candidate = widget.nextInFocusChain()
    while candidate is not widget and candidate.focusPolicy() == Qt.NoFocus:
        candidate = candidate.nextInFocusChain()
    return candidate


def test_small_adjustment_targets_have_names_and_are_at_least_36_pixels():
    QApplication.instance() or QApplication([])
    panel = ControlPanelQt()
    names = [
        "calibration_wizard_button",
        "calibrate_unlit_button",
        "white_height_dec_button",
        "white_height_inc_button",
        "white_width_dec_button",
        "white_width_inc_button",
        "black_height_dec_button",
        "black_height_inc_button",
        "black_width_dec_button",
        "black_width_inc_button",
        "left_slant_dec_button",
        "left_slant_inc_button",
        "right_slant_dec_button",
        "right_slant_inc_button",
    ]
    try:
        for name in names:
            button = getattr(panel, name)
            assert button.minimumWidth() >= 36
            assert button.minimumHeight() >= 36
            if button.text() in {"+", "-"}:
                assert button.accessibleName()
    finally:
        panel.close()


def test_main_settings_target_is_40_pixels_and_accessibly_named(monkeypatch):
    QApplication.instance() or QApplication([])
    monkeypatch.setattr(QTimer, "singleShot", lambda *_args, **_kwargs: None)
    app = Video2MidiApp()
    try:
        assert app.settings_toggle_button.width() >= 40
        assert app.settings_toggle_button.height() >= 40
        assert app.settings_toggle_button.accessibleName() == "Settings"
    finally:
        app.close()


def test_required_dialogs_define_beginner_path_focus_order():
    QApplication.instance() or QApplication([])
    startup = StartupDialog()
    wizard = CalibrationWizard(None, AppState())
    youtube = YouTubeDownloadDialog(default_output_dir="/tmp")
    panel = ControlPanelQt()
    try:
        assert _next(startup.local_file_btn) is startup.youtube_btn
        assert _next(wizard.auto_selection_button) is wizard.edit_current_calibration_button
        assert _next(wizard.edit_current_calibration_button) is wizard.leftmost_note_combo
        assert _next(youtube.url_input) is youtube.fetch_info_btn
        assert _next(youtube.fetch_info_btn) is youtube.quality_combo
        guide_buttons = [row.primary_button for row in panel.guide_page.step_rows]
        assert _next(panel.settings_section_rail) is guide_buttons[0]
        assert _next(guide_buttons[0]) is panel.guide_page.youtube_button
        assert _next(panel.guide_page.youtube_button) is guide_buttons[1]
        assert _next(guide_buttons[1]) is guide_buttons[2]
        assert _next(guide_buttons[2]) is guide_buttons[3]
        assert _next(guide_buttons[3]) is guide_buttons[4]
        assert _next(guide_buttons[4]) is panel.convert_button
    finally:
        startup.close()
        wizard.close()
        youtube.close()
        panel.close()


def test_custom_status_styles_do_not_use_known_low_contrast_tokens():
    QApplication.instance() or QApplication([])
    panel = ControlPanelQt()
    try:
        styles = "\n".join(widget.styleSheet() for widget in panel.findChildren(type(panel.unlit_status_label)))
        assert "#888" not in styles
        assert "#4CAF50" not in styles
        assert "color: grey" not in styles
        assert "color: green" not in styles
    finally:
        panel.close()
