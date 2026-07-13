from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from synthesia2midi.gui.startup_dialog import StartupDialog


def test_startup_dialog_language_selector_is_prominent_and_saves_restart_notice(monkeypatch, tmp_path):
    from synthesia2midi.localization import APP_LOCALE_SETTINGS_KEY, locale_display_name, supported_user_locales

    QApplication.instance() or QApplication([])
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path))
    settings = QSettings(QSettings.IniFormat, QSettings.UserScope, "Synthesia2MIDI", "test-startup-language")
    notices = []
    monkeypatch.setattr(
        "synthesia2midi.gui.startup_dialog.QMessageBox.information",
        lambda parent, title, message: notices.append((title, message)),
    )

    dialog = StartupDialog(settings=settings)

    try:
        layout = dialog.layout()
        assert layout.indexOf(dialog.title_label) < layout.indexOf(dialog.language_widget)
        assert layout.indexOf(dialog.language_widget) < layout.indexOf(dialog.subtitle_label)

        locale_values = [
            dialog.language_combo.itemData(index)
            for index in range(dialog.language_combo.count())
        ]
        locale_labels = [
            dialog.language_combo.itemText(index)
            for index in range(dialog.language_combo.count())
        ]

        assert locale_values == supported_user_locales()
        assert locale_labels == [locale_display_name(locale_name) for locale_name in supported_user_locales()]
        assert "qps" not in locale_values
        assert dialog.language_combo.currentData() == "en"

        dialog.language_combo.setCurrentIndex(locale_values.index("es"))

        assert settings.value(APP_LOCALE_SETTINGS_KEY) == "es"
        assert notices == [
            (
                "Language",
                "Restart Synthesia2MIDI to apply the selected language.",
            )
        ]
    finally:
        dialog.close()
        dialog.deleteLater()


def test_startup_dialog_explains_suitable_input_without_tooltip():
    QApplication.instance() or QApplication([])
    dialog = StartupDialog()
    try:
        assert not hasattr(dialog, "midi_touchup_button")
        assert [dialog.local_file_btn.text(), dialog.youtube_btn.text()] == [
            "Open Video File",
            "Download from YouTube",
        ]
        assert dialog.input_cue_label.text() == (
            "Choose a Synthesia-style piano video with visible keys and falling notes."
        )
        assert dialog.input_cue_label.wordWrap()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_startup_source_buttons_request_actions_without_closing_dialog():
    QApplication.instance() or QApplication([])
    dialog = StartupDialog()
    requests = []
    finished = []
    dialog.open_local_file.connect(lambda: requests.append("local"))
    dialog.download_from_youtube.connect(lambda: requests.append("youtube"))
    dialog.finished.connect(finished.append)

    try:
        dialog.local_file_btn.click()
        dialog.youtube_btn.click()

        assert requests == ["local", "youtube"]
        assert finished == []
    finally:
        dialog.close()
        dialog.deleteLater()
