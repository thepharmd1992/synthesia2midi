from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from synthesia2midi.gui.controls_qt import ControlPanelQt


def test_edit_midi_button_emits_touchup_request_directly():
    QApplication.instance() or QApplication([])
    panel = ControlPanelQt()
    emitted = []

    panel.midi_touchup_requested.connect(lambda: emitted.append(True))
    panel.midi_touchup_button.click()

    assert emitted == [True]


def test_language_selector_is_first_class_section_and_saves_restart_notice(monkeypatch, tmp_path):
    from synthesia2midi.localization import APP_LOCALE_SETTINGS_KEY

    QApplication.instance() or QApplication([])
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path))
    settings = QSettings(QSettings.IniFormat, QSettings.UserScope, "Synthesia2MIDI", "test-language-selector")
    notices = []
    monkeypatch.setattr(
        "synthesia2midi.gui.controls_qt.QMessageBox.information",
        lambda parent, title, message: notices.append((title, message)),
    )

    panel = ControlPanelQt(settings=settings)

    try:
        rail_labels = [
            panel.settings_section_rail.item(index).text()
            for index in range(panel.settings_section_rail.count())
        ]
        language_section_index = rail_labels.index("Language")
        optional_section_index = rail_labels.index("Optional")

        assert rail_labels[0] == "Language"
        assert language_section_index != optional_section_index
        assert panel.tab_widget.widget(language_section_index).findChild(type(panel.language_combo), "language_combo") is panel.language_combo
        assert panel.tab_widget.widget(optional_section_index).findChild(type(panel.language_combo), "language_combo") is None

        locale_values = [
            panel.language_combo.itemData(index)
            for index in range(panel.language_combo.count())
        ]
        locale_labels = [
            panel.language_combo.itemText(index)
            for index in range(panel.language_combo.count())
        ]

        assert locale_values == ["en", "es"]
        assert locale_labels == ["English", "Español"]
        assert "qps" not in locale_values
        assert panel.language_combo.currentData() == "en"

        panel.language_combo.setCurrentIndex(locale_values.index("es"))

        assert settings.value(APP_LOCALE_SETTINGS_KEY) == "es"
        assert notices == [
            (
                "Language",
                "Restart Synthesia2MIDI to apply the selected language.",
            )
        ]
    finally:
        panel.close()
        panel.deleteLater()
