from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.gui.controls_qt import ControlPanelQt
from synthesia2midi.core.app_state import AppState


def test_edit_midi_button_emits_touchup_request_directly():
    QApplication.instance() or QApplication([])
    panel = ControlPanelQt()
    emitted = []

    panel.midi_touchup_requested.connect(lambda: emitted.append(True))
    panel.midi_touchup_button.click()

    assert emitted == [True]


def test_language_selector_is_bottom_settings_section_and_saves_restart_notice(monkeypatch, tmp_path):
    from synthesia2midi.localization import APP_LOCALE_SETTINGS_KEY, locale_display_name, supported_user_locales

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

        assert rail_labels[-2:] == ["Optional", "Language"]
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

        assert locale_values == supported_user_locales()
        assert locale_labels == [locale_display_name(locale_name) for locale_name in supported_user_locales()]
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


def _panel_with_state(app_state: AppState) -> ControlPanelQt:
    QApplication.instance() or QApplication([])
    return ControlPanelQt(app_state=app_state)


def _basic_overlay(*, unlit=True, unlit_hist=None) -> OverlayConfig:
    return OverlayConfig(
        key_id=1,
        note_octave=4,
        note_name_in_octave="C",
        x=0,
        y=0,
        width=10,
        height=40,
        key_type="white",
        unlit_reference_color=(12, 12, 12) if unlit else None,
        unlit_hist=unlit_hist,
    )


def _calibrate_all_exemplars(app_state: AppState) -> None:
    app_state.detection.exemplar_lit_colors = {
        "LW": (255, 0, 0),
        "LB": (160, 0, 0),
        "RW": (0, 120, 255),
        "RB": (0, 70, 180),
    }


def _prepare_conversion_ready_state(
    state: AppState,
    *,
    use_histogram_detection: bool = False,
    detection_threshold: float = 0.8,
    tempo: int = 120,
    unlit_hist=None,
) -> None:
    state.video.filepath = "/tmp/source.mp4"
    state.overlays = [_basic_overlay(unlit=True, unlit_hist=unlit_hist)]
    _calibrate_all_exemplars(state)
    state.detection.use_histogram_detection = use_histogram_detection
    state.detection.detection_threshold = detection_threshold
    state.midi.tempo = tempo


def test_conversion_readiness_explains_first_missing_prerequisite():
    state = AppState()
    panel = _panel_with_state(state)
    try:
        assert not panel.convert_button.isEnabled()
        assert panel.conversion_status.text() == "Load a video to convert."

        state.video.filepath = "/tmp/source.mp4"
        panel.update_controls_from_state()
        assert not panel.convert_button.isEnabled()
        assert panel.conversion_status.text() == "Create key overlays first."

        state.overlays = [_basic_overlay(unlit=False)]
        panel.update_controls_from_state()
        assert not panel.convert_button.isEnabled()
        assert panel.conversion_status.text() == "Capture a no-key frame."

        state.overlays = [_basic_overlay(unlit=True)]
        panel.update_controls_from_state()
        assert not panel.convert_button.isEnabled()
        assert panel.conversion_status.text() == "Capture at least one pressed-key example."

        _calibrate_all_exemplars(state)
        panel.update_controls_from_state()
        assert panel.convert_button.isEnabled()
        assert panel.conversion_status.text() == "Ready to create MIDI."
    finally:
        panel.close()
        panel.deleteLater()


def test_conversion_readiness_requires_histogram_data_when_histogram_detection_is_enabled():
    state = AppState()
    panel = _panel_with_state(state)
    try:
        _prepare_conversion_ready_state(state, use_histogram_detection=True, unlit_hist=None)

        panel.update_controls_from_state()

        assert not panel.convert_button.isEnabled()
        assert panel.conversion_status.text() == "Capture a no-key frame."
    finally:
        panel.close()
        panel.deleteLater()


def test_conversion_readiness_rejects_detection_threshold_out_of_range():
    state = AppState()
    panel = _panel_with_state(state)
    try:
        _prepare_conversion_ready_state(state, detection_threshold=1.5)

        panel.update_controls_from_state()

        assert not panel.convert_button.isEnabled()
        assert panel.conversion_status.text() == "Check detection sensitivity."
    finally:
        panel.close()
        panel.deleteLater()


def test_conversion_readiness_rejects_non_positive_midi_tempo():
    state = AppState()
    panel = _panel_with_state(state)
    try:
        _prepare_conversion_ready_state(state, tempo=0)

        panel.update_controls_from_state()

        assert not panel.convert_button.isEnabled()
        assert panel.conversion_status.text() == "Check MIDI tempo."
    finally:
        panel.close()
        panel.deleteLater()
