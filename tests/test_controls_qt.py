from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QGroupBox, QLabel, QPushButton, QScrollArea

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
        assert panel.conversion_status.text() == (
            "Capture a pressed-key example for Color 1 Natural."
        )

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


def test_conversion_readiness_names_enabled_higher_family_missing_color():
    state = AppState()
    panel = _panel_with_state(state)
    try:
        _prepare_conversion_ready_state(state)
        state.detection.exemplar_key_type_enabled["COLOR_4_B"] = True
        state.detection.exemplar_lit_colors["COLOR_4_B"] = None

        panel.update_controls_from_state()

        assert not panel.convert_button.isEnabled()
        assert panel.conversion_status.text() == (
            "Capture a pressed-key example for Color 4 Sharp / Flat."
        )
    finally:
        panel.close()
        panel.deleteLater()


def test_conversion_readiness_ignores_unchecked_higher_family_slot():
    state = AppState()
    panel = _panel_with_state(state)
    try:
        _prepare_conversion_ready_state(state)
        state.detection.exemplar_key_type_enabled["COLOR_4_B"] = False
        state.detection.exemplar_lit_colors["COLOR_4_B"] = None

        panel.update_controls_from_state()

        assert panel.convert_button.isEnabled()
        assert panel.conversion_status.text() == "Ready to create MIDI."
    finally:
        panel.close()
        panel.deleteLater()


def test_calibration_section_shows_visible_step_instructions():
    QApplication.instance() or QApplication([])
    panel = ControlPanelQt()
    try:
        assert panel.calibration_wizard_button.text() == "Draw Keyboard Box and Find Keys"
        assert (
            panel.calibration_wizard_button.minimumWidth()
            >= panel.calibration_wizard_button.sizeHint().width()
        )
        assert panel.calibrate_unlit_button.text() == "Capture No-Key Frame"
        assert panel.calibration_instruction_labels["keyboard"].text() == (
            "Pause on a clear frame where the full keyboard is visible."
        )
        assert panel.calibration_instruction_labels["unlit"].text() == "Pause where no keys are glowing."
        assert panel.calibration_instruction_labels["pressed"].text() == (
            "Pause where a key is glowing, then click that key."
        )
        calibration_page = panel.tab_widget.widget(1)
        button_texts = [
            button.text() for button in calibration_page.findChildren(QPushButton)
        ]
        assert "Set Left White" not in button_texts
        assert "Set Left Black" not in button_texts
    finally:
        panel.close()
        panel.deleteLater()


def test_calibration_panel_uses_dynamic_color_family_grid_and_forwards_actions():
    state = AppState()
    panel = _panel_with_state(state)
    calibrated = []
    added = []
    panel.calibrate_lit_exemplar_requested.connect(calibrated.append)
    panel.add_additional_color_requested.connect(lambda: added.append(True))
    try:
        assert panel.color_family_grid.family_heading(1).text() == "Color 1"
        assert panel.color_family_grid.family_heading(2).text() == "Color 2"
        assert panel.exemplar_buttons["LW"].text() == "Set"
        assert panel.color_family_grid.rows["LW"].label.text() == "Natural"
        assert panel.color_family_grid.rows["LB"].label.text() == "Sharp / Flat"
        assert 1 not in panel.color_family_grid.remove_family_buttons

        panel.exemplar_buttons["LW"].click()
        panel.color_family_grid.add_family_button.click()

        assert calibrated == ["LW"]
        assert added == [True]
    finally:
        panel.close()
        panel.deleteLater()


def test_calibration_panel_hides_add_at_four_families():
    state = AppState()
    state.detection.exemplar_key_type_enabled.update(
        {slot: True for slot in state.detection.exemplar_key_type_enabled}
    )
    panel = _panel_with_state(state)
    try:
        assert set(panel.color_family_grid.remove_family_buttons) == {2, 3, 4}
        assert not hasattr(panel.color_family_grid, "add_family_button")
    finally:
        panel.close()
        panel.deleteLater()


def test_one_family_state_hides_color_two_without_deleting_compatibility_slots():
    state = AppState()
    state.detection.exemplar_key_type_enabled["RW"] = False
    state.detection.exemplar_key_type_enabled["RB"] = False
    panel = _panel_with_state(state)
    try:
        assert list(panel.color_family_grid.rows) == ["LW", "LB"]
        assert "RW" in state.detection.exemplar_lit_colors
        assert "RB" in state.detection.exemplar_lit_histograms
        assert "RW" not in panel.exemplar_buttons
        assert "RB" not in panel.exemplar_presence_checkboxes
    finally:
        panel.close()
        panel.deleteLater()


def _all_label_texts(widget):
    return [label.text() for label in widget.findChildren(QLabel)]


def _all_group_titles(widget):
    return [group.title() for group in widget.findChildren(QGroupBox)]


def test_detection_section_uses_sensitivity_and_symptom_copy():
    QApplication.instance() or QApplication([])
    panel = ControlPanelQt()
    try:
        texts = _all_label_texts(panel)
        titles = _all_group_titles(panel)

        assert "Detection Sensitivity" in titles
        assert "Detection Threshold" not in titles
        assert "Detection Sensitivity:" in texts
        assert "Missing notes? Lower it. Extra notes? Raise it." in texts
    finally:
        panel.close()
        panel.deleteLater()


def test_spark_midi_trim_optional_sections_use_plain_recovery_copy():
    QApplication.instance() or QApplication([])
    panel = ControlPanelQt()
    try:
        texts = _all_label_texts(panel)
        titles = _all_group_titles(panel)

        assert "Use this only if repeated notes merge into one long note." in texts
        assert panel.spark_roi_select_button.text() == "Select Flash Area Above Keys"
        assert "Repeated Notes Fix" in titles
        assert "Repeated-Note Setup" in titles
        assert "Convert Only Part of the Video" in titles
        assert "This affects MIDI creation only. It does not trim or change the video session." in texts
        assert "Permanently Trim Project" in titles
        assert (
            "Most users should use MIDI range instead. Trim changes the working video session, not the original video file."
        ) in texts
        assert panel.trim_video_button.text() == "Permanently Trim Project"
        assert panel.hand_assignment_cb.text() == "Put each hand/color on a separate MIDI channel"
        assert "Use this only if the video uses different colors for left and right hand notes." in texts
    finally:
        panel.close()
        panel.deleteLater()


def test_advanced_settings_are_symptom_led_and_collapsed_by_default():
    QApplication.instance() or QApplication([])
    panel = ControlPanelQt()
    try:
        rail_labels = [
            panel.settings_section_rail.item(index).text()
            for index in range(panel.settings_section_rail.count())
        ]
        assert rail_labels == [
            "Guide",
            "Calibration",
            "Overlays",
            "Detection",
            "MIDI",
            "Advanced",
            "Optional",
            "Language",
        ]
        assert set(panel.advanced_sections) == {
            "histogram",
            "delta",
            "black_keys",
            "repeated_notes",
            "trim",
            "glossary",
        }
        assert all(not section._content.isVisible() for section in panel.advanced_sections.values())
        assert panel.advanced_sections["histogram"]._content.isAncestorOf(panel.histogram_detection_cb)
        assert panel.advanced_sections["delta"]._content.isAncestorOf(panel.delta_detection_cb)
        assert panel.advanced_sections["black_keys"]._content.isAncestorOf(panel.black_key_filter_cb)
        assert panel.advanced_sections["repeated_notes"]._content.isAncestorOf(
            panel.open_repeated_notes_tool_button
        )
        assert not panel.advanced_sections["repeated_notes"]._content.isAncestorOf(
            panel.spark_detection_cb
        )
        assert panel.advanced_sections["trim"]._content.isAncestorOf(panel.trim_video_button)

        detection_index = rail_labels.index("Detection")
        detection_page = panel.tab_widget.widget(detection_index)
        assert not detection_page.isAncestorOf(panel.histogram_detection_cb)
        assert not detection_page.isAncestorOf(panel.delta_detection_cb)
        assert not detection_page.isAncestorOf(panel.black_key_filter_cb)
    finally:
        panel.close()
        panel.deleteLater()


def test_repeated_notes_uses_one_dedicated_tool_scroller():
    QApplication.instance() or QApplication([])
    panel = ControlPanelQt()
    try:
        panel.show()
        panel.open_repeated_notes_tool_button.click()
        QApplication.processEvents()

        tool = panel.repeated_notes_tool_window
        assert tool.isVisible()
        assert tool.isAncestorOf(panel.spark_detection_cb)
        scroll_areas = tool.findChildren(QScrollArea)
        assert len(scroll_areas) == 1
        assert not any(
            outer is not inner and outer.isAncestorOf(inner)
            for outer in scroll_areas
            for inner in scroll_areas
        )
        advanced_index = [
            panel.settings_section_rail.item(index).text()
            for index in range(panel.settings_section_rail.count())
        ].index("Advanced")
        assert not panel.tab_widget.widget(advanced_index).isAncestorOf(
            panel.spark_detection_cb
        )
    finally:
        panel.repeated_notes_tool_window.close()
        panel.close()
        panel.deleteLater()
