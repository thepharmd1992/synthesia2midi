from pathlib import Path
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

from PySide6.QtCore import QCoreApplication
from PySide6.QtCore import QSettings
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


EXPECTED_USER_LOCALE_DISPLAY_NAMES = [
    ("en", "English"),
    ("es", "Español"),
    ("ja", "日本語"),
    ("ru", "Русский"),
    ("zh_CN", "简体中文"),
    ("ko", "한국어"),
    ("pt_BR", "Português (Brasil)"),
]

TASK_5_PRODUCTION_LOCALE_STRINGS = [
    "Before tuning detection: capture a no-key frame and at least one pressed-key example.",
    "Detection Sensitivity: main setting for pressed vs unpressed keys.",
    "Missing notes? Lower it. Extra notes? Raise it.",
    "Histogram Detection helps when pressed colors have gradients or uneven lighting.",
    "Delta Detection helps when pressed colors fade in or out instead of switching cleanly.",
    "Black Key Filter reduces false black-key notes caused by nearby overlays.",
    "Detection Sensitivity",
    "Detection Sensitivity:",
    "Select Spark Area Above Keys",
    "Select the area above the keys where spark bars and flashes appear.",
    "Show or hide the spark area overlay on the video.",
    "Convert Only Part of the Video",
    "This affects MIDI creation only. It does not trim or change the video session.",
    "Permanently Trim Project",
    "Most users should use MIDI range instead. Trim changes the working video session, not the original video file.",
    "Put each Color family on a separate MIDI channel",
    "Use this when the video contains more than one Color family.",
    (
        "<b>This will permanently trim the working video session.</b><br><br>"
        "Frames outside {start_frame} to {end_text} will be unavailable in this project session.<br><br>"
        "Most users should cancel and use the MIDI range controls instead."
    ),
    "Trim Project",
    "Use this only if repeated notes merge into one long note.",
]

TASK_6_PRODUCTION_LOCALE_STRINGS = [
    "If YouTube blocks the download",
    "Synthesia2MIDI can retry using saved browser cookies only if YouTube blocks the normal download.",
    "1080p - recommended for best MIDI detection",
    "720p - faster, may be less accurate",
    "480p - fastest, highest risk of bad calibration",
    "recommended for best MIDI detection",
    "faster, may be less accurate",
    "fastest, highest risk of bad calibration",
    "Up to {preset} ({actual_height}p source) - {note}",
]

SCANNER_WARNING_PRODUCTION_LOCALE_STRINGS = [
    "More than four stable color families were found.",
    "Evidence conflicts with two saved color family identities.",
]

TASK_10_PRODUCTION_LOCALE_STRINGS = [
    "Color {number}",
    "Natural",
    "Sharp / Flat",
    "Add Color Family",
    "Found",
    "Missing",
    "Set",
    "Present",
    "Remove Color {number}",
    "Remove Color {number} and delete its saved calibration data?",
    "Capture a pressed-key example for {label}.",
    (
        "3) Capture Pressed-Key Examples: for each Color family, capture a "
        "Natural and Sharp / Flat example that appears in the video."
    ),
    "Found {count} Synthesia note color families.",
    "Scanning for lit key examples...",
    (
        "Start from the no-key frame. The scan looks ahead for Natural and "
        "Sharp / Flat examples in every enabled Color family."
    ),
    (
        "Color family: one Synthesia note color, with separate Natural and "
        "Sharp / Flat examples."
    ),
    (
        "Click a glowing key that matches {label}. The application will sample "
        "the color and histogram for {label}."
    ),
    *SCANNER_WARNING_PRODUCTION_LOCALE_STRINGS,
]

SPARK_AUTO_CALIBRATION_PRODUCTION_LOCALE_STRINGS = [
    "Color {number}",
    "Natural",
    "Sharp / Flat",
    "Auto-Calibrate {label}",
    (
        "Auto-calibration for {label} started.\n\n"
        "1. Navigate to a frame where a {label} key first turns on.\n"
        "2. Click that key overlay.\n"
        "3. The application will capture the bar-only frame, dimmest sparks, "
        "and brightest sparks, then save the calibration."
    ),
]
SPARK_AUTO_CALIBRATION_RUNTIME_STRINGS = (
    SPARK_AUTO_CALIBRATION_PRODUCTION_LOCALE_STRINGS[-2:]
)

TOUCHUP_ACCESS_PRODUCTION_LOCALE_STRINGS = [
    ("Video2MidiApp", "Open MIDI in Touch-Up Editor..."),
    ("ControlPanelQt", "Open Touch-Up Editor"),
]

FINAL_REVIEW_RUNTIME_STRINGS = [
    (
        "CalibrationGuideWidget",
        "Start from the no-key frame. The scan looks ahead for Natural and "
        "Sharp / Flat examples in every enabled Color family.",
    ),
    (
        "ControlPanelQt",
        "Put each Color family on a separate MIDI channel",
    ),
    (
        "ControlPanelQt",
        "Use this when the video contains more than one Color family.",
    ),
    (
        "UiGlossary",
        "Color family: one Synthesia note color, with separate Natural and "
        "Sharp / Flat examples.",
    ),
    (
        "CalibrationWorkflow",
        "Click a glowing key that matches {label}. The application will sample "
        "the color and histogram for {label}.",
    ),
]

FINAL_REVIEW_COPY_FILES = [
    Path("synthesia2midi/synthesia2midi/workflows/calibration.py"),
    Path("synthesia2midi/synthesia2midi/gui/controls_qt.py"),
    Path("synthesia2midi/synthesia2midi/gui/calibration_guide.py"),
    Path("synthesia2midi/synthesia2midi/gui/ui_glossary.py"),
    Path("synthesia2midi/synthesia2midi/gui/spark_calibration_controller.py"),
    Path("synthesia2midi/synthesia2midi/workflows/auto_calibration.py"),
]

OBSOLETE_FAMILY_COPY = [
    "Left Hand White",
    "Left Hand Black",
    "Right Hand White",
    "Right Hand Black",
    "Left/Right color family",
    "the two Synthesia note colors",
    "different colors for left and right hand notes",
    "Put each hand/color",
    "System detects hand via hue analysis",
    "Detect if it's left/right hand based on color",
    "Key Type Legend",
]


def _production_translation_locales():
    from synthesia2midi.localization import supported_user_locales

    return [locale for locale in supported_user_locales() if locale != "en"]


def _catalog_messages(locale_name):
    ts_path = Path(f"synthesia2midi/synthesia2midi/translations/synthesia2midi_{locale_name}.ts")
    tree = ET.parse(ts_path)
    for context in tree.findall("context"):
        context_name = context.findtext("name") or ""
        for message in context.findall("message"):
            translation = message.find("translation")
            yield context_name, message.findtext("source") or "", translation


def test_available_locales_includes_english_and_pseudo_locale():
    from synthesia2midi.localization import available_locales

    locales = available_locales()
    assert "en" in locales
    assert "qps" in available_locales()
    for locale_name, _display_name in EXPECTED_USER_LOCALE_DISPLAY_NAMES:
        assert locale_name in locales


def test_translation_dir_points_to_package_translations():
    from synthesia2midi.localization import translation_dir

    path = translation_dir()

    assert path.name == "translations"
    assert path.is_dir()


def test_pseudo_translator_changes_qt_translate_output():
    from synthesia2midi.localization import install_translator

    app = QApplication.instance() or QApplication([])
    selected = install_translator(app, "qps")

    try:
        translated = QCoreApplication.translate("Video2MidiApp", "File")

        assert selected == "qps"
        assert translated != "File"
        assert translated.startswith("[!! ")
        assert translated.endswith(" !!]")
    finally:
        install_translator(app, "en")


def test_install_translator_defaults_to_english_for_missing_locale():
    from synthesia2midi.localization import install_translator

    app = QApplication.instance() or QApplication([])

    assert install_translator(app, "missing-locale") == "en"


def test_spanish_translator_loads_known_source_texts():
    from synthesia2midi.localization import install_translator

    app = QApplication.instance() or QApplication([])
    selected = install_translator(app, "es")

    try:
        assert selected == "es"
        assert QCoreApplication.translate("Video2MidiApp", "File") == "Archivo"
        assert QCoreApplication.translate("ControlPanelQt", "Convert") == "Convertir"
    finally:
        install_translator(app, "en")


def test_production_translators_load_known_source_texts():
    from synthesia2midi.localization import install_translator

    app = QApplication.instance() or QApplication([])

    try:
        for locale_name in _production_translation_locales():
            selected = install_translator(app, locale_name)
            translated = QCoreApplication.translate("Video2MidiApp", "File")

            assert selected == locale_name
            assert translated != "File"
            assert QCoreApplication.translate(
                "ColorFamilyGrid", "Add Color Family"
            ) != "Add Color Family"
            assert QCoreApplication.translate(
                "CalibrationGuideWidget", "Download from YouTube"
            ) != "Download from YouTube"
            for source in SCANNER_WARNING_PRODUCTION_LOCALE_STRINGS:
                assert QCoreApplication.translate(
                    "AssistedCalibrationDialog", source
                ) != source
            for context, source in FINAL_REVIEW_RUNTIME_STRINGS:
                assert QCoreApplication.translate(context, source) != source
            for source in SPARK_AUTO_CALIBRATION_RUNTIME_STRINGS:
                assert QCoreApplication.translate(
                    "SparkCalibrationController", source
                ) != source
    finally:
        install_translator(app, "en")


def test_final_review_copy_removes_obsolete_family_guidance_but_keeps_slants():
    source_text = "\n".join(
        path.read_text(encoding="utf-8") for path in FINAL_REVIEW_COPY_FILES
    )

    for obsolete_text in OBSOLETE_FAMILY_COPY:
        assert obsolete_text not in source_text

    assert 'translate("ControlPanelQt", "Left Slant")' in source_text
    assert 'translate("ControlPanelQt", "Right Slant")' in source_text


def test_supported_user_locales_hide_pseudo_locale():
    from synthesia2midi.localization import locale_display_name, supported_user_locales

    assert supported_user_locales() == [
        locale_name for locale_name, _display_name in EXPECTED_USER_LOCALE_DISPLAY_NAMES
    ]
    assert "qps" not in supported_user_locales()
    for locale_name, display_name in EXPECTED_USER_LOCALE_DISPLAY_NAMES:
        assert locale_display_name(locale_name) == display_name


def test_locale_preference_helpers_use_qsettings(tmp_path):
    from synthesia2midi.localization import (
        APP_LOCALE_SETTINGS_KEY,
        load_preferred_locale,
        resolve_startup_locale,
        save_preferred_locale,
    )

    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path))
    settings = QSettings(QSettings.IniFormat, QSettings.UserScope, "Synthesia2MIDI", "test-locales")

    assert load_preferred_locale(settings) == "en"

    save_preferred_locale("es", settings)

    assert settings.value(APP_LOCALE_SETTINGS_KEY) == "es"
    assert load_preferred_locale(settings) == "es"
    assert resolve_startup_locale(None, settings) == "es"
    assert resolve_startup_locale("qps", settings) == "qps"

    settings.setValue(APP_LOCALE_SETTINGS_KEY, "missing-locale")

    assert load_preferred_locale(settings) == "en"


def test_translation_assets_are_collected_by_packaging_spec():
    spec_text = Path("packaging/Synthesia2MIDI.spec").read_text(encoding="utf-8")

    assert "translations" in spec_text


def test_production_translation_sources_are_complete_and_preserve_placeholders():
    unfinished = []
    placeholder_mismatches = []

    for locale_name in _production_translation_locales():
        ts_path = Path(f"synthesia2midi/synthesia2midi/translations/synthesia2midi_{locale_name}.ts")
        qm_path = Path(f"synthesia2midi/synthesia2midi/translations/synthesia2midi_{locale_name}.qm")

        assert ts_path.is_file()
        assert qm_path.is_file()

        tree = ET.parse(ts_path)
        for message in tree.findall(".//message"):
            source = message.findtext("source") or ""
            translation = message.find("translation")
            if translation is None or translation.get("type") == "unfinished" or not (translation.text or "").strip():
                unfinished.append((locale_name, source))
                continue

            source_placeholders = sorted(re.findall(r"\{[^{}]+\}", source))
            translated_placeholders = sorted(re.findall(r"\{[^{}]+\}", translation.text or ""))
            if source_placeholders != translated_placeholders:
                placeholder_mismatches.append((locale_name, source, translation.text or ""))

    assert unfinished == []
    assert placeholder_mismatches == []


def test_task_5_production_strings_are_localized_in_non_english_catalogs():
    identical_english = []

    for locale_name in _production_translation_locales():
        ts_path = Path(f"synthesia2midi/synthesia2midi/translations/synthesia2midi_{locale_name}.ts")
        tree = ET.parse(ts_path)
        translations_by_source = {}

        for message in tree.findall(".//message"):
            source = message.findtext("source") or ""
            translation = message.findtext("translation") or ""
            if source:
                translations_by_source[source] = translation

        for source in TASK_5_PRODUCTION_LOCALE_STRINGS:
            translated = translations_by_source.get(source)
            assert translated is not None, f"{locale_name} missing Task 5 source: {source}"
            if translated == source:
                identical_english.append((locale_name, source))

    assert identical_english == []


def test_task_6_production_strings_are_localized_in_non_english_catalogs():
    identical_english = []

    for locale_name in _production_translation_locales():
        ts_path = Path(f"synthesia2midi/synthesia2midi/translations/synthesia2midi_{locale_name}.ts")
        tree = ET.parse(ts_path)
        translations_by_source = {}

        for message in tree.findall(".//message"):
            source = message.findtext("source") or ""
            translation = message.findtext("translation") or ""
            if source:
                translations_by_source[source] = translation

        for source in TASK_6_PRODUCTION_LOCALE_STRINGS:
            translated = translations_by_source.get(source)
            assert translated is not None, f"{locale_name} missing Task 6 source: {source}"
            if translated == source:
                identical_english.append((locale_name, source))

    assert identical_english == []


def test_touchup_access_strings_are_localized_in_their_qt_contexts():
    incomplete = []

    for locale_name in _production_translation_locales():
        translations = {
            (context_name, source): "" if translation is None else translation.text or ""
            for context_name, source, translation in _catalog_messages(locale_name)
            if translation is not None and translation.get("type") != "vanished"
        }
        for context_name, source in TOUCHUP_ACCESS_PRODUCTION_LOCALE_STRINGS:
            translated = translations.get((context_name, source), "")
            if not translated.strip() or translated == source:
                incomplete.append((locale_name, context_name, source))

    assert incomplete == []


def test_task_10_color_family_strings_are_active_in_production_catalogs():
    incomplete = []
    placeholder_mismatches = []

    for locale_name in _production_translation_locales():
        active_sources = set()
        for context_name, source, translation in _catalog_messages(locale_name):
            if source not in TASK_10_PRODUCTION_LOCALE_STRINGS:
                continue
            if translation is not None and translation.get("type") == "vanished":
                continue

            active_sources.add(source)
            translated = "" if translation is None else translation.text or ""
            if (
                translation is None
                or translation.get("type") == "unfinished"
                or not translated.strip()
            ):
                incomplete.append((locale_name, context_name, source))

            source_placeholders = sorted(re.findall(r"\{[^{}]+\}", source))
            translated_placeholders = sorted(re.findall(r"\{[^{}]+\}", translated))
            if source_placeholders != translated_placeholders:
                placeholder_mismatches.append((locale_name, context_name, source, translated))

        missing_sources = set(TASK_10_PRODUCTION_LOCALE_STRINGS) - active_sources
        for source in sorted(missing_sources):
            incomplete.append((locale_name, "missing", source))

    assert incomplete == []
    assert placeholder_mismatches == []


def test_spark_auto_calibration_strings_are_active_in_their_qt_context():
    for locale_name in _production_translation_locales():
        messages = {
            source: translation
            for context_name, source, translation in _catalog_messages(locale_name)
            if context_name == "SparkCalibrationController"
            and source in SPARK_AUTO_CALIBRATION_PRODUCTION_LOCALE_STRINGS
            and translation is not None
            and translation.get("type") != "vanished"
        }

        assert set(messages) == set(SPARK_AUTO_CALIBRATION_PRODUCTION_LOCALE_STRINGS)
        for source, translation in messages.items():
            assert translation.get("type") != "unfinished"
            translated = translation.text or ""
            assert translated.strip()
            assert sorted(re.findall(r"\{[^{}]+\}", translated)) == sorted(
                re.findall(r"\{[^{}]+\}", source)
            )


def test_color_family_grid_uses_installed_translator_for_all_locales():
    from synthesia2midi.gui.color_family_grid import ColorFamilyGrid
    from synthesia2midi.localization import install_translator

    app = QApplication.instance() or QApplication([])
    colors = {
        "LW": (30, 80, 220),
        "LB": (20, 45, 150),
        "RW": (220, 60, 70),
        "RB": (160, 35, 50),
        "COLOR_3_W": (230, 180, 30),
        "COLOR_3_B": (170, 120, 20),
    }
    enabled = {slot: True for slot in colors}

    try:
        for locale_name in [*_production_translation_locales(), "qps"]:
            assert install_translator(app, locale_name) == locale_name
            grid = ColorFamilyGrid(mode="calibration")
            try:
                grid.set_families((1, 2, 3), colors=colors, enabled=enabled)
                texts = [
                    grid.family_heading(1).text(),
                    grid.family_heading(2).text(),
                    grid.rows["LW"].label.text(),
                    grid.rows["LB"].label.text(),
                    grid.rows["LW"].set_button.text(),
                    grid.rows["LW"].present.text(),
                    grid.add_family_button.text(),
                    grid.remove_family_buttons[2].toolTip(),
                ]

                assert all(text.strip() for text in texts)
                if locale_name == "qps":
                    assert all(text.startswith("[!! ") and text.endswith(" !!]") for text in texts)
            finally:
                grid.close()
                grid.deleteLater()
                app.processEvents()
    finally:
        install_translator(app, "en")


def test_translation_agent_packet_matches_source_catalog():
    packet_path = Path("docs/localization/translation-agent-packet.json")
    ts_path = Path("synthesia2midi/synthesia2midi/translations/synthesia2midi_es.ts")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    tree = ET.parse(ts_path)

    packet_keys = {
        (entry["context"], entry["source"])
        for entry in packet["entries"]
    }
    source_keys = set()
    for context in tree.findall("context"):
        context_name = context.findtext("name") or ""
        for message in context.findall("message"):
            translation = message.find("translation")
            if translation is not None and translation.get("type") == "vanished":
                continue
            source = message.findtext("source") or ""
            if source:
                source_keys.add((context_name, source))

    assert packet["schema_version"] == 1
    assert packet_keys == source_keys


def test_apply_translation_json_writes_qt_ts_and_validates_placeholders(tmp_path):
    from synthesia2midi.tools.apply_translation_json import apply_translation_json

    source_ts = tmp_path / "source.ts"
    source_ts.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<TS version="2.1" language="es_419">
  <context>
    <name>SampleDialog</name>
    <message>
      <source>File</source>
      <translation>Archivo</translation>
    </message>
    <message>
      <source>Error: {message}</source>
      <translation>Error: {message}</translation>
    </message>
  </context>
</TS>
""",
        encoding="utf-8",
    )
    translation_json = tmp_path / "ja.json"
    translation_json.write_text(
        json.dumps(
            [
                {
                    "context": "SampleDialog",
                    "source": "File",
                    "translation": "ファイル",
                    "notes": "",
                },
                {
                    "context": "SampleDialog",
                    "source": "Error: {message}",
                    "translation": "エラー: {message}",
                    "notes": "",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_ts = tmp_path / "output.ts"

    apply_translation_json(
        source_ts=source_ts,
        translations_json=translation_json,
        output_ts=output_ts,
        language_code="ja",
    )

    tree = ET.parse(output_ts)
    root = tree.getroot()
    translations = [
        message.findtext("translation")
        for message in root.findall(".//message")
    ]

    assert root.get("language") == "ja"
    assert translations == ["ファイル", "エラー: {message}"]


def test_lupdate_extracts_source_texts(tmp_path):
    lupdate = Path(sys.executable).parent / "pyside6-lupdate"
    output_ts = tmp_path / "probe.ts"

    result = subprocess.run(
        [str(lupdate), "-extensions", "py", "synthesia2midi/synthesia2midi", "-ts", str(output_ts)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Found 0 source text" not in result.stdout
    assert "<source>File</source>" in output_ts.read_text(encoding="utf-8")


def test_main_window_menu_text_uses_installed_translator(monkeypatch):
    from synthesia2midi.localization import install_translator
    from synthesia2midi.main import Video2MidiApp

    monkeypatch.setattr(QTimer, "singleShot", lambda *args, **kwargs: None)
    app = QApplication.instance() or QApplication([])
    install_translator(app, "qps")
    window = Video2MidiApp()

    try:
        menu_texts = [action.text() for action in window.menuBar().actions()]

        assert any(text.startswith("[!! Fìlè") for text in menu_texts)
    finally:
        install_translator(app, "en")
        window.app_state.unsaved_changes = False
        window.close()
        window.deleteLater()
        app.processEvents()


def test_core_user_facing_widgets_use_installed_translator(monkeypatch):
    import numpy as np

    from synthesia2midi.core.app_state import AppState
    from synthesia2midi.gui.auto_detect_tuning_dialog import AutoDetectTuningDialog
    from synthesia2midi.gui.manual_keyboard_fit_dialog import ManualKeyboardFitDialog
    from synthesia2midi.gui.startup_dialog import StartupDialog
    from synthesia2midi.gui.wizard import CalibrationWizard
    from synthesia2midi.gui.youtube_download_dialog import YouTubeDownloadDialog
    from synthesia2midi.localization import install_translator
    from synthesia2midi.main import Video2MidiApp

    monkeypatch.setattr(QTimer, "singleShot", lambda *args, **kwargs: None)
    app = QApplication.instance() or QApplication([])
    install_translator(app, "qps")

    widgets = [
        Video2MidiApp(),
        StartupDialog(recent_video_paths=[]),
        YouTubeDownloadDialog(default_output_dir="/tmp"),
        ManualKeyboardFitDialog(),
        CalibrationWizard(None, AppState()),
        AutoDetectTuningDialog(
            None,
            AppState(),
            np.zeros((8, 8, 3), dtype=np.uint8),
            (0, 0, 8, 8),
            initial_detection_results={"total_keys": 88},
            fallback_used=True,
            apply_detection_callback=lambda _results: True,
        ),
    ]

    try:
        main_window = widgets[0]
        startup = widgets[1]
        youtube = widgets[2]
        manual_fit = widgets[3]
        wizard = widgets[4]
        auto_tune = widgets[5]

        assert main_window.control_panel.convert_button.text().startswith("[!! Cònvèrt")
        assert startup.windowTitle().startswith("[!! Synthèsìà tò MÌDÌ")
        assert youtube.windowTitle().startswith("[!! Dòwnlòàd YòùTùbè Vìdèò")
        assert manual_fit.windowTitle().startswith("[!! Mànùàl Fìt")
        assert wizard.windowTitle().startswith("[!! Càlìbràtìòn Wìzàrd")
        assert auto_tune.windowTitle().startswith("[!! Àùtò-Dètèct Tùnìng")
    finally:
        install_translator(app, "en")
        for widget in widgets:
            if hasattr(widget, "app_state"):
                widget.app_state.unsaved_changes = False
            widget.close()
            widget.deleteLater()
        app.processEvents()
