from pathlib import Path

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication


def test_available_locales_includes_english_and_pseudo_locale():
    from synthesia2midi.localization import available_locales

    assert "en" in available_locales()
    assert "qps" in available_locales()


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


def test_translation_assets_are_collected_by_packaging_spec():
    spec_text = Path("packaging/Synthesia2MIDI.spec").read_text(encoding="utf-8")

    assert "translations" in spec_text
