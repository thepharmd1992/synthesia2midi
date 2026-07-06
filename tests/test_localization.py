from pathlib import Path
import subprocess
import sys

from PySide6.QtCore import QCoreApplication
from PySide6.QtCore import QTimer
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
