from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication


ROOT = Path(__file__).resolve().parents[1]


def test_app_icon_assets_exist_with_expected_formats():
    package_icon = ROOT / "synthesia2midi" / "synthesia2midi" / "assets" / "app_icon.png"
    mac_icon = ROOT / "packaging" / "assets" / "Synthesia2MIDI.icns"
    windows_icon = ROOT / "packaging" / "assets" / "Synthesia2MIDI.ico"

    assert package_icon.is_file()
    assert mac_icon.is_file()
    assert windows_icon.is_file()
    assert package_icon.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert mac_icon.read_bytes().startswith(b"icns")
    assert windows_icon.read_bytes().startswith(b"\x00\x00\x01\x00")


def test_app_icon_helper_loads_non_null_icon():
    QApplication.instance() or QApplication([])

    from synthesia2midi.app_icon import app_icon, app_icon_path

    assert app_icon_path().name == "app_icon.png"

    icon = app_icon()

    assert isinstance(icon, QIcon)
    assert not icon.isNull()


def test_pyinstaller_spec_references_platform_icons_and_runtime_icon_data():
    spec_text = (ROOT / "packaging" / "Synthesia2MIDI.spec").read_text(encoding="utf-8")

    assert 'includes=["translations/*", "assets/*"]' in spec_text
    assert "icon=str(ROOT / \"packaging\" / \"assets\" / \"Synthesia2MIDI.ico\")" in spec_text
    assert "icon=str(ROOT / \"packaging\" / \"assets\" / \"Synthesia2MIDI.icns\")" in spec_text
