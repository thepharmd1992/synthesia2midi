"""Application icon helpers."""

from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication


def app_icon_path() -> Path:
    return Path(__file__).resolve().parent / "assets" / "app_icon.png"


def app_icon() -> QIcon:
    return QIcon(str(app_icon_path()))


def install_app_icon(app: QApplication) -> bool:
    icon = app_icon()
    if icon.isNull():
        return False
    app.setWindowIcon(icon)
    return True
