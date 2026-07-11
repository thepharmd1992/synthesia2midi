"""Dedicated modeless tool for repeated-note detection setup."""
from __future__ import annotations

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout, QWidget

from synthesia2midi.gui.dialog_positioning import (
    move_to_top_center_safe_zone,
    screen_for_widget,
)

translate = QCoreApplication.translate


class RepeatedNotesToolWindow(QDialog):
    """Host the repeated-notes workflow outside the Settings page scroller."""

    def __init__(self, parent: QWidget, content: QWidget):
        super().__init__(parent, Qt.Tool | Qt.WindowCloseButtonHint)
        self.setWindowTitle(translate("RepeatedNotesToolWindow", "Repeated Notes"))
        self.setModal(False)
        self.setMinimumSize(520, 480)
        self.resize(640, 680)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        heading = QLabel(
            translate(
                "RepeatedNotesToolWindow",
                "Set up the repeated-notes fix, then close this window when the preview looks right.",
            )
        )
        heading.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(content, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    def show_near_parent(self) -> None:
        screen = screen_for_widget(self.parentWidget(), self)
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(
                min(680, max(520, available.width() - 80)),
                min(720, max(480, available.height() - 120)),
            )
        move_to_top_center_safe_zone(self, self.parentWidget())
        self.show()
        self.raise_()
        self.activateWindow()
