"""Floating settings window for the main control panel."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QApplication, QDialog, QScrollArea, QVBoxLayout, QWidget


class SettingsToolWindow(QDialog):
    """Non-modal tool window that hosts the settings panel without taking canvas width."""

    visibility_changed = Signal(bool)

    def __init__(self, parent: QWidget):
        super().__init__(parent, Qt.Tool | Qt.WindowCloseButtonHint)
        self.setWindowTitle("Settings")
        self.setModal(False)
        self.setMinimumSize(360, 420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        layout.addWidget(self.scroll_area)

    def set_settings_widget(self, widget: QWidget) -> None:
        """Install the settings widget into the scrollable tool window."""
        self.scroll_area.setWidget(widget)

    def fit_to_available_screen(self) -> None:
        """Keep the pop-out usable on laptop-sized displays."""
        screen = QApplication.primaryScreen()
        if not screen:
            self.resize(760, 560)
            return

        rect = screen.availableGeometry()
        width = min(780, max(380, rect.width() - 80))
        height = min(600, max(420, rect.height() - 160))
        self.resize(width, height)

    def show_near_parent(self) -> None:
        """Show the tool window near the main window without reserving layout space."""
        if self.isVisible():
            self.raise_()
            self.activateWindow()
            return

        self.fit_to_available_screen()
        screen = QApplication.primaryScreen()
        if screen is not None:
            rect = screen.availableGeometry()
            self.move(
                max(rect.left(), rect.right() - self.frameGeometry().width() - 20),
                rect.top() + 40,
            )
        self.show()
        self.raise_()
        self.activateWindow()

    def show_preserving_geometry(self) -> None:
        """Restore after workflow dialogs without changing the user's window placement."""
        self.show()
        self.raise_()
        self.activateWindow()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.visibility_changed.emit(True)

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self.visibility_changed.emit(False)
