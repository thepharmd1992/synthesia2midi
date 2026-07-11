"""Floating settings window for the main control panel."""
from __future__ import annotations

from PySide6.QtCore import QCoreApplication, Qt, Signal
from PySide6.QtWidgets import QDialog, QVBoxLayout, QWidget

from synthesia2midi.gui.dialog_positioning import screen_for_widget

translate = QCoreApplication.translate


class SettingsToolWindow(QDialog):
    """Non-modal tool window that hosts the settings panel without taking canvas width."""

    visibility_changed = Signal(bool)

    def __init__(self, parent: QWidget):
        super().__init__(parent, Qt.Tool | Qt.WindowCloseButtonHint)
        self.setWindowTitle(translate("SettingsToolWindow", "Settings"))
        self.setModal(False)
        self.setMinimumSize(360, 420)

        self._content_layout = QVBoxLayout(self)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)
        self.settings_widget: QWidget | None = None

    def set_settings_widget(self, widget: QWidget) -> None:
        """Install the settings panel as the tool window's direct content."""
        if self.settings_widget is not None:
            self._content_layout.removeWidget(self.settings_widget)
        self.settings_widget = widget
        self._content_layout.addWidget(widget)

    def fit_to_available_screen(self) -> None:
        """Keep the pop-out usable on laptop-sized displays."""
        screen = screen_for_widget(self.parentWidget(), self)
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
        screen = screen_for_widget(self.parentWidget(), self)
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
