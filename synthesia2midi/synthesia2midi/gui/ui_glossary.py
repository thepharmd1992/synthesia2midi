"""Small user-facing glossary for necessary Synthesia2MIDI terms."""

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class UiGlossary(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        entries = [
            self.tr("Keyboard box: the rectangle around the full visible keyboard."),
            self.tr("Overlay: a small box that follows one piano key."),
            self.tr("Color family: one Synthesia note color, with separate Natural and Sharp / Flat examples."),
            self.tr("Detection sensitivity: how easily a key counts as pressed."),
            self.tr("Repeated-notes flashes: brief flashes above a key that separate repeated notes."),
        ]
        self.labels = []
        for entry in entries:
            label = QLabel(entry)
            label.setWordWrap(True)
            layout.addWidget(label)
            self.labels.append(label)
