"""No-video presentation for the main canvas area."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class VideoEmptyState(QWidget):
    open_video_requested = Signal()
    youtube_requested = Signal()
    settings_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("video_empty_state")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.addStretch(1)

        title = QLabel(self.tr("Open a Synthesia-style video to begin"))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        layout.addWidget(title)

        cue = QLabel(
            self.tr("Choose a Synthesia-style piano video with visible keys and falling notes.")
        )
        cue.setAlignment(Qt.AlignCenter)
        cue.setWordWrap(True)
        layout.addWidget(cue)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.open_video_button = QPushButton(self.tr("Open Video"))
        self.open_video_button.setMinimumHeight(36)
        self.open_video_button.clicked.connect(self.open_video_requested.emit)
        actions.addWidget(self.open_video_button)
        self.youtube_button = QPushButton(self.tr("Download from YouTube"))
        self.youtube_button.setMinimumHeight(36)
        self.youtube_button.clicked.connect(self.youtube_requested.emit)
        actions.addWidget(self.youtube_button)
        self.settings_button = QPushButton(self.tr("Settings"))
        self.settings_button.setMinimumHeight(40)
        self.settings_button.clicked.connect(self.settings_requested.emit)
        actions.addWidget(self.settings_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        layout.addStretch(1)

