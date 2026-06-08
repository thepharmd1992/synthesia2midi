"""Startup dialog for Synthesia2MIDI - Choose between local file or YouTube download"""
import os

# Third-party imports
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class StartupDialog(QDialog):
    """Initial dialog shown on startup to choose video source"""
    
    # Signals for different choices
    open_local_file = Signal()
    download_from_youtube = Signal()
    open_recent_file = Signal(str)
    
    def __init__(self, parent=None, *, recent_video_paths=None):
        super().__init__(parent)
        self.recent_video_paths = list(recent_video_paths or [])
        self.recent_video_buttons = []
        self.setWindowTitle("Synthesia to MIDI - Select Video Source")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Title
        title_label = QLabel("Welcome to Synthesia to MIDI")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("How would you like to load a video?")
        subtitle_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle_label)
        
        # Add some spacing
        layout.addSpacing(20)
        
        # Buttons container
        button_layout = QVBoxLayout()
        button_layout.setSpacing(15)
        
        # Local file button
        self.local_file_btn = QPushButton("Open Video File")
        self.local_file_btn.setMinimumHeight(50)
        self.local_file_btn.setToolTip("Browse for a video file on your computer")
        self.local_file_btn.clicked.connect(self._on_local_file_clicked)
        button_layout.addWidget(self.local_file_btn)
        
        # YouTube download button
        self.youtube_btn = QPushButton("Download from YouTube")
        self.youtube_btn.setMinimumHeight(50)
        self.youtube_btn.setToolTip("Download a video from YouTube")
        self.youtube_btn.clicked.connect(self._on_youtube_clicked)
        button_layout.addWidget(self.youtube_btn)
        
        layout.addLayout(button_layout)

        if self.recent_video_paths:
            recent_separator = QFrame()
            recent_separator.setFrameShape(QFrame.HLine)
            recent_separator.setFrameShadow(QFrame.Sunken)
            layout.addWidget(recent_separator)

            recent_label = QLabel("Recent Videos")
            recent_font = QFont()
            recent_font.setBold(True)
            recent_label.setFont(recent_font)
            layout.addWidget(recent_label)

            recent_layout = QVBoxLayout()
            recent_layout.setSpacing(4)
            for path in self.recent_video_paths:
                recent_button = self._create_recent_video_button(path)
                recent_layout.addWidget(recent_button)
                self.recent_video_buttons.append(recent_button)
            layout.addLayout(recent_layout)
        
        # Add separator
        layout.addSpacing(20)
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)
        
        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        cancel_layout = QHBoxLayout()
        # Keep the cancel button left-aligned.
        cancel_layout.addWidget(cancel_btn)
        layout.addLayout(cancel_layout)
        
        # Set default button
        self.local_file_btn.setDefault(True)
        self.local_file_btn.setFocus()
        
    def _on_local_file_clicked(self):
        """Handle local file button click"""
        self.accept()
        self.open_local_file.emit()
        
    def _on_youtube_clicked(self):
        """Handle YouTube button click"""
        self.accept()
        self.download_from_youtube.emit()

    def _create_recent_video_button(self, path: str) -> QPushButton:
        filename = os.path.basename(path) or path
        button = QPushButton(filename)
        button.setMinimumHeight(28)
        button.setMaximumHeight(28)
        button.setStyleSheet("QPushButton { text-align: left; padding: 3px 8px; }")
        button.setToolTip(path)
        button.clicked.connect(lambda checked=False, selected_path=path: self._on_recent_file_clicked(selected_path))
        return button

    def _on_recent_file_clicked(self, path: str):
        """Handle recent file click"""
        self.accept()
        self.open_recent_file.emit(path)
