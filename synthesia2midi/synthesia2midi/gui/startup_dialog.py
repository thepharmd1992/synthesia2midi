"""Startup dialog for Synthesia2MIDI - Choose between local file or YouTube download"""
# Third-party imports
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFrame, QHBoxLayout, 
    QLabel, QPushButton, QVBoxLayout
)


class StartupDialog(QDialog):
    """Initial dialog shown on startup to choose video source"""
    
    # Signals for different choices
    open_local_file = Signal()
    download_from_youtube = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Synthesia2MIDI - Select Video Source")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Title
        title_label = QLabel("Welcome to Synthesia2MIDI")
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
        self.local_file_btn = QPushButton("Open Local Video File")
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
        self.open_local_file.emit()
        self.accept()
        
    def _on_youtube_clicked(self):
        """Handle YouTube button click"""
        self.download_from_youtube.emit()
        self.accept()
