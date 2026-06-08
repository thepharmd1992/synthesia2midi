"""YouTube download dialog for Synthesia2MIDI"""
# Standard library imports
import logging
import os
from pathlib import Path

# Third-party imports
from PySide6.QtCore import Qt, Signal, QTimer, QThread
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QGroupBox, QHBoxLayout, 
    QLabel, QLineEdit, QMessageBox, QProgressBar, 
    QPushButton, QTextEdit, QVBoxLayout
)

from ..youtube_downloader import YouTubeDownloader, YouTubeDownloaderThread


class YouTubeInfoFetcherThread(QThread):
    """Thread for fetching YouTube video info without blocking UI"""

    info_fetched = Signal(str, dict)
    error = Signal(str, str)

    def __init__(self, url: str, output_dir: str):
        super().__init__()
        self.url = url
        self.output_dir = output_dir

    def run(self):
        try:
            downloader = YouTubeDownloader(self.output_dir)
            info = downloader.get_video_info(self.url)
            self.info_fetched.emit(self.url, info)
        except Exception as exc:
            self.error.emit(self.url, str(exc))


class YouTubeDownloadDialog(QDialog):
    """Dialog for downloading YouTube videos"""

    AUTO_FETCH_DELAY_MS = 350
    
    # Signal emitted when download completes with file path
    video_downloaded = Signal(str)
    
    def __init__(self, parent=None, default_output_dir='videos'):
        super().__init__(parent)
        self.downloader = YouTubeDownloader(default_output_dir)
        self.download_thread = None
        self.info_fetch_thread = None
        self._current_info_url = None
        self._active_info_url = None
        self._queued_info_url = None
        self._queued_info_show_dialog = False
        self._show_info_error_dialog = True
        self._auto_fetching = False
        self.auto_fetch_timer = QTimer(self)
        self.auto_fetch_timer.setSingleShot(True)
        self.auto_fetch_timer.timeout.connect(self._auto_fetch_video_info)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the dialog UI"""
        self.setWindowTitle("Download YouTube Video")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        # Set a reasonable initial size to prevent Windows sizing warnings
        self.resize(650, 450)
        
        layout = QVBoxLayout(self)
        
        # URL input section
        url_group = QGroupBox("YouTube URL")
        url_layout = QVBoxLayout()
        
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://www.youtube.com/watch?v=...")
        self.url_input.textChanged.connect(self.on_url_changed)
        url_layout.addWidget(self.url_input)
        
        # Video info (hidden initially)
        self.info_widget = QGroupBox("Video Information")
        self.info_widget.hide()
        # Reserve space even when hidden to prevent dialog resizing
        self.info_widget.setMinimumHeight(100)
        info_layout = QVBoxLayout()
        
        self.title_label = QLabel()
        self.title_label.setWordWrap(True)
        self.duration_label = QLabel()
        self.uploader_label = QLabel()
        
        info_layout.addWidget(self.title_label)
        info_layout.addWidget(self.duration_label)
        info_layout.addWidget(self.uploader_label)
        self.info_widget.setLayout(info_layout)
        
        url_layout.addWidget(self.info_widget)
        url_group.setLayout(url_layout)
        layout.addWidget(url_group)
        
        # Fetch info button
        self.fetch_info_btn = QPushButton("Fetch Video Info")
        self.fetch_info_btn.clicked.connect(self.fetch_video_info)
        self.fetch_info_btn.setEnabled(False)
        layout.addWidget(self.fetch_info_btn)
        
        # Progress section
        progress_group = QGroupBox("Download Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready to download")
        self.status_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.download_btn = QPushButton("Download Video")
        self.download_btn.clicked.connect(self.start_download)
        self.download_btn.setEnabled(False)
        button_layout.addWidget(self.download_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_download)
        self.cancel_btn.setEnabled(False)
        button_layout.addWidget(self.cancel_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
    def on_url_changed(self, text):
        """Handle URL input changes"""
        self.auto_fetch_timer.stop()

        url = text.strip()
        is_valid = bool(url) and self.downloader.validate_url(url)
        self.fetch_info_btn.setEnabled(is_valid)

        has_current_info = is_valid and url == self._current_info_url
        if not has_current_info:
            self.info_widget.hide()
        self.download_btn.setEnabled(has_current_info)
        
        if not is_valid and text:
            self.status_label.setText("Invalid YouTube URL")
            self._current_info_url = None
        elif has_current_info:
            self.status_label.setText("Ready to download")
        elif is_valid:
            self.status_label.setText("Fetching video information...")
            self.auto_fetch_timer.start(self.AUTO_FETCH_DELAY_MS)
        else:
            self.status_label.setText("Ready to download")

    def _auto_fetch_video_info(self):
        self._auto_fetching = True
        try:
            self.fetch_video_info()
        finally:
            self._auto_fetching = False
    
    def fetch_video_info(self):
        """Fetch and display video information"""
        self.auto_fetch_timer.stop()
        url = self.url_input.text().strip()

        if not self.downloader.validate_url(url):
            self.fetch_info_btn.setEnabled(False)
            self.download_btn.setEnabled(False)
            self.info_widget.hide()
            self._current_info_url = None
            self.status_label.setText("Invalid YouTube URL" if url else "Ready to download")
            return

        self._start_video_info_fetch(url, show_error_dialog=not self._auto_fetching)

    def _start_video_info_fetch(self, url, show_error_dialog=True):
        if self.info_fetch_thread and self.info_fetch_thread.isRunning():
            self._queued_info_url = url
            self._queued_info_show_dialog = show_error_dialog
            return

        self.fetch_info_btn.setEnabled(False)
        self.download_btn.setEnabled(False)
        self.status_label.setText("Fetching video information...")
        self._active_info_url = url
        self._show_info_error_dialog = show_error_dialog
        self.info_fetch_thread = YouTubeInfoFetcherThread(url, str(self.downloader.output_dir))
        self.info_fetch_thread.info_fetched.connect(self._on_video_info_fetched)
        self.info_fetch_thread.error.connect(self._on_video_info_error)
        if hasattr(self.info_fetch_thread, "finished"):
            self.info_fetch_thread.finished.connect(self._on_video_info_thread_finished)
        self.info_fetch_thread.start()

    def _on_video_info_fetched(self, url, info):
        if url != self.url_input.text().strip():
            return

        self.title_label.setText(f"<b>Title:</b> {info['title']}")

        duration = info['duration']
        minutes = duration // 60
        seconds = duration % 60
        self.duration_label.setText(f"<b>Duration:</b> {minutes}:{seconds:02d}")

        self.uploader_label.setText(f"<b>Uploader:</b> {info['uploader']}")

        self._current_info_url = url
        self.info_widget.show()
        self.download_btn.setEnabled(True)
        self.status_label.setText("Ready to download")

    def _on_video_info_error(self, url, error):
        if url != self.url_input.text().strip():
            return

        self._current_info_url = None
        self.info_widget.hide()
        self.download_btn.setEnabled(False)
        if self._show_info_error_dialog:
            QMessageBox.warning(self, "Error", f"Failed to fetch video info: {error}")
        self.status_label.setText("Failed to fetch video info")

    def _on_video_info_thread_finished(self):
        self.info_fetch_thread = None
        self._active_info_url = None

        queued_url = self._queued_info_url
        queued_show_dialog = self._queued_info_show_dialog
        self._queued_info_url = None
        self._queued_info_show_dialog = False

        current_url = self.url_input.text().strip()
        if queued_url and queued_url == current_url and self.downloader.validate_url(queued_url):
            self._start_video_info_fetch(queued_url, queued_show_dialog)
            return

        self.fetch_info_btn.setEnabled(bool(current_url) and self.downloader.validate_url(current_url))

    def start_download(self):
        """Start downloading the video"""
        url = self.url_input.text()
        # Use default quality of 1080p
        quality = '1080p'
        
        # Create download thread
        output_dir = str(self.downloader.output_dir)
        self.download_thread = YouTubeDownloaderThread(url, output_dir, quality)
        
        # Connect signals
        self.download_thread.progress_handler.progress.connect(self.update_progress)
        self.download_thread.progress_handler.status.connect(self.update_status)
        self.download_thread.progress_handler.finished.connect(self.on_download_finished)
        self.download_thread.progress_handler.error.connect(self.on_download_error)
        
        # Update UI
        self.download_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.url_input.setEnabled(False)
        self.fetch_info_btn.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        
        # Start download
        self.download_thread.start()
    
    def cancel_download(self):
        """Cancel the current download"""
        if self.download_thread and self.download_thread.isRunning():
            self.download_thread.cancel()
            self.download_thread.quit()
            self.download_thread.wait()
            
        self.reset_ui()
        self.status_label.setText("Download cancelled")
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def update_status(self, status):
        """Update status label"""
        self.status_label.setText(status)
    
    def on_download_finished(self, file_path):
        """Handle successful download"""
        self.reset_ui()
        self.status_label.setText(f"Download complete!")
        
        # Store the file path for later use
        self.downloaded_file_path = file_path
        
        # Ask if user wants to load the video immediately
        reply = QMessageBox.question(
            self, 
            "Download Complete", 
            "Video downloaded successfully. Load it now?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            # Close this modal before loading the video; loading may open progress dialogs.
            self.accept()
            self.video_downloaded.emit(file_path)
        else:
            # Just close the dialog without loading
            self.accept()
        
    def on_download_error(self, error):
        """Handle download error"""
        self.reset_ui()
        self.status_label.setText("Download failed")
        QMessageBox.critical(self, "Download Error", f"Failed to download video: {error}")
    
    def reset_ui(self):
        """Reset UI to initial state"""
        url = self.url_input.text().strip()
        is_valid = bool(url) and self.downloader.validate_url(url)
        self.download_btn.setEnabled(is_valid and url == self._current_info_url)
        self.cancel_btn.setEnabled(False)
        self.url_input.setEnabled(True)
        self.fetch_info_btn.setEnabled(is_valid)
        self.progress_bar.hide()
        
    def closeEvent(self, event):
        """Handle dialog close"""
        self.auto_fetch_timer.stop()
        if self.info_fetch_thread and self.info_fetch_thread.isRunning():
            self.info_fetch_thread.quit()
            self.info_fetch_thread.wait()
            self.info_fetch_thread = None

        if self.download_thread and self.download_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Download in Progress",
                "A download is in progress. Cancel and close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.cancel_download()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
