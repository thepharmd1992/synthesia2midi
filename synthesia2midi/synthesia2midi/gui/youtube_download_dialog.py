"""YouTube download dialog for Synthesia2MIDI"""
# Standard library imports
import logging
import os
from pathlib import Path

# Third-party imports
from PySide6.QtCore import Qt, Signal, QTimer, QThread, QSettings
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QGroupBox, QHBoxLayout,
    QCheckBox, QLabel, QLineEdit, QMessageBox, QProgressBar,
    QPushButton, QTextEdit, QVBoxLayout
)

from ..youtube_downloader import (
    SUPPORTED_COOKIE_BROWSERS,
    YouTubeDownloader,
    YouTubeDownloaderThread,
)


YOUTUBE_PREFERRED_BROWSER_KEY = "youtube/preferred_browser"
YOUTUBE_AUTO_COOKIE_RETRY_KEY = "youtube/auto_cookie_retry"
DEFAULT_COOKIE_BROWSER = "chrome"


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
            downloader = YouTubeDownloader(
                self.output_dir,
                preferred_browser=getattr(self, "preferred_browser", None),
                auto_cookie_retry=getattr(self, "auto_cookie_retry", True),
            )
            info = downloader.get_video_info(self.url)
            self.info_fetched.emit(self.url, info)
        except Exception as exc:
            self.error.emit(self.url, str(exc))


class YouTubeDownloadDialog(QDialog):
    """Dialog for downloading YouTube videos"""

    AUTO_FETCH_DELAY_MS = 350
    DOWNLOAD_STALL_DELAY_MS = 20000
    QUALITY_ORDER = ("1080p", "720p", "480p")
    
    # Signal emitted when download completes with file path
    video_downloaded = Signal(str)
    
    def __init__(self, parent=None, default_output_dir='videos', settings=None):
        super().__init__(parent)
        self.settings = settings or QSettings("Synthesia2MIDI", "Synthesia2MIDI")
        self.downloader = YouTubeDownloader(default_output_dir)
        self.download_thread = None
        self.info_fetch_thread = None
        self._current_info_url = None
        self._current_video_title = None
        self._active_info_url = None
        self._queued_info_url = None
        self._queued_info_show_dialog = False
        self._show_info_error_dialog = True
        self._auto_fetching = False
        self.auto_fetch_timer = QTimer(self)
        self.auto_fetch_timer.setSingleShot(True)
        self.auto_fetch_timer.timeout.connect(self._auto_fetch_video_info)
        self.download_stall_timer = QTimer(self)
        self.download_stall_timer.setSingleShot(True)
        self.download_stall_timer.timeout.connect(self._on_download_stall)
        self._preferred_browser = self._load_preferred_browser()
        self._auto_cookie_retry = self._load_auto_cookie_retry()
        self.downloader.preferred_browser = self._preferred_browser
        self.downloader.auto_cookie_retry = self._auto_cookie_retry
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
        
        self.quality_combo = QComboBox()
        self._reset_quality_options()
        self.quality_combo.setEnabled(False)
        layout.addWidget(self.quality_combo)

        fallback_group = QGroupBox("YouTube Access Fallback")
        fallback_layout = QVBoxLayout()

        self.browser_combo = QComboBox()
        self.browser_combo.addItem("Chrome", "chrome")
        self.browser_combo.addItem("Edge", "edge")
        self.browser_combo.addItem("Safari", "safari")
        browser_index = self.browser_combo.findData(self._preferred_browser)
        self.browser_combo.setCurrentIndex(browser_index if browser_index >= 0 else 0)
        self.browser_combo.currentIndexChanged.connect(self._on_browser_changed)
        fallback_layout.addWidget(self.browser_combo)

        self.auto_retry_checkbox = QCheckBox("Auto-retry with saved browser cookies if YouTube blocks access")
        self.auto_retry_checkbox.setChecked(self._auto_cookie_retry)
        self.auto_retry_checkbox.toggled.connect(self._on_auto_retry_toggled)
        fallback_layout.addWidget(self.auto_retry_checkbox)

        fallback_group.setLayout(fallback_layout)
        layout.addWidget(fallback_group)

        # Fetch info button
        self.fetch_info_btn = QPushButton("Refresh Info")
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
            self._reset_quality_options()
            self.quality_combo.setEnabled(False)
            self._current_video_title = None
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
            self._current_video_title = None
            self.quality_combo.setEnabled(False)
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
        self.info_fetch_thread.preferred_browser = self.preferred_browser()
        self.info_fetch_thread.auto_cookie_retry = self.auto_cookie_retry_enabled()
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
        self._current_video_title = info['title']
        self._apply_available_qualities(info.get("available_qualities"))
        self.info_widget.show()
        self.quality_combo.setEnabled(True)
        self.download_btn.setEnabled(True)
        self.status_label.setText("Ready to download")

    def _on_video_info_error(self, url, error):
        if url != self.url_input.text().strip():
            return

        self._current_info_url = None
        self._current_video_title = None
        self.info_widget.hide()
        self._reset_quality_options()
        self.quality_combo.setEnabled(False)
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
        url = self.url_input.text().strip()
        quality = self.quality_combo.currentData() or "1080p"
        overwrite = False

        if self._current_video_title:
            existing_path = self.downloader.get_download_path(self._current_video_title, quality)
            if existing_path.exists():
                reply = QMessageBox.question(
                    self,
                    "Video Already Downloaded",
                    "This quality is already downloaded. Use existing file?",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                    QMessageBox.Yes,
                )
                if reply == QMessageBox.Yes:
                    self.accept()
                    self.video_downloaded.emit(str(existing_path))
                    return
                if reply == QMessageBox.Cancel:
                    self.status_label.setText("Download cancelled")
                    return
                overwrite = True
        
        # Create download thread
        output_dir = str(self.downloader.output_dir)
        self.download_thread = YouTubeDownloaderThread(url, output_dir, quality, overwrite=overwrite)
        self.download_thread.preferred_browser = self.preferred_browser()
        self.download_thread.auto_cookie_retry = self.auto_cookie_retry_enabled()
        
        # Connect signals
        self.download_thread.progress_handler.progress.connect(self.update_progress)
        self.download_thread.progress_handler.status.connect(self.update_status)
        self.download_thread.progress_handler.finished.connect(self.on_download_finished)
        self.download_thread.progress_handler.error.connect(self.on_download_error)
        
        # Update UI
        self.download_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setText("Cancel Download")
        self.url_input.setEnabled(False)
        self.fetch_info_btn.setEnabled(False)
        self.quality_combo.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Starting download...")
        self.download_stall_timer.start(self.DOWNLOAD_STALL_DELAY_MS)
        
        # Start download
        self.download_thread.start()
    
    def cancel_download(self):
        """Cancel the current download"""
        self.download_stall_timer.stop()
        if self.download_thread and self.download_thread.isRunning():
            self.download_thread.cancel()
            self.download_thread.quit()
            self.download_thread.wait()
            
        self.reset_ui()
        self.status_label.setText("Download cancelled")
    
    def update_progress(self, value):
        """Update progress bar"""
        self._restart_download_stall_timer()
        if value < 0:
            self.progress_bar.setRange(0, 0)
            return
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(value)
    
    def update_status(self, status):
        """Update status label"""
        self._restart_download_stall_timer()
        self.status_label.setText(status)

    def _restart_download_stall_timer(self):
        if self.download_thread and self._download_thread_is_running():
            self.download_stall_timer.start(self.DOWNLOAD_STALL_DELAY_MS)

    def _download_thread_is_running(self):
        return hasattr(self.download_thread, "isRunning") and self.download_thread.isRunning()

    def _on_download_stall(self):
        if self.download_thread and self._download_thread_is_running():
            self.status_label.setText("Still waiting for YouTube...")
    
    def on_download_finished(self, file_path):
        """Handle successful download"""
        self.download_stall_timer.stop()
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
        self.download_stall_timer.stop()
        self.reset_ui()
        self.status_label.setText("Download failed")
        QMessageBox.critical(self, "Download Error", f"Failed to download video: {error}")
    
    def reset_ui(self):
        """Reset UI to initial state"""
        url = self.url_input.text().strip()
        is_valid = bool(url) and self.downloader.validate_url(url)
        self.download_btn.setEnabled(is_valid and url == self._current_info_url)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Cancel")
        self.url_input.setEnabled(True)
        self.fetch_info_btn.setEnabled(is_valid)
        self.quality_combo.setEnabled(is_valid and url == self._current_info_url)
        self.progress_bar.hide()
        self.progress_bar.setRange(0, 100)

    def _reset_quality_options(self):
        self.quality_combo.clear()
        self.quality_combo.addItem("1080p", "1080p")
        self.quality_combo.addItem("720p - faster processing, higher calibration risk", "720p")
        self.quality_combo.addItem("480p - fastest processing, highest calibration risk", "480p")

    def _apply_available_qualities(self, available_qualities):
        if not available_qualities:
            self._reset_quality_options()
            return

        visible_options = {}
        for preset in self.QUALITY_ORDER:
            quality_info = available_qualities.get(preset) or {}
            if not quality_info.get("available"):
                continue
            actual_height = quality_info.get("actual_height")
            if not actual_height:
                continue
            existing = visible_options.get(actual_height)
            if existing is None or self._prefer_quality_option(preset, actual_height, existing[0]):
                visible_options[actual_height] = (preset, quality_info)

        if not visible_options:
            self._reset_quality_options()
            return

        current_quality = self.quality_combo.currentData()
        self.quality_combo.clear()
        selected_index = 0
        for index, actual_height in enumerate(sorted(visible_options.keys(), reverse=True)):
            preset, quality_info = visible_options[actual_height]
            self.quality_combo.addItem(self._quality_option_label(preset, quality_info), preset)
            if current_quality == preset:
                selected_index = index
        self.quality_combo.setCurrentIndex(selected_index)

    def _prefer_quality_option(self, preset, actual_height, existing_preset):
        preset_height = YouTubeDownloader.QUALITY_PRESETS[preset]["height"]
        existing_height = YouTubeDownloader.QUALITY_PRESETS[existing_preset]["height"]
        preset_exact = preset_height == actual_height
        existing_exact = existing_height == actual_height
        if preset_exact != existing_exact:
            return preset_exact
        return preset_height < existing_height

    def _quality_option_label(self, preset, quality_info):
        actual_height = quality_info.get("actual_height")
        note = quality_info.get("note", "")
        target_height = YouTubeDownloader.QUALITY_PRESETS[preset]["height"]
        if actual_height and actual_height != target_height:
            label = f"Up to {target_height}p ({actual_height}p source)"
        else:
            label = preset
        if note:
            return f"{label} - {note.lower()}"
        return label

    def preferred_browser(self):
        current = self.browser_combo.currentData() if hasattr(self, "browser_combo") else self._preferred_browser
        return current if current in SUPPORTED_COOKIE_BROWSERS else DEFAULT_COOKIE_BROWSER

    def auto_cookie_retry_enabled(self):
        if hasattr(self, "auto_retry_checkbox"):
            return self.auto_retry_checkbox.isChecked()
        return self._auto_cookie_retry

    def _load_preferred_browser(self):
        stored = str(self.settings.value(YOUTUBE_PREFERRED_BROWSER_KEY, DEFAULT_COOKIE_BROWSER) or "")
        stored = stored.strip().lower()
        return stored if stored in SUPPORTED_COOKIE_BROWSERS else DEFAULT_COOKIE_BROWSER

    def _load_auto_cookie_retry(self):
        value = self.settings.value(YOUTUBE_AUTO_COOKIE_RETRY_KEY, True)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in {"0", "false", "no", "off", ""}

    def _on_browser_changed(self):
        self._preferred_browser = self.preferred_browser()
        self.downloader.preferred_browser = self._preferred_browser
        self.settings.setValue(YOUTUBE_PREFERRED_BROWSER_KEY, self._preferred_browser)

    def _on_auto_retry_toggled(self, checked):
        self._auto_cookie_retry = bool(checked)
        self.downloader.auto_cookie_retry = self._auto_cookie_retry
        self.settings.setValue(YOUTUBE_AUTO_COOKIE_RETRY_KEY, self._auto_cookie_retry)
        
    def closeEvent(self, event):
        """Handle dialog close"""
        self.auto_fetch_timer.stop()
        self.download_stall_timer.stop()
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
