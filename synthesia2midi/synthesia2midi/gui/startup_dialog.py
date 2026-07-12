"""Startup dialog for Synthesia2MIDI - Choose between local file or YouTube download"""
import os
from collections import Counter

# Third-party imports
from PySide6.QtCore import QCoreApplication, QSettings, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QWidget,
    QVBoxLayout,
)

from synthesia2midi.localization import (
    load_preferred_locale,
    locale_display_name,
    save_preferred_locale,
    supported_user_locales,
)


class StartupDialog(QDialog):
    """Initial dialog shown on startup to choose video source"""
    
    # Signals for different choices
    open_local_file = Signal()
    download_from_youtube = Signal()
    open_recent_file = Signal(str)
    
    def __init__(self, parent=None, *, recent_video_paths=None, settings=None):
        super().__init__(parent)
        self.recent_video_paths = list(recent_video_paths or [])
        self.recent_video_buttons = []
        self.settings = settings or QSettings("Synthesia2MIDI", "Synthesia2MIDI")
        self.setWindowTitle(QCoreApplication.translate("StartupDialog", "Synthesia to MIDI - Select Video Source"))
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Title
        self.title_label = QLabel(QCoreApplication.translate("StartupDialog", "Welcome to Synthesia to MIDI"))
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)

        self.language_widget = QWidget()
        self.language_widget.setObjectName("startup_language_widget")
        language_layout = QHBoxLayout(self.language_widget)
        language_layout.setContentsMargins(0, 0, 0, 0)
        language_layout.addStretch()
        self.language_label = QLabel(QCoreApplication.translate("StartupDialog", "Language:"))
        self.language_label.setMinimumWidth(self.language_label.sizeHint().width())
        language_layout.addWidget(self.language_label)
        self.language_combo = QComboBox()
        self.language_combo.setObjectName("language_combo")
        current_locale = load_preferred_locale(self.settings)
        self.language_combo.blockSignals(True)
        for locale_name in supported_user_locales():
            self.language_combo.addItem(locale_display_name(locale_name), locale_name)
        selected_index = self.language_combo.findData(current_locale)
        if selected_index >= 0:
            self.language_combo.setCurrentIndex(selected_index)
        self.language_combo.blockSignals(False)
        self.language_combo.currentIndexChanged.connect(self._handle_language_changed)
        language_layout.addWidget(self.language_combo)
        language_layout.addStretch()
        layout.addWidget(self.language_widget)
        
        # Subtitle
        self.subtitle_label = QLabel(QCoreApplication.translate("StartupDialog", "How would you like to load a video?"))
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setWordWrap(True)
        layout.addWidget(self.subtitle_label)

        self.input_cue_label = QLabel(
            QCoreApplication.translate(
                "StartupDialog",
                "Choose a Synthesia-style piano video with visible keys and falling notes.",
            )
        )
        self.input_cue_label.setAlignment(Qt.AlignCenter)
        self.input_cue_label.setWordWrap(True)
        layout.addWidget(self.input_cue_label)
        
        # Add some spacing
        layout.addSpacing(20)
        
        # Buttons container
        button_layout = QVBoxLayout()
        button_layout.setSpacing(15)
        
        # Local file button
        self.local_file_btn = QPushButton(QCoreApplication.translate("StartupDialog", "Open Video File"))
        self.local_file_btn.setMinimumHeight(50)
        self.local_file_btn.setToolTip(QCoreApplication.translate("StartupDialog", "Browse for a video file on your computer"))
        self.local_file_btn.clicked.connect(self._on_local_file_clicked)
        button_layout.addWidget(self.local_file_btn)
        
        # YouTube download button
        self.youtube_btn = QPushButton(QCoreApplication.translate("StartupDialog", "Download from YouTube"))
        self.youtube_btn.setMinimumHeight(50)
        self.youtube_btn.setToolTip(QCoreApplication.translate("StartupDialog", "Download a video from YouTube"))
        self.youtube_btn.clicked.connect(self._on_youtube_clicked)
        button_layout.addWidget(self.youtube_btn)
        
        layout.addLayout(button_layout)

        if self.recent_video_paths:
            recent_separator = QFrame()
            recent_separator.setFrameShape(QFrame.HLine)
            recent_separator.setFrameShadow(QFrame.Sunken)
            layout.addWidget(recent_separator)

            recent_label = QLabel(QCoreApplication.translate("StartupDialog", "Recent Videos"))
            recent_font = QFont()
            recent_font.setBold(True)
            recent_label.setFont(recent_font)
            layout.addWidget(recent_label)

            recent_layout = QVBoxLayout()
            recent_layout.setSpacing(4)
            filename_counts = Counter(
                (os.path.basename(path) or path).casefold()
                for path in self.recent_video_paths
            )
            for path in self.recent_video_paths:
                recent_button = self._create_recent_video_button(
                    path,
                    duplicate_name=filename_counts[(os.path.basename(path) or path).casefold()] > 1,
                )
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
        cancel_btn = QPushButton(QCoreApplication.translate("StartupDialog", "Cancel"))
        cancel_btn.clicked.connect(self.reject)
        cancel_layout = QHBoxLayout()
        # Keep the cancel button left-aligned.
        cancel_layout.addWidget(cancel_btn)
        layout.addLayout(cancel_layout)
        
        # Set default button
        self.local_file_btn.setDefault(True)
        self.local_file_btn.setFocus()
        QWidget.setTabOrder(self.local_file_btn, self.youtube_btn)
        if self.recent_video_buttons:
            QWidget.setTabOrder(self.youtube_btn, self.recent_video_buttons[0])
        layout.activate()
        opening_hint = self.sizeHint()
        self.resize(
            max(self.width(), opening_hint.width()),
            max(self.height(), opening_hint.height()),
        )
        
    def _on_local_file_clicked(self):
        """Handle local file button click"""
        self.accept()
        self.open_local_file.emit()
        
    def _on_youtube_clicked(self):
        """Handle YouTube button click"""
        self.accept()
        self.download_from_youtube.emit()

    def _handle_language_changed(self, index: int):
        """Persist the selected UI language for the next app launch."""
        locale_name = self.language_combo.itemData(index)
        if not locale_name:
            return
        save_preferred_locale(str(locale_name), self.settings)
        QMessageBox.information(
            self,
            QCoreApplication.translate("StartupDialog", "Language"),
            QCoreApplication.translate("StartupDialog", "Restart Synthesia2MIDI to apply the selected language."),
        )

    def _create_recent_video_button(self, path: str, *, duplicate_name: bool = False) -> QPushButton:
        filename = os.path.basename(path) or path
        exists = os.path.exists(path)
        if not exists:
            label = QCoreApplication.translate("StartupDialog", "{filename} (missing)").format(
                filename=filename
            )
        elif duplicate_name:
            parent_name = os.path.basename(os.path.dirname(path)) or os.path.dirname(path)
            label = QCoreApplication.translate("StartupDialog", "{filename} — {folder}").format(
                filename=filename,
                folder=parent_name,
            )
        else:
            label = filename
        button = QPushButton(label)
        if button.fontMetrics().horizontalAdvance(label) > 430:
            extension = os.path.splitext(filename)[1]
            stem = filename[: -len(extension)] if extension else filename
            label_suffix = label[len(filename):] if label.startswith(filename) else ""
            preserved_suffix = extension + label_suffix
            stem_width = max(
                24,
                430 - button.fontMetrics().horizontalAdvance(preserved_suffix),
            )
            visible_stem = button.fontMetrics().elidedText(
                stem, Qt.ElideMiddle, stem_width
            )
            button.setText(visible_stem + preserved_suffix)
        button.setMinimumHeight(36)
        button.setStyleSheet("QPushButton { text-align: left; padding: 5px 8px; }")
        button.setToolTip(path)
        button.setAccessibleName(label)
        button.setAccessibleDescription(path)
        button.setEnabled(exists)
        if exists:
            button.clicked.connect(lambda checked=False, selected_path=path: self._on_recent_file_clicked(selected_path))
        return button

    def _on_recent_file_clicked(self, path: str):
        """Handle recent file click"""
        self.accept()
        self.open_recent_file.emit(path)
