"""
synthesia2midi - Main Application Entry Point.

Primary application module that orchestrates the complete synthesia2midi system.
Manages the Qt-based GUI, coordinates workflows, and integrates all detection,
calibration, and MIDI generation components into a unified desktop application.

Key Responsibilities:
- QMainWindow management and GUI initialization
- Video loading and session management
- Detection method coordination and switching
- Calibration workflow integration
- MIDI conversion process management
- User interaction handling and state updates
- Cross-component signal routing and communication

This is the central hub that connects all synthesia2midi subsystems while
maintaining clean separation of concerns through workflow patterns.
"""
import datetime
import glob
import logging
import os
import subprocess
import sys
from typing import List, Optional, Tuple

import cv2  # For HSV color space conversion
import numpy as np  # For image data
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QDialog, QFileDialog, QHBoxLayout,
    QInputDialog, QLabel, QListView, QMainWindow, QMessageBox, QSlider,
    QTreeView, QVBoxLayout, QWidget
)

from synthesia2midi.app_config import (
    APP_NAME, DEFAULT_BLACK_KEY_STYLE, DEFAULT_WHITE_KEY_STYLE,
    FRAME_NAV_INTERVALS, LOG_DIR, NOTE_NAMES_SHARP, OverlayConfig
)
from synthesia2midi.config_manager import ConfigManager
from synthesia2midi.core.app_state import AppState
from synthesia2midi.core.logging_config import LoggingConfig
from synthesia2midi.core.state_manager import StateManager
from synthesia2midi.detection.factory import DetectionFactory
from synthesia2midi.detection.roi_utils import get_hist_feature
from synthesia2midi.gui.controls_qt import ControlPanelQt, KEY_TYPES
from synthesia2midi.gui.display_manager import DisplayManager
from synthesia2midi.gui.keyboard_canvas import KeyboardCanvas
from synthesia2midi.gui.signal_manager import ControlSignalManager
from synthesia2midi.gui.startup_dialog import StartupDialog
from synthesia2midi.gui.youtube_download_dialog import YouTubeDownloadDialog
from synthesia2midi.gui.ui_update_interface import UIUpdateInterface
from synthesia2midi.gui.video_controls import VideoControls
from synthesia2midi.gui.window_manager import WindowManager
from synthesia2midi.video_loader import VideoSession
from synthesia2midi.workflows.auto_calibration import AutoCalibrationWorkflow
from synthesia2midi.workflows.calibration import CalibrationWorkflow
from synthesia2midi.workflows.conversion import ConversionWorkflow
from synthesia2midi.workflows.detection_manager import DetectionManager
from synthesia2midi.workflows.overlay_manager import OverlayManager
from synthesia2midi.workflows.parameter_manager import ParameterManager
from synthesia2midi.workflows.video_loading import VideoLoadingWorkflow

# Configure application logging. Enable console logging during development if needed.
log_filename = LoggingConfig.setup_logging(
    log_to_file=True,
    log_to_console=False,  # Set to True for development
    log_level=logging.INFO
)
LoggingConfig.suppress_verbose_libraries()  # Suppress third-party library logs


class VideoToFramesWorker(QThread):
    """Worker thread for video to frames conversion to avoid blocking the GUI."""
    
    # Signals for progress updates
    progress_updated = Signal(str)  # progress message
    conversion_finished = Signal(bool, str)  # success, result_message
    
    def __init__(self, video_path: str, output_dir: str, quality: int = 90):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.quality = quality
    
    def run(self):
        """Run the video conversion in a separate thread."""
        try:
            self.progress_updated.emit("Starting video to frame series conversion...")
            
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Build ffmpeg command  
            from synthesia2midi.utils.ffmpeg_helper import find_ffmpeg
            ffmpeg_path = find_ffmpeg()
            if not ffmpeg_path:
                self.conversion_finished.emit(False, "FFmpeg not found. Please install FFmpeg.")
                return
                
            output_pattern = os.path.join(self.output_dir, "frame_%06d.jpg")
            cmd = [
                ffmpeg_path, '-y', '-i', self.video_path,
                '-q:v', str(100 - self.quality),  # ffmpeg uses inverse scale
                '-vf', 'format=bgr24',  # Ensure BGR format for OpenCV
                output_pattern
            ]
            
            self.progress_updated.emit("Running ffmpeg conversion...")
            
            # Run ffmpeg conversion
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Count frames to verify success
                frame_count = len(glob.glob(os.path.join(self.output_dir, "frame_*.jpg")))
                success_msg = f"Successfully converted {frame_count} frames to {self.output_dir}"
                self.progress_updated.emit(success_msg)
                self.conversion_finished.emit(True, success_msg)
            else:
                # Clean up stderr message for user-friendly display
                stderr = result.stderr.strip()
                if "Is a directory" in stderr:
                    error_msg = "Error: Input path is a directory, not a video file. Please load the original video file."
                elif "No such file or directory" in stderr:
                    error_msg = "Error: Video file not found. Please check the file path."
                elif stderr:
                    # Show only the last few lines of stderr to avoid overwhelming the user
                    stderr_lines = stderr.split('\n')
                    relevant_lines = [line for line in stderr_lines[-10:] if line.strip() and not line.startswith('[')]
                    error_msg = f"FFmpeg conversion failed:\n{chr(10).join(relevant_lines)}"
                else:
                    error_msg = f"FFmpeg conversion failed with return code: {result.returncode}"
                self.conversion_finished.emit(False, error_msg)
                
        except Exception as e:
            error_msg = f"Error during conversion: {str(e)}"
            self.conversion_finished.emit(False, error_msg)


class Video2MidiApp(QMainWindow, UIUpdateInterface):
    """Main application class with UI update interface implementation."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        # Set initial size that will be adjusted when video loads
        self.resize(1200, 800)  # Reduced width to match control panel constraints

        self.app_state = AppState()
        self.state_manager = StateManager(self.app_state)
        self.video_session: VideoSession | None = None
        self.config_manager = ConfigManager(self.app_state)
        
        # Initialize ROI utils with app_state reference for downsampling
        from synthesia2midi.detection.roi_utils import set_app_state_reference
        set_app_state_reference(self.app_state)
        
        
        # Initialize workflow modules
        self.video_loading_workflow = VideoLoadingWorkflow(self.app_state, self.config_manager, self)
        self.parameter_manager = ParameterManager(self.app_state, self.state_manager, self)
        self.window_manager = WindowManager(self.app_state, self)
        self.calibration_workflow = None  # Will be initialized when video is loaded
        self.auto_calibration_workflow = None  # Will be initialized when video is loaded
        self.conversion_workflow = None  # Will be initialized when video is loaded
        self.debug_tools = None  # Will be initialized when video is loaded
        self.detection_manager = None  # Will be initialized when video is loaded
        self.overlay_manager = OverlayManager(self.app_state, self)
        self.display_manager = DisplayManager(self.app_state, self)
        
        # Video to frames conversion worker
        self.video_to_frames_worker = None
        
        # Frame slider handling is now done by VideoControls class
        
        # Detection parameter logging
        self._detection_logging_enabled = False
        self._detection_log_data = []
        self._detection_log_start_time = None
        
        self._init_ui()
        self._bind_hotkeys()

        logging.info(f"{APP_NAME} started.")
        
        
        # Show startup dialog instead of directly opening file dialog
        QTimer.singleShot(100, self._show_startup_dialog)


    def _init_ui(self):
        # --- Menu ---
        menubar = self.menuBar()
        
        # File menu
        filemenu = menubar.addMenu("File")
        open_action = QAction("Open Video (MP4)...", self)
        open_action.triggered.connect(self._open_video_file)
        filemenu.addAction(open_action)
        
        youtube_action = QAction("Download Youtube Video...", self)
        youtube_action.triggered.connect(self._show_youtube_download_dialog)
        filemenu.addAction(youtube_action)
        
        save_action = QAction("Save Settings (Ctrl+S)", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_settings)
        filemenu.addAction(save_action)
        
        filemenu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        filemenu.addAction(exit_action)


        # View menu
        view_menu = menubar.addMenu("View")
        self.show_overlays_action = QAction("Show Overlays", self)
        self.show_overlays_action.setCheckable(True)
        self.show_overlays_action.setChecked(self.app_state.ui.show_overlays)
        self.show_overlays_action.triggered.connect(self._toggle_overlays)
        view_menu.addAction(self.show_overlays_action)
        
        self.live_detection_action = QAction("Live Detection Feedback", self)
        self.live_detection_action.setCheckable(True)
        self.live_detection_action.setChecked(self.app_state.ui.live_detection_feedback)
        self.live_detection_action.triggered.connect(self._toggle_live_detection_feedback)
        view_menu.addAction(self.live_detection_action)
        
        view_menu.addSeparator()
        
        
        view_menu.addSeparator()
        
        # Frame Navigation menu
        frame_nav_menu = menubar.addMenu("Frame Navigation")
        
        # Create interval menu items with checkmarks
        self.frame_nav_actions = {}
        for interval in FRAME_NAV_INTERVALS:
            action = QAction(f"{interval} frame{'s' if interval != 1 else ''}", self)
            action.setCheckable(True)
            action.setChecked(self.app_state.video.current_nav_interval == interval)
            action.triggered.connect(lambda checked, val=interval: self._handle_frame_nav_interval(val))
            frame_nav_menu.addAction(action)
            self.frame_nav_actions[interval] = action

        # Visual Threshold Monitor menu
        debug_menu = menubar.addMenu("Visual Threshold Monitor")
        
        # Visual Threshold Monitor toggle
        self.visual_threshold_monitor_action = QAction("Enable", self)
        self.visual_threshold_monitor_action.setCheckable(True)
        self.visual_threshold_monitor_action.setChecked(self.app_state.ui.visual_threshold_monitor_enabled)
        self.visual_threshold_monitor_action.triggered.connect(self._handle_visual_threshold_monitor_menu)
        debug_menu.addAction(self.visual_threshold_monitor_action)
        
        debug_menu.addSeparator()
        
        # Screenshot capture action
        capture_action = QAction("Capture Window Screenshot", self)
        capture_action.setShortcut("Ctrl+Shift+C")
        capture_action.triggered.connect(self._capture_window_screenshot)
        debug_menu.addAction(capture_action)

        # --- Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create a main vertical layout to control alignment
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)  # No margins on the central layout
        central_layout.setSpacing(0)
        
        # Create the actual content widget
        content_widget = QWidget()
        main_layout = QHBoxLayout(content_widget)
        main_layout.setContentsMargins(5, 5, 5, 10)  # Limit bottom padding to 10px
        main_layout.setSpacing(10)  # Add spacing between canvas and controls
        
        # Add content widget to central layout with stretch to fill available space
        from PySide6.QtCore import Qt
        central_layout.addWidget(content_widget, 1)
        
        # Left side layout for canvas and frame slider
        left_layout = QVBoxLayout()
        left_layout.setSpacing(5)
        
        # Canvas - Variable width
        self.keyboard_canvas = KeyboardCanvas(self.app_state, width=800, height=600,
                                              on_color_pick_callback=self._handle_color_pick,
                                              on_overlay_select_callback=self._handle_overlay_selection,
                                              detect_pressed_func=self._create_detection_wrapper()
                                              )
        # Set up additional callbacks
        self.keyboard_canvas.on_spark_roi_callback = self._handle_spark_roi_updated
        # Give canvas stretch factor so it expands to fill available vertical space
        left_layout.addWidget(self.keyboard_canvas, 1)  # Stretch factor 1
        
        # Frame slider with time display
        slider_layout = QHBoxLayout()
        slider_layout.setSpacing(10)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)  # Will be updated when video loads
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)  # Disabled until video loads
        # Frame slider signals are wired via ControlSignalManager/VideoControls.
        # Enable tracking for real-time time display updates (frame loading is still debounced)
        self.frame_slider.setTracking(True)  
        # Set a reasonable height for the slider
        self.frame_slider.setMaximumHeight(30)
        slider_layout.addWidget(self.frame_slider, 0)  # No stretch factor
        
        # Time display label
        self.time_label = QLabel("0:00")
        self.time_label.setMinimumWidth(60)
        self.time_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        slider_layout.addWidget(self.time_label, 0)  # No stretch
        
        # Navigation instructions
        nav_instructions = QLabel("Move forward – Page Down\nMove backward – Page Up")
        nav_instructions.setStyleSheet("font-size: 12px; color: #666; margin-left: 15px;")
        slider_layout.addWidget(nav_instructions, 0)  # No stretch
        
        left_layout.addLayout(slider_layout, 0)  # No stretch
        
        # Create a widget to contain the left layout
        left_widget = QWidget()
        # Set size policy to prevent vertical expansion beyond content
        from PySide6.QtWidgets import QSizePolicy
        left_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        left_widget.setLayout(left_layout)
        main_layout.addWidget(left_widget, 0)  # No stretch factor

        # Control Panel (Right) - Fixed width container
        self.control_panel = ControlPanelQt(self, self.app_state, self.state_manager)
        
        # Connect control panel to canvas for ROI visualization updates
        self.control_panel.canvas_refresh_callback = lambda: self.keyboard_canvas.refresh_spark_roi_visualization()
        
        # Initialize video controls module
        self.video_controls = VideoControls(
            self.app_state, 
            self.video_session, 
            self.keyboard_canvas, 
            self.frame_slider, 
            self.time_label
        )
        
        # Set up detection logging callback for video controls
        self.video_controls.set_detection_logging_callback(self._log_detection_parameters)
        
        # Connect video controls to control panel for trim functionality
        self.control_panel.video_controls = self.video_controls
        self.control_panel.keyboard_canvas = self.keyboard_canvas
        
        # Use signal manager for all control panel connections
        self.signal_manager = ControlSignalManager(self.control_panel, self)
        
        
        # Initial width, will be adjusted when video loads
        self.control_panel.setFixedWidth(900)  # Increased for 14pt font readability
        # Prevent control panel from expanding vertically - keep it compact
        from PySide6.QtWidgets import QSizePolicy
        self.control_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        main_layout.addWidget(self.control_panel, 0)  # No stretch factor
    
    def resizeEvent(self, event):
        """Delegate window resize handling to WindowManager."""
        self.window_manager.handle_resize_event(event)
    
    def showEvent(self, event):
        """Delegate window show handling to WindowManager."""
        self.window_manager.handle_show_event(event)


    def _bind_hotkeys(self):
        # Control+S is already handled by the menu action
        # Space key for conversion
        space_action = QAction(self)
        space_action.setShortcut(Qt.Key_Space)
        space_action.triggered.connect(self._start_conversion_process)
        self.addAction(space_action)
        
        # Page Up/Down for navigation
        pgup_action = QAction(self)
        pgup_action.setShortcut(Qt.Key_PageUp)
        pgup_action.triggered.connect(self._navigate_frame_pgup)
        self.addAction(pgup_action)
        
        pgdn_action = QAction(self)
        pgdn_action.setShortcut(Qt.Key_PageDown)
        pgdn_action.triggered.connect(self._navigate_frame_pgdn)
        self.addAction(pgdn_action)

    def _show_startup_dialog(self):
        """Show the startup dialog for choosing video source."""
        logging.info("_show_startup_dialog: Showing startup dialog.")
        
        dialog = StartupDialog(self)
        dialog.open_local_file.connect(self._open_video_file)
        dialog.download_from_youtube.connect(self._show_youtube_download_dialog)
        
        # If user cancels, just continue with empty application
        if dialog.exec() != QDialog.Accepted:
            logging.info("_show_startup_dialog: User cancelled startup dialog, continuing with empty application.")
            # No video loaded, but app remains open
    
    def _show_youtube_download_dialog(self):
        """Show the YouTube download dialog."""
        logging.info("_show_youtube_download_dialog: Showing YouTube download dialog.")
        
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        videos_dir = os.path.join(project_root, 'videos')
        
        dialog = YouTubeDownloadDialog(self, default_output_dir=videos_dir)
        dialog.video_downloaded.connect(self._handle_youtube_video_downloaded)
        
        if dialog.exec() != QDialog.Accepted:
            # If user cancels YouTube dialog, just continue with empty application
            logging.info("_show_youtube_download_dialog: User cancelled YouTube dialog, continuing with empty application.")
            # No video loaded, but app remains open

    def _open_video_file(self):
        """Open a video file or image sequence directory using VideoLoadingWorkflow."""
        logging.info("_open_video_file: Method started.")
        
        # Get the project root directory as starting location
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Custom dialog that shows both files and directories
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Select Video File or Image Sequence Directory")
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setOption(QFileDialog.ShowDirsOnly, False)
        dialog.setNameFilter("Video/Images (*.mp4 *.avi *.mov *.jpg *.png);;All files (*.*)")
        
        # Set dialog size - expand width by factor of 2
        dialog.resize(1200, 800)  # Default QFileDialog is typically around 600x400
        
        # Set default directory to project root
        if os.path.exists(project_root):
            dialog.setDirectory(project_root)
            logging.info(f"_open_video_file: Set default directory to: {project_root}")
        else:
            logging.warning(f"_open_video_file: Project root directory not found: {project_root}")
        
        # Try non-native dialog first for better file/directory selection
        use_native = False
        try:
            dialog.setOption(QFileDialog.DontUseNativeDialog, True)
            file_view = dialog.findChild(QListView)
            if file_view:
                file_view.setSelectionMode(QAbstractItemView.SingleSelection)
            tree_view = dialog.findChild(QTreeView)
            if tree_view:
                tree_view.setSelectionMode(QAbstractItemView.SingleSelection)
        except Exception as e:
            logging.warning(f"_open_video_file: Non-native dialog setup failed: {e}")
            use_native = True
            
        # Show dialog
        if not dialog.exec():
            logging.info("_open_video_file: Dialog cancelled.")
            # If non-native dialog failed and user cancelled, try native dialog
            if not use_native:
                logging.info("_open_video_file: Trying native Windows dialog as fallback...")
                filepath, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select Video File",
                    project_root if os.path.exists(project_root) else "",
                    "Video/Images (*.mp4 *.avi *.mov *.jpg *.png);;All files (*.*)"
                )
                if not filepath:
                    logging.info("_open_video_file: Native dialog also cancelled.")
                    return
            else:
                return
        else:
            selected = dialog.selectedFiles()
            if not selected:
                logging.info("_open_video_file: No file selected.")
                return
            filepath = selected[0]
        logging.info(f"_open_video_file: filedialog returned: '{filepath if filepath else "Dialog cancelled or no file selected"}'")
        if not filepath:
            logging.info("_open_video_file: No filepath selected, returning.")
            return

        logging.info("_open_video_file: Proceeding with filepath.")

        # Close existing video session if any
        try:
            if self.video_session:
                logging.info("_open_video_file: Closing existing video session.")
                self.video_session.close()
                self.video_session = None
                logging.info("_open_video_file: Existing video session closed.")
        except Exception as e_close_session:
            logging.error(f"_open_video_file: Error closing existing video session: {e_close_session}", exc_info=True)

        # Reset all state so video-specific settings don't carry over to the next video.
        # The new video's INI (if present) will repopulate state after load.
        self.state_manager.reset_to_defaults()

        # Use VideoLoadingWorkflow to load the video and configuration
        success, video_session = self.video_loading_workflow.load_video_file(filepath)
        if not success:
            return  # Error already shown by workflow
            
        # Set the video session
        self.video_session = video_session
        
        # Update video controls with new video session
        self.video_controls.set_video_session(video_session)
        
        # Set video session in canvas
        try:
            logging.info("_open_video_file: Setting video session in KeyboardCanvas.")
            if self.keyboard_canvas:
                self.keyboard_canvas.set_video_session(self.video_session)
                logging.info("_open_video_file: Video session set in KeyboardCanvas.")
            else:
                logging.warning("_open_video_file: KeyboardCanvas not initialized when trying to set video session.")
        except Exception as e_set_canvas_session:
            logging.error(f"_open_video_file: Error setting video session in KeyboardCanvas: {e_set_canvas_session}", exc_info=True)
        
        # Update frame slider for new video (but don't set position yet)
        self._update_frame_slider_for_video()
        
        # Update video frame limits for trim controls
        self.control_panel.update_video_frame_limits()
        
        # Update FPS display with detected FPS
        if self.video_session:
            self.control_panel.update_video_info(self.video_session.fps)
        
        # Initialize workflow objects now that video is loaded
        self.calibration_workflow = CalibrationWorkflow(self.app_state, self.video_session, self)
        self.auto_calibration_workflow = AutoCalibrationWorkflow(self.app_state, self.video_session, self)
        # Create detection manager first since conversion workflow needs it
        self.detection_manager = DetectionManager(self.app_state, self._update_current_frame_display, self)
        self.conversion_workflow = ConversionWorkflow(
            self.app_state, 
            self.video_session, 
            self,
            self.detection_manager
        )
        
        # Update keyboard canvas with detection function now that detection_manager exists
        self.keyboard_canvas.detect_pressed_func = self._create_detection_wrapper()
        
        # Check if configuration was loaded (workflow handles this)
        video_info = self.video_loading_workflow.get_video_info()
        config_loaded_from_specific_ini = video_info.get('config_file') is not None

        if config_loaded_from_specific_ini:
            # Standard procedure if a video-specific config was found and loaded
            self.control_panel.update_controls_from_state()
            # Update frame range controls to reflect loaded processing/trim ranges
            self.control_panel.update_trim_controls_from_state()
            # Set processing range defaults based on trim range if not already set
            self._initialize_processing_range_defaults()
            # Start at processing start frame if set, otherwise use saved frame index
            initial_frame = self.app_state.video.processing_start_frame if self.app_state.video.processing_start_frame > 0 else self.app_state.video.current_frame_index
            self._display_frame_with_slider_update(initial_frame)
            if self.app_state.overlays:
                 self.control_panel.convert_button.setEnabled(True)
                 self.control_panel.wizard_button.setEnabled(True) # Allow re-running wizard
            else:
                 self.control_panel.convert_button.setEnabled(False)
                 self.control_panel.wizard_button.setEnabled(True) # Allow running wizard
            
            # Resize window based on screen size to ensure everything is visible
            self._resize_and_position_window()
            
            # Initialize calibration frame range display and buttons
            
        else:
            # No video-specific INI found or failed to load.
            # Overlays will be empty. User must run wizard manually.
            logging.info("Video-specific INI not found or failed to load. User must run calibration wizard.")
            self.app_state.overlays.clear() # Ensure overlays are empty
            self.app_state.unsaved_changes = False # No config loaded, no initial changes
            
            self.control_panel.update_controls_from_state() # Update UI (e.g. disable convert)
            # Update frame range controls to reflect default/current state
            self.control_panel.update_trim_controls_from_state()
            # Display first frame or trim start if video is trimmed
            initial_frame = self.app_state.video.trim_start_frame if self.app_state.video.video_is_trimmed else 0
            self._display_frame_with_slider_update(initial_frame) # Display initial frame of video
            self.control_panel.convert_button.setEnabled(False)
            self.control_panel.wizard_button.setEnabled(True) # Enable wizard button
            
            # No automatic styling here; styling will happen after manual wizard invocation.
            
            # Resize window based on screen size to ensure everything is visible
            self._resize_and_position_window()
            
            # Initialize calibration frame range display and buttons
            

    def _handle_youtube_video_downloaded(self, filepath: str):
        """Handle a video downloaded from YouTube."""
        logging.info(f"_handle_youtube_video_downloaded: Loading YouTube video from {filepath}")
        logging.info(f"_handle_youtube_video_downloaded: Auto-convert setting: {self.app_state.ui.auto_convert_to_frames}")
        
        # Close existing video session if any
        try:
            if self.video_session:
                logging.info("_handle_youtube_video_downloaded: Closing existing video session.")
                self.video_session.close()
                self.video_session = None
                logging.info("_handle_youtube_video_downloaded: Existing video session closed.")
        except Exception as e_close_session:
            logging.error(f"_handle_youtube_video_downloaded: Error closing existing video session: {e_close_session}", exc_info=True)

        # Reset all state so video-specific settings don't carry over to the next video.
        # The new video's INI (if present) will repopulate state after load.
        self.state_manager.reset_to_defaults()
        
        # Use VideoLoadingWorkflow to load the video and configuration
        logging.info(f"_handle_youtube_video_downloaded: Calling VideoLoadingWorkflow.load_video_file({filepath})")
        success, video_session = self.video_loading_workflow.load_video_file(filepath)
        if not success:
            return  # Error already shown by workflow
        
        # Set the video session
        self.video_session = video_session
        
        # Update video controls with new video session
        self.video_controls.set_video_session(video_session)
        
        # Set video session in canvas
        try:
            logging.info("_handle_youtube_video_downloaded: Setting video session in KeyboardCanvas.")
            if self.keyboard_canvas:
                self.keyboard_canvas.set_video_session(self.video_session)
                logging.info("_handle_youtube_video_downloaded: Video session set in KeyboardCanvas.")
            else:
                logging.warning("_handle_youtube_video_downloaded: KeyboardCanvas not initialized when trying to set video session.")
        except Exception as e_set_canvas_session:
            logging.error(f"_handle_youtube_video_downloaded: Error setting video session in KeyboardCanvas: {e_set_canvas_session}", exc_info=True)
        
        # Update frame slider for new video (but don't set position yet)
        self._update_frame_slider_for_video()
        
        # Update video frame limits for trim controls
        self.control_panel.update_video_frame_limits()
        
        # Initialize workflow objects now that video is loaded
        self.calibration_workflow = CalibrationWorkflow(self.app_state, self.video_session, self)
        self.auto_calibration_workflow = AutoCalibrationWorkflow(self.app_state, self.video_session, self)
        # Create detection manager first since conversion workflow needs it
        self.detection_manager = DetectionManager(self.app_state, self._update_current_frame_display, self)
        self.conversion_workflow = ConversionWorkflow(
            self.app_state, 
            self.video_session, 
            self,
            self.detection_manager
        )
        
        # Update keyboard canvas with detection function now that detection_manager exists
        self.keyboard_canvas.detect_pressed_func = self._create_detection_wrapper()
        
        # Check if configuration was loaded (workflow handles this)
        video_info = self.video_loading_workflow.get_video_info()
        config_loaded_from_specific_ini = video_info.get('config_file') is not None

        if config_loaded_from_specific_ini:
            # Standard procedure if a video-specific config was found and loaded
            self.control_panel.update_controls_from_state()
            # Update frame range controls to reflect loaded processing/trim ranges
            self.control_panel.update_trim_controls_from_state()
            # Set processing range defaults based on trim range if not already set
            self._initialize_processing_range_defaults()
            # Start at processing start frame if set, otherwise use saved frame index
            initial_frame = self.app_state.video.processing_start_frame if self.app_state.video.processing_start_frame > 0 else self.app_state.video.current_frame_index
            self._display_frame_with_slider_update(initial_frame)
            if self.app_state.overlays:
                 self.control_panel.convert_button.setEnabled(True)
                 self.control_panel.wizard_button.setEnabled(True) # Allow re-running wizard
            else:
                 self.control_panel.convert_button.setEnabled(False)
                 self.control_panel.wizard_button.setEnabled(True) # Allow running wizard
            
            # Resize window based on screen size to ensure everything is visible
            self._resize_and_position_window()
        else:
            # No video-specific INI found or failed to load.
            # Overlays will be empty. User must run wizard manually.
            logging.info("Video-specific INI not found or failed to load. User must run calibration wizard.")
            self.app_state.overlays.clear() # Ensure overlays are empty
            self.app_state.unsaved_changes = False # No config loaded, no initial changes
            
            self.control_panel.update_controls_from_state() # Update UI (e.g. disable convert)
            # Update frame range controls to reflect default/current state
            self.control_panel.update_trim_controls_from_state()
            # Display first frame or trim start if video is trimmed
            initial_frame = self.app_state.video.trim_start_frame if self.app_state.video.video_is_trimmed else 0
            self._display_frame_with_slider_update(initial_frame) # Display initial frame of video
            self.control_panel.convert_button.setEnabled(False)
            self.control_panel.wizard_button.setEnabled(True) # Enable wizard button
            
            # Resize window based on screen size to ensure everything is visible
            self._resize_and_position_window()

    def _handle_video_to_frames_request(self):
        """Handle request to convert current video to frame series."""
        if not self.app_state.video.filepath:
            QMessageBox.warning(self, "Video to Frames", "No video file is open. Open a video first.")
            return
            
        # Check if ffmpeg is available
        from synthesia2midi.utils.ffmpeg_helper import check_ffmpeg_available
        is_available, message = check_ffmpeg_available()
        if not is_available:
            QMessageBox.critical(
                self, "FFmpeg Not Found", 
                f"{message}\n\n"
                "Please install FFmpeg:\n"
                "• Windows: Download from https://ffmpeg.org/download.html\n"
                "• macOS: brew install ffmpeg\n"
                "• Linux: sudo apt install ffmpeg"
            )
            return
        
        # Check if current loaded "video" is actually a frame series directory
        video_path = self.app_state.video.filepath
        
        if os.path.isdir(video_path):
            # User has loaded a frame series, try to find the original video file
            # Frame series directories typically end with "_frames"
            if video_path.endswith("_frames"):
                # Remove "_frames" suffix to get base video name
                base_path = video_path[:-7]  # Remove "_frames"
                parent_dir = os.path.dirname(base_path)
                base_name = os.path.basename(base_path)
                
                # Look for video files with common extensions
                video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.m4v']
                original_video = None
                
                for ext in video_extensions:
                    candidate_path = base_path + ext
                    if os.path.isfile(candidate_path):
                        original_video = candidate_path
                        break
                
                if original_video:
                    # Found the original video file, use it instead
                    video_path = original_video
                    QMessageBox.information(
                        self, "Video to Frames",
                        f"Frame series is currently loaded. Found original video file:\n\n"
                        f"{os.path.basename(original_video)}\n\n"
                        f"Will convert this video to update the frame series."
                    )
                else:
                    # Could not find original video file
                    QMessageBox.warning(
                        self, "Video to Frames",
                        f"A frame series is currently loaded, but the original video file could not be found.\n\n"
                        f"Frame series path: {video_path}\n"
                        f"Expected video in: {parent_dir}/\n"
                        f"With name: {base_name}.mp4 (or .mov, .avi, etc.)\n\n"
                        f"Please load the original video file manually."
                    )
                    return
            else:
                # Directory doesn't end with "_frames", unclear what this is
                QMessageBox.warning(
                    self, "Video to Frames",
                    f"A directory is currently loaded, but it doesn't appear to be a frame series.\n\n"
                    f"Current path: {video_path}\n\n"
                    f"Please load a video file (.mp4, .mov, etc.) to convert it to frames."
                )
                return
            
        if not os.path.isfile(video_path):
            QMessageBox.warning(
                self, "Video to Frames",
                f"The video file path is not valid:\n{video_path}\n\n"
                "Please load a valid video file first."
            )
            return
            
        # Determine output directory based on video filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(os.path.dirname(video_path), f"{base_name}_frames")
        
        # Confirm with user before proceeding
        reply = QMessageBox.question(
            self, "Convert Video to Frame Series",
            f"This will convert the current video to a frame series:\n\n"
            f"Video: {os.path.basename(video_path)}\n"
            f"Output: {output_dir}\n\n"
            f"This may take several minutes and will overwrite any existing frame series.\n\n"
            f"Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Disable the button during conversion
        self.control_panel.video_to_frames_button.setEnabled(False)
        self.control_panel.video_to_frames_button.setText("Converting...")
        
        # Create and start worker thread
        self.video_to_frames_worker = VideoToFramesWorker(video_path, output_dir, quality=90)
        self.video_to_frames_worker.progress_updated.connect(self._on_conversion_progress)
        self.video_to_frames_worker.conversion_finished.connect(self._on_conversion_finished)
        self.video_to_frames_worker.start()
    
    def _on_conversion_progress(self, message: str):
        """Handle progress updates from video conversion."""
        # Update status bar or show progress dialog
        logging.info(f"Video conversion progress: {message}")
        # You could add a progress dialog here if needed
    
    def _on_conversion_finished(self, success: bool, message: str):
        """Handle completion of video conversion."""
        # Re-enable the button
        self.control_panel.video_to_frames_button.setEnabled(True)
        self.control_panel.video_to_frames_button.setText("Reset Video -> Frame Series")
        
        # Clean up worker
        if self.video_to_frames_worker:
            self.video_to_frames_worker.deleteLater()
            self.video_to_frames_worker = None
        
        if success:
            QMessageBox.information(
                self, "Conversion Complete",
                f"Video conversion completed successfully!\n\n{message}\n\n"
                f"You can now load the frame series for faster playback."
            )
            logging.info(f"Video to frames conversion completed: {message}")
        else:
            QMessageBox.critical(
                self, "Conversion Failed",
                f"Video conversion failed:\n\n{message}"
            )
            logging.error(f"Video to frames conversion failed: {message}")

    def _save_settings(self):
        if not self.app_state.video.filepath:
            QMessageBox.warning(self, "Save Settings", "No video file is open. Open a video first.")
            return
        
        success = self.video_loading_workflow.save_current_config()
        if success:
            QMessageBox.information(self, "Save Settings", "Settings saved successfully.")
            logging.info(f"Settings saved for {self.app_state.video.filepath}")
        else:
            QMessageBox.critical(self, "Save Settings", "Failed to save settings.")

    def closeEvent(self, event):
        if self.app_state.unsaved_changes:
            reply = QMessageBox.question(self, "Exit", "You have unsaved changes. Save before exiting?",
                                         QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Yes:
                self._save_settings()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
            # If No or save complete, proceed to close
        
        # Clean up canvas resources
        if hasattr(self, 'keyboard_canvas') and self.keyboard_canvas:
            self.keyboard_canvas.cleanup()
        
        if self.video_session:
            self.video_session.release()
        logging.info(f"{APP_NAME} closing.")
        event.accept()

    def _update_tempo(self, value: int):
        """Delegate tempo update to ParameterManager."""
        self.parameter_manager.update_tempo(value)

    def _update_nav_interval(self, value: int):
        """Delegate navigation interval update to ParameterManager and update menu."""
        self.parameter_manager.update_nav_interval(value)
        
        # Update menu check states when interval changes from other sources
        if hasattr(self, 'frame_nav_actions'):
            for nav_interval, action in self.frame_nav_actions.items():
                action.setChecked(nav_interval == value)


    def _navigate_frame_pgup(self):
        """Navigate backwards by current navigation interval."""
        self.video_controls.navigate_frame_pgup()
    
    def _navigate_frame_pgdn(self):
        """Navigate forwards by current navigation interval."""
        self.video_controls.navigate_frame_pgdn()
    
    def _update_frame_slider_for_video(self):
        """Update frame slider range and state when video is loaded."""
        self.video_controls.update_frame_slider_for_video()

    # Frame slider events are handled by VideoControls via ControlSignalManager.
    
    def _display_frame_lightweight(self, frame_index: int) -> bool:
        """Display frame without expensive live detection for smooth navigation."""
        return self.video_controls.display_frame_lightweight(frame_index)
    
    def _update_frame_slider_position(self):
        """Update frame slider position to match current frame without triggering events."""
        self.video_controls.update_frame_slider_position()
    
    def _update_time_display(self, frame_index: int):
        """Update the time display label based on frame index."""
        self.video_controls.update_time_display(frame_index)
    
    def _display_frame_with_slider_update(self, frame_index: int) -> bool:
        """Wrapper for display_frame that also updates the frame slider."""
        return self.video_controls.display_frame_with_slider_update(frame_index)
    
    def _update_current_frame_display(self):
        """Centralized function to reprocess and redisplay the current frame."""
        return self.video_controls.update_current_frame_display()

    def _handle_color_pick(self, color_rgb: Tuple[int, int, int], coordinates: Tuple[int, int]):
        """Delegate color picking to CalibrationWorkflow."""
        if self.calibration_workflow:
            self.calibration_workflow.handle_color_pick(color_rgb, coordinates)

    def _handle_overlay_selection(self, selected_key_id: Optional[int]):
        self.app_state.ui.selected_overlay_id = selected_key_id
        logging.debug(f"Overlay selected: key_id {selected_key_id}")

        if self.app_state.calibration.calibration_mode == "lit_exemplar" and \
           self.app_state.calibration.current_calibration_key_type and \
           selected_key_id is not None:
            
            overlay_to_sample = next((o for o in self.app_state.overlays if o.key_id == selected_key_id), None)
            if overlay_to_sample and self.keyboard_canvas:
                sampled_color = self.keyboard_canvas.get_average_color_for_overlay(self.keyboard_canvas.current_frame_rgb, overlay_to_sample)
                if sampled_color:
                    key_type_to_cal = self.app_state.calibration.current_calibration_key_type
                    self.app_state.detection.exemplar_lit_colors[key_type_to_cal] = sampled_color
                    logging.info(f"Calibrated exemplar lit color for {key_type_to_cal} to {sampled_color} from overlay {selected_key_id}.")
                    
                    # Also capture histogram from the same overlay
                    if self.keyboard_canvas.current_frame_rgb is not None:
                        from synthesia2midi.detection.roi_utils import get_hist_feature
                        try:
                            roi_bgr = self.keyboard_canvas.get_roi_bgr(overlay_to_sample)
                            if roi_bgr is not None and roi_bgr.size > 0:
                                exemplar_hist = get_hist_feature(roi_bgr)
                                if exemplar_hist is not None:
                                    self.app_state.detection.exemplar_lit_histograms[key_type_to_cal] = exemplar_hist
                                    logging.info(f"Captured exemplar lit histogram for {key_type_to_cal} from overlay {selected_key_id}: shape={exemplar_hist.shape}, sum={exemplar_hist.sum()}")
                                else:
                                    logging.warning(f"get_hist_feature returned None for {key_type_to_cal}")
                                
                                # Extract hue for hand detection calibration (only if hand assignment is enabled)
                                if self.app_state.detection.hand_assignment_enabled:
                                    logging.debug(f"[HUE-CALIBRATION] Extracting hue for {key_type_to_cal} from overlay {selected_key_id}")
                                    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
                                    avg_hue = np.mean(hsv[:, :, 0])
                                    logging.debug(f"[HUE-CALIBRATION] Average hue extracted: {avg_hue}")
                                    
                                    # Update hand detection calibration based on key type
                                    if key_type_to_cal.startswith('L'):  # Left hand
                                        logging.debug(f"[HUE-CALIBRATION] Updating left_hand_hue_mean from {self.app_state.detection.left_hand_hue_mean} to {avg_hue}")
                                        self.app_state.detection.left_hand_hue_mean = avg_hue
                                    elif key_type_to_cal.startswith('R'):  # Right hand
                                        logging.debug(f"[HUE-CALIBRATION] Updating right_hand_hue_mean from {self.app_state.detection.right_hand_hue_mean} to {avg_hue}")
                                        self.app_state.detection.right_hand_hue_mean = avg_hue
                                    else:
                                        logging.error(f"[HUE-CALIBRATION] ERROR: Invalid key type: {key_type_to_cal}")
                                    
                                    # If both hands are calibrated, mark as calibrated
                                    if self.app_state.detection.left_hand_hue_mean > 0 and self.app_state.detection.right_hand_hue_mean > 0:
                                        logging.info(f"[HUE-CALIBRATION] Both hands calibrated. Marking hand_detection_calibrated = True")
                                        self.app_state.detection.hand_detection_calibrated = True
                                        
                                    logging.debug(f"[HUE-CALIBRATION] Current hand detection state:")
                                    logging.debug(f"[HUE-CALIBRATION]   left_hand_hue_mean: {self.app_state.detection.left_hand_hue_mean}")
                                    logging.debug(f"[HUE-CALIBRATION]   right_hand_hue_mean: {self.app_state.detection.right_hand_hue_mean}")
                                    logging.debug(f"[HUE-CALIBRATION]   hand_detection_calibrated: {self.app_state.detection.hand_detection_calibrated}")
                                else:
                                    logging.debug(f"[HUE-CALIBRATION] Hand assignment disabled - skipping hue extraction")
                            else:
                                logging.warning(f"Could not extract valid ROI BGR from overlay {selected_key_id} for {key_type_to_cal}")
                        except Exception as e:
                            logging.error(f"Error capturing exemplar histogram for {key_type_to_cal}: {e}")
                            import traceback
                            logging.error(f"Traceback: {traceback.format_exc()}")
                    
                    # Auto-save calibration changes
                    if hasattr(self, 'video_loading_workflow') and self.video_loading_workflow:
                        success = self.video_loading_workflow.save_current_config()
                        if success:
                            logging.info("Exemplar calibration changes automatically saved to config file.")
                            save_msg = "\\nCalibration data automatically saved."
                        else:
                            logging.warning("Auto-save of exemplar calibration changes failed.")
                            save_msg = "\\nWarning: Auto-save failed."
                    else:
                        save_msg = "\\nWarning: Auto-save not available."
                    
                    # Update GUI display before resetting calibration mode
                    self.control_panel.update_advanced_calibration_display()
                    self.control_panel.update_controls_from_state()  # Update color swatches
                    
                    # Check if histogram was captured for user feedback
                    hist_captured = "Yes" if self.app_state.detection.exemplar_lit_histograms.get(key_type_to_cal) is not None else "No"
                    
                    # Reset calibration state after successful completion
                    self.app_state.calibration.calibration_mode = None
                    self.app_state.calibration.current_calibration_key_type = None
                    self.app_state.unsaved_changes = False  # Reset since we auto-saved
                    
                    # Simple success message
                    QMessageBox.information(self, "Calibration Successful", 
                                          f"Calibration of {key_type_to_cal} successful")
                else:
                    QMessageBox.warning(self, "Calibration Error", "Could not sample color from the selected overlay.")
            # Always reset calibration mode after an attempt
            self.app_state.calibration.calibration_mode = None
            self.app_state.calibration.current_calibration_key_type = None

        # Handle spark calibration modes
        elif self.app_state.calibration.calibration_mode in ["spark_bar_only", "spark_dimmest_sparks", 
                                                              "spark_lh_bar_only", "spark_lh_dimmest_sparks",
                                                              "spark_lh_brightest_sparks",
                                                              "spark_rh_bar_only", "spark_rh_dimmest_sparks",
                                                              "spark_rh_brightest_sparks"] and \
             selected_key_id is not None:
            
            overlay_to_sample = next((o for o in self.app_state.overlays if o.key_id == selected_key_id), None)
            if overlay_to_sample:
                self._capture_spark_overlay_calibration(overlay_to_sample, self.app_state.calibration.calibration_mode)
            else:
                QMessageBox.warning(self, "Calibration Error", f"Could not find overlay for key {selected_key_id}.")
                self.app_state.calibration.calibration_mode = None
        
        # Handle shadow calibration modes
        elif (self.app_state.calibration.calibration_mode and 
              self.app_state.calibration.calibration_mode.startswith("shadow_") and
              selected_key_id is not None):
            
            overlay_to_sample = next((o for o in self.app_state.overlays if o.key_id == selected_key_id), None)
            if overlay_to_sample:
                self._capture_shadow_overlay_calibration(overlay_to_sample, self.app_state.calibration.calibration_mode)
            else:
                QMessageBox.warning(self, "Calibration Error", f"Could not find overlay for key {selected_key_id}.")
                self.app_state.calibration.calibration_mode = None
        
        # Handle auto-calibration modes
        elif (self.app_state.calibration.calibration_mode and 
              self.app_state.calibration.calibration_mode.startswith("auto_calibrate_") and 
              selected_key_id is not None):
            
            overlay_to_sample = next((o for o in self.app_state.overlays if o.key_id == selected_key_id), None)
            if overlay_to_sample and self.auto_calibration_workflow:
                # Get ROI from the selected overlay
                roi_bgr = self.keyboard_canvas.get_roi_bgr(overlay_to_sample)
                if roi_bgr is not None:
                    # Handle the overlay click through auto-calibration workflow
                    logging.debug(f"[MAIN-AUTO-CAL] About to call handle_overlay_click for overlay {overlay_to_sample.key_id}")
                    success = self.auto_calibration_workflow.handle_overlay_click(overlay_to_sample, roi_bgr)
                    logging.debug(f"[MAIN-AUTO-CAL] handle_overlay_click returned: {success}")
                    
                    if success:
                        logging.info(f"[MAIN-AUTO-CAL] Auto-calibration successful - showing success message")
                        QMessageBox.information(self, "Auto-Calibration Complete", 
                                               f"Auto-calibration completed successfully for {overlay_to_sample.key_id}!")
                        logging.debug(f"[MAIN-AUTO-CAL] Triggering control panel update after successful calibration")
                        # Also explicitly trigger control panel update
                        self.control_panel.update_controls_from_state()
                        logging.debug(f"[MAIN-AUTO-CAL] Control panel update completed")
                    else:
                        logging.warning(f"[MAIN-AUTO-CAL] Auto-calibration failed - showing error message")
                        QMessageBox.warning(self, "Auto-Calibration Failed", 
                                           "Auto-calibration failed. Please check the logs for details.")
                else:
                    QMessageBox.warning(self, "ROI Error", "Could not extract ROI data from the selected overlay.")
            else:
                QMessageBox.warning(self, "Calibration Error", f"Could not find overlay for key {selected_key_id} or auto-calibration workflow not available.")
                self.app_state.calibration.calibration_mode = None

        # Potentially update UI elements that depend on selected overlay
        self.control_panel.update_selected_overlay_display()


    
    def _prepare_frame_for_detection(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[OverlayConfig]]:
        """Delegate frame preparation to DetectionManager."""
        if self.detection_manager:
            return self.detection_manager.prepare_frame_for_detection(frame_bgr)
        return frame_bgr, self.app_state.overlays

    def _start_conversion_process(self):
        """Start MIDI conversion using ConversionWorkflow."""
        logging.warning("[MIDI-BUTTON-CLICKED] === MIDI CONVERSION BUTTON CLICKED ===")
        logging.warning(f"[MIDI-BUTTON-CLICKED] User initiated MIDI conversion at {datetime.datetime.now()}")
        
        if not self.conversion_workflow:
            logging.error("[MIDI-BUTTON-CLICKED] FAILED: No conversion workflow available")
            QMessageBox.information(self, "Error", "Please open a video file first.")
            self.control_panel.set_conversion_result(False, "Please open a video file first.")
            return

        logging.warning("[MIDI-BUTTON-CLICKED] Conversion workflow available - proceeding with conversion")

        # Generate output path for MIDI file
        # Use original video path if available (when using frame sequences)
        video_path_for_output = getattr(self.app_state.video, 'original_video_path', None) or self.app_state.video.filepath
        completed_midi_dir = os.path.join(os.path.dirname(video_path_for_output), "Completed MIDI Files")
        os.makedirs(completed_midi_dir, exist_ok=True)
        
        video_basename = os.path.splitext(os.path.basename(video_path_for_output))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        midi_filename = f"{video_basename}_{timestamp}.mid"
        midi_output_path = os.path.join(completed_midi_dir, midi_filename)

        logging.warning(f"[MIDI-CONVERSION-START] === Starting MIDI conversion process ===")
        logging.warning(f"[MIDI-CONVERSION-START] Output path: {midi_output_path}")
        logging.warning(f"[MIDI-CONVERSION-START] Video path: {video_path_for_output}")
        
        try:
            # Use ConversionWorkflow to perform the conversion
            logging.warning("[MIDI-CONVERSION-START] Calling conversion_workflow.convert_to_midi()...")
            success = self.conversion_workflow.convert_to_midi(midi_output_path)
            logging.warning(f"[MIDI-CONVERSION-RESULT] convert_to_midi() returned: {success}")
            
            # Update UI state - reset button whether success or failure
            if success:
                self.control_panel.set_conversion_result(True, f"MIDI file saved to:\n{midi_output_path}")
                QMessageBox.information(self, "Conversion Complete", f"MIDI file saved to:\n{midi_output_path}")
                logging.warning(f"[MIDI-CONVERSION-SUCCESS] MIDI conversion successful. Output: {midi_output_path}")
            else:
                self.control_panel.set_conversion_result(False, "MIDI conversion failed. Check logs for details.")
                QMessageBox.critical(self, "Conversion Failed", "MIDI conversion failed. Check logs for details.")
                logging.error("[MIDI-CONVERSION-FAILED] MIDI conversion failed - convert_to_midi returned False")
        except Exception as e:
            # Ensure button is reset even if an exception occurs
            self.control_panel.set_conversion_result(False, f"MIDI conversion error: {str(e)}")
            QMessageBox.critical(self, "Conversion Error", f"MIDI conversion error: {str(e)}")
            logging.error(f"[MIDI-CONVERSION-EXCEPTION] MIDI conversion exception: {e}", exc_info=True)
    
    def _toggle_overlays(self):
        """Delegate overlay visibility toggle to DisplayManager."""
        if self.display_manager:
            self.display_manager.toggle_overlays()

    def _toggle_live_detection_feedback(self):
        """Delegate live detection feedback toggle to DisplayManager."""
        if self.display_manager:
            self.display_manager.toggle_live_detection_feedback()
    

    def _handle_frame_nav_interval(self, interval: int):
        """Handle frame navigation interval selection from menu."""
        # Update state
        self.app_state.video.current_nav_interval = interval
        self.app_state.unsaved_changes = True
        
        # Update menu check states (mutual exclusivity)
        for nav_interval, action in self.frame_nav_actions.items():
            action.setChecked(nav_interval == interval)
        
        # Emit signal to update control panel and other components
        if hasattr(self.control_panel, 'nav_interval_changed'):
            self.control_panel.nav_interval_changed.emit(interval)
        
        logging.info(f"Frame navigation interval changed to: {interval}")

    def _handle_visual_threshold_monitor_menu(self, checked: bool):
        """Handle visual threshold monitor toggle from menu."""
        # Update state
        self.app_state.ui.visual_threshold_monitor_enabled = checked
        self.app_state.unsaved_changes = True
        
        # Update menu check state
        self.visual_threshold_monitor_action.setChecked(checked)
        
        # Emit signal to update display manager and other components
        if hasattr(self, 'display_manager') and self.display_manager:
            self.display_manager.handle_visual_threshold_monitor_toggle(checked)
        
        logging.info(f"Visual threshold monitor: {'enabled' if checked else 'disabled'}")

    def _capture_window_screenshot(self):
        """Capture a screenshot of the current window and save with timestamp."""
        try:
            # Create screenshots directory if it doesn't exist
            screenshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "screenshots")
            os.makedirs(screenshot_dir, exist_ok=True)
            
            # Generate timestamp filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            filename = f"gui_capture_{timestamp}.png"
            filepath = os.path.join(screenshot_dir, filename)
            
            # Capture the main window
            pixmap = self.grab()
            
            # Save the screenshot
            if pixmap.save(filepath):
                QMessageBox.information(
                    self, 
                    "Screenshot Saved", 
                    f"Window screenshot saved to:\n{filepath}"
                )
                logging.info(f"Window screenshot saved: {filepath}")
            else:
                QMessageBox.warning(
                    self, 
                    "Screenshot Failed", 
                    f"Failed to save screenshot to:\n{filepath}"
                )
                logging.error(f"Failed to save window screenshot: {filepath}")
                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Screenshot Error", 
                f"Error capturing window screenshot:\n{str(e)}"
            )
            logging.error(f"Error capturing window screenshot: {e}")

    def _handle_calibrate_unlit_all_keys(self):
        """Delegate unlit calibration to CalibrationWorkflow."""
        if self.calibration_workflow:
            self.calibration_workflow.handle_calibrate_unlit_all_keys()

    def _handle_calibrate_lit_exemplar_key_start(self, key_type: str):
        """Delegate lit exemplar calibration start to CalibrationWorkflow."""
        if self.calibration_workflow:
            self.calibration_workflow.handle_calibrate_lit_exemplar_key_start(key_type)

    def _handle_spark_roi_selection_request(self):
        """Handle spark ROI selection request from control panel."""
        logging.info("Spark ROI selection requested - entering ROI selection mode")
        if hasattr(self, 'keyboard_canvas') and self.keyboard_canvas:
            # Enter ROI selection mode on the canvas
            if hasattr(self.keyboard_canvas, 'interaction') and self.keyboard_canvas.interaction:
                self.keyboard_canvas.interaction.enter_spark_roi_selection_mode()
                QMessageBox.information(self, "Spark ROI Selection", 
                                       "Click and drag on the video to select the spark detection region.\n"
                                       "Right-click to cancel.")
            else:
                QMessageBox.warning(self, "Canvas Error", "Canvas interaction system not available.")
        else:
            QMessageBox.warning(self, "No Canvas", "No video canvas available for ROI selection.")
    
    def _handle_spark_roi_visibility_toggle(self, visible: bool):
        """Handle spark ROI visibility toggle from control panel."""
        logging.info(f"Spark ROI visibility toggled to: {visible}")
        self.app_state.detection.spark_roi_visible = visible
        # Update canvas to show/hide the ROI
        if self.keyboard_canvas:
            self.keyboard_canvas.update()

    def _handle_shadow_roi_selection_request(self):
        """Handle shadow ROI selection request from control panel."""
        logging.info("Shadow ROI selection requested - entering ROI selection mode")
        if hasattr(self, 'keyboard_canvas') and self.keyboard_canvas:
            # Enter ROI selection mode on the canvas
            if hasattr(self.keyboard_canvas, 'interaction') and self.keyboard_canvas.interaction:
                self.keyboard_canvas.interaction.enter_shadow_roi_selection_mode()
                QMessageBox.information(self, "Shadow ROI Selection", 
                                       "Click and drag on the video to select the shadow detection region.\n"
                                       "Shadow zones will be created for each key overlay.\n"
                                       "Right-click to cancel.")
            else:
                QMessageBox.warning(self, "Canvas Error", "Canvas interaction system not available.")
        else:
            QMessageBox.warning(self, "No Canvas", "No video canvas available for ROI selection.")
    
    def _handle_shadow_white_roi_selection_request(self):
        """Handle white key shadow ROI selection request from control panel."""
        logging.info("White key shadow ROI selection requested - entering ROI selection mode")
        if hasattr(self, 'keyboard_canvas') and self.keyboard_canvas:
            # Enter ROI selection mode on the canvas
            if hasattr(self.keyboard_canvas, 'interaction') and self.keyboard_canvas.interaction:
                self.keyboard_canvas.interaction.enter_shadow_white_roi_selection_mode()
                QMessageBox.information(self, "White Key Shadow ROI Selection", 
                                       "Click and drag on the video to select the white key shadow detection region.\n"
                                       "This will define the vertical region where white key shadows are detected.\n"
                                       "Right-click to cancel.")
            else:
                QMessageBox.warning(self, "Canvas Error", "Canvas interaction system not available.")
        else:
            QMessageBox.warning(self, "No Canvas", "No video canvas available for ROI selection.")
    
    def _handle_shadow_black_roi_selection_request(self):
        """Handle black key shadow ROI selection request from control panel."""
        logging.info("Black key shadow ROI selection requested - entering ROI selection mode")
        if hasattr(self, 'keyboard_canvas') and self.keyboard_canvas:
            # Enter ROI selection mode on the canvas
            if hasattr(self.keyboard_canvas, 'interaction') and self.keyboard_canvas.interaction:
                self.keyboard_canvas.interaction.enter_shadow_black_roi_selection_mode()
                QMessageBox.information(self, "Black Key Shadow ROI Selection", 
                                       "Click and drag on the video to select the black key shadow detection region.\n"
                                       "This will define the vertical region where black key shadows are detected.\n"
                                       "Right-click to cancel.")
            else:
                QMessageBox.warning(self, "Canvas Error", "Canvas interaction system not available.")
        else:
            QMessageBox.warning(self, "No Canvas", "No video canvas available for ROI selection.")
    
    def _handle_spark_roi_updated(self, top_y: int, bottom_y: int):
        """Handle spark ROI coordinates updated from canvas selection."""
        logging.info(f"Spark ROI updated from canvas: top={top_y}, bottom={bottom_y}")
        
        # Update control panel to reflect new values
        if hasattr(self, 'control_panel') and self.control_panel:
            self.control_panel.update_controls_from_state()
        
        # Show confirmation message
        QMessageBox.information(self, "Spark ROI Set", 
                               f"Spark detection region set:\n"
                               f"Top: {top_y} pixels\n"
                               f"Bottom: {bottom_y} pixels\n"
                               f"Height: {bottom_y - top_y} pixels")

    def _handle_spark_calibration_request(self, step_type: str):
        """Handle spark calibration request from control panel."""
        logging.info(f"Spark calibration requested: {step_type}")
        
        if not hasattr(self, 'keyboard_canvas') or not self.keyboard_canvas:
            QMessageBox.warning(self, "No Canvas", "No video canvas available for calibration.")
            return
        
        if self.keyboard_canvas.current_frame_rgb is None:
            QMessageBox.warning(self, "No Frame", "No video frame loaded. Please open a video and navigate to a frame.")
            return
        
        # Map step type to calibration modes
        calibration_mode_map = {
            "background": "spark_background",
            "bar_only": "spark_bar_only", 
            "dimmest_sparks": "spark_dimmest_sparks",
            "lh_bar_only": "spark_lh_bar_only",
            "lh_dimmest_sparks": "spark_lh_dimmest_sparks",
            "lh_brightest_sparks": "spark_lh_brightest_sparks",
            "rh_bar_only": "spark_rh_bar_only",
            "rh_dimmest_sparks": "spark_rh_dimmest_sparks",
            "rh_brightest_sparks": "spark_rh_brightest_sparks"
        }
        
        if step_type not in calibration_mode_map:
            QMessageBox.warning(self, "Invalid Step", f"Unknown calibration step: {step_type}")
            return
        
        calibration_mode = calibration_mode_map[step_type]
        
        # Handle background calibration differently (immediate capture)
        if step_type == "background":
            self._capture_spark_background_calibration()
            return
        
        # For bar_only and dimmest_sparks, set mode and wait for user click
        self.app_state.calibration.calibration_mode = calibration_mode
        
        # Show instruction dialog
        step_display_names = {
            "bar_only": "Bar-Only",
            "dimmest_sparks": "Dimmest Sparks",
            "lh_bar_only": "Left Hand Bar-Only",
            "lh_dimmest_sparks": "Left Hand Dimmest Sparks",
            "lh_brightest_sparks": "Left Hand Brightest Sparks",
            "rh_bar_only": "Right Hand Bar-Only",
            "rh_dimmest_sparks": "Right Hand Dimmest Sparks",
            "rh_brightest_sparks": "Right Hand Brightest Sparks"
        }
        
        display_name = step_display_names.get(step_type, step_type)
        
        # Determine hand type
        hand_type = ""
        if "lh_" in step_type:
            hand_type = "LEFT HAND "
        elif "rh_" in step_type:
            hand_type = "RIGHT HAND "
        
        instruction_msg = (f"Now click on a {hand_type}key that shows {display_name.lower()} condition.\n\n"
                          f"For {display_name}:\n")
        
        if "bar_only" in step_type:
            instruction_msg += ("- Key should show colored bars WITHOUT any sparks\n"
                              "- Navigate to a frame where this condition is visible\n"
                              "- Click on the key showing this exact condition")
        elif "dimmest_sparks" in step_type:
            instruction_msg += ("- Key should show colored bars WITH barely visible sparks\n"
                              "- Navigate to a frame where sparks are just starting to appear\n" 
                              "- Click on the key showing this exact condition")
        else:  # brightest_sparks
            instruction_msg += ("- Key should show colored bars WITH very bright/intense sparks\n"
                              "- Navigate to a frame where sparks are at their brightest\n" 
                              "- Click on the key showing this exact condition")
        
        QMessageBox.information(self, f"Spark {display_name} Calibration", instruction_msg)
        logging.info(f"Set calibration mode to {calibration_mode}, waiting for user to click on key")
    
    def _handle_auto_spark_calibration_request(self, key_type: str):
        """Handle auto-spark calibration request from control panel."""
        logging.info(f"Auto-spark calibration requested for key type: {key_type}")
        
        if not hasattr(self, 'auto_calibration_workflow') or not self.auto_calibration_workflow:
            QMessageBox.warning(self, "Workflow Error", "Auto-calibration workflow not available.")
            return
        
        # Start the auto-calibration process
        success = self.auto_calibration_workflow.start_auto_calibration(key_type)
        
        if success:
            instruction_msg = (f"Auto-Calibration for {key_type} Started\n\n"
                              f"Instructions:\n"
                              f"1. Navigate to a frame where a {key_type} key FIRST turns ON\n"
                              f"2. Click on that key overlay\n"
                              f"3. The system will automatically:\n"
                              f"   - Detect if it's left/right hand based on color\n"
                              f"   - Capture bar-only (frame +0)\n"
                              f"   - Capture dimmest sparks (frame +2)\n"
                              f"   - Find brightest sparks (frames +3 to +22)\n"
                              f"   - Save calibration data\n\n"
                              f"Key Type Legend:\n"
                              f"LW = Left White, LB = Left Black\n"
                              f"RW = Right White, RB = Right Black")
            
            QMessageBox.information(self, f"Auto-Calibrate {key_type}", instruction_msg)
        else:
            QMessageBox.warning(self, "Calibration Error", 
                               "Failed to start auto-calibration. Please check video and overlays are loaded.")
    
    def _handle_spark_detection_toggle(self, enabled: bool):
        """Handle spark detection enable/disable toggle."""
        logging.info(f"Spark detection {'enabled' if enabled else 'disabled'}")
        
        # Update the app state
        self.app_state.detection.spark_detection_enabled = enabled
        self.app_state.unsaved_changes = True
        
        # The detector will be recreated automatically on the next frame processing 
        # when factory.create_from_app_state is called
    
    def _handle_spark_detection_sensitivity_change(self, value: float):
        """Handle spark detection sensitivity change."""
        logging.info(f"Spark detection sensitivity changed to {value:.2f}")
        
        # Update the app state
        self.app_state.detection.spark_detection_sensitivity = value
        self.app_state.unsaved_changes = True
        
        # The sensitivity will be used on the next frame processing
    
    def _handle_shadow_detection_toggle(self, enabled: bool):
        """Handle shadow detection enable/disable toggle."""
        logging.info(f"Shadow detection {'enabled' if enabled else 'disabled'}")
        
        # The app_state is already updated by the control panel
        # We don't need to recreate the detector here as it will be recreated
        # automatically on the next frame processing when factory.create_from_app_state is called
    
    def _handle_shadow_detection_sensitivity_change(self, value: float):
        """Handle shadow detection sensitivity change."""
        logging.info(f"Shadow detection sensitivity changed to {value:.2f}")
        # The app_state is already updated by the control panel
        # The sensitivity will be used on the next frame processing
    
    def _handle_shadow_darkness_threshold_change(self, value: float):
        """Handle shadow darkness threshold change."""
        logging.info(f"Shadow darkness threshold changed to {value:.2f}")
        # The app_state is already updated by the control panel
        # The threshold will be used on the next frame processing
    
    def _handle_shadow_calibration_request(self, key_type: str, calibration_type: str):
        """Handle shadow calibration request for specific key type."""
        logging.info(f"Shadow calibration requested: key_type={key_type}, calibration_type={calibration_type}")
        
        if not hasattr(self, 'keyboard_canvas') or not self.keyboard_canvas:
            QMessageBox.warning(self, "No Canvas", "No video canvas available for calibration.")
            return
        
        if self.keyboard_canvas.current_frame_rgb is None:
            QMessageBox.warning(self, "No Frame", "No video frame loaded. Please open a video and navigate to a frame.")
            return
        
        # Map calibration type and key type to calibration mode
        calibration_mode = f"shadow_{key_type.lower()}_{calibration_type}"
        
        # Set calibration mode
        self.app_state.calibration.calibration_mode = calibration_mode
        
        # Show instruction dialog
        key_type_display = {
            "LW": "Left White",
            "LB": "Left Black",
            "RW": "Right White",
            "RB": "Right Black"
        }.get(key_type, key_type)
        
        calibration_display = {
            "unpressed": "Unpressed (lit but not pressed)",
            "pressed": "Pressed (lit and fully pressed)"
        }.get(calibration_type, calibration_type)
        
        instruction_msg = (f"Now click on a {key_type_display} key that shows {calibration_display} condition.\n\n"
                          f"For {calibration_display}:\n")
        
        if calibration_type == "unpressed":
            instruction_msg += ("- Key should be lit (colored bars visible)\n"
                              "- Key should NOT be pressed down\n"
                              "- No shadow should be visible underneath")
        else:  # pressed
            instruction_msg += ("- Key should be lit (colored bars visible)\n"
                              "- Key should be FULLY pressed down\n"
                              "- Dark shadow should be visible underneath")
        
        QMessageBox.information(self, f"Shadow Calibration - {key_type_display}", instruction_msg)
        logging.info(f"Set calibration mode to {calibration_mode}, waiting for user to click on key")
    
    def _handle_overlay_type_change(self, overlay_type: str):
        """Handle overlay type change (key/spark/shadow)."""
        logging.info(f"Overlay type changed to: {overlay_type}")
        
        # Update the current drawing mode in app_state
        if hasattr(self.app_state, 'ui'):
            self.app_state.ui.overlay_drawing_type = overlay_type
            
        # Update the canvas drawing mode if available
        if hasattr(self, 'keyboard_canvas') and self.keyboard_canvas:
            if hasattr(self.keyboard_canvas, 'interaction') and self.keyboard_canvas.interaction:
                self.keyboard_canvas.interaction.overlay_drawing_type = overlay_type
                logging.info(f"Updated canvas interaction overlay type to: {overlay_type}")
        
        # Refresh the display to show overlay colors correctly
        if hasattr(self, 'keyboard_canvas'):
            self.keyboard_canvas.draw_overlays()
    
    def _capture_spark_background_calibration(self):
        """Capture background calibration immediately (no user interaction needed)."""
        logging.info("Capturing spark background calibration")
        
        # Import calibration classes
        from synthesia2midi.detection.spark_calibration import SparkCalibrationManager, CalibrationStep
        
        # Create calibration manager
        calibration_manager = SparkCalibrationManager(self.app_state)
        
        # Start background calibration step
        if not calibration_manager.start_calibration_step(CalibrationStep.BACKGROUND):
            QMessageBox.warning(self, "Calibration Failed", 
                              "Could not start background calibration.\n\n"
                              "Requirements:\n"
                              "- Spark ROI must be set (top < bottom)\n"
                              "- Key overlays must be configured")
            return
        
        # Capture current frame
        current_frame = self.keyboard_canvas.current_frame_rgb
        frame_index = getattr(self.keyboard_canvas, 'current_frame_index', 0)
        
        if calibration_manager.capture_calibration_frame(current_frame, frame_index, "spark_calibration_background"):
            # Update UI display
            self.control_panel.update_spark_calibration_display()
            
            # Show success message
            QMessageBox.information(self, "Background Calibration Complete", 
                                   "Background calibration captured successfully!")
            logging.info("Spark background calibration completed successfully")
        else:
            QMessageBox.critical(self, "Calibration Failed", 
                               "Failed to capture background calibration data.\n\n"
                               "Please check that spark ROI is properly set.")
            logging.error("Spark background calibration failed")
    
    def _capture_spark_overlay_calibration(self, overlay, calibration_mode: str):
        """Capture spark calibration from selected overlay's spark zone."""
        logging.info(f"Capturing {calibration_mode} calibration from overlay {overlay.key_id}")
        
        # Import calibration classes
        from synthesia2midi.detection.spark_calibration import SparkCalibrationManager, CalibrationStep
        from synthesia2midi.detection.spark_mapper import get_spark_zones
        
        try:
            # Get spark zones to find the zone for this overlay
            spark_zones = get_spark_zones(self.app_state)
            target_zone = next((zone for zone in spark_zones if zone.key_id == overlay.key_id), None)
            
            if not target_zone:
                QMessageBox.warning(self, "Calibration Error", 
                                   f"No spark zone found for key {overlay.key_id}. "
                                   f"Please ensure spark ROI is properly set.")
                self.app_state.calibration.calibration_mode = None
                return
            
            # Create calibration manager
            calibration_manager = SparkCalibrationManager(self.app_state)
            
            # Map calibration mode to step and field
            mode_map = {
                "spark_bar_only": (CalibrationStep.BAR_ONLY, "spark_calibration_bar_only"),
                "spark_dimmest_sparks": (CalibrationStep.DIMMEST_SPARKS, "spark_calibration_dimmest_sparks"),
                "spark_lh_bar_only": (CalibrationStep.BAR_ONLY, "spark_calibration_lh_bar_only"),
                "spark_lh_dimmest_sparks": (CalibrationStep.DIMMEST_SPARKS, "spark_calibration_lh_dimmest_sparks"),
                "spark_lh_brightest_sparks": (CalibrationStep.BRIGHTEST_SPARKS, "spark_calibration_lh_brightest_sparks"),
                "spark_rh_bar_only": (CalibrationStep.BAR_ONLY, "spark_calibration_rh_bar_only"),
                "spark_rh_dimmest_sparks": (CalibrationStep.DIMMEST_SPARKS, "spark_calibration_rh_dimmest_sparks"),
                "spark_rh_brightest_sparks": (CalibrationStep.BRIGHTEST_SPARKS, "spark_calibration_rh_brightest_sparks")
            }
            
            if calibration_mode not in mode_map:
                QMessageBox.warning(self, "Invalid Mode", f"Unknown calibration mode: {calibration_mode}")
                self.app_state.calibration.calibration_mode = None
                return
            
            calibration_step, field_name = mode_map[calibration_mode]
            
            # Start calibration step
            if not calibration_manager.start_calibration_step(calibration_step):
                QMessageBox.warning(self, "Calibration Failed", 
                                   "Could not start calibration step. Please check spark ROI configuration.")
                self.app_state.calibration.calibration_mode = None
                return
            
            # Capture single zone calibration
            current_frame = self.keyboard_canvas.current_frame_rgb
            frame_index = getattr(self.keyboard_canvas, 'current_frame_index', 0)
            
            # Extract calibration sample from the target zone only
            zone_sample = calibration_manager._extract_zone_sample(current_frame, target_zone)
            if not zone_sample:
                QMessageBox.critical(self, "Calibration Failed", 
                                   f"Could not extract calibration data from key {overlay.key_id}.")
                self.app_state.calibration.calibration_mode = None
                return
            
            # Create calibration data from single zone sample
            zone_samples = {target_zone.key_id: zone_sample}
            calib_data = calibration_manager._create_calibration_data(
                calibration_step, frame_index, zone_samples
            )
            
            # Store calibration data
            calibration_manager._store_calibration_data(calib_data, field_name)
            
            # Update UI display
            self.control_panel.update_spark_calibration_display()
            
            # Show success message
            step_names = {
                "spark_bar_only": "Bar-Only",
                "spark_dimmest_sparks": "Dimmest Sparks",
                "spark_lh_bar_only": "Left Hand Bar-Only",
                "spark_lh_dimmest_sparks": "Left Hand Dimmest Sparks",
                "spark_lh_brightest_sparks": "Left Hand Brightest Sparks",
                "spark_rh_bar_only": "Right Hand Bar-Only",
                "spark_rh_dimmest_sparks": "Right Hand Dimmest Sparks",
                "spark_rh_brightest_sparks": "Right Hand Brightest Sparks"
            }
            step_name = step_names.get(calibration_mode, calibration_mode)
            
            QMessageBox.information(self, f"{step_name} Calibration Complete", 
                                   f"{step_name} calibration captured successfully from key {overlay.key_id}!\n\n"
                                   f"Quality: {calib_data.confidence_score:.1%}\n"
                                   f"Brightness: {calib_data.mean_brightness:.3f}")
            
            logging.info(f"Spark {calibration_mode} calibration completed successfully from key {overlay.key_id}")
            
        except Exception as e:
            logging.error(f"Error during spark calibration: {e}")
            QMessageBox.critical(self, "Calibration Error", 
                               f"An error occurred during calibration: {str(e)}")
        finally:
            # Always reset calibration mode
            self.app_state.calibration.calibration_mode = None

    def _capture_shadow_overlay_calibration(self, overlay, calibration_mode: str):
        """Capture shadow calibration from selected overlay."""
        logging.info(f"Capturing {calibration_mode} calibration from overlay {overlay.key_id}")
        
        # Parse calibration mode: shadow_{key_type}_{unpressed/pressed}
        parts = calibration_mode.split('_')
        if len(parts) != 3 or parts[0] != 'shadow':
            QMessageBox.warning(self, "Invalid Mode", f"Invalid shadow calibration mode: {calibration_mode}")
            self.app_state.calibration.calibration_mode = None
            return
        
        key_type = parts[1].upper()  # lw -> LW
        calibration_type = parts[2]  # unpressed or pressed
        
        logging.info(f"Shadow calibration for {key_type} type using key {overlay.key_id} ({getattr(overlay, 'note_name_in_octave', 'Unknown')})")
        
        # Note: LW/LB/RW/RB types are organizational helpers only, not technical requirements
        # Any key can be used to calibrate any type - the types are just visual guides for the user
        
        # Get the current frame
        current_frame = self.keyboard_canvas.current_frame_rgb
        if current_frame is None:
            QMessageBox.warning(self, "No Frame", "No frame available for calibration.")
            self.app_state.calibration.calibration_mode = None
            return
        
        # Get shadow overlay for this key if it exists
        logging.info(f"=== SHADOW OVERLAY LOOKUP DEBUG ===")
        logging.info(f"Looking for shadow overlay for key {overlay.key_id}")
        logging.info(f"Total overlays in app_state: {len(self.app_state.overlays)}")
        
        # Log all overlays for debugging
        for i, ov in enumerate(self.app_state.overlays):
            overlay_type = getattr(ov, 'overlay_type', 'None')
            logging.info(f"  Overlay {i}: key_id={ov.key_id}, type={overlay_type}")
        
        shadow_overlay = None
        for ov in self.app_state.overlays:
            if hasattr(ov, 'overlay_type') and ov.overlay_type == 'shadow' and ov.key_id == overlay.key_id:
                shadow_overlay = ov
                logging.info(f"Found matching shadow overlay for key {overlay.key_id}")
                break
        
        if not shadow_overlay:
            logging.warning(f"No shadow overlay found for key {overlay.key_id}")
            logging.info(f"Shadow overlays in system:")
            shadow_overlays = [ov for ov in self.app_state.overlays if hasattr(ov, 'overlay_type') and ov.overlay_type == 'shadow']
            for i, sov in enumerate(shadow_overlays):
                logging.info(f"  Shadow overlay {i}: key_id={sov.key_id}")
            
            QMessageBox.warning(self, "No Shadow Overlay", 
                               f"No shadow overlay found for key {overlay.key_id}. "
                               "Please create shadow overlays before calibration.")
            self.app_state.calibration.calibration_mode = None
            return
        
        # Extract shadow ROI
        shadow_roi = self._extract_roi(current_frame, shadow_overlay)
        if shadow_roi is None or shadow_roi.size == 0:
            QMessageBox.warning(self, "ROI Error", "Could not extract shadow region.")
            self.app_state.calibration.calibration_mode = None
            return
        
        # Calculate darkness ratio for the shadow region
        gray = cv2.cvtColor(shadow_roi, cv2.COLOR_RGB2GRAY)
        black_pixel_threshold = 30  # Pixels below this are considered "black"
        black_pixels = np.sum(gray < black_pixel_threshold)
        total_pixels = gray.size
        darkness_ratio = black_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # Store calibration data based on type
        calibration_data = {
            'darkness_ratio': darkness_ratio,
            'mean_brightness': np.mean(gray),
            'std_brightness': np.std(gray),
            'black_pixel_count': int(black_pixels),
            'total_pixels': int(total_pixels),
            'frame_index': self.app_state.video.current_frame_index
        }
        
        # Store in appropriate app_state field
        if calibration_type == 'unpressed':
            setattr(self.app_state.detection, f'shadow_calibration_{key_type.lower()}_unpressed', calibration_data)
            state_desc = "unpressed (no shadow)"
        else:  # pressed
            setattr(self.app_state.detection, f'shadow_calibration_{key_type.lower()}_pressed', calibration_data)
            state_desc = "pressed (with shadow)"
        
        # Update UI
        self.control_panel.update_shadow_calibration_display()
        
        # Auto-save calibration
        if hasattr(self, 'video_loading_workflow') and self.video_loading_workflow:
            success = self.video_loading_workflow.save_current_config()
            save_msg = "\nCalibration data automatically saved." if success else "\nWarning: Auto-save failed."
        else:
            save_msg = "\nWarning: Auto-save not available."
        
        # Show success message
        QMessageBox.information(self, f"Shadow Calibration - {key_type}", 
                               f"Shadow calibration captured for {key_type} key in {state_desc} state.\n\n"
                               f"Darkness ratio: {darkness_ratio:.1%}\n"
                               f"Mean brightness: {calibration_data['mean_brightness']:.1f}\n"
                               f"Black pixels: {black_pixels}/{total_pixels}"
                               f"{save_msg}")
        
        logging.info(f"Shadow calibration completed for {key_type} {calibration_type}: darkness_ratio={darkness_ratio:.3f}")
        
        # Reset calibration mode
        self.app_state.calibration.calibration_mode = None
        self.app_state.unsaved_changes = False  # Reset since we auto-saved
    
    def _extract_roi(self, frame: np.ndarray, overlay: OverlayConfig) -> Optional[np.ndarray]:
        """Extract region of interest from frame based on overlay coordinates."""
        if frame is None or overlay is None:
            return None
        
        x, y = int(overlay.x), int(overlay.y)
        w, h = int(overlay.width), int(overlay.height)
        
        # Ensure coordinates are within frame bounds
        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_w, x + w)
        y2 = min(frame_h, y + h)
        
        if x1 >= x2 or y1 >= y2:
            return None
        
        return frame[y1:y2, x1:x2]

    def _get_calibration_instructions(self, step_type: str) -> str:
        """Get user instructions for each calibration step."""
        instructions = {
            "background": "Navigate to a frame with no bars visible and no sparks.\nThe spark ROI should show only background content.",
            "bar_only": "Navigate to a frame with colored bars visible but NO sparks.\nBars should be clearly visible in the spark ROI without any bright flashes.",
            "dimmest_sparks": "Navigate to a frame with the DIMMEST visible sparks.\nSparks should be just barely noticeable as bright flashes in the ROI."
        }
        return instructions.get(step_type, "Unknown calibration step")

    def _handle_detection_threshold_change(self, threshold: float):
        """Delegate detection threshold change to DetectionManager."""
        if self.detection_manager:
            self.detection_manager.handle_detection_threshold_change(threshold)

    def _handle_rise_delta_threshold_change(self, threshold: float):
        """Handle rise delta threshold change."""
        self.app_state.detection.rise_delta_threshold = threshold
        self.app_state.unsaved_changes = True

    def _handle_fall_delta_threshold_change(self, threshold: float):
        """Handle fall delta threshold change."""
        self.app_state.detection.fall_delta_threshold = threshold
        self.app_state.unsaved_changes = True

    def _handle_start_frame_change(self, frame: int):
        """Compatibility handler for start-frame signals.

        The app uses `processing_start_frame` for non-destructive processing ranges. This
        handler keeps `video.start_frame` and the processing range in sync.
        """
        self.app_state.video.start_frame = frame
        self.app_state.video.processing_start_frame = frame  # Keep in sync with processing range
        self.app_state.unsaved_changes = True
    
    def _handle_end_frame_change(self, frame: int):
        """Compatibility handler for end-frame signals.

        The app uses `processing_end_frame` for non-destructive processing ranges. This
        handler keeps `video.end_frame` and the processing range in sync.
        """
        self.app_state.video.end_frame = frame
        self.app_state.video.processing_end_frame = frame  # Keep in sync with processing range
        self.app_state.unsaved_changes = True

    def _handle_refresh_selected_overlay_display(self):
        """Delegate overlay display refresh to DisplayManager."""
        if self.display_manager:
            self.display_manager.handle_refresh_selected_overlay_display()

    def _align_overlays_vertically(self, master_overlay: OverlayConfig, target_key_color_type: str):
        """Delegate vertical overlay alignment to OverlayManager."""
        if self.overlay_manager:
            self.overlay_manager.align_overlays_vertically(master_overlay, target_key_color_type)

    def _handle_align_white_keys_to_selected(self):
        """Delegate white key alignment to OverlayManager."""
        if self.overlay_manager:
            self.overlay_manager.handle_align_white_keys_to_selected()

    def _handle_align_black_keys_to_selected(self):
        """Delegate black key alignment to OverlayManager."""
        if self.overlay_manager:
            self.overlay_manager.handle_align_black_keys_to_selected()
    
    def _handle_spinbox_overlay_size_change(self, key_suffix: str, dimension: str, value: int):
        """Handle real-time spinbox value changes for overlay dimensions.
        
        Args:
            key_suffix: 'W' for white keys or 'B' for black keys
            dimension: 'width' or 'height' 
            value: The new absolute value from the spinbox
        """
        # Update all overlays of the specified type
        for overlay in self.app_state.overlays:
            # Match both left and right keys with the suffix
            if overlay.key_type.endswith(key_suffix):
                if dimension == 'width':
                    overlay.width = value
                elif dimension == 'height':
                    overlay.height = value
        
        # Update canvas display
        if self.keyboard_canvas:
            self.keyboard_canvas.update()
        
        # Mark as unsaved changes
        self.app_state.unsaved_changes = True
    
    def _handle_overlay_size_adjustment(self, key_color: str, dimension: str, delta: int):
        """Handle overlay size adjustment request from control panel.
        
        Args:
            key_color: The key color ('white' or 'black')
            dimension: 'width' or 'height'
            delta: Amount to adjust (typically +2 or -2 pixels)
        """
        # Use the overlay manager to handle the adjustment
        self.overlay_manager.adjust_overlay_sizes(key_color, dimension, delta)

    def _invoke_calibration_wizard(self):
        """Called by the Run/Reset Calibration Wizard button using CalibrationWorkflow."""
        if not self.calibration_workflow:
            QMessageBox.warning(self, "Wizard Error", "Please open a video file first.")
            # Ensure wizard button is disabled if no video session somehow
            if hasattr(self.control_panel, 'wizard_button'):
                 self.control_panel.wizard_button.setEnabled(False)
            return

        logging.info("Starting manual calibration wizard invocation.")
        
        # Use CalibrationWorkflow to create the wizard
        self.calibration_wizard = self.calibration_workflow.run_calibration_wizard()
        
        if not self.calibration_wizard:
            logging.error("Failed to create calibration wizard")
            return
        
        # Connect signals for keyboard region selection
        self.calibration_wizard.keyboard_region_selection_requested.connect(
            self._handle_keyboard_region_selection_request
        )
        
        # Mark that keyboard region was not requested yet
        self.calibration_wizard._keyboard_region_requested = False
        
        # Show the wizard
        result = self.calibration_wizard.exec()
        
        # Check if keyboard region selection was requested
        if hasattr(self.calibration_wizard, '_keyboard_region_requested') and self.calibration_wizard._keyboard_region_requested:
            # Don't cleanup wizard yet - we need it for keyboard region selection
            logging.info("Keyboard region selection was requested, keeping wizard instance alive")
            return
        
        wizard_success = self.calibration_wizard.result is True
        
        # Handle wizard completion
        if self.calibration_workflow.handle_wizard_completed(wizard_success):
            logging.info("Wizard submitted successfully and overlays generated.")
            # Apply template styles (kept in main class for UI coordination)
            self._apply_template_styles_to_overlays()
            
            
            self.control_panel.convert_button.setEnabled(True)
            self.keyboard_canvas.draw_overlays() # Explicitly redraw overlays
        else:
            logging.info("Wizard was cancelled or did not generate overlays. Convert button remains disabled.")
            self.control_panel.convert_button.setEnabled(False)
        
        # Cleanup
        self.calibration_wizard = None
        
        # Refresh UI elements
        self.keyboard_canvas.display_frame(self.app_state.video.current_frame_index) # Redraw frame and overlays
    
    def _handle_keyboard_region_selection_request(self):
        """Handle request to select keyboard region from wizard."""
        logging.info("Starting keyboard region selection mode")
        
        # Mark that keyboard region was requested
        if self.calibration_wizard:
            self.calibration_wizard._keyboard_region_requested = True
        
        # Connect the keyboard region selected signal
        if hasattr(self.keyboard_canvas, 'interaction') and self.keyboard_canvas.interaction:
            logging.info("Canvas interaction available, setting up signal connection")
            
            # Disconnect any previous connections
            try:
                # Check if signal has any connections before disconnecting
                if hasattr(self.keyboard_canvas.interaction.keyboard_region_selected, '__self__'):
                    self.keyboard_canvas.interaction.keyboard_region_selected.disconnect()
                    logging.debug("Disconnected previous keyboard_region_selected connections")
                else:
                    logging.debug("No connections to disconnect")
            except (TypeError, RuntimeError):
                # No connections to disconnect, which is fine
                logging.debug("No previous connections to disconnect")
                pass
            
            # Connect to our handler
            logging.info("Connecting keyboard_region_selected signal to _handle_keyboard_region_selected")
            self.keyboard_canvas.interaction.keyboard_region_selected.connect(
                self._handle_keyboard_region_selected
            )
            logging.info("Signal connected successfully")
            
            # Enter selection mode
            logging.info("Entering keyboard region selection mode")
            self.keyboard_canvas.interaction.enter_keyboard_region_selection_mode()
            self.keyboard_canvas.setCursor(Qt.CrossCursor)
            logging.info("Selection mode activated with crosshair cursor")
        else:
            QMessageBox.warning(self, "Canvas Error", "Canvas interaction system not available.")
            logging.error("Canvas interaction system not available")
    
    def _handle_keyboard_region_selected(self, x: int, y: int, width: int, height: int):
        """Handle the keyboard region selection from canvas."""
        logging.info(f"=== KEYBOARD REGION SELECTION RECEIVED IN MAIN APP ===")
        logging.info(f"User-drawn ROI rectangle coordinates: x={x}, y={y}, width={width}, height={height}")
        logging.info(f"ROI rectangle bounds: left={x}, right={x+width}, top={y}, bottom={y+height}")
        logging.info("This ROI rectangle will be used to crop the frame for auto-detection")
        
        # Reset cursor
        self.keyboard_canvas.setCursor(Qt.ArrowCursor)
        logging.debug("Reset cursor to arrow")
        
        # Call the wizard's handler if it exists
        if self.calibration_wizard:
            logging.info(f"Calibration wizard exists, calling handle_keyboard_region_selected")
            try:
                self.calibration_wizard.handle_keyboard_region_selected(x, y, width, height)
                logging.info("Successfully called wizard's keyboard region handler")
                
                # Force canvas refresh to show the new overlays
                logging.info("Forcing canvas refresh to display new overlays")
                
                # Check if overlays were created successfully
                if self.app_state.overlays:
                    logging.info(f"Successfully created {len(self.app_state.overlays)} overlays")
                    
                    # Enable convert button since we have overlays
                    self.control_panel.convert_button.setEnabled(True)
                    logging.info("Enabled convert button")
                    
                    # Apply template styles to the new overlays
                    self._apply_template_styles_to_overlays()
                    logging.info("Applied template styles to overlays")
                    
                    # Ensure overlays are visible
                    self.app_state.ui.show_overlays = True
                    self.show_overlays_action.setChecked(True)
                    logging.info("Ensured show_overlays is True")
                    
                    # Cleanup the wizard instance
                    self.calibration_wizard = None
                    logging.info("Cleaned up calibration wizard instance")
                
                # Update canvas to show overlays
                if self.keyboard_canvas:
                    # Force immediate full canvas update
                    logging.info("Forcing immediate full canvas update")
                    self.keyboard_canvas.update()  # Schedule full repaint
                    
                    # Use QTimer to do a second update with display_frame
                    def delayed_full_redraw():
                        current_frame = self.app_state.video.current_frame_index
                        if current_frame is not None:
                            logging.info(f"Delayed full redraw: Redisplaying frame {current_frame} with overlays")
                            # Call display_frame which recreates the base pixmap
                            self.keyboard_canvas.display_frame(current_frame)
                    
                    # Schedule full redraw after 100ms to ensure UI has settled
                    from PySide6.QtCore import QTimer
                    QTimer.singleShot(100, delayed_full_redraw)
                    logging.info("Scheduled delayed full redraw")
            except Exception as e:
                logging.error(f"Error calling wizard's keyboard region handler: {e}", exc_info=True)
        else:
            logging.error("No calibration wizard available to handle keyboard region selection")
            logging.error(f"calibration_wizard is: {self.calibration_wizard}")
        
        self.control_panel.update_controls_from_state() # Reflect any changes
        self.control_panel.update_trim_controls_from_state() # Update frame range controls
        self.control_panel.update_selected_overlay_display() # Refresh selected overlay info
    
    def _apply_template_styles_to_overlays(self):
        """Delegate template style application to CalibrationWorkflow."""
        if self.calibration_workflow:
            self.calibration_workflow.apply_template_styles_to_overlays()


    def _on_toggle_hist_detection(self):
        """Delegate histogram detection toggle to DetectionManager."""
        if self.detection_manager:
            self.detection_manager.toggle_histogram_detection()

    def _on_toggle_delta_detection(self):
        """Delegate delta detection toggle to DetectionManager."""
        if self.detection_manager:
            self.detection_manager.toggle_delta_detection()
    
    def _on_toggle_winner_takes_black(self, enabled: bool):
        """Toggle black key filter (winner takes black) mode."""
        self.app_state.detection.winner_takes_black_enabled = enabled
        self.app_state.unsaved_changes = True
        logging.info(f"Black key filter (winner takes black) is now {enabled}")
    
    def _handle_hand_assignment_toggle(self, enabled: bool):
        """Toggle hand assignment mode for MIDI channel separation."""
        self.app_state.detection.hand_assignment_enabled = enabled
        self.app_state.unsaved_changes = True
        logging.info(f"Hand assignment is now {enabled}")
    
    
    def _handle_visual_threshold_monitor_toggle(self, enabled: bool):
        """Delegate visual threshold monitor toggle to DisplayManager and update menu."""
        if self.display_manager:
            self.display_manager.handle_visual_threshold_monitor_toggle(enabled)
        
        # Update menu check state when changed from control panel
        if hasattr(self, 'visual_threshold_monitor_action'):
            self.visual_threshold_monitor_action.setChecked(enabled)
    
    def _handle_overlay_color_change(self, color: str):
        """Handle overlay color change from control panel."""
        logging.debug(f"Overlay color changed to: {color}")
        # Update the app state with the new color
        self.app_state.ui.overlay_color = color.lower()
        # Refresh the keyboard canvas to apply the new color
        if self.keyboard_canvas:
            self.keyboard_canvas.update()
        # Mark as unsaved changes
        self.app_state.unsaved_changes = True
    
    def _handle_fps_override_change(self, fps_override):
        """Handle FPS override change from control panel.
        
        Args:
            fps_override: The FPS override value (float) or None for auto-detect
        """
        logging.info(f"Setting FPS override to: {fps_override}")
        
        # Update the app state with the new FPS override
        if hasattr(self.app_state, 'video'):
            self.app_state.video.fps_override = fps_override
            self.app_state.unsaved_changes = True
            
            # Update the FPS display if we have a video loaded
            if self.video_session:
                effective_fps = fps_override if fps_override else self.video_session.fps
                if fps_override:
                    logging.info(f"FPS override set to {fps_override} (detected: {self.video_session.fps})")
                else:
                    logging.info(f"FPS override disabled, using detected: {self.video_session.fps}")
                
                # Update the control panel display to show effective FPS
                self.control_panel.update_video_info(self.video_session.fps)
    
    def _handle_octave_transpose_change(self, transpose_value: int):
        """Handle octave transpose change from control panel.
        
        Args:
            transpose_value: The current octave transpose value (-8 to +8)
        """
        logging.info(f"Applying octave transpose: {transpose_value}")
        
        # Update the app state with the new transpose value
        if hasattr(self.app_state, 'midi') and hasattr(self.app_state.midi, 'octave_transpose'):
            self.app_state.midi.octave_transpose = transpose_value
        
        # Force a full redraw of the canvas to update overlay labels
        if self.keyboard_canvas:
            # Force recreation of the display to ensure labels are updated
            self.keyboard_canvas.draw_overlays()
        
        # Mark state as changed
        self.app_state.mark_unsaved()
    
    def _handle_processing_start_frame_change(self, frame_value: int):
        """Handle processing start frame change from control panel.
        
        Args:
            frame_value: The new start frame for MIDI processing
        """
        if hasattr(self.app_state, 'video'):
            video = self.app_state.video
            
            # If video is trimmed, constrain to trim range
            if video.video_is_trimmed:
                min_frame = video.trim_start_frame
                max_frame = video.trim_end_frame
                frame_value = max(min_frame, min(frame_value, max_frame))
            else:
                # Validate bounds (0 to total frames)
                total_frames = getattr(video, 'total_frames', 0)
                if total_frames > 0:
                    frame_value = max(0, min(frame_value, total_frames - 1))
            
            # Validate that start < end (if end is set)
            if (video.processing_end_frame > 0 and 
                frame_value >= video.processing_end_frame):
                logging.warning(f"Processing start frame {frame_value} must be less than end frame {video.processing_end_frame}")
                return
            
            video.processing_start_frame = frame_value
            self.app_state.mark_unsaved()
            logging.info(f"Set MIDI processing start frame to: {frame_value}")
    
    def _handle_processing_end_frame_change(self, frame_value: int):
        """Handle processing end frame change from control panel.
        
        Args:
            frame_value: The new end frame for MIDI processing
        """
        if hasattr(self.app_state, 'video'):
            video = self.app_state.video
            
            # If video is trimmed, constrain to trim range
            if video.video_is_trimmed:
                min_frame = video.trim_start_frame
                max_frame = video.trim_end_frame
                frame_value = max(min_frame, min(frame_value, max_frame))
            else:
                # Validate bounds (0 to total frames)
                total_frames = getattr(video, 'total_frames', 0)
                if total_frames > 0:
                    frame_value = max(0, min(frame_value, total_frames - 1))
            
            # Validate that end > start (if start is set)
            if (video.processing_start_frame > 0 and 
                frame_value <= video.processing_start_frame):
                logging.warning(f"Processing end frame {frame_value} must be greater than start frame {video.processing_start_frame}")
                return
            
            video.processing_end_frame = frame_value
            self.app_state.mark_unsaved()
            logging.info(f"Set MIDI processing end frame to: {frame_value}")
    
    def _handle_trim_video_request(self, start_frame: int, end_frame: int):
        """Handle video trimming request - makes trim range permanent for session."""
        if not hasattr(self.app_state, 'video'):
            return
        
        video = self.app_state.video
        
        # Set trim parameters
        video.trim_start_frame = start_frame
        video.trim_end_frame = end_frame if end_frame != -1 else video.total_frames - 1
        video.video_is_trimmed = True
        
        # Update MIDI processing range to match trim range
        video.processing_start_frame = video.trim_start_frame
        video.processing_end_frame = video.trim_end_frame
        
        # Update UI controls to reflect new ranges
        self.control_panel.update_controls_from_state()
        self.control_panel.update_video_frame_limits()
        
        # Update frame slider to respect new trim range
        self._update_frame_slider_for_video()
        
        # Navigate to start of trimmed range
        self._display_frame_with_slider_update(start_frame)
        
        # Save the changes
        self.app_state.mark_unsaved()
        
        # Auto-save trim settings
        if hasattr(self, 'video_loading_workflow') and self.video_loading_workflow:
            success = self.video_loading_workflow.save_current_config()
            if success:
                logging.info("Video trim settings automatically saved to config file.")
            else:
                logging.warning("Auto-save of video trim settings failed.")
        
        logging.info(f"Video trimmed to frames {start_frame} to {video.trim_end_frame}. MIDI processing range updated accordingly.")
    
    def _initialize_processing_range_defaults(self):
        """Initialize processing range defaults based on trim range if not already set."""
        if not hasattr(self.app_state, 'video'):
            return
        
        video = self.app_state.video
        
        # If processing range is not set (both are 0), set defaults based on trim range
        if video.processing_start_frame == 0 and video.processing_end_frame == 0:
            if video.video_is_trimmed and video.trim_start_frame > 0:
                # Use trim range as default
                video.processing_start_frame = video.trim_start_frame
                video.processing_end_frame = video.trim_end_frame if video.trim_end_frame > 0 else video.total_frames - 1
                logging.info(f"Set processing range defaults from trim range: {video.processing_start_frame} to {video.processing_end_frame}")
            else:
                # Use full video range as default
                video.processing_start_frame = 0
                video.processing_end_frame = video.total_frames - 1 if video.total_frames > 0 else 0
                logging.info(f"Set processing range defaults to full video: {video.processing_start_frame} to {video.processing_end_frame}")
            
            # Update the UI controls
            if hasattr(self.control_panel, 'processing_start_frame_spin'):
                self.control_panel.processing_start_frame_spin.setValue(video.processing_start_frame)
                self.control_panel.processing_end_frame_spin.setValue(video.processing_end_frame)
    
    def _handle_detection_logging_toggle(self, enabled: bool):
        """No-op placeholder (kept for compatibility with older UI/menu hooks)."""
        pass
    
    def _log_detection_parameters(self):
        """No-op placeholder (kept for compatibility with older UI/menu hooks)."""
        pass
    
    
    
    def _resize_and_position_window(self):
        """Delegate window resize and positioning to WindowManager."""
        self.window_manager.resize_and_position_window()
    



    def _create_detection_wrapper(self):
        """Delegate detection wrapper creation to DetectionManager."""
        if self.detection_manager:
            return self.detection_manager.create_detection_wrapper()
        return None
    
    # UIUpdateInterface implementations
    def update_overlay_action(self, checked: bool) -> None:
        """Update the overlay visibility action state."""
        if hasattr(self, 'show_overlays_action'):
            self.show_overlays_action.setChecked(checked)
    
    def refresh_canvas(self) -> None:
        """Refresh the keyboard canvas display."""
        if hasattr(self, 'keyboard_canvas') and self.app_state.video.current_frame_index is not None:
            self.keyboard_canvas.display_frame(self.app_state.video.current_frame_index)
    
    def update_control_panel(self) -> None:
        """Update the control panel display."""
        if hasattr(self, 'control_panel'):
            self.control_panel.update_controls_from_state()
    
    def update_selected_overlay_display(self) -> None:
        """Update the selected overlay display in control panel."""
        if hasattr(self, 'control_panel'):
            self.control_panel.update_selected_overlay_display()
    
    
    def update_live_detection_action(self, checked: bool) -> None:
        """Update live detection action state."""
        if hasattr(self, 'live_detection_action'):
            self.live_detection_action.setChecked(checked)
    
    def update_detection_threshold(self, value: float) -> None:
        """Update detection threshold spinner value."""
        if hasattr(self, 'control_panel'):
            self.control_panel.detection_threshold_spin.setValue(value)
    
    def show_message(self, title: str, message: str) -> None:
        """Show a message to the user."""
        QMessageBox.information(self, title, message)
    
    def get_video_session(self) -> Optional[object]:
        """Get current video session if available."""
        return self.video_session
    
    def has_video_loaded(self) -> bool:
        """Check if a video is currently loaded."""
        return self.video_session is not None
    
    def get_total_frames(self) -> Optional[int]:
        """Get total frames in current video."""
        return self.video_session.total_frames if self.video_session else None
    
    def get_roi_bgr(self, overlay: object) -> Optional[object]:
        """Get ROI BGR from keyboard canvas for given overlay."""
        if hasattr(self, 'keyboard_canvas'):
            return self.keyboard_canvas.get_roi_bgr(overlay)
        return None
    


if __name__ == "__main__":
    qapp = QApplication(sys.argv)
    app = Video2MidiApp()
    app.show()
    sys.exit(qapp.exec())
