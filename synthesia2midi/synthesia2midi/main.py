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
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2  # For HSV color space conversion
import numpy as np  # For image data
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QDialog, QFileDialog, QHBoxLayout,
    QLabel, QListView, QMainWindow, QMessageBox, QSlider,
    QTreeView, QVBoxLayout, QWidget
)

from synthesia2midi.app_config import (
    APP_NAME, FRAME_NAV_INTERVALS, OverlayConfig
)
from synthesia2midi.config_manager import ConfigManager
from synthesia2midi.core.app_state import AppState
from synthesia2midi.core.logging_config import LoggingConfig
from synthesia2midi.core.state_manager import StateManager
from synthesia2midi.detection.factory import DetectionFactory
from synthesia2midi.detection.roi_utils import get_hist_feature
from synthesia2midi.gui.controls_qt import ControlPanelQt
from synthesia2midi.gui.display_manager import DisplayManager
from synthesia2midi.gui.keyboard_canvas import KeyboardCanvas
from synthesia2midi.gui.calibration_effects_controller import CalibrationEffectsController
from synthesia2midi.gui.calibration_interaction_controller import CalibrationInteractionController
from synthesia2midi.gui.calibration_wizard_controller import CalibrationWizardController
from synthesia2midi.gui.midi_touchup_controller import MidiTouchupController
from synthesia2midi.gui.signal_manager import ControlSignalManager
from synthesia2midi.gui.startup_dialog import StartupDialog
from synthesia2midi.gui.youtube_download_dialog import YouTubeDownloadDialog
from synthesia2midi.gui.ui_update_interface import UIUpdateInterface
from synthesia2midi.gui.video_controls import VideoControls
from synthesia2midi.gui.window_manager import WindowManager
from synthesia2midi.video_loader import VideoSession
from synthesia2midi.workflows.overlay_manager import OverlayManager
from synthesia2midi.workflows.parameter_manager import ParameterManager
from synthesia2midi.workflows.video_loading import VideoLoadingWorkflow
from synthesia2midi.workflows.video_session_coordinator import VideoSessionCoordinator
from synthesia2midi.workflows.video_to_frames import VideoToFramesController

# Configure application logging. Enable console logging during development if needed.
log_filename = LoggingConfig.setup_logging(
    log_to_file=True,
    log_to_console=False,  # Set to True for development
    log_level=logging.INFO
)
LoggingConfig.suppress_verbose_libraries()  # Suppress third-party library logs


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
        self.video_session_coordinator = VideoSessionCoordinator(self)
        self.parameter_manager = ParameterManager(self.app_state, self.state_manager, self)
        self.window_manager = WindowManager(self.app_state, self)
        self.calibration_workflow = None  # Will be initialized when video is loaded
        self.auto_calibration_workflow = None  # Will be initialized when video is loaded
        self.conversion_workflow = None  # Will be initialized when video is loaded
        self.debug_tools = None  # Will be initialized when video is loaded
        self.detection_manager = None  # Will be initialized when video is loaded
        self.overlay_manager = OverlayManager(self.app_state, self)
        self.display_manager = DisplayManager(self.app_state, self)

        # Video to frames conversion controller
        self.video_to_frames_worker = None
        self.video_to_frames_controller = VideoToFramesController(self)
        self.calibration_interaction_controller = CalibrationInteractionController(self)
        self.calibration_effects_controller = CalibrationEffectsController(self)
        self.calibration_wizard_controller = CalibrationWizardController(self)
        self._midi_touchup_processes: List[Any] = []
        self.midi_touchup_controller = MidiTouchupController(self)
        self._is_closing = False

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
        self.keyboard_canvas = KeyboardCanvas(self.app_state, width=720, height=450,
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
        nav_instructions = QLabel("Move forward - Page Down or Right Arrow\nMove backward - Page Up or Left Arrow")
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


        # Let the control panel shrink on smaller screens while keeping a reasonable preferred size
        from PySide6.QtWidgets import QSizePolicy
        self.control_panel.setMinimumWidth(380)
        self.control_panel.setMaximumWidth(1000)
        self.control_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
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

        # Page Up/Down and Left/Right for navigation
        pgup_action = QAction(self)
        pgup_action.setShortcut(Qt.Key_PageUp)
        pgup_action.triggered.connect(self._navigate_frame_pgup)
        self.addAction(pgup_action)
        left_action = QAction(self)
        left_action.setShortcut(Qt.Key_Left)
        left_action.triggered.connect(self._navigate_frame_pgup)
        self.addAction(left_action)


        pgdn_action = QAction(self)
        pgdn_action.setShortcut(Qt.Key_PageDown)
        pgdn_action.triggered.connect(self._navigate_frame_pgdn)
        self.addAction(pgdn_action)
        right_action = QAction(self)
        right_action.setShortcut(Qt.Key_Right)
        right_action.triggered.connect(self._navigate_frame_pgdn)
        self.addAction(right_action)

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
        logging.info(
            f"_open_video_file: filedialog returned: '{filepath if filepath else 'Dialog cancelled or no file selected'}'"
        )
        if not filepath:
            logging.info("_open_video_file: No filepath selected, returning.")
            return

        logging.info("_open_video_file: Proceeding with filepath.")
        self.video_session_coordinator.load_path(
            filepath,
            log_prefix="_open_video_file",
            update_fps_display=True,
        )


    def _handle_youtube_video_downloaded(self, filepath: str):
        """Handle a video downloaded from YouTube."""
        logging.info(f"_handle_youtube_video_downloaded: Loading YouTube video from {filepath}")
        logging.info(f"_handle_youtube_video_downloaded: Auto-convert setting: {self.app_state.ui.auto_convert_to_frames}")
        self.video_session_coordinator.load_path(
            filepath,
            log_prefix="_handle_youtube_video_downloaded",
            update_fps_display=False,
        )

    def _handle_video_to_frames_request(self):
        """Handle request to convert current video to frame series."""
        return self.video_to_frames_controller.handle_request()

    def _on_conversion_progress(self, message: str):
        """Handle progress updates from video conversion."""
        return self.video_to_frames_controller.on_progress(message)

    def _on_conversion_finished(self, success: bool, message: str):
        """Handle completion of video conversion."""
        return self.video_to_frames_controller.on_finished(success, message)

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
                self._is_closing = False
                event.ignore()
                return
            # If No or save complete, proceed to close

        self._is_closing = True
        self.midi_touchup_controller.shutdown_processes()

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
        return self.calibration_interaction_controller._handle_color_pick(color_rgb, coordinates)

    def _handle_overlay_selection(self, selected_key_id: Optional[int]):
        return self.calibration_interaction_controller._handle_overlay_selection(selected_key_id)



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
                self._show_conversion_complete_dialog_with_touchup(midi_output_path)
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

    def _show_conversion_complete_dialog_with_touchup(self, midi_output_path: str) -> None:
        return self.midi_touchup_controller.show_conversion_complete_dialog(midi_output_path)


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
        return self.calibration_effects_controller._handle_spark_roi_selection_request()

    def _handle_spark_roi_visibility_toggle(self, visible: bool):
        return self.calibration_effects_controller._handle_spark_roi_visibility_toggle(visible)

    def _handle_shadow_roi_selection_request(self):
        return self.calibration_effects_controller._handle_shadow_roi_selection_request()

    def _handle_shadow_white_roi_selection_request(self):
        return self.calibration_effects_controller._handle_shadow_white_roi_selection_request()

    def _handle_shadow_black_roi_selection_request(self):
        return self.calibration_effects_controller._handle_shadow_black_roi_selection_request()

    def _handle_spark_roi_updated(self, top_y: int, bottom_y: int):
        return self.calibration_effects_controller._handle_spark_roi_updated(top_y, bottom_y)

    def _handle_spark_calibration_request(self, step_type: str):
        return self.calibration_effects_controller._handle_spark_calibration_request(step_type)

    def _handle_auto_spark_calibration_request(self, key_type: str):
        return self.calibration_effects_controller._handle_auto_spark_calibration_request(key_type)

    def _handle_spark_detection_toggle(self, enabled: bool):
        return self.calibration_effects_controller._handle_spark_detection_toggle(enabled)

        # The detector will be recreated automatically on the next frame processing
        # when factory.create_from_app_state is called

    def _handle_spark_detection_sensitivity_change(self, value: float):
        return self.calibration_effects_controller._handle_spark_detection_sensitivity_change(value)

        # The sensitivity will be used on the next frame processing

    def _handle_shadow_detection_toggle(self, enabled: bool):
        return self.calibration_effects_controller._handle_shadow_detection_toggle(enabled)

        # The app_state is already updated by the control panel
        # We don't need to recreate the detector here as it will be recreated
        # automatically on the next frame processing when factory.create_from_app_state is called

    def _handle_shadow_detection_sensitivity_change(self, value: float):
        return self.calibration_effects_controller._handle_shadow_detection_sensitivity_change(value)
        # The app_state is already updated by the control panel
        # The sensitivity will be used on the next frame processing

    def _handle_shadow_darkness_threshold_change(self, value: float):
        return self.calibration_effects_controller._handle_shadow_darkness_threshold_change(value)
        # The app_state is already updated by the control panel
        # The threshold will be used on the next frame processing

    def _handle_shadow_calibration_request(self, key_type: str, calibration_type: str):
        return self.calibration_effects_controller._handle_shadow_calibration_request(key_type, calibration_type)

    def _handle_overlay_type_change(self, overlay_type: str):
        return self.calibration_effects_controller._handle_overlay_type_change(overlay_type)

    def _capture_spark_background_calibration(self):
        return self.calibration_effects_controller._capture_spark_background_calibration()

    def _capture_spark_overlay_calibration(self, overlay, calibration_mode: str):
        return self.calibration_effects_controller._capture_spark_overlay_calibration(overlay, calibration_mode)

    def _capture_shadow_overlay_calibration(self, overlay, calibration_mode: str):
        return self.calibration_effects_controller._capture_shadow_overlay_calibration(overlay, calibration_mode)

    def _extract_roi(self, frame: np.ndarray, overlay: OverlayConfig) -> Optional[np.ndarray]:
        return self.calibration_effects_controller._extract_roi(frame, overlay)

    def _get_calibration_instructions(self, step_type: str) -> str:
        return self.calibration_effects_controller._get_calibration_instructions(step_type)

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
        return self.calibration_wizard_controller._invoke_calibration_wizard()

    def _handle_keyboard_region_selection_request(self):
        return self.calibration_wizard_controller._handle_keyboard_region_selection_request()

    def _handle_edit_current_calibration_request(self):
        return self.calibration_wizard_controller._handle_edit_current_calibration_request()

    def _clone_auto_detect_tuning_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return self.calibration_wizard_controller._clone_auto_detect_tuning_context(context)

    def _cache_auto_detect_tuning_context(self, context: Dict[str, Any]) -> None:
        return self.calibration_wizard_controller._cache_auto_detect_tuning_context(context)

    def _get_current_frame_rgb_for_tuning(self) -> Optional[np.ndarray]:
        return self.calibration_wizard_controller._get_current_frame_rgb_for_tuning()

    def _build_auto_detect_tuning_context_from_state(self) -> Optional[Dict[str, Any]]:
        return self.calibration_wizard_controller._build_auto_detect_tuning_context_from_state()

    def _resolve_auto_detect_tuning_context(self, *, use_wizard_context: bool) -> Optional[Dict[str, Any]]:
        return self.calibration_wizard_controller._resolve_auto_detect_tuning_context(use_wizard_context=use_wizard_context)

    def _has_editable_auto_detect_tuning_context(self) -> bool:
        return self.calibration_wizard_controller._has_editable_auto_detect_tuning_context()

    def _handle_keyboard_region_selected(self, x: int, y: int, width: int, height: int):
        return self.calibration_wizard_controller._handle_keyboard_region_selected(x, y, width, height)

    def _apply_auto_detect_preview_result(self, detection_results: Dict[str, Any]) -> bool:
        return self.calibration_wizard_controller._apply_auto_detect_preview_result(detection_results)

    def _open_auto_detect_tuning_dialog(self, *, use_wizard_context: bool=True) -> bool:
        return self.calibration_wizard_controller._open_auto_detect_tuning_dialog(use_wizard_context=use_wizard_context)

    def _on_auto_detect_tuning_dialog_finished(self, _result: int) -> None:
        return self.calibration_wizard_controller._on_auto_detect_tuning_dialog_finished(_result)

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

    def _handle_exemplar_key_type_enabled_change(self, key_type: str, enabled: bool):
        """Handle per-key-type exemplar availability changes from the control panel."""
        if key_type not in {"LW", "LB", "RW", "RB"}:
            logging.warning(f"Ignoring invalid exemplar key type toggle: {key_type}")
            return

        self.app_state.detection.exemplar_key_type_enabled[key_type] = enabled
        self.app_state.unsaved_changes = True
        logging.info(f"Exemplar key type {key_type} availability set to {enabled}")

        # If user disabled the key type currently being calibrated, cancel that calibration.
        if (
            not enabled
            and self.app_state.calibration.calibration_mode == "lit_exemplar"
            and self.app_state.calibration.current_calibration_key_type == key_type
        ):
            self.app_state.calibration.calibration_mode = None
            self.app_state.calibration.current_calibration_key_type = None
            logging.info(f"Cancelled lit exemplar calibration for disabled key type {key_type}")

        if self.control_panel:
            self.control_panel.update_controls_from_state()

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
