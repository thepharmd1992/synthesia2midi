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
from typing import Any, List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import QApplication, QDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QSlider, QVBoxLayout, QWidget

from .app_config import APP_NAME, FRAME_NAV_INTERVALS
from synthesia2midi.config_manager import ConfigManager
from synthesia2midi.core.app_state import AppState
from synthesia2midi.core.logging_config import LoggingConfig
from synthesia2midi.core.state_manager import StateManager

from synthesia2midi.gui.controls_qt import ControlPanelQt
from synthesia2midi.gui.display_manager import DisplayManager
from synthesia2midi.gui.keyboard_canvas import KeyboardCanvas
from synthesia2midi.gui.main_action_controller import MainActionController
from synthesia2midi.gui.calibration_effects_controller import CalibrationEffectsController
from synthesia2midi.gui.calibration_interaction_controller import CalibrationInteractionController
from synthesia2midi.gui.calibration_wizard_controller import CalibrationWizardController
from synthesia2midi.gui.midi_conversion_controller import MidiConversionController
from synthesia2midi.gui.midi_touchup_controller import MidiTouchupController
from synthesia2midi.gui.signal_manager import ControlSignalManager
from synthesia2midi.gui.startup_dialog import StartupDialog
from synthesia2midi.gui.ui_update_interface import UIUpdateInterface
from synthesia2midi.gui.video_session_ui_controller import VideoSessionUiController
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
        self.video_session_ui_controller = VideoSessionUiController(self)
        self.main_action_controller = MainActionController(self)
        self.calibration_interaction_controller = CalibrationInteractionController(self)
        self.calibration_effects_controller = CalibrationEffectsController(self)
        self.calibration_wizard_controller = CalibrationWizardController(self)
        self._midi_touchup_processes: List[Any] = []
        self.midi_touchup_controller = MidiTouchupController(self)
        self.midi_conversion_controller = MidiConversionController(self)
        self._is_closing = False

        # Frame slider handling is now done by VideoControls class


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
        open_action.triggered.connect(self.video_session_ui_controller.open_video_file)
        filemenu.addAction(open_action)

        youtube_action = QAction("Download Youtube Video...", self)
        youtube_action.triggered.connect(self.video_session_ui_controller.show_youtube_download_dialog)
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
        self.show_overlays_action.triggered.connect(self.main_action_controller.toggle_overlays)
        view_menu.addAction(self.show_overlays_action)

        self.live_detection_action = QAction("Live Detection Feedback", self)
        self.live_detection_action.setCheckable(True)
        self.live_detection_action.setChecked(self.app_state.ui.live_detection_feedback)
        self.live_detection_action.triggered.connect(self.main_action_controller.toggle_live_detection_feedback)
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
            action.triggered.connect(lambda checked, val=interval: self.video_session_ui_controller.handle_frame_nav_interval(val))
            frame_nav_menu.addAction(action)
            self.frame_nav_actions[interval] = action

        # Visual Threshold Monitor menu
        debug_menu = menubar.addMenu("Visual Threshold Monitor")

        # Visual Threshold Monitor toggle
        self.visual_threshold_monitor_action = QAction("Enable", self)
        self.visual_threshold_monitor_action.setCheckable(True)
        self.visual_threshold_monitor_action.setChecked(self.app_state.ui.visual_threshold_monitor_enabled)
        self.visual_threshold_monitor_action.triggered.connect(self.main_action_controller.handle_visual_threshold_monitor_menu)
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
                                              on_color_pick_callback=self.calibration_interaction_controller._handle_color_pick,
                                              on_overlay_select_callback=self.calibration_interaction_controller._handle_overlay_selection,
                                              detect_pressed_func=self.main_action_controller.create_detection_wrapper()
                                              )
        # Set up additional callbacks
        self.keyboard_canvas.on_spark_roi_callback = self.calibration_effects_controller._handle_spark_roi_updated
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
        space_action.triggered.connect(self.midi_conversion_controller.start_conversion_process)
        self.addAction(space_action)

        # Page Up/Down and Left/Right for navigation
        pgup_action = QAction(self)
        pgup_action.setShortcut(Qt.Key_PageUp)
        pgup_action.triggered.connect(self.video_controls.navigate_frame_pgup)
        self.addAction(pgup_action)
        left_action = QAction(self)
        left_action.setShortcut(Qt.Key_Left)
        left_action.triggered.connect(self.video_controls.navigate_frame_pgup)
        self.addAction(left_action)


        pgdn_action = QAction(self)
        pgdn_action.setShortcut(Qt.Key_PageDown)
        pgdn_action.triggered.connect(self.video_controls.navigate_frame_pgdn)
        self.addAction(pgdn_action)
        right_action = QAction(self)
        right_action.setShortcut(Qt.Key_Right)
        right_action.triggered.connect(self.video_controls.navigate_frame_pgdn)
        self.addAction(right_action)

    def _show_startup_dialog(self):
        """Show the startup dialog for choosing video source."""
        logging.info("_show_startup_dialog: Showing startup dialog.")

        dialog = StartupDialog(self)
        dialog.open_local_file.connect(self.video_session_ui_controller.open_video_file)
        dialog.download_from_youtube.connect(self.video_session_ui_controller.show_youtube_download_dialog)

        # If user cancels, just continue with empty application
        if dialog.exec() != QDialog.Accepted:
            logging.info("_show_startup_dialog: User cancelled startup dialog, continuing with empty application.")
            # No video loaded, but app remains open

            # No video loaded, but app remains open



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



    # Frame slider events are handled by VideoControls via ControlSignalManager.

    def _update_current_frame_display(self):
        """Centralized function to reprocess and redisplay the current frame."""
        return self.video_controls.update_current_frame_display()








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










        # The detector will be recreated automatically on the next frame processing
        # when factory.create_from_app_state is called


        # The sensitivity will be used on the next frame processing


        # The app_state is already updated by the control panel
        # We don't need to recreate the detector here as it will be recreated
        # automatically on the next frame processing when factory.create_from_app_state is called

        # The app_state is already updated by the control panel
        # The sensitivity will be used on the next frame processing

        # The app_state is already updated by the control panel
        # The threshold will be used on the next frame processing































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
