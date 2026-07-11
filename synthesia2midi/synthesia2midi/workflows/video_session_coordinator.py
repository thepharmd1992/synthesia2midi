"""Shared video-session setup for the main Qt window.

The file-open and YouTube-download paths need the same fragile post-load ordering.
This coordinator centralizes that ordering so future agents do not patch two large,
near-identical blocks in ``main.py``.
"""
from __future__ import annotations

import logging

from synthesia2midi.video_loader import VideoSession
from synthesia2midi.workflows.auto_calibration import AutoCalibrationWorkflow
from synthesia2midi.workflows.calibration import CalibrationWorkflow
from synthesia2midi.workflows.conversion import ConversionWorkflow
from synthesia2midi.workflows.detection_manager import DetectionManager


class VideoSessionCoordinator:
    """Load a video path and wire the resulting session into the app."""

    def __init__(self, app):
        self.app = app

    def load_path(
        self,
        filepath: str,
        *,
        log_prefix: str,
        update_fps_display: bool,
    ) -> bool:
        """Load ``filepath`` and apply all UI/workflow session wiring.

        Args:
            filepath: Video file or frame-directory path selected by the user.
            log_prefix: Existing log prefix to preserve source-specific messages.
            update_fps_display: Preserve the historical file-open behavior of
                updating the FPS label while leaving the YouTube path unchanged.
        """
        app = self.app
        self._close_existing_session(log_prefix)
        app.state_manager.reset_to_defaults()
        if hasattr(app, "set_video_loaded_state"):
            app.set_video_loaded_state(False)

        logging.info("%s: Calling VideoLoadingWorkflow.load_video_file(%s)", log_prefix, filepath)
        success, video_session = app.video_loading_workflow.load_video_file(filepath)
        if not success:
            return False

        self.apply_loaded_session(
            video_session,
            log_prefix=log_prefix,
            update_fps_display=update_fps_display,
        )
        if hasattr(app, "set_video_loaded_state"):
            app.set_video_loaded_state(True)
        return True

    def apply_loaded_session(
        self,
        video_session: VideoSession,
        *,
        log_prefix: str,
        update_fps_display: bool,
    ) -> None:
        """Apply a successfully loaded session to controls, canvas, workflows, and UI."""
        app = self.app
        app.video_session = video_session
        app.video_controls.set_video_session(video_session)
        self._set_canvas_video_session(video_session, log_prefix)

        app.video_controls.update_frame_slider_for_video()
        app.control_panel.update_video_frame_limits()

        if update_fps_display and app.video_session:
            app.control_panel.update_video_info(app.video_session.fps)

        self._initialize_video_bound_workflows(video_session)
        if hasattr(app.detection_manager, "create_detection_wrapper"):
            app.keyboard_canvas.detect_pressed_func = app.detection_manager.create_detection_wrapper()
        else:
            app.keyboard_canvas.detect_pressed_func = app.main_action_controller.create_detection_wrapper()
        self._apply_loaded_configuration_ui()
        # Loading a session is not a user edit. Some Qt control sync calls emit
        # change signals while reflecting loaded state; clear any dirty flag they set.
        app.app_state.unsaved_changes = False

    def _close_existing_session(self, log_prefix: str) -> None:
        app = self.app
        try:
            if app.video_session:
                logging.info("%s: Closing existing video session.", log_prefix)
                app.video_session.close()
                app.video_session = None
                logging.info("%s: Existing video session closed.", log_prefix)
        except Exception as exc:
            logging.error("%s: Error closing existing video session: %s", log_prefix, exc, exc_info=True)

    def _set_canvas_video_session(self, video_session: VideoSession, log_prefix: str) -> None:
        app = self.app
        try:
            logging.info("%s: Setting video session in KeyboardCanvas.", log_prefix)
            if app.keyboard_canvas:
                app.keyboard_canvas.set_video_session(video_session)
                logging.info("%s: Video session set in KeyboardCanvas.", log_prefix)
            else:
                logging.warning(
                    "%s: KeyboardCanvas not initialized when trying to set video session.",
                    log_prefix,
                )
        except Exception as exc:
            logging.error("%s: Error setting video session in KeyboardCanvas: %s", log_prefix, exc, exc_info=True)

    def _initialize_video_bound_workflows(self, video_session: VideoSession) -> None:
        app = self.app
        app.calibration_workflow = CalibrationWorkflow(app.app_state, video_session, app)
        app.auto_calibration_workflow = AutoCalibrationWorkflow(app.app_state, video_session, app)
        if getattr(app, "detection_manager", None) is None:
            app.detection_manager = DetectionManager(app.app_state, app._update_current_frame_display, app)
        else:
            app.detection_manager.reset_detector_cache()
        if hasattr(app.detection_manager, "set_navigation_mode"):
            app.detection_manager.set_navigation_mode(True)
        app.conversion_workflow = ConversionWorkflow(
            app.app_state,
            video_session,
            app,
            app.detection_manager,
        )

    def _apply_loaded_configuration_ui(self) -> None:
        app = self.app
        video_info = app.video_loading_workflow.get_video_info()
        config_loaded_from_specific_ini = video_info.get("config_file") is not None

        if config_loaded_from_specific_ini:
            app.control_panel.update_controls_from_state()
            app.control_panel.update_trim_controls_from_state()
            app.video_session_ui_controller.initialize_processing_range_defaults()
            initial_frame = (
                app.app_state.video.processing_start_frame
                if app.app_state.video.processing_start_frame > 0
                else app.app_state.video.current_frame_index
            )
            app.video_controls.display_frame_with_slider_update(initial_frame)
            app.control_panel.convert_button.setEnabled(app.control_panel._can_convert())
            app.control_panel.wizard_button.setEnabled(True)
            app.window_manager.resize_and_position_window()
            return

        logging.info("Video-specific INI not found or failed to load. User must run calibration wizard.")
        app.app_state.overlays.clear()
        app.app_state.unsaved_changes = False
        app.control_panel.update_controls_from_state()
        app.control_panel.update_trim_controls_from_state()
        initial_frame = app.app_state.video.trim_start_frame if app.app_state.video.video_is_trimmed else 0
        app.video_controls.display_frame_with_slider_update(initial_frame)
        app.control_panel.convert_button.setEnabled(False)
        app.control_panel.wizard_button.setEnabled(True)
        app.window_manager.resize_and_position_window()
