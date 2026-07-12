"""
Centralized signal management for control panel connections.
Makes it obvious where to add new controls and how they connect.
"""
import logging

from PySide6.QtCore import QObject


class ControlSignalManager(QObject):
    """
    All control panel signal connections in one place.
    
    No more hunting through 2000 lines of main.py!
    """
    
    def __init__(self, control_panel, main_window):
        super().__init__()
        self.control_panel = control_panel
        self.main_window = main_window
        self.logger = logging.getLogger(f"{__name__}.ControlSignalManager")
        
        # Connect all signals in organized groups
        self._connect_guide_signals()
        self._connect_video_signals()
        self._connect_detection_signals()
        self._connect_calibration_signals()
        self._connect_midi_signals()
        self._connect_ui_signals()
        
        self.logger.info("All control signals connected")

    def _connect_guide_signals(self):
        guide = getattr(self.control_panel, "__dict__", {}).get("guide_page")
        if guide is None:
            return

        mw = self.main_window
        guide.open_video_requested.connect(mw.video_session_ui_controller.open_video_file)
        guide.youtube_requested.connect(mw.video_session_ui_controller.show_youtube_download_dialog)
        guide.find_keyboard_requested.connect(mw.calibration_wizard_controller.run_calibration_wizard)
        guide.review_alignment_requested.connect(mw.calibration_wizard_controller.review_current_alignment)
        guide.capture_unlit_requested.connect(mw.main_action_controller.handle_calibrate_unlit_all_keys)
        guide.assisted_scan_requested.connect(
            mw.calibration_wizard_controller.run_assisted_calibration_from_current_frame
        )
        guide.convert_requested.connect(mw.midi_conversion_controller.start_conversion_process)
    
    def _connect_video_signals(self):
        """Video-related control signals"""
        cp = self.control_panel  # Shorthand
        mw = self.main_window
        
        # Frame navigation interval
        cp.nav_interval_changed.connect(mw.video_session_ui_controller.update_nav_interval)
        
        # Frame slider (owned by main window, not control panel)
        mw.frame_slider.valueChanged.connect(mw.video_controls.on_frame_slider_changed)
        
        # YouTube video download
        cp.youtube_video_downloaded.connect(mw.video_session_ui_controller.handle_youtube_video_downloaded)
        
        # Video to frame series conversion
        cp.video_to_frames_requested.connect(mw.video_session_ui_controller.handle_video_to_frames_request)
        
        # Video trim frame controls
        cp.start_frame_changed.connect(mw.video_session_ui_controller.handle_start_frame_change)
        cp.end_frame_changed.connect(mw.video_session_ui_controller.handle_end_frame_change)
        
        self.logger.debug("Video signals connected")
    
    def _connect_detection_signals(self):
        """Detection-related control signals"""
        cp = self.control_panel  # Shorthand
        mw = self.main_window

        detection_manager = mw.detection_manager

        # Basic detection parameters
        cp.detection_threshold_changed.connect(detection_manager.set_detection_threshold)
        cp.rise_delta_threshold_changed.connect(detection_manager.set_rise_delta_threshold)
        cp.fall_delta_threshold_changed.connect(detection_manager.set_fall_delta_threshold)

        # Detection method toggles
        cp.histogram_detection_toggled.connect(detection_manager.set_histogram_detection_enabled)
        cp.delta_detection_toggled.connect(detection_manager.set_delta_detection_enabled)

        # Black key filter toggle
        cp.winner_takes_black_changed.connect(detection_manager.set_winner_takes_black_enabled)

        # Hand assignment toggle
        cp.hand_assignment_toggled.connect(detection_manager.set_hand_assignment_enabled)
        cp.histogram_threshold_changed.connect(detection_manager.set_histogram_threshold)
        cp.similarity_ratio_changed.connect(detection_manager.set_similarity_ratio)

        self.logger.debug("Detection signals connected")
    
    def _connect_calibration_signals(self):
        """Calibration-related control signals"""
        cp = self.control_panel  # Shorthand
        mw = self.main_window
        
        # Basic calibration actions
        cp.calibrate_unlit_requested.connect(mw.main_action_controller.handle_calibrate_unlit_all_keys)
        cp.calibrate_lit_exemplar_requested.connect(mw.main_action_controller.handle_calibrate_lit_exemplar_key_start)
        cp.exemplar_key_type_enabled_changed.connect(mw.main_action_controller.handle_exemplar_key_type_enabled_change)
        cp.calibration_wizard_requested.connect(mw.calibration_wizard_controller.run_calibration_wizard)
        
        # Overlay management actions
        cp.refresh_overlay_display_requested.connect(mw.main_action_controller.handle_refresh_selected_overlay_display)
        cp.align_white_keys_requested.connect(mw.main_action_controller.handle_align_white_keys_to_selected)
        cp.align_black_keys_requested.connect(mw.main_action_controller.handle_align_black_keys_to_selected)
        cp.manual_fit_requested.connect(mw.main_action_controller.handle_manual_fit_request)
        
        # Overlay size adjustments
        cp.overlay_size_adjustment_requested.connect(mw.main_action_controller.handle_overlay_size_adjustment)
        
        # Conversion and testing actions
        cp.conversion_requested.connect(mw.midi_conversion_controller.start_conversion_process)
        cp.midi_touchup_requested.connect(mw.midi_touchup_controller.open_from_picker)
        
        # Video trimming action
        cp.trim_video_requested.connect(mw.video_session_ui_controller.handle_trim_video_request)
        
        # Spark ROI selection
        cp.spark_roi_selection_requested.connect(mw.calibration_effects_controller.spark.select_spark_roi)
        cp.spark_roi_visibility_toggled.connect(mw.calibration_effects_controller.spark.set_spark_roi_visible)
        
        
        # Spark calibration
        cp.spark_calibration_requested.connect(mw.calibration_effects_controller.spark.request_spark_calibration)
        
        # Auto-spark calibration
        cp.auto_spark_calibration_requested.connect(mw.calibration_effects_controller.spark.start_auto_spark_calibration)
        
        # Spark detection toggle
        cp.spark_detection_toggled.connect(mw.calibration_effects_controller.spark.set_spark_detection_enabled)
        
        # Spark-off threshold
        cp.spark_detection_sensitivity_changed.connect(mw.calibration_effects_controller.spark.set_spark_detection_sensitivity)
        
        cp.overlay_type_changed.connect(mw.calibration_effects_controller.overlay.handle_overlay_type_change)
        
        # Frame-range and trim controls manage their own state updates.
        
        self.logger.debug("Calibration & action signals connected")
    
    def _connect_midi_signals(self):
        """MIDI-related control signals"""
        cp = self.control_panel  # Shorthand
        mw = self.main_window
        
        # MIDI output settings (tempo is part of app state and not exposed here)
        
        # Octave transpose
        cp.octave_transpose_changed.connect(mw.main_action_controller.handle_octave_transpose_change)
        
        # FPS override
        cp.fps_override_changed.connect(mw.main_action_controller.handle_fps_override_change)
        
        # Custom MIDI processing range
        cp.processing_start_frame_changed.connect(mw.video_session_ui_controller.handle_processing_start_frame_change)
        cp.processing_end_frame_changed.connect(mw.video_session_ui_controller.handle_processing_end_frame_change)
        
        self.logger.debug("MIDI signals connected")
    
    def _connect_ui_signals(self):
        """UI state signals (debug, overlays, etc.)"""
        cp = self.control_panel  # Shorthand
        mw = self.main_window
        
        # Debug and visualization toggles (Visual Threshold Monitor is controlled via the main menu)
        
        # Overlay color change
        cp.overlay_color_changed.connect(mw.main_action_controller.handle_overlay_color_change)
        
        self.logger.debug("UI state signals connected")
    
