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
        self._connect_video_signals()
        self._connect_detection_signals()
        self._connect_calibration_signals()
        self._connect_midi_signals()
        self._connect_ui_signals()
        
        self.logger.info("All control signals connected")
    
    def _connect_video_signals(self):
        """Video-related control signals"""
        cp = self.control_panel  # Shorthand
        mw = self.main_window
        
        # Frame navigation interval
        cp.nav_interval_changed.connect(mw._update_nav_interval)
        
        # Frame slider (owned by main window, not control panel)
        mw.frame_slider.valueChanged.connect(mw.video_controls.on_frame_slider_changed)
        
        # YouTube video download
        cp.youtube_video_downloaded.connect(mw._handle_youtube_video_downloaded)
        
        # Video to frame series conversion
        cp.video_to_frames_requested.connect(mw._handle_video_to_frames_request)
        
        # Video trim frame controls
        cp.start_frame_changed.connect(mw._handle_start_frame_change)
        cp.end_frame_changed.connect(mw._handle_end_frame_change)
        
        self.logger.debug("Video signals connected")
    
    def _connect_detection_signals(self):
        """Detection-related control signals"""
        cp = self.control_panel  # Shorthand
        mw = self.main_window
        
        # Basic detection parameters
        cp.detection_threshold_changed.connect(mw._handle_detection_threshold_change)
        cp.rise_delta_threshold_changed.connect(mw._handle_rise_delta_threshold_change)
        cp.fall_delta_threshold_changed.connect(mw._handle_fall_delta_threshold_change)
        
        # Detection method toggles
        cp.histogram_detection_toggled.connect(mw._on_toggle_hist_detection)
        cp.delta_detection_toggled.connect(mw._on_toggle_delta_detection)
        
        # Black key filter toggle
        cp.winner_takes_black_changed.connect(mw._on_toggle_winner_takes_black)
        
        # Hand assignment toggle
        cp.hand_assignment_toggled.connect(mw._handle_hand_assignment_toggle)
        
        
        # Manual refresh button
        
        self.logger.debug("Detection signals connected")
    
    def _connect_calibration_signals(self):
        """Calibration-related control signals"""
        cp = self.control_panel  # Shorthand
        mw = self.main_window
        
        # Basic calibration actions
        cp.calibrate_unlit_requested.connect(mw._handle_calibrate_unlit_all_keys)
        cp.calibrate_lit_exemplar_requested.connect(mw._handle_calibrate_lit_exemplar_key_start)
        cp.calibration_wizard_requested.connect(mw._invoke_calibration_wizard)
        
        # Overlay management actions
        cp.refresh_overlay_display_requested.connect(mw._handle_refresh_selected_overlay_display)
        cp.align_white_keys_requested.connect(mw._handle_align_white_keys_to_selected)
        cp.align_black_keys_requested.connect(mw._handle_align_black_keys_to_selected)
        
        # Overlay size adjustments
        cp.overlay_size_adjustment_requested.connect(mw._handle_overlay_size_adjustment)
        
        # Conversion and testing actions
        cp.conversion_requested.connect(mw._start_conversion_process)
        
        # Video trimming action
        cp.trim_video_requested.connect(mw._handle_trim_video_request)
        
        # Spark ROI selection
        cp.spark_roi_selection_requested.connect(mw._handle_spark_roi_selection_request)
        cp.spark_roi_visibility_toggled.connect(mw._handle_spark_roi_visibility_toggle)
        
        
        # Spark calibration
        cp.spark_calibration_requested.connect(mw._handle_spark_calibration_request)
        
        # Auto-spark calibration
        cp.auto_spark_calibration_requested.connect(mw._handle_auto_spark_calibration_request)
        
        # Spark detection toggle
        cp.spark_detection_toggled.connect(mw._handle_spark_detection_toggle)
        
        # Spark-off threshold
        cp.spark_detection_sensitivity_changed.connect(mw._handle_spark_detection_sensitivity_change)
        
        cp.overlay_type_changed.connect(mw._handle_overlay_type_change)
        
        # Frame-range and trim controls manage their own state updates.
        
        self.logger.debug("Calibration & action signals connected")
    
    def _connect_midi_signals(self):
        """MIDI-related control signals"""
        cp = self.control_panel  # Shorthand
        mw = self.main_window
        
        # MIDI output settings (tempo is part of app state and not exposed here)
        
        # Octave transpose
        cp.octave_transpose_changed.connect(mw._handle_octave_transpose_change)
        
        # FPS override
        cp.fps_override_changed.connect(mw._handle_fps_override_change)
        
        # Custom MIDI processing range
        cp.processing_start_frame_changed.connect(mw._handle_processing_start_frame_change)
        cp.processing_end_frame_changed.connect(mw._handle_processing_end_frame_change)
        
        self.logger.debug("MIDI signals connected")
    
    def _connect_ui_signals(self):
        """UI state signals (debug, overlays, etc.)"""
        cp = self.control_panel  # Shorthand
        mw = self.main_window
        
        # Debug and visualization toggles (Visual Threshold Monitor is controlled via the main menu)
        
        # Overlay color change
        cp.overlay_color_changed.connect(mw._handle_overlay_color_change)
        
        self.logger.debug("UI state signals connected")
    
