"""
Detection parameter management for Synthesia2MIDI application.

This module owns detection-related parameter updates and toggle behavior.
"""

import logging
from typing import Callable, Tuple, List, Optional
import numpy as np

from ..core.app_state import AppState
from ..app_config import OverlayConfig
from ..detection.factory import DetectionFactory
from ..gui.ui_update_interface import UIUpdateInterface


class DetectionManager:
    """Manages detection parameter updates and toggles."""
    
    def __init__(self, app_state: AppState, update_frame_callback: Callable[[], None], ui_updater: Optional[UIUpdateInterface] = None):
        """
        Initialize DetectionManager.
        
        Args:
            app_state: The application state object
            update_frame_callback: Callback to update current frame display 
            ui_updater: UI update interface for clean dependency injection
        """
        self.app_state = app_state
        self.update_frame_callback = update_frame_callback
        self.ui_updater = ui_updater
        
        # Cache the detector instance to avoid recreating on every frame
        self._cached_detector = None
        self._detector_config_state = None
        self._is_navigation_mode = True  # Default to navigation mode for GUI interaction
    
    def handle_detection_threshold_change(self, threshold: float):
        """Handle detection threshold change with validation."""
        try:
            new_threshold = float(threshold)
            if 0.0 <= new_threshold <= 1.0:
                self.app_state.detection.detection_threshold = new_threshold
                self.app_state.unsaved_changes = True
                logging.info(f"Detection threshold changed to: {new_threshold}")
                
                # Trigger live visual update if video is loaded
                if self.ui_updater and self.ui_updater.has_video_loaded():
                    self.update_frame_callback()
            else:
                if self.ui_updater:
                    self.ui_updater.show_message("Invalid Threshold", 
                                            "Detection threshold must be between 0.0 and 1.0.")
                # Revert UI value
                if self.ui_updater:
                    self.ui_updater.update_detection_threshold(
                        self.app_state.detection.detection_threshold)
        except ValueError:
            if self.ui_updater:
                self.ui_updater.show_message("Invalid Input", 
                                        "Detection threshold must be a valid number.")
                self.ui_updater.update_detection_threshold(
                    self.app_state.detection.detection_threshold)
    
    def toggle_histogram_detection(self):
        """Toggle histogram detection mode."""
        # The app_state.detection.use_histogram_detection is already updated by ControlPanel's handler
        logging.info(f"DetectionManager: Histogram detection is now {self.app_state.detection.use_histogram_detection}")
        self.app_state.unsaved_changes = True
        
        # Trigger live visual update if video is loaded
        if self.ui_updater and self.ui_updater.has_video_loaded():
            self.update_frame_callback()
    
    def toggle_delta_detection(self):
        """Toggle delta detection mode."""
        # The app_state.detection.use_delta_detection is already updated by ControlPanel's handler
        logging.info(f"DetectionManager: Delta detection is now {self.app_state.detection.use_delta_detection}")
        self.app_state.unsaved_changes = True
        
        # Trigger live visual update if video is loaded
        if self.ui_updater and self.ui_updater.has_video_loaded():
            self.update_frame_callback()
    

    def prepare_frame_for_detection(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[OverlayConfig]]:
        """
        Prepares a frame for detection.
        Returns (processed_frame, overlays).
        """
        # Detection uses full-frame coordinates; return the frame unchanged.
        return frame_bgr, self.app_state.overlays
    
    def set_navigation_mode(self, navigation_mode: bool):
        """Set whether detection is in navigation mode (lightweight) or conversion mode (full)."""
        if self._is_navigation_mode != navigation_mode:
            self._is_navigation_mode = navigation_mode
            # Clear cached detector when mode changes
            self._cached_detector = None
            self._detector_config_state = None
            logging.info(f"Detection mode changed to: {'navigation' if navigation_mode else 'conversion'}")
    
    def _get_detector_config_state(self):
        """Get a hashable representation of detector configuration state."""
        # Include all configuration that affects detector creation
        detection_state = self.app_state.detection
        config_tuple = (
            self._is_navigation_mode,  # Include navigation mode in state
            detection_state.spark_detection_enabled,
            detection_state.spark_roi_top,
            detection_state.spark_roi_bottom,
            detection_state.spark_brightness_threshold,
            # Include calibration state
            detection_state.spark_calibration_lw_bar_only is not None,
            detection_state.spark_calibration_lw_brightest_sparks is not None,
            detection_state.spark_calibration_lb_bar_only is not None,
            detection_state.spark_calibration_lb_brightest_sparks is not None,
            detection_state.spark_calibration_rw_bar_only is not None,
            detection_state.spark_calibration_rw_brightest_sparks is not None,
            detection_state.spark_calibration_rb_bar_only is not None,
            detection_state.spark_calibration_rb_brightest_sparks is not None,
            # Number of overlays can affect detection
            len(self.app_state.overlays)
        )
        return config_tuple

    def create_detection_wrapper(self):
        """
        Create a detection function compatible with KeyboardCanvas expectations.
        
        This wrapper uses the new detection factory while maintaining the same
        interface that the canvas expects.
        """
        def detection_wrapper(frame, overlays, exemplars, exemplar_histograms, thresh, hist_thresh,
                            rise_delta_thresh, fall_delta_thresh,
                            use_histogram_detection, use_delta_detection,
                            similarity_ratio, apply_black_filter,
                            ):
            try:
                # Check if we need to create or recreate the detector
                current_config_state = self._get_detector_config_state()
                
                if self._cached_detector is None or self._detector_config_state != current_config_state:
                    # Configuration changed or first time - create new detector
                    self._cached_detector = DetectionFactory.create_from_app_state(
                        self.app_state, overlays, navigation_mode=self._is_navigation_mode
                    )
                    self._detector_config_state = current_config_state
                    logging.info(f"Created new detector due to configuration change (navigation_mode={self._is_navigation_mode})")
                
                # Use cached detector
                detector = self._cached_detector
                
                # Use the unified detection interface
                # Build kwargs dict to conditionally include hand detection parameters
                kwargs = {
                    'hist_ratio_threshold': hist_thresh,
                    'rise_delta_threshold': rise_delta_thresh,
                    'fall_delta_threshold': fall_delta_thresh,
                    'use_histogram_detection': use_histogram_detection,
                    'use_delta_detection': use_delta_detection,
                    'similarity_ratio': similarity_ratio,
                    'apply_black_filter': apply_black_filter,
                    'navigation_mode': self._is_navigation_mode,
                    'frame_index': self.app_state.video.current_frame_index,
                }
                
                # Only include hand detection parameters during conversion mode (not navigation)
                # This improves performance during frame navigation
                if not self._is_navigation_mode:
                    kwargs.update({
                        'hand_assignment_enabled': self.app_state.detection.hand_assignment_enabled,
                        'hand_detection_calibrated': self.app_state.detection.hand_detection_calibrated,
                        'left_hand_hue_mean': self.app_state.detection.left_hand_hue_mean,
                        'right_hand_hue_mean': self.app_state.detection.right_hand_hue_mean,
                    })
                
                return detector.detect_frame(
                    frame, overlays, exemplars, exemplar_histograms, thresh,
                    **kwargs
                )
            except Exception as e:
                logging.error(f"Detection wrapper failed: {e}")
                return set()
        
        return detection_wrapper
