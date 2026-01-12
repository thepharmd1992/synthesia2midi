"""
Auto-calibration workflow for key-type-specific spark detection.

Implements the automated calibration process:
1. User clicks auto-calibrate button for key type (LW/LB/RW/RB)
2. User clicks on overlay at frame where it first turns ON
3. System detects hand via hue analysis
4. Automated capture: Frame +0 (bar-only), Frame +2 (dimmest), Frames +3-22 (brightest)
5. Save calibration data to appropriate key type
"""
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np

from synthesia2midi.core.app_state import AppState
from synthesia2midi.video_loader import VideoSession
from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.detection.hand_detection import (
    detect_hand_for_overlay_frame, 
    get_key_type_for_overlay
)
from synthesia2midi.detection.spark_calibration import SparkCalibrationManager


@dataclass
class AutoCalibrationRequest:
    """Request for auto-calibration of a specific key type."""
    requested_key_type: str  # LW, LB, RW, RB
    overlay_config: OverlayConfig
    start_frame: int
    roi_bgr: np.ndarray


class AutoCalibrationWorkflow:
    """
    Handles the complete auto-calibration workflow for key-type-specific spark detection.
    """
    
    def __init__(self, app_state: AppState, video_session: VideoSession, parent_widget=None):
        self.app_state = app_state
        self.video_session = video_session
        self.parent_widget = parent_widget
        self.logger = logging.getLogger(f"{__name__}.AutoCalibrationWorkflow")
        
        # Calibration manager for spark analysis
        self.spark_calibration_manager = SparkCalibrationManager(self.app_state)
        
        # Auto-calibration state
        self.current_request: Optional[AutoCalibrationRequest] = None
        self.calibration_in_progress = False
    
    def start_auto_calibration(self, requested_key_type: str) -> bool:
        """
        Start auto-calibration process for the specified key type.
        
        Args:
            requested_key_type: "LW", "LB", "RW", or "RB"
            
        Returns:
            True if calibration started successfully, False otherwise
        """
        if not self._validate_prerequisites():
            return False
        
        if self.calibration_in_progress:
            self.logger.warning("Auto-calibration already in progress")
            return False
        
        valid_key_types = ["LW", "LB", "RW", "RB"]
        if requested_key_type not in valid_key_types:
            self.logger.error(f"Invalid key type: {requested_key_type}")
            return False
        
        # Set calibration mode
        self.app_state.calibration.calibration_mode = f"auto_calibrate_{requested_key_type.lower()}"
        self.app_state.calibration.current_calibration_key_type = requested_key_type
        self.calibration_in_progress = True
        
        self.logger.info(f"Started auto-calibration for key type: {requested_key_type}")
        self.logger.info("Please click on an overlay at the frame where it first turns ON")
        
        return True
    
    def handle_overlay_click(self, overlay_config: OverlayConfig, roi_bgr: np.ndarray) -> bool:
        """
        Handle user clicking on an overlay during auto-calibration.
        
        Args:
            overlay_config: The overlay that was clicked
            roi_bgr: BGR image data from the overlay region
            
        Returns:
            True if calibration completed successfully, False otherwise
        """
        try:
            self.logger.debug(f"[AUTO-CAL] Starting overlay click handler for overlay {overlay_config.key_id}")
            
            if not self.calibration_in_progress:
                error_msg = "Overlay click received but no auto-calibration in progress"
                self.logger.error(f"[AUTO-CAL ERROR] {error_msg}")
                self.logger.warning(error_msg)
                return False
            
            requested_key_type = self.app_state.calibration.current_calibration_key_type
            current_frame = self.app_state.video.current_frame_index
            
            self.logger.debug(f"[AUTO-CAL] Processing overlay click for {requested_key_type} at frame {current_frame}")
            self.logger.info(f"Processing overlay click for {requested_key_type} at frame {current_frame}")
            
            # Check ROI data
            if roi_bgr is None or roi_bgr.size == 0:
                error_msg = "ROI data is None or empty"
                self.logger.error(f"[AUTO-CAL ERROR] {error_msg}")
                self.logger.error(error_msg)
                self._cleanup_calibration()
                return False
            
            self.logger.debug(f"[AUTO-CAL] ROI shape: {roi_bgr.shape}")
            
            # Detect hand for this overlay at this frame
            detected_hand, confidence = detect_hand_for_overlay_frame(roi_bgr)
            
            if detected_hand is None:
                error_msg = "Failed to detect hand from overlay region"
                self.logger.error(f"[AUTO-CAL ERROR] {error_msg}")
                self.logger.error(error_msg)
                self._cleanup_calibration()
                return False
            
            self.logger.debug(f"[AUTO-CAL] Hand detection: {detected_hand} (confidence: {confidence:.2f})")
            
            if confidence < 0.5:
                warning_msg = f"Low confidence hand detection: {confidence:.2f}"
                self.logger.warning(f"[AUTO-CAL WARNING] {warning_msg}")
                self.logger.warning(warning_msg)
            
            # Get actual key type based on detection
            actual_key_type = get_key_type_for_overlay(overlay_config, detected_hand)
            
            self.logger.debug(f"[AUTO-CAL] Detected hand: {detected_hand}, actual key type: {actual_key_type}")
            self.logger.info(f"Detected hand: {detected_hand}, actual key type: {actual_key_type}")
            
            # Verify the detected key type matches the requested key type
            if actual_key_type != requested_key_type:
                warning_msg = f"Key type mismatch: requested {requested_key_type}, detected {actual_key_type}"
                self.logger.warning(f"[AUTO-CAL WARNING] {warning_msg}")
                self.logger.warning(warning_msg)
                # Continue with detected key type for now
            
            # Create calibration request
            self.current_request = AutoCalibrationRequest(
                requested_key_type=actual_key_type,
                overlay_config=overlay_config,
                start_frame=current_frame,
                roi_bgr=roi_bgr.copy()
            )
            
            self.logger.info(f"[AUTO-CAL] Created calibration request, executing auto-calibration...")
            
            # Execute the auto-calibration sequence
            return self._execute_auto_calibration()
            
        except Exception as e:
            error_msg = f"Exception in handle_overlay_click: {str(e)}"
            self.logger.debug(f"[AUTO-CAL ERROR] {error_msg}")
            self.logger.error(error_msg)
            import traceback
            traceback.print_exc()
            self._cleanup_calibration()
            return False
    
    def _execute_auto_calibration(self) -> bool:
        """
        Execute the automated calibration sequence.
        
        Returns:
            True if calibration completed successfully, False otherwise
        """
        try:
            if not self.current_request:
                error_msg = "No calibration request available"
                self.logger.error(f"[AUTO-CAL ERROR] {error_msg}")
                self.logger.error(error_msg)
                return False
            
            request = self.current_request
            key_type = request.requested_key_type
            overlay = request.overlay_config
            start_frame = request.start_frame
            
            self.logger.info(f"[AUTO-CAL] Executing auto-calibration for {key_type} starting at frame {start_frame}")
            self.logger.info(f"Executing auto-calibration for {key_type} starting at frame {start_frame}")
            
            # Step 1: Capture bar-only baseline (Frame +0)
            self.logger.debug(f"[AUTO-CAL] Step 1: Capturing bar-only baseline at frame {start_frame}")
            bar_only_data = self._capture_bar_only(overlay, start_frame)
            if not bar_only_data:
                error_msg = "Failed to capture bar-only data"
                self.logger.error(f"[AUTO-CAL ERROR] {error_msg}")
                self.logger.error(error_msg)
                return False
            self.logger.debug(f"[AUTO-CAL] Step 1 completed successfully")
            
            # Step 2: Capture dimmest sparks (Frame +2)
            self.logger.debug(f"[AUTO-CAL] Step 2: Capturing dimmest sparks at frame {start_frame + 2}")
            dimmest_data = self._capture_dimmest_sparks(overlay, start_frame + 2)
            if not dimmest_data:
                error_msg = "Failed to capture dimmest sparks data"
                self.logger.error(f"[AUTO-CAL ERROR] {error_msg}")
                self.logger.error(error_msg)
                return False
            self.logger.debug(f"[AUTO-CAL] Step 2 completed successfully")
            
            # Step 3: Find brightest sparks (Frames +0 to +10)
            self.logger.debug(f"[AUTO-CAL] Step 3: Finding brightest sparks in frames {start_frame} to {start_frame + 10}")
            brightest_data = self._find_brightest_sparks(overlay, start_frame, start_frame + 10)
            if not brightest_data:
                error_msg = "Failed to find brightest sparks data"
                self.logger.error(f"[AUTO-CAL ERROR] {error_msg}")
                self.logger.error(error_msg)
                return False
            self.logger.debug(f"[AUTO-CAL] Step 3 completed successfully")
            
            # Step 4: Save calibration data
            self.logger.debug(f"[AUTO-CAL] Step 4: Saving calibration data for {key_type}")
            success = self._save_calibration_data(key_type, bar_only_data, dimmest_data, brightest_data)
            if not success:
                error_msg = "Failed to save calibration data"
                self.logger.error(f"[AUTO-CAL ERROR] {error_msg}")
                self.logger.error(error_msg)
                return False
            self.logger.debug(f"[AUTO-CAL] Step 4 completed successfully")
            
            self.logger.info(f"[AUTO-CAL SUCCESS] Auto-calibration completed successfully for {key_type}")
            self.logger.info(f"Auto-calibration completed successfully for {key_type}")
            
            # Trigger UI update to apply new calibration immediately
            self.logger.debug(f"[AUTO-CAL] About to trigger UI update after calibration")
            self._trigger_ui_update_after_calibration()
            self.logger.debug(f"[AUTO-CAL] UI update trigger completed")
            
            return True
            
        except Exception as e:
            error_msg = f"Auto-calibration failed with exception: {str(e)}"
            self.logger.debug(f"[AUTO-CAL ERROR] {error_msg}")
            self.logger.error(error_msg)
            import traceback
            traceback.print_exc()
            return False
        finally:
            self._cleanup_calibration()
    
    def _capture_bar_only(self, overlay: OverlayConfig, frame_index: int) -> Optional[Dict[str, Any]]:
        """Capture bar-only calibration data at specified frame."""
        roi_bgr = self._get_roi_at_frame(overlay, frame_index)
        if roi_bgr is None:
            return None
        
        return self._create_simple_calibration_data(roi_bgr, "bar_only", frame_index)
    
    def _capture_dimmest_sparks(self, overlay: OverlayConfig, frame_index: int) -> Optional[Dict[str, Any]]:
        """Capture dimmest sparks calibration data at specified frame."""
        roi_bgr = self._get_roi_at_frame(overlay, frame_index)
        if roi_bgr is None:
            return None
        
        return self._create_simple_calibration_data(roi_bgr, "dimmest_sparks", frame_index)
    
    def _find_brightest_sparks(self, overlay: OverlayConfig, start_frame: int, end_frame: int) -> Optional[Dict[str, Any]]:
        """
        Find the frame with brightest sparks in the specified range.
        
        Args:
            overlay: Overlay to analyze
            start_frame: Start of search range (frame +3)
            end_frame: End of search range (frame +22)
            
        Returns:
            Calibration data from the frame with brightest sparks
        """
        brightest_frame = None
        brightest_value = -1
        
        self.logger.info(f"Searching for brightest sparks in frames {start_frame}-{end_frame}")
        
        for frame_idx in range(start_frame, end_frame + 1):
            roi_bgr = self._get_roi_at_frame(overlay, frame_idx)
            if roi_bgr is None:
                continue
            
            # Calculate brightness metric (average value in HSV)
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            avg_value = np.mean(hsv[:, :, 2])  # V channel in HSV
            
            if avg_value > brightest_value:
                brightest_value = avg_value
                brightest_frame = frame_idx
        
        if brightest_frame is None:
            self.logger.error("No valid frames found for brightest sparks detection")
            return None
        
        self.logger.info(f"Found brightest sparks at frame {brightest_frame} (value: {brightest_value:.2f})")
        
        # Get calibration data from the brightest frame
        roi_bgr = self._get_roi_at_frame(overlay, brightest_frame)
        if roi_bgr is None:
            return None
        return self._create_simple_calibration_data(roi_bgr, "brightest_sparks", brightest_frame)
    
    def _get_roi_at_frame(self, overlay: OverlayConfig, frame_index: int) -> Optional[np.ndarray]:
        """
        Get BGR ROI data for overlay at specified frame.
        
        Args:
            overlay: Overlay configuration
            frame_index: Frame to extract ROI from
            
        Returns:
            BGR image data or None if extraction failed
        """
        try:
            self.logger.debug(f"[AUTO-CAL] Getting ROI for overlay {overlay.key_id} at frame {frame_index}")
            
            if not self.video_session:
                error_msg = "Video session not available"
                self.logger.error(f"[AUTO-CAL ERROR] {error_msg}")
                self.logger.error(error_msg)
                return None
            
            # Use the proper interface method to get frame
            self.logger.debug(f"[AUTO-CAL] Getting frame {frame_index}")
            ret, frame = self.video_session.get_frame(frame_index)
            
            if not ret or frame is None:
                error_msg = f"Failed to get frame {frame_index}"
                self.logger.error(f"[AUTO-CAL ERROR] {error_msg}")
                self.logger.error(error_msg)
                return None
            
            self.logger.debug(f"[AUTO-CAL] Successfully read frame {frame_index}, shape: {frame.shape}")
            
            # Extract ROI from frame
            x, y, w, h = int(overlay.x), int(overlay.y), int(overlay.width), int(overlay.height)
            self.logger.debug(f"[AUTO-CAL] Extracting ROI at x={x}, y={y}, w={w}, h={h}")
            
            # Validate ROI coordinates
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                error_msg = f"ROI coordinates out of bounds: x={x}, y={y}, w={w}, h={h}, frame_shape={frame.shape}"
                self.logger.error(f"[AUTO-CAL ERROR] {error_msg}")
                self.logger.error(error_msg)
                return None
            
            roi_bgr = frame[y:y+h, x:x+w]
            self.logger.debug(f"[AUTO-CAL] Extracted ROI shape: {roi_bgr.shape}")
            
            return roi_bgr
            
        except Exception as e:
            error_msg = f"Failed to get ROI at frame {frame_index}: {str(e)}"
            self.logger.debug(f"[AUTO-CAL ERROR] {error_msg}")
            self.logger.error(error_msg)
            import traceback
            traceback.print_exc()
            return None
    
    def _create_simple_calibration_data(self, roi_bgr: np.ndarray, step_type: str, frame_index: int) -> Dict[str, Any]:
        """
        Create simple calibration data from BGR ROI.
        
        Args:
            roi_bgr: BGR image region
            step_type: Type of calibration step
            frame_index: Frame number
            
        Returns:
            Dictionary with calibration data
        """
        try:
            self.logger.debug(f"[AUTO-CAL] Creating calibration data for {step_type} at frame {frame_index}")
            
            # Convert BGR to HSV
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            
            # Extract saturation and value channels (normalized to 0-1)
            saturation = hsv[:, :, 1].astype(np.float32) / 255.0
            value = hsv[:, :, 2].astype(np.float32) / 255.0
            
            # Calculate statistics
            mean_saturation = float(np.mean(saturation))
            max_saturation = float(np.max(saturation))
            min_saturation = float(np.min(saturation))
            saturation_std = float(np.std(saturation))
            mean_brightness = float(np.mean(value))
            
            self.logger.debug(f"[AUTO-CAL] Calibration stats - mean_sat: {mean_saturation:.3f}, max_sat: {max_saturation:.3f}, brightness: {mean_brightness:.3f}")
            
            # Create calibration data dictionary
            calibration_data = {
                "step_type": step_type,
                "frame_index": frame_index,
                "timestamp": __import__('time').time(),
                "mean_saturation": mean_saturation,
                "max_saturation": max_saturation,
                "min_saturation": min_saturation,
                "saturation_std": saturation_std,
                "mean_brightness": mean_brightness,
                "pixel_count": int(saturation.size)
            }
            
            return calibration_data
            
        except Exception as e:
            error_msg = f"Failed to create calibration data: {str(e)}"
            self.logger.debug(f"[AUTO-CAL ERROR] {error_msg}")
            self.logger.error(error_msg)
            return {}
    
    def _save_calibration_data(self, key_type: str, bar_only: Dict, dimmest: Dict, brightest: Dict) -> bool:
        """
        Save calibration data to the appropriate key-type-specific fields.
        
        Args:
            key_type: "LW", "LB", "RW", or "RB"
            bar_only: Bar-only calibration data
            dimmest: Dimmest sparks calibration data
            brightest: Brightest sparks calibration data
            
        Returns:
            True if save was successful
        """
        try:
            # Map key type to app_state fields
            field_prefix = f"spark_calibration_{key_type.lower()}"
            
            self.logger.debug(f"[AUTO-CAL-SAVE] Saving calibration data for {key_type}")
            self.logger.debug(f"[AUTO-CAL-SAVE] Field prefix: {field_prefix}")
            self.logger.debug(f"[AUTO-CAL-SAVE] Bar-only data: {bar_only is not None}")
            self.logger.debug(f"[AUTO-CAL-SAVE] Dimmest data: {dimmest is not None}")
            self.logger.debug(f"[AUTO-CAL-SAVE] Brightest data: {brightest is not None}")
            
            setattr(self.app_state.detection, f"{field_prefix}_bar_only", bar_only)
            setattr(self.app_state.detection, f"{field_prefix}_dimmest_sparks", dimmest)
            setattr(self.app_state.detection, f"{field_prefix}_brightest_sparks", brightest)
            
            # Verify the data was saved
            saved_bar_only = getattr(self.app_state.detection, f"{field_prefix}_bar_only")
            saved_brightest = getattr(self.app_state.detection, f"{field_prefix}_brightest_sparks")
            self.logger.debug(f"[AUTO-CAL-SAVE] Verification - bar_only saved: {saved_bar_only is not None}")
            self.logger.debug(f"[AUTO-CAL-SAVE] Verification - brightest saved: {saved_brightest is not None}")
            
            # Mark as having unsaved changes
            self.app_state.unsaved_changes = True
            
            self.logger.info(f"Saved calibration data for {key_type}")
            self.logger.info(f"[AUTO-CAL-SAVE] Successfully saved calibration data for {key_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save calibration data for {key_type}: {e}")
            return False
    
    def _validate_prerequisites(self) -> bool:
        """Validate that auto-calibration can proceed."""
        if not self.video_session:
            self.logger.error("No video session available")
            return False
        
        if not self.app_state.overlays:
            self.logger.error("No overlays defined")
            return False
        
        return True
    
    def _trigger_ui_update_after_calibration(self):
        """Trigger UI update to apply new calibration data immediately."""
        try:
            self.logger.debug(f"[AUTO-CAL-UI] _trigger_ui_update_after_calibration called")
            self.logger.debug(f"[AUTO-CAL-UI] parent_widget: {self.parent_widget}")
            self.logger.debug(f"[AUTO-CAL-UI] parent_widget type: {type(self.parent_widget)}")
            
            # Access parent widget (should be main app) to trigger frame update
            if self.parent_widget and hasattr(self.parent_widget, 'video_controls'):
                video_controls = self.parent_widget.video_controls
                self.logger.debug(f"[AUTO-CAL-UI] Found video_controls: {video_controls}")
                if hasattr(video_controls, 'update_current_frame_display'):
                    self.logger.debug(f"[AUTO-CAL-UI] Calling video_controls.update_current_frame_display()")
                    self.logger.info("Triggering UI update to apply new calibration data")
                    video_controls.update_current_frame_display()
                    return
            
            # Also try to update the control panel display directly
            if self.parent_widget and hasattr(self.parent_widget, 'control_panel'):
                control_panel = self.parent_widget.control_panel
                self.logger.debug(f"[AUTO-CAL-UI] Found control_panel: {control_panel}")
                if hasattr(control_panel, 'update_controls_from_state'):
                    self.logger.debug(f"[AUTO-CAL-UI] Calling control_panel.update_controls_from_state()")
                    control_panel.update_controls_from_state()
                    self.logger.debug(f"[AUTO-CAL-UI] Called control_panel.update_controls_from_state() successfully")
                    return
                elif hasattr(control_panel, 'update_auto_calibration_display'):
                    self.logger.debug(f"[AUTO-CAL-UI] Calling control_panel.update_auto_calibration_display() directly")
                    control_panel.update_auto_calibration_display()
                    self.logger.debug(f"[AUTO-CAL-UI] Called control_panel.update_auto_calibration_display() successfully")
                    return
            
            self.logger.warning(f"[AUTO-CAL-UI] Could not find method to trigger UI update after calibration")
            self.logger.warning("Could not find method to trigger UI update after calibration")
            
        except Exception as e:
            self.logger.error(f"[AUTO-CAL-UI] Exception in _trigger_ui_update_after_calibration: {e}")
            self.logger.warning(f"Failed to trigger UI update after calibration: {e}")
    
    def _cleanup_calibration(self):
        """Clean up calibration state."""
        self.calibration_in_progress = False
        self.current_request = None
        self.app_state.calibration.calibration_mode = None
        self.app_state.calibration.current_calibration_key_type = None
        
        self.logger.debug("Auto-calibration state cleaned up")
    
    def cancel_calibration(self):
        """Cancel any in-progress auto-calibration."""
        if self.calibration_in_progress:
            self.logger.info("Auto-calibration cancelled by user")
            self._cleanup_calibration()
    
    def is_calibration_in_progress(self) -> bool:
        """Check if auto-calibration is currently in progress."""
        return self.calibration_in_progress
    
    def get_calibration_summary(self, key_type: str) -> Dict[str, Any]:
        """
        Get summary of calibration data for a specific key type.
        
        Args:
            key_type: "LW", "LB", "RW", or "RB"
            
        Returns:
            Dictionary with calibration status and data summary
        """
        field_prefix = f"spark_calibration_{key_type.lower()}"
        
        bar_only = getattr(self.app_state.detection, f"{field_prefix}_bar_only", None)
        dimmest = getattr(self.app_state.detection, f"{field_prefix}_dimmest_sparks", None)
        brightest = getattr(self.app_state.detection, f"{field_prefix}_brightest_sparks", None)
        
        return {
            "key_type": key_type,
            "bar_only_set": bar_only is not None,
            "dimmest_sparks_set": dimmest is not None,
            "brightest_sparks_set": brightest is not None,
            "fully_calibrated": all([bar_only, dimmest, brightest])
        }
