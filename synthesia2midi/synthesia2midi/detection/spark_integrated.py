"""
Spark-integrated detection method.

Combines standard color detection with spark-based split detection to solve
the "continuous key press problem" where rapid successive keypresses appear
as one continuous note instead of separate musical events.

This detection method runs standard detection first, then applies spark-based
analysis to identify and split continuous notes that should be separate events.
"""
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.core.app_state import AppState

from .base import DetectionMethod
from .spark_mapper import get_spark_zones
from .standard import StandardDetection


class SparkIntegratedDetection(DetectionMethod):
    """
    Spark-integrated detection for precise note boundary detection.
    
    Simplified Algorithm:
    1. Run standard color detection to get base keypress events
    2. Analyze spark zones for max-sparking → bar-only transitions
    3. When transition detected while key overlay remains ON: insert 1-frame OFF event
    4. Uses conservative calibration approach (most restrictive thresholds from both hands)
    
    Key Insights:
    - Spark detection never adds notes, only splits false continuous notes
    - Uses simple saturation change detection (reverted to perfect algorithm)
    - Detects spark-off when saturation increases > threshold (0.06-0.18 based on sensitivity)
    - Conservative thresholds work for both green/blue hands without color distinction
    - Single 1-frame OFF insertion provides minimal, precise note splitting
    - Sensitivity: 0.0=threshold 0.06, 0.5=threshold 0.12, 1.0=threshold 0.18
    """
    
    def __init__(self, app_state: AppState):
        super().__init__("Spark-Integrated Detection")
        self.app_state = app_state
        self.standard_detector = StandardDetection()
        
        # Spark detection state
        self.previous_spark_states: Dict[int, bool] = {}  # key_id -> was_sparking
        self.previous_detected_keys: Set[int] = set()  # Keys detected in previous frame
        
        # Spark-off detection state
        self.previous_saturations: Dict[int, float] = {}  # key_id -> previous saturation value
        self.spark_off_events: Dict[int, bool] = {}  # key_id -> spark just turned off this frame
        
    def detect_frame(self, 
                    frame_bgr: np.ndarray, 
                    overlays: List[OverlayConfig],
                    exemplar_lit_colors: Dict[str, Optional[Tuple[int, int, int]]],
                    exemplar_lit_histograms: Dict[str, Optional[np.ndarray]],
                    detection_threshold: float,
                    **kwargs) -> Set[int]:
        """
        Spark-integrated detection combining standard detection with spark analysis.
        
        Args:
            frame_bgr: Current video frame in BGR format
            overlays: List of key overlay configurations
            exemplar_lit_colors: Calibrated lit colors for each key type
            exemplar_lit_histograms: Calibrated lit histograms for each key type
            detection_threshold: Standard detection threshold
            **kwargs: Additional parameters for standard detection
            
        Returns:
            Set of key IDs that are currently pressed (with spark-based splitting)
        """
        # Log every 100th frame to reduce spam
        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
        self._frame_count += 1
        
        # Track frame index to detect non-consecutive frame navigation
        current_frame_index = kwargs.get('frame_index', None)
        if hasattr(self, '_last_processed_frame_index') and current_frame_index is not None:
            # Clear state if frames are not consecutive (allows for reverse playback)
            if abs(current_frame_index - self._last_processed_frame_index) > 1:
                self.logger.debug(f"[SPARK] Non-consecutive frame detected: {self._last_processed_frame_index} -> {current_frame_index}, clearing state")
                self.previous_spark_states.clear()
                self.previous_saturations.clear()
                self.spark_off_events.clear()
                self.previous_detected_keys.clear()
        
        # Store current frame index for next call
        if current_frame_index is not None:
            self._last_processed_frame_index = current_frame_index
        
        # Step 1: Run standard detection to get base results
        standard_pressed = self.standard_detector.detect_frame(
            frame_bgr, overlays, exemplar_lit_colors, exemplar_lit_histograms,
            detection_threshold, **kwargs
        )
        
        if self._frame_count % 100 == 0:
            self.logger.debug(f"[SPARK-INTEGRATED] Frame {self._frame_count}: Standard detected {len(standard_pressed)} keys")
        
        # Step 2: Apply spark-based analysis if configured and calibrated
        if self._is_spark_detection_ready():
            if self._frame_count % 100 == 0:
                self.logger.debug(f"[SPARK-INTEGRATED] Spark detection is READY, applying spark analysis")
            spark_modified_pressed = self._apply_spark_analysis(
                frame_bgr, overlays, standard_pressed
            )
            if self._frame_count % 100 == 0:
                self.logger.debug(f"[SPARK-INTEGRATED] After spark analysis: {len(spark_modified_pressed)} keys")
        else:
            # Spark detection not ready - use standard results
            if self._frame_count % 100 == 0:
                ready = self._is_spark_detection_ready()
                self.logger.debug(f"[SPARK-INTEGRATED] Spark detection NOT ready (ready={ready}), using standard results")
            spark_modified_pressed = standard_pressed
        
        # Step 3: Update state for next frame
        self.previous_detected_keys = spark_modified_pressed.copy()
        
        return spark_modified_pressed
    
    def _is_spark_detection_ready(self) -> bool:
        """Check if spark detection is properly configured and calibrated."""
        detection_state = self.app_state.detection
        
        # Check ROI is set
        if (detection_state.spark_roi_top <= 0 or 
            detection_state.spark_roi_bottom <= detection_state.spark_roi_top):
            return False
        
        # Check calibration is complete (requires at least one key type calibrated)
        has_lw_cal = (detection_state.spark_calibration_lw_bar_only is not None and
                      detection_state.spark_calibration_lw_brightest_sparks is not None)
        has_lb_cal = (detection_state.spark_calibration_lb_bar_only is not None and
                      detection_state.spark_calibration_lb_brightest_sparks is not None)
        has_rw_cal = (detection_state.spark_calibration_rw_bar_only is not None and
                      detection_state.spark_calibration_rw_brightest_sparks is not None)
        has_rb_cal = (detection_state.spark_calibration_rb_bar_only is not None and
                      detection_state.spark_calibration_rb_brightest_sparks is not None)
        
        has_required_calibration = has_lw_cal or has_lb_cal or has_rw_cal or has_rb_cal
        
        if self._frame_count % 100 == 0:
            self.logger.debug(f"[SPARK-READY-CHECK] LW={has_lw_cal}, LB={has_lb_cal}, RW={has_rw_cal}, RB={has_rb_cal}")
        
        if not has_required_calibration:
            return False
        
        # Check threshold is calculated
        if detection_state.spark_brightness_threshold <= 0:
            return False
        
        # Check overlays exist
        if not self.app_state.overlays:
            return False
        
        return True
    
    def _apply_spark_analysis(self, frame_bgr: np.ndarray, overlays: List[OverlayConfig], 
                             standard_pressed: Set[int]) -> Set[int]:
        """
        Apply spark-based analysis to modify standard detection results.
        
        Args:
            frame_bgr: Current video frame
            overlays: Key overlay configurations
            standard_pressed: Results from standard detection
            
        Returns:
            Modified pressed key set with spark-based splitting applied
        """
        try:
            # Get spark zones for current overlays
            spark_zones = get_spark_zones(self.app_state)
            if not spark_zones:
                self.logger.warning("No spark zones available for analysis")
                return standard_pressed
            
            # Analyze current spark states
            current_spark_states = self._analyze_spark_states(frame_bgr, spark_zones)
            
            # Apply spark-based modifications
            modified_pressed = self._apply_spark_modifications(
                standard_pressed, current_spark_states, overlays
            )
            
            
            # Update previous spark states
            self.previous_spark_states = current_spark_states.copy()
            
            # Debug: Log state update for key 42
            if 42 in current_spark_states:
                self.logger.debug(f"Updated spark state for key 42: {current_spark_states[42]}")
            
            return modified_pressed
            
        except Exception as e:
            self.logger.error(f"Error in spark analysis: {e}", exc_info=True)
            return standard_pressed
    
    def _analyze_spark_states(self, frame_bgr: np.ndarray, spark_zones: List[Any]) -> Dict[int, bool]:
        """
        Analyze spark zones for max-sparking to bar-only transitions.
        
        Simple saturation change approach (reverted to perfect algorithm):
        - Detect spark-off events using simple saturation increase detection
        - Use conservative thresholds calculated from both hands' calibration data
        - Focus solely on spark-off events that indicate missed note boundaries
        - Sensitivity scales the 0.12 threshold (0.06-0.18 range)
        
        Args:
            frame_bgr: Current video frame
            spark_zones: List of spark zone objects
            
        Returns:
            Dictionary mapping key_id to current spark state (True if max sparking)
        """
        current_spark_states = {}
        
        # Clear spark-off events from previous frame
        self.spark_off_events.clear()
        
        # Get key-type-specific calibration thresholds
        # Collect all available calibration data
        calibration_data = []
        
        # Check each key type
        for key_type in ['lw', 'lb', 'rw', 'rb']:
            bar_attr = f'spark_calibration_{key_type}_bar_only'
            sparks_attr = f'spark_calibration_{key_type}_brightest_sparks'
            
            bar_data = getattr(self.app_state.detection, bar_attr, None)
            sparks_data = getattr(self.app_state.detection, sparks_attr, None)
            
            if bar_data and sparks_data:
                calibration_data.append((key_type, bar_data, sparks_data))
        
        if not calibration_data:
            self.logger.warning("No key-type-specific calibration data available for spark analysis")
            return current_spark_states
        
        if self._frame_count % 100 == 0:
            self.logger.debug(f"[SPARK-ANALYSIS] Using calibration data from {len(calibration_data)} key types: {[d[0] for d in calibration_data]}")
        
        # Extract conservative thresholds from all available calibration data
        bar_saturations = []
        spark_saturations = []
        
        for key_type, bar_data, sparks_data in calibration_data:
            bar_sat = bar_data.get("mean_saturation", bar_data.get("mean_brightness", 0.5))
            sparks_sat = sparks_data.get("mean_saturation", sparks_data.get("mean_brightness", 0.5))
            bar_saturations.append(bar_sat)
            spark_saturations.append(sparks_sat)
        
        # Conservative bounds (most restrictive detection window)
        conservative_bar_threshold = min(bar_saturations)      # Lowest bar-only level
        conservative_sparks_threshold = max(spark_saturations)  # Highest spark level
        
        # Simple threshold calculation (similar to original perfect algorithm)
        # Use midpoint between bar-only and brightest sparks for basic spark detection
        sensitivity = self.app_state.detection.spark_detection_sensitivity
        max_sparking_threshold = (conservative_bar_threshold + conservative_sparks_threshold) / 2.0
        
        for zone in spark_zones:
            try:
                # Extract zone pixels - convert to integers for array slicing
                x = int(round(zone.x))
                y = int(round(zone.y))
                w = int(round(zone.width))
                h = int(round(zone.height))
                
                # Clamp to frame bounds
                frame_h, frame_w = frame_bgr.shape[:2]
                x = max(0, min(x, frame_w - 1))
                y = max(0, min(y, frame_h - 1))
                w = max(1, min(w, frame_w - x))
                h = max(1, min(h, frame_h - y))
                
                # Extract RGB pixels and convert to HSV
                rgb_pixels = frame_bgr[y:y+h, x:x+w][:, :, ::-1]  # BGR to RGB
                
                hsv_pixels = cv2.cvtColor(rgb_pixels, cv2.COLOR_RGB2HSV)
                
                # Calculate mean saturation (S channel)
                s_channel = hsv_pixels[:, :, 1].astype(np.float32) / 255.0
                mean_saturation = float(np.mean(s_channel))
                
                # Determine current state: max sparking or not
                is_max_sparking = mean_saturation <= max_sparking_threshold
                current_spark_states[zone.key_id] = is_max_sparking
                
                # Detect spark-off transitions using simple saturation change (PERFECT ALGORITHM)
                if zone.key_id in self.previous_saturations:
                    prev_saturation = self.previous_saturations[zone.key_id]
                    saturation_change = mean_saturation - prev_saturation
                    
                    # Simple spark-off detection: large saturation increase
                    # This indicates spark turned off (went from white/low sat to colored/high sat)
                    # Use user-adjustable sensitivity to scale the 0.12 threshold
                    spark_off_threshold = 0.12 * (0.5 + sensitivity)  # Range: 0.06 to 0.18
                    
                    if saturation_change > spark_off_threshold:
                        self.spark_off_events[zone.key_id] = True
                        self.logger.debug(f"Key {zone.key_id}: SPARK-OFF EVENT detected - "
                                        f"saturation {prev_saturation:.3f} → {mean_saturation:.3f} "
                                        f"(change: +{saturation_change:.3f}, threshold: {spark_off_threshold:.3f})")
                
                # Update saturation tracking
                self.previous_saturations[zone.key_id] = mean_saturation
                
                # Debug logging for state changes
                prev_max_sparking = self.previous_spark_states.get(zone.key_id, False)
                if is_max_sparking != prev_max_sparking:
                    if is_max_sparking:
                        self.logger.debug(f"Key {zone.key_id}: MAX SPARKING detected - "
                                        f"saturation={mean_saturation:.3f} <= {max_sparking_threshold:.3f}")
                    else:
                        self.logger.debug(f"Key {zone.key_id}: MAX SPARKING ended - "
                                        f"saturation={mean_saturation:.3f} > {max_sparking_threshold:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error analyzing spark zone {zone.key_id}: {e}")
                current_spark_states[zone.key_id] = False
        
        return current_spark_states
    
    def _apply_spark_modifications(self, standard_pressed: Set[int], 
                                  current_spark_states: Dict[int, bool],
                                  overlays: List[OverlayConfig]) -> Set[int]:
        """
        Apply spark-based modifications to standard detection results.
        
        New simplified approach:
        - Detect max-sparking → bar-only transitions while key overlay remains ON
        - These transitions indicate missed note boundaries (false continuous presses)
        - Insert exactly 1-frame OFF event to split the continuous note
        
        Args:
            standard_pressed: Keys detected by standard method
            current_spark_states: Current max sparking states for each key
            overlays: Key overlay configurations
            
        Returns:
            Modified pressed key set with 1-frame OFF events for note splitting
        """
        modified_pressed = standard_pressed.copy()
        
        # Process spark-off events (max sparking → bar-only transitions)
        for key_id in self.spark_off_events:
            if key_id in standard_pressed:
                # Max sparking ended but key overlay still shows pressed
                # This indicates a missed note boundary - insert 1-frame OFF event
                modified_pressed.discard(key_id)
                self.logger.debug(f"Key {key_id}: Inserting 1-frame OFF event due to max-sparking → bar-only transition")
        
        return modified_pressed
    
    
    def reset_state(self):
        """Reset detection state."""
        self.standard_detector.reset_state()
        self.previous_spark_states.clear()
        self.previous_frame_bgr = None
        self.previous_detected_keys.clear()
        self.previous_saturations.clear()
        self.spark_off_events.clear()
        self.logger.debug("Spark-integrated detection state reset")
    
    def get_method_info(self) -> Dict[str, Any]:
        """Get information about spark-integrated detection method."""
        standard_info = self.standard_detector.get_method_info()
        
        return {
            "name": self.name,
            "description": "Standard detection enhanced with max-sparking to bar-only transition detection",
            "parameters": {
                **standard_info["parameters"],
                "spark_brightness_threshold": "Conservative saturation threshold calculated from calibration (0.0-1.0)",
                "spark_roi_top": "Top Y coordinate of spark detection region",
                "spark_roi_bottom": "Bottom Y coordinate of spark detection region",
            },
            "algorithm": "Standard detection + max-sparking → bar-only transition detection for 1-frame note splitting",
            "calibration_required": [
                "lh_bar_only", "lh_brightest_sparks", "rh_bar_only", "rh_brightest_sparks"
            ]
        }
    
    def validate_parameters(self, **kwargs) -> List[str]:
        """Validate spark-integrated detection parameters."""
        errors = self.standard_detector.validate_parameters(**kwargs)
        
        if 'spark_brightness_threshold' in kwargs:
            threshold = kwargs['spark_brightness_threshold']
            if not 0.0 <= threshold <= 1.0:
                errors.append(f"Spark saturation threshold {threshold} must be between 0.0 and 1.0")
        
        if 'spark_roi_top' in kwargs and 'spark_roi_bottom' in kwargs:
            top = kwargs['spark_roi_top']
            bottom = kwargs['spark_roi_bottom']
            if top >= bottom:
                errors.append(f"Spark ROI top ({top}) must be less than bottom ({bottom})")
        
        return errors