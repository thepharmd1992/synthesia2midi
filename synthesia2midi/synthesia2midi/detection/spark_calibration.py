"""
Spark calibration system for three-point brightness detection.

This module implements the three-point calibration system needed for reliable
spark detection:

1. Background calibration: Captures baseline brightness when no bars or sparks present
2. Bar-only calibration: Captures brightness of colored bars without spark activity  
3. Dimmest-sparks calibration: Captures brightness when sparks are just barely visible

The calibration data is used to calculate detection thresholds that can distinguish
between bar-only states and spark-present states across different lighting conditions.
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from synthesia2midi.core.app_state import AppState
from synthesia2midi.detection.spark_mapper import get_spark_zones


class CalibrationStep(Enum):
    """Enumeration of calibration steps."""
    BACKGROUND = "background"
    BAR_ONLY = "bar_only" 
    DIMMEST_SPARKS = "dimmest_sparks"
    BRIGHTEST_SPARKS = "brightest_sparks"


@dataclass
class SparkCalibrationSample:
    """
    Simplified calibration sample focusing on saturation statistics.
    
    Zone-size independent measurements for robust calibration.
    """
    zone_id: int
    note_name: str
    
    # Core saturation statistics (zone-size independent)
    mean_saturation: float  # Average saturation (0.0-1.0)
    max_saturation: float   # Peak saturation (0.0-1.0)
    min_saturation: float   # Minimum saturation (0.0-1.0)
    saturation_std: float   # Standard deviation of saturation
    
    # Also keep brightness for reference/debugging
    mean_brightness: float  # Average brightness (0.0-1.0)
    
    # Quality metrics
    pixel_count: int
    confidence_score: float  # Quality of this sample (0.0-1.0)


@dataclass  
class SparkCalibrationData:
    """
    Simplified calibration data for one calibration step.
    
    Focuses on core saturation statistics needed for threshold calculation.
    """
    step_type: CalibrationStep
    frame_index: int
    timestamp: float
    
    # Core saturation statistics from calibrated zone
    mean_saturation: float    # Primary value for threshold calculation
    max_saturation: float     # Peak saturation observed
    min_saturation: float     # Minimum saturation observed
    saturation_std: float     # Saturation variation
    
    # Also keep brightness for reference
    mean_brightness: float    # Average brightness for debugging
    
    # Calibration metadata
    calibrated_zone_id: int   # Which zone was used for calibration
    calibrated_note_name: str # Note name of calibrated zone
    pixel_count: int          # Number of pixels sampled
    confidence_score: float   # Quality metric (0.0-1.0)


class SparkCalibrationManager:
    """
    Manages the three-point spark calibration process.
    
    Handles capture, analysis, and storage of calibration data for
    background, bar-only, and dimmest-sparks conditions.
    """
    
    def __init__(self, app_state: AppState):
        self.app_state = app_state
        self.logger = logging.getLogger(f"{__name__}.SparkCalibrationManager")
        
        # Current calibration state
        self._current_step: Optional[CalibrationStep] = None
        self._calibration_in_progress = False
        
    def start_calibration_step(self, step: CalibrationStep) -> bool:
        """
        Start a calibration step.
        
        Args:
            step: Which calibration step to perform
            
        Returns:
            True if step started successfully
        """
        if self._calibration_in_progress:
            self.logger.warning(f"Cannot start {step.value} - calibration already in progress")
            return False
        
        # Validate prerequisites
        if not self._validate_calibration_prerequisites():
            return False
        
        self._current_step = step
        self._calibration_in_progress = True
        
        self.logger.info(f"Started {step.value} calibration step")
        return True
    
    def capture_calibration_frame(self, frame_rgb: np.ndarray, frame_index: int, target_field: str = None) -> bool:
        """
        Capture calibration data from current frame.
        
        Args:
            frame_rgb: Current frame in RGB format
            frame_index: Frame index for reference
            
        Returns:
            True if capture successful
        """
        if not self._calibration_in_progress or not self._current_step:
            self.logger.error("No calibration step in progress")
            return False
        
        try:
            # Get spark zones for sampling
            spark_zones = get_spark_zones(self.app_state)
            if not spark_zones:
                self.logger.error("No spark zones available for calibration")
                return False
            
            # Extract samples from each zone
            zone_samples = {}
            for zone in spark_zones:
                sample = self._extract_zone_sample(frame_rgb, zone)
                if sample:
                    zone_samples[zone.key_id] = sample
            
            if not zone_samples:
                self.logger.error("Failed to extract any zone samples")
                return False
            
            # Create calibration data
            calib_data = self._create_calibration_data(
                self._current_step, frame_index, zone_samples
            )
            
            # Store calibration data
            self._store_calibration_data(calib_data, target_field)
            
            # Complete calibration step
            self._calibration_in_progress = False
            self._current_step = None
            
            self.logger.info(f"Captured {calib_data.step_type.value} calibration data "
                           f"from {len(zone_samples)} zones")
            return True
            
        except Exception as e:
            self.logger.error(f"Error capturing calibration frame: {e}", exc_info=True)
            self._calibration_in_progress = False
            self._current_step = None
            return False
    
    def analyze_calibration_quality(self) -> Dict[str, Any]:
        """
        Analyze quality of current calibration data.
        
        Returns:
            Dictionary with quality metrics and recommendations
        """
        analysis = {
            "background_quality": 0.0,
            "bar_only_quality": 0.0, 
            "dimmest_sparks_quality": 0.0,
            "overall_quality": 0.0,
            "brightness_separation": 0.0,
            "recommendations": []
        }
        
        # Analyze each calibration step
        bg_data = self.app_state.detection.spark_calibration_background
        
        # Prioritize universal calibration data
        universal_bar_data = self.app_state.detection.spark_calibration_bar_only
        universal_spark_data = self.app_state.detection.spark_calibration_dimmest_sparks
        
        # Legacy data for fallback
        lh_bar_data = self.app_state.detection.spark_calibration_lh_bar_only
        lh_spark_data = self.app_state.detection.spark_calibration_lh_dimmest_sparks
        rh_bar_data = self.app_state.detection.spark_calibration_rh_bar_only
        rh_spark_data = self.app_state.detection.spark_calibration_rh_dimmest_sparks
        
        if bg_data:
            analysis["background_quality"] = bg_data.get("confidence_score", 0.0)
        
        # Use universal data if available, otherwise use best legacy data
        bar_qualities = []
        spark_qualities = []
        
        if universal_bar_data:
            bar_qualities.append(universal_bar_data.get("confidence_score", 0.0))
        else:
            if lh_bar_data:
                bar_qualities.append(lh_bar_data.get("confidence_score", 0.0))
            if rh_bar_data:
                bar_qualities.append(rh_bar_data.get("confidence_score", 0.0))
        
        if universal_spark_data:
            spark_qualities.append(universal_spark_data.get("confidence_score", 0.0))
        else:
            if lh_spark_data:
                spark_qualities.append(lh_spark_data.get("confidence_score", 0.0))
            if rh_spark_data:
                spark_qualities.append(rh_spark_data.get("confidence_score", 0.0))
            
        analysis["bar_only_quality"] = max(bar_qualities) if bar_qualities else 0.0
        analysis["dimmest_sparks_quality"] = max(spark_qualities) if spark_qualities else 0.0
        
        # Calculate overall quality
        qualities = [analysis["background_quality"], 
                    analysis["bar_only_quality"],
                    analysis["dimmest_sparks_quality"]]
        analysis["overall_quality"] = sum(q for q in qualities if q > 0) / max(1, sum(1 for q in qualities if q > 0))
        
        # Analyze saturation separation - prefer universal data
        bar_data = universal_bar_data or lh_bar_data or rh_bar_data
        spark_data = universal_spark_data or lh_spark_data or rh_spark_data
        
        if bar_data and spark_data:
            bar_saturation = bar_data.get("mean_saturation", bar_data.get("mean_brightness", 0.5))
            spark_saturation = spark_data.get("mean_saturation", spark_data.get("mean_brightness", 0.5))
            analysis["brightness_separation"] = abs(bar_saturation - spark_saturation)  # Keep field name for compatibility
        
        # Generate recommendations
        if analysis["overall_quality"] < 0.5:
            analysis["recommendations"].append("Consider recalibrating in better lighting conditions")
        
        if analysis["brightness_separation"] < 0.1:
            analysis["recommendations"].append("Insufficient saturation difference - ensure sparks are visible as white flashes")
        
        return analysis
    
    def calculate_detection_threshold(self) -> Optional[float]:
        """
        Calculate conservative detection threshold using brightest sparks and bar-only data.
        
        Uses conservative approach: min(LH_bar, RH_bar) and max(LH_sparks, RH_sparks)
        to create the most restrictive detection window that works for both hands.
        
        Returns:
            Conservative saturation threshold value, or None if insufficient calibration
        """
        # Get LH and RH calibration data
        lh_bar_data = self.app_state.detection.spark_calibration_lh_bar_only
        lh_sparks_data = self.app_state.detection.spark_calibration_lh_brightest_sparks
        rh_bar_data = self.app_state.detection.spark_calibration_rh_bar_only
        rh_sparks_data = self.app_state.detection.spark_calibration_rh_brightest_sparks
        
        # Check if we have both hands' data
        if not (lh_bar_data and lh_sparks_data and rh_bar_data and rh_sparks_data):
            self.logger.warning("Insufficient calibration data - need both LH and RH bar-only and brightest-sparks")
            return None
        
        # Extract saturation values
        lh_bar_sat = lh_bar_data.get("mean_saturation", lh_bar_data.get("mean_brightness", 0.5))
        lh_sparks_sat = lh_sparks_data.get("mean_saturation", lh_sparks_data.get("mean_brightness", 0.5))
        rh_bar_sat = rh_bar_data.get("mean_saturation", rh_bar_data.get("mean_brightness", 0.5))
        rh_sparks_sat = rh_sparks_data.get("mean_saturation", rh_sparks_data.get("mean_brightness", 0.5))
        
        # Conservative threshold selection (most restrictive detection window)
        conservative_bar_threshold = min(lh_bar_sat, rh_bar_sat)      # Lowest bar-only level
        conservative_sparks_threshold = max(lh_sparks_sat, rh_sparks_sat)  # Highest spark level
        
        # Validation: sparks should have lower saturation than bars
        if conservative_sparks_threshold >= conservative_bar_threshold:
            self.logger.warning(f"Invalid calibration: sparks ({conservative_sparks_threshold:.3f}) "
                              f"should have lower saturation than bars ({conservative_bar_threshold:.3f})")
            return None
        
        # Calculate threshold midway between conservative bounds
        threshold = (conservative_bar_threshold + conservative_sparks_threshold) / 2.0
        
        # Clamp to valid range
        threshold = max(0.0, min(1.0, threshold))
        
        self.logger.info(f"Calculated conservative detection threshold: {threshold:.3f}")
        self.logger.info(f"  LH: bar={lh_bar_sat:.3f}, sparks={lh_sparks_sat:.3f}")
        self.logger.info(f"  RH: bar={rh_bar_sat:.3f}, sparks={rh_sparks_sat:.3f}")
        self.logger.info(f"  Conservative bounds: sparks={conservative_sparks_threshold:.3f}, bar={conservative_bar_threshold:.3f}")
        
        # Store the conservative thresholds for detection logic to use
        self.app_state.detection.spark_brightness_threshold = threshold
        self._conservative_bar_threshold = conservative_bar_threshold
        self._conservative_sparks_threshold = conservative_sparks_threshold
        
        # Note: Spark detection sensitivity is now manually controlled by user via GUI
        
        return threshold
    
    
    def _validate_calibration_prerequisites(self) -> bool:
        """Validate that calibration can be performed."""
        # Check ROI is set
        if (self.app_state.detection.spark_roi_top <= 0 or 
            self.app_state.detection.spark_roi_bottom <= self.app_state.detection.spark_roi_top):
            self.logger.error("Spark ROI must be set before calibration")
            return False
        
        # Check overlays exist
        if not self.app_state.overlays:
            self.logger.error("No key overlays available for calibration")
            return False
        
        # Check zones can be generated
        spark_zones = get_spark_zones(self.app_state)
        if not spark_zones:
            self.logger.error("No spark zones available for calibration")
            return False
        
        return True
    
    def _extract_zone_sample(self, frame_rgb: np.ndarray, zone) -> Optional[SparkCalibrationSample]:
        """Extract saturation calibration sample from a spark zone."""
        try:
            # Extract zone pixels - convert to integers for array slicing
            x = int(round(zone.x))
            y = int(round(zone.y))
            w = int(round(zone.width))
            h = int(round(zone.height))
            
            # Clamp to frame bounds
            frame_h, frame_w = frame_rgb.shape[:2]
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = max(1, min(w, frame_w - x))
            h = max(1, min(h, frame_h - y))
            
            rgb_pixels = frame_rgb[y:y+h, x:x+w].copy()
            
            # Convert to HSV
            hsv_pixels = cv2.cvtColor(rgb_pixels, cv2.COLOR_RGB2HSV)
            s_channel = hsv_pixels[:, :, 1].astype(np.float32) / 255.0  # Saturation
            v_channel = hsv_pixels[:, :, 2].astype(np.float32) / 255.0  # Brightness
            
            # Calculate core saturation statistics
            mean_saturation = float(np.mean(s_channel))
            max_saturation = float(np.max(s_channel))
            min_saturation = float(np.min(s_channel))
            saturation_std = float(np.std(s_channel))
            
            # Also calculate brightness for reference
            mean_brightness = float(np.mean(v_channel))
            
            # Quality metrics
            pixel_count = rgb_pixels.shape[0] * rgb_pixels.shape[1]
            saturation_range = max_saturation - min_saturation
            confidence_score = min(1.0, saturation_range * 2.0)  # Higher range = better sample
            
            return SparkCalibrationSample(
                zone_id=zone.key_id,
                note_name=zone.note_name,
                mean_saturation=mean_saturation,
                max_saturation=max_saturation,
                min_saturation=min_saturation,
                saturation_std=saturation_std,
                mean_brightness=mean_brightness,
                pixel_count=pixel_count,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting zone sample: {e}")
            return None
    
    def _create_calibration_data(self, step_type: CalibrationStep, frame_index: int, 
                               zone_samples: Dict[int, SparkCalibrationSample]) -> SparkCalibrationData:
        """Create simplified calibration data object from single zone sample."""
        import time
        
        # Since we're using single-zone calibration, get the one sample
        if not zone_samples:
            raise ValueError("No zone samples provided for calibration")
        
        sample = next(iter(zone_samples.values()))  # Get the single sample
        
        return SparkCalibrationData(
            step_type=step_type,
            frame_index=frame_index,
            timestamp=time.time(),
            mean_saturation=sample.mean_saturation,
            max_saturation=sample.max_saturation,
            min_saturation=sample.min_saturation,
            saturation_std=sample.saturation_std,
            mean_brightness=sample.mean_brightness,
            calibrated_zone_id=sample.zone_id,
            calibrated_note_name=sample.note_name,
            pixel_count=sample.pixel_count,
            confidence_score=sample.confidence_score
        )
    
    def _store_calibration_data(self, calib_data: SparkCalibrationData, target_field: str = None):
        """Store calibration data in app state."""
        # Convert to serializable format
        data_dict = {
            "step_type": calib_data.step_type.value,
            "frame_index": calib_data.frame_index,
            "timestamp": calib_data.timestamp,
            "mean_saturation": calib_data.mean_saturation,
            "max_saturation": calib_data.max_saturation,
            "min_saturation": calib_data.min_saturation,
            "saturation_std": calib_data.saturation_std,
            "mean_brightness": calib_data.mean_brightness,
            "calibrated_zone_id": calib_data.calibrated_zone_id,
            "calibrated_note_name": calib_data.calibrated_note_name,
            "pixel_count": calib_data.pixel_count,
            "confidence_score": calib_data.confidence_score,
            # Legacy fields for backward compatibility with UI
            "overall_mean_brightness": calib_data.mean_brightness,
            "sample_count": 1
        }
        
        # Store in specified target field or use default mapping
        if target_field:
            setattr(self.app_state.detection, target_field, data_dict)
        else:
            # Default legacy behavior
            if calib_data.step_type == CalibrationStep.BACKGROUND:
                self.app_state.detection.spark_calibration_background = data_dict
            elif calib_data.step_type == CalibrationStep.BAR_ONLY:
                self.app_state.detection.spark_calibration_bar_only = data_dict
            elif calib_data.step_type == CalibrationStep.DIMMEST_SPARKS:
                self.app_state.detection.spark_calibration_dimmest_sparks = data_dict
            elif calib_data.step_type == CalibrationStep.BRIGHTEST_SPARKS:
                # For brightest sparks, we need target_field to know LH vs RH
                self.logger.warning("Brightest sparks calibration requires target_field to be specified")
        
        # Update derived values
        threshold = self.calculate_detection_threshold()
        if threshold is not None:
            self.app_state.detection.spark_brightness_threshold = threshold
        
        # Update confidence score
        analysis = self.analyze_calibration_quality()
        self.app_state.detection.spark_detection_confidence = analysis["overall_quality"]
        
        # Mark as unsaved
        self.app_state.unsaved_changes = True