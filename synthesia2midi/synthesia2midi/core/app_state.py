"""
Organized application state with validation and clear ownership.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from synthesia2midi.app_config import DEFAULT_MIDI_TEMPO, FRAME_NAV_INTERVALS, OverlayConfig


@dataclass
class DetectionConfig:
    """All detection-related settings grouped together."""
    
    # Core detection parameters
    detection_threshold: float = 0.8
    hist_ratio_threshold: float = 0.8
    rise_delta_threshold: float = 0.15
    fall_delta_threshold: float = 0.15
    
    # Detection method toggles
    use_histogram_detection: bool = False
    use_delta_detection: bool = False
    spark_detection_enabled: bool = False  # Toggle for spark detection system
    hand_assignment_enabled: bool = False  # Toggle for hand detection/MIDI channel assignment
    
    
    # Black key filtering
    winner_takes_black_enabled: bool = False
    similarity_ratio: float = 0.60
    
    
    
    # Exemplar colors for lit state detection
    # Supports unlimited exemplars - keys are "LW", "LB", "RW", "RB" for the first 4,
    # then "COLOR_3", "COLOR_4", etc. for additional colors
    exemplar_lit_colors: Dict[str, Optional[Tuple[int, int, int]]] = field(
        default_factory=lambda: {"LW": None, "LB": None, "RW": None, "RB": None}
    )
    
    # Exemplar histograms for lit state detection
    exemplar_lit_histograms: Dict[str, Optional[Any]] = field(  # Using Any for numpy arrays
        default_factory=lambda: {"LW": None, "LB": None, "RW": None, "RB": None}
    )
    
    # Hand detection hue calibration
    left_hand_hue_mean: float = 0.0
    right_hand_hue_mean: float = 0.0
    hand_detection_calibrated: bool = False
    
    # Spark ROI detection area
    spark_roi_top: int = 0
    spark_roi_bottom: int = 0
    spark_roi_visible: bool = False  # Toggle for showing/hiding ROI overlay
    
    
    # Spark calibration data (three-point universal system)
    spark_calibration_background: Optional[Dict[str, Any]] = None  # Background (no bars, no sparks)
    spark_calibration_bar_only: Optional[Dict[str, Any]] = None      # Universal bars without sparks
    spark_calibration_dimmest_sparks: Optional[Dict[str, Any]] = None # Universal dimmest sparks
    
    # Compatibility fields for older per-hand calibration data
    spark_calibration_lh_bar_only: Optional[Dict[str, Any]] = None    # Left hand bars without sparks
    spark_calibration_lh_dimmest_sparks: Optional[Dict[str, Any]] = None  # Left hand dimmest sparks
    spark_calibration_lh_brightest_sparks: Optional[Dict[str, Any]] = None  # Left hand brightest sparks
    spark_calibration_rh_bar_only: Optional[Dict[str, Any]] = None    # Right hand bars without sparks
    spark_calibration_rh_dimmest_sparks: Optional[Dict[str, Any]] = None  # Right hand dimmest sparks
    spark_calibration_rh_brightest_sparks: Optional[Dict[str, Any]] = None  # Right hand brightest sparks
    
    # Key-type-specific calibration fields (LW = Left White, LB = Left Black, RW = Right White, RB = Right Black)
    spark_calibration_lw_bar_only: Optional[Dict[str, Any]] = None    # Left White bars without sparks
    spark_calibration_lw_dimmest_sparks: Optional[Dict[str, Any]] = None  # Left White dimmest sparks
    spark_calibration_lw_brightest_sparks: Optional[Dict[str, Any]] = None  # Left White brightest sparks
    spark_calibration_lb_bar_only: Optional[Dict[str, Any]] = None    # Left Black bars without sparks
    spark_calibration_lb_dimmest_sparks: Optional[Dict[str, Any]] = None  # Left Black dimmest sparks
    spark_calibration_lb_brightest_sparks: Optional[Dict[str, Any]] = None  # Left Black brightest sparks
    spark_calibration_rw_bar_only: Optional[Dict[str, Any]] = None    # Right White bars without sparks
    spark_calibration_rw_dimmest_sparks: Optional[Dict[str, Any]] = None  # Right White dimmest sparks
    spark_calibration_rw_brightest_sparks: Optional[Dict[str, Any]] = None  # Right White brightest sparks
    spark_calibration_rb_bar_only: Optional[Dict[str, Any]] = None    # Right Black bars without sparks
    spark_calibration_rb_dimmest_sparks: Optional[Dict[str, Any]] = None  # Right Black dimmest sparks
    spark_calibration_rb_brightest_sparks: Optional[Dict[str, Any]] = None  # Right Black brightest sparks
    
    # Derived calibration thresholds
    spark_brightness_threshold: float = 0.7  # Threshold between bar-only and spark states
    spark_detection_sensitivity: float = 0.5  # Sensitivity for spark-off detection (0.0=conservative, 1.0=aggressive)
    spark_detection_confidence: float = 0.0  # Quality metric for calibration data
    
    
    def validate(self) -> List[str]:
        """Validate detection configuration and return error messages."""
        errors = []
        
        # Threshold validations
        if not 0.1 <= self.detection_threshold <= 0.99:
            errors.append(f"Detection threshold {self.detection_threshold} must be between 0.1 and 0.99")
        
        if not 0.1 <= self.hist_ratio_threshold <= 1.0:
            errors.append(f"Histogram ratio threshold {self.hist_ratio_threshold} must be between 0.1 and 1.0")
        
        if not 0.01 <= self.rise_delta_threshold <= 0.5:
            errors.append(f"Rise delta threshold {self.rise_delta_threshold} must be between 0.01 and 0.5")
        
        if not 0.01 <= self.fall_delta_threshold <= 0.5:
            errors.append(f"Fall delta threshold {self.fall_delta_threshold} must be between 0.01 and 0.5")
        
        if not 0.1 <= self.similarity_ratio <= 1.0:
            errors.append(f"Similarity ratio {self.similarity_ratio} must be between 0.1 and 1.0")
        
        # Spark ROI validations
        if self.spark_roi_top < 0:
            errors.append("Spark ROI top must be non-negative")
        
        if self.spark_roi_bottom < 0:
            errors.append("Spark ROI bottom must be non-negative")
        
        if self.spark_roi_top >= self.spark_roi_bottom and self.spark_roi_bottom > 0:
            errors.append("Spark ROI top must be less than bottom")
        
        # Spark calibration validations
        if not 0.0 <= self.spark_brightness_threshold <= 1.0:
            errors.append(f"Spark saturation threshold {self.spark_brightness_threshold} must be between 0.0 and 1.0")
        
        if not 0.0 <= self.spark_detection_sensitivity <= 1.0:
            errors.append(f"Spark detection sensitivity {self.spark_detection_sensitivity} must be between 0.0 and 1.0")
        
        if not 0.0 <= self.spark_detection_confidence <= 1.0:
            errors.append(f"Spark detection confidence {self.spark_detection_confidence} must be between 0.0 and 1.0")
        
        
        
        return errors


@dataclass
class VideoConfig:
    """All video-related settings and state."""
    
    # Video file information
    filepath: str = ""
    filepath_ini_used: Optional[str] = None  # Path to .ini file loaded for this video
    original_video_path: Optional[str] = None  # Original video path when using frame sequences
    
    # Frame navigation
    current_frame_index: int = 0
    total_frames: int = 0
    fps: float = 30.0
    fps_override: Optional[float] = None  # Manual FPS override when auto-detection is wrong
    
    # Compatibility fields used by some UI/configs; processing_* is the primary range for MIDI generation.
    start_frame: int = 0
    end_frame: int = 0
    
    # Processing range for MIDI generation (non-destructive)
    processing_start_frame: int = 0
    processing_end_frame: int = 0
    
    # Video trimming range (destructive, permanent modification)
    trim_start_frame: int = 0
    trim_end_frame: int = 0
    video_is_trimmed: bool = False  # Track if video has been permanently trimmed
    
    # Navigation settings
    current_nav_interval: int = FRAME_NAV_INTERVALS[0]
    
    def validate(self) -> List[str]:
        """Validate video configuration."""
        errors = []
        
        if self.current_frame_index < 0:
            errors.append("Current frame index cannot be negative")
        
        if self.total_frames > 0 and self.current_frame_index >= self.total_frames:
            errors.append(f"Current frame {self.current_frame_index} exceeds total frames {self.total_frames}")
        
        if self.start_frame > self.end_frame and self.end_frame > 0:
            errors.append("Start frame cannot be after end frame")
        
        if self.processing_start_frame > self.processing_end_frame and self.processing_end_frame > 0:
            errors.append("Processing start frame cannot be after processing end frame")
            
        if self.trim_start_frame > self.trim_end_frame and self.trim_end_frame > 0:
            errors.append("Trim start frame cannot be after trim end frame")
        
        if self.fps <= 0:
            errors.append("FPS must be positive")
        
        return errors


@dataclass
class MidiConfig:
    """All MIDI-related settings."""
    
    # Basic MIDI settings
    tempo: int = DEFAULT_MIDI_TEMPO
    
    # Key mapping
    total_keys: int = 88
    leftmost_note_name: str = "A"
    leftmost_note_octave: int = 1
    
    # Octave transpose adjustment (-8 to +8)
    octave_transpose: int = 0
    
    # Color to channel mapping
    color_to_channel_map: Dict[Tuple[int, int, int], int] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Validate MIDI configuration."""
        errors = []
        
        if not 60 <= self.tempo <= 200:
            errors.append(f"Tempo {self.tempo} should be between 60 and 200 BPM")
        
        if not 1 <= self.total_keys <= 128:
            errors.append(f"Total keys {self.total_keys} must be between 1 and 128")
        
        if self.leftmost_note_octave < 0 or self.leftmost_note_octave > 9:
            errors.append(f"Leftmost note octave {self.leftmost_note_octave} must be between 0 and 9")
        
        # Validate octave transpose range
        if not -8 <= self.octave_transpose <= 8:
            errors.append(f"Octave transpose {self.octave_transpose} must be between -8 and +8")
        
        # Validate note name
        valid_notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
        if self.leftmost_note_name not in valid_notes:
            errors.append(f"Leftmost note name '{self.leftmost_note_name}' must be one of {valid_notes}")
        
        return errors


@dataclass
class UIConfig:
    """All UI-related settings and state."""
    
    # Display settings
    show_overlays: bool = False
    live_detection_feedback: bool = False
    visual_threshold_monitor_enabled: bool = False
    overlay_color: str = "red"  # Default color for overlays
    
    # Performance settings
    auto_convert_to_frames: bool = True  # Automatically convert videos to frame sequences
    
    # Selection state
    selected_overlay_id: Optional[int] = None
    
    def validate(self) -> List[str]:
        """Validate UI configuration."""
        errors = []
        
        return errors


@dataclass
class CalibrationConfig:
    """All calibration-related settings and state."""
    
    # Calibration mode and state
    calibration_mode: Optional[str] = None  # e.g., "lit_exemplar", "spark_bar_only", "spark_dimmest_sparks"
    current_calibration_key_type: Optional[str] = None  # e.g., "LW", "LB", "RW", "RB"
    
    # Calibration frame range
    calib_start_frame: int = 0
    calib_end_frame: int = 0
    
    
    def validate(self) -> List[str]:
        """Validate calibration configuration."""
        errors = []
        
        
        # Validate calibration key types
        # Allow the 4 base types plus any COLOR_N_W or COLOR_N_B format
        if self.current_calibration_key_type is not None:
            valid_base_types = ["LW", "LB", "RW", "RB"]
            is_additional_color = (self.current_calibration_key_type.startswith("COLOR_") and 
                                 (self.current_calibration_key_type.endswith("_W") or 
                                  self.current_calibration_key_type.endswith("_B")))
            
            if (self.current_calibration_key_type not in valid_base_types and 
                not is_additional_color):
                errors.append(f"Calibration key type '{self.current_calibration_key_type}' must be one of {valid_base_types} or COLOR_N_W/COLOR_N_B format")
        
        return errors


@dataclass
class AppState:
    """
    Organized application state with clear boundaries and validation.
    """
    
    # Organized state groups
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    midi: MidiConfig = field(default_factory=MidiConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    
    # Complex state that doesn't fit neatly into groups
    overlays: List[OverlayConfig] = field(default_factory=list)
    
    # Meta state
    unsaved_changes: bool = False
    
    def validate(self) -> List[str]:
        """
        Comprehensive validation of all state groups.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        all_errors = []
        
        # Validate each state group
        all_errors.extend([f"Detection: {err}" for err in self.detection.validate()])
        all_errors.extend([f"Video: {err}" for err in self.video.validate()])
        all_errors.extend([f"MIDI: {err}" for err in self.midi.validate()])
        all_errors.extend([f"UI: {err}" for err in self.ui.validate()])
        all_errors.extend([f"Calibration: {err}" for err in self.calibration.validate()])
        
        # Cross-group validations
        all_errors.extend(self._validate_cross_group())
        
        return all_errors
    
    def _validate_cross_group(self) -> List[str]:
        """Validate relationships between different state groups."""
        errors = []
        
        # Video/UI consistency
        
        
        # Overlay validation
        if self.ui.selected_overlay_id is not None:
            overlay_ids = {overlay.key_id for overlay in self.overlays}
            if self.ui.selected_overlay_id not in overlay_ids:
                errors.append(f"Selected overlay ID {self.ui.selected_overlay_id} does not exist")
        
        return errors
    
    def mark_unsaved(self):
        """Mark state as having unsaved changes."""
        self.unsaved_changes = True
        logging.debug("AppState marked as unsaved")
    
    def mark_saved(self):
        """Mark state as saved."""
        self.unsaved_changes = False
        logging.debug("AppState marked as saved")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of current state for debugging."""
        return {
            "detection": {
                "threshold": self.detection.detection_threshold,
                "histogram_enabled": self.detection.use_histogram_detection,
                "delta_enabled": self.detection.use_delta_detection
            },
            "video": {
                "filepath": self.video.filepath,
                "current_frame": self.video.current_frame_index,
                "total_frames": self.video.total_frames,
                "fps": self.video.fps
            },
            "midi": {
                "tempo": self.midi.tempo,
                "total_keys": self.midi.total_keys
            },
            "ui": {
                "selected_overlay": self.ui.selected_overlay_id,
                "live_feedback": self.ui.live_detection_feedback
            },
            "overlays_count": len(self.overlays),
            "unsaved_changes": self.unsaved_changes
        }
