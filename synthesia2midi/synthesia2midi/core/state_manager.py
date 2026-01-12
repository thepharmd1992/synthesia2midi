"""
State manager for coordinating state changes and ensuring consistency.

Provides safe state mutations with validation, change tracking, and
notification to prevent the system from getting into invalid states.
"""
import logging
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional

from .app_state import AppState, CalibrationConfig, DetectionConfig, MidiConfig, UIConfig, VideoConfig


class StateChangeEvent:
    """Represents a state change event for notifications."""
    
    def __init__(self, field_path: str, old_value: Any, new_value: Any, timestamp: float = None):
        self.field_path = field_path
        self.old_value = old_value
        self.new_value = new_value
        self.timestamp = timestamp or __import__('time').time()
    
    def __str__(self):
        return f"StateChange({self.field_path}: {self.old_value} -> {self.new_value})"


class StateValidationError(Exception):
    """Exception raised when state validation fails."""
    pass


class StateManager:
    """
    Coordinates state changes and ensures consistency.
    
    Prevents invalid state by validating changes and provides
    change notifications for UI updates and persistence.
    """
    
    def __init__(self, app_state: AppState):
        self.app_state = app_state
        self.logger = logging.getLogger(f"{__name__}.StateManager")
        
        # Change listeners
        self._change_listeners: List[Callable[[StateChangeEvent], None]] = []
        
        # Validation settings
        self._validate_on_change = True
        self._auto_mark_unsaved = True
        
        self.logger.debug("StateManager initialized")
    
    def add_change_listener(self, callback: Callable[[StateChangeEvent], None]):
        """Add a callback that gets called on any state change."""
        self._change_listeners.append(callback)
        self.logger.debug(f"Added change listener: {callback.__name__}")
    
    def remove_change_listener(self, callback: Callable[[StateChangeEvent], None]):
        """Remove a change listener."""
        if callback in self._change_listeners:
            self._change_listeners.remove(callback)
            self.logger.debug(f"Removed change listener: {callback.__name__}")
    
    def set_validation_enabled(self, enabled: bool):
        """Enable or disable validation on state changes."""
        self._validate_on_change = enabled
        self.logger.debug(f"Validation on change: {enabled}")
    
    def set_auto_mark_unsaved(self, enabled: bool):
        """Enable or disable automatic unsaved marking on changes."""
        self._auto_mark_unsaved = enabled
        self.logger.debug(f"Auto mark unsaved: {enabled}")
    
    # Detection state updates
    def update_detection_threshold(self, value: float):
        """Safely update detection threshold with validation."""
        old_value = self.app_state.detection.detection_threshold
        
        if self._validate_on_change:
            # Create temporary config for validation
            temp_config = replace(self.app_state.detection, detection_threshold=value)
            errors = temp_config.validate()
            if errors:
                raise StateValidationError(f"Invalid detection threshold: {errors}")
        
        self.app_state.detection.detection_threshold = value
        self._notify_change("detection.detection_threshold", old_value, value)
        
        self.logger.info(f"Detection threshold: {old_value} -> {value}")
    
    def update_histogram_detection(self, enabled: bool):
        """Toggle histogram detection."""
        old_value = self.app_state.detection.use_histogram_detection
        self.app_state.detection.use_histogram_detection = enabled
        self._notify_change("detection.use_histogram_detection", old_value, enabled)
        
        self.logger.info(f"Histogram detection: {old_value} -> {enabled}")
    
    def update_delta_detection(self, enabled: bool):
        """Toggle delta detection."""
        old_value = self.app_state.detection.use_delta_detection
        self.app_state.detection.use_delta_detection = enabled
        self._notify_change("detection.use_delta_detection", old_value, enabled)
        
        self.logger.info(f"Delta detection: {old_value} -> {enabled}")
    
    
    # Video state updates
    def update_current_frame(self, frame_index: int):
        """Update current video frame with validation."""
        old_value = self.app_state.video.current_frame_index
        
        if self._validate_on_change:
            if frame_index < 0:
                raise StateValidationError("Frame index cannot be negative")
            if (self.app_state.video.total_frames > 0 and 
                frame_index >= self.app_state.video.total_frames):
                raise StateValidationError(f"Frame {frame_index} exceeds total frames {self.app_state.video.total_frames}")
        
        self.app_state.video.current_frame_index = frame_index
        self._notify_change("video.current_frame_index", old_value, frame_index)
        
        # Don't log every frame change (too verbose)
        if abs(frame_index - old_value) > 10:
            self.logger.debug(f"Current frame: {old_value} -> {frame_index}")
    
    def update_video_file(self, filepath: str, total_frames: int = 0, fps: float = 30.0):
        """Update video file information."""
        old_filepath = self.app_state.video.filepath
        old_total = self.app_state.video.total_frames
        old_fps = self.app_state.video.fps
        
        self.app_state.video.filepath = filepath
        self.app_state.video.total_frames = total_frames
        self.app_state.video.fps = fps
        self.app_state.video.current_frame_index = 0  # Reset to start
        
        self._notify_change("video.filepath", old_filepath, filepath)
        self._notify_change("video.total_frames", old_total, total_frames)
        self._notify_change("video.fps", old_fps, fps)
        
        self.logger.info(f"Video file updated: {filepath} ({total_frames} frames, {fps} fps)")
    
    # UI state updates
    def update_selected_overlay(self, overlay_id: Optional[int]):
        """Update selected overlay with validation."""
        old_value = self.app_state.ui.selected_overlay_id
        
        if self._validate_on_change and overlay_id is not None:
            overlay_ids = {overlay.key_id for overlay in self.app_state.overlays}
            if overlay_id not in overlay_ids:
                raise StateValidationError(f"Overlay ID {overlay_id} does not exist")
        
        self.app_state.ui.selected_overlay_id = overlay_id
        self._notify_change("ui.selected_overlay_id", old_value, overlay_id)
        
        self.logger.debug(f"Selected overlay: {old_value} -> {overlay_id}")
    
    
    # MIDI state updates
    def update_tempo(self, tempo: int):
        """Update MIDI tempo with validation."""
        old_value = self.app_state.midi.tempo
        
        if self._validate_on_change:
            if not 60 <= tempo <= 200:
                raise StateValidationError(f"Tempo {tempo} must be between 60 and 200 BPM")
        
        self.app_state.midi.tempo = tempo
        self._notify_change("midi.tempo", old_value, tempo)
        
        self.logger.info(f"MIDI tempo: {old_value} -> {tempo}")
    
    # Batch operations
    def update_detection_config(self, **kwargs):
        """Update multiple detection settings in one operation."""
        old_config = replace(self.app_state.detection)
        
        # Apply changes
        for key, value in kwargs.items():
            if hasattr(self.app_state.detection, key):
                setattr(self.app_state.detection, key, value)
            else:
                raise ValueError(f"Unknown detection config field: {key}")
        
        # Validate entire config
        if self._validate_on_change:
            errors = self.app_state.detection.validate()
            if errors:
                # Rollback changes
                self.app_state.detection = old_config
                raise StateValidationError(f"Invalid detection config: {errors}")
        
        # Notify of all changes
        for key, value in kwargs.items():
            old_value = getattr(old_config, key)
            self._notify_change(f"detection.{key}", old_value, value)
        
        self.logger.info(f"Detection config updated: {list(kwargs.keys())}")
    
    def validate_current_state(self) -> List[str]:
        """Validate current state and return any errors."""
        return self.app_state.validate()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get current state summary for debugging."""
        return self.app_state.get_state_summary()
    
    def _notify_change(self, field_path: str, old_value: Any, new_value: Any):
        """Notify listeners of state changes."""
        if old_value == new_value:
            return  # No actual change
        
        # Mark as unsaved if enabled
        if self._auto_mark_unsaved and not field_path.endswith('unsaved_changes'):
            self.app_state.mark_unsaved()
        
        # Create and send change event
        event = StateChangeEvent(field_path, old_value, new_value)
        
        for listener in self._change_listeners:
            try:
                listener(event)
            except Exception as e:
                self.logger.error(f"State change listener failed for {field_path}: {e}")
    
    def reset_to_defaults(self):
        """Reset app state to default values."""
        old_summary = self.get_state_summary()
        
        # Create new default state
        self.app_state.detection = DetectionConfig()
        self.app_state.video = VideoConfig()
        self.app_state.midi = MidiConfig()
        self.app_state.ui = UIConfig()
        self.app_state.calibration = CalibrationConfig()
        self.app_state.overlays.clear()
        self.app_state.unsaved_changes = True
        
        new_summary = self.get_state_summary()
        self._notify_change("app_state", old_summary, new_summary)
        
        self.logger.info("App state reset to defaults")