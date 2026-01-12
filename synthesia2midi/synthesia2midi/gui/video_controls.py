"""
Video navigation and frame display helpers.

Handles frame navigation, slider events, time display, and frame rendering.
"""
# Standard library imports
import logging
from typing import Optional

# Third-party imports
from PySide6.QtCore import QTimer

class VideoControls:
    """Manages video navigation, frame slider, and time display functionality."""
    
    def __init__(self, app_state, video_session, keyboard_canvas, frame_slider, time_label):
        self.app_state = app_state
        self.video_session = video_session
        self.keyboard_canvas = keyboard_canvas
        self.frame_slider = frame_slider
        self.time_label = time_label
        
        # Frame slider debouncing and throttling
        self._slider_timer = QTimer()
        self._slider_timer.setSingleShot(True)
        self._slider_timer.timeout.connect(self._on_slider_timer_timeout)
        self._pending_frame_index = None
        
        # Aggressive throttling for slider movement
        self._last_slider_update_time = 0
        self._slider_throttle_ms = 50  # Minimum 50ms between frame updates
        self._slider_is_being_dragged = False
        
        # Detection logging callback (will be set by main app)
        self._log_detection_callback: Optional[callable] = None
        
    def set_detection_logging_callback(self, callback: callable):
        """Set callback for detection parameter logging."""
        self._log_detection_callback = callback
        
    def set_video_session(self, video_session):
        """Update video session reference."""
        self.video_session = video_session
    
    def navigate_frame_pgup(self):
        """Page up navigation + detection logging."""
        if not self.video_session: 
            return
        # Respect trim boundaries if video is trimmed
        min_frame = self.app_state.video.trim_start_frame if self.app_state.video.video_is_trimmed else 0
        new_idx = max(min_frame, self.app_state.video.current_frame_index - self.app_state.video.current_nav_interval)
        try:
            if not self.display_frame_with_slider_update(new_idx):
                logging.warning(f"Failed to display frame {new_idx} during page up navigation")
        except Exception as e:
            logging.error(f"Error during page up navigation to frame {new_idx}: {e}")
            return
        # Log detection parameters if logging is enabled and callback is set
        if self._log_detection_callback:
            self._log_detection_callback()
    
    def navigate_frame_pgdn(self):
        """Page down navigation + detection logging."""
        if not self.video_session: 
            return
        # Respect trim boundaries if video is trimmed
        max_frame = self.app_state.video.trim_end_frame if self.app_state.video.video_is_trimmed else self.video_session.total_frames - 1
        new_idx = min(max_frame, 
                      self.app_state.video.current_frame_index + self.app_state.video.current_nav_interval)
        try:
            if not self.display_frame_with_slider_update(new_idx):
                logging.warning(f"Failed to display frame {new_idx} during page down navigation")
        except Exception as e:
            logging.error(f"Error during page down navigation to frame {new_idx}: {e}")
            return
        # Log detection parameters if logging is enabled and callback is set
        if self._log_detection_callback:
            self._log_detection_callback()
    
    def update_frame_slider_for_video(self):
        """Updates slider range when video loads."""
        if self.video_session:
            # Check if video is trimmed and use trim boundaries
            if self.app_state.video.video_is_trimmed:
                min_frame = self.app_state.video.trim_start_frame
                max_frame = self.app_state.video.trim_end_frame
            else:
                min_frame = 0
                max_frame = self.video_session.total_frames - 1
            
            self.frame_slider.setMinimum(min_frame)
            self.frame_slider.setMaximum(max_frame)
            # Set slider to current frame index temporarily to avoid slider event issues
            # The actual initial frame will be set by display_frame_with_slider_update
            current_frame = self.app_state.video.current_frame_index
            if current_frame < min_frame or current_frame > max_frame:
                # If current frame is out of bounds, use min_frame
                self.frame_slider.setValue(min_frame)
            else:
                # Use current frame to avoid slider change events
                self.frame_slider.setValue(current_frame)
            self.frame_slider.setEnabled(True)
            logging.info(f"Frame slider updated: {min_frame}-{max_frame}, initial value: {self.frame_slider.value()}")
        else:
            self.frame_slider.setEnabled(False)
            self.frame_slider.setValue(0)
            self.time_label.setText("0:00")
    
    def on_frame_slider_changed(self, frame_index: int):
        """Frame slider event handling - NO frame loading during drag for max performance."""
        if not self.video_session:
            return
        
        # Prevent recursive updates when we programmatically set the slider value
        if frame_index == self.app_state.video.current_frame_index:
            return
        
        # Store the target frame
        self._pending_frame_index = frame_index
        self._slider_is_being_dragged = True
        
        # Cancel any previous timer
        self._slider_timer.stop()
        
        # Update time display immediately for responsive feedback
        self.update_time_display(frame_index)
        
        # DO NOT load frame during drag - just start/restart the timer
        # Frame will only be loaded when user stops dragging (timer fires)
        self._slider_timer.start(200)  # Wait 200ms after last slider change
        
        logging.debug(f"Frame slider target: {frame_index} (loading deferred)")
    
    def _on_slider_timer_timeout(self):
        """Debounced slider handling - called after slider has stopped moving."""
        if self._pending_frame_index is not None and self.video_session:
            # Now actually load and display the target frame
            target_frame = self._pending_frame_index
            self._pending_frame_index = None
            self._slider_is_being_dragged = False
            
            # Load frame with detection if enabled
            if self.app_state.ui.live_detection_feedback:
                self.display_frame_with_slider_update(target_frame)
            else:
                self.display_frame_lightweight(target_frame)
                
            logging.debug(f"Frame slider loaded frame: {target_frame}")
    
    def display_frame_lightweight(self, frame_index: int) -> bool:
        """Frame display without detection for smooth navigation."""
        if not self.video_session:
            return False
        
        # Use the optimized no-detection display method
        result = self.keyboard_canvas.display_frame_no_detection(frame_index)
        return result
    
    def update_frame_slider_position(self):
        """Syncs slider with current frame without triggering events."""
        if hasattr(self, 'frame_slider') and self.video_session:
            # Use blockSignals instead of disconnect/reconnect to avoid Qt connection overhead
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.app_state.video.current_frame_index)
            self.frame_slider.blockSignals(False)
            # Update time display
            self.update_time_display(self.app_state.video.current_frame_index)
    
    def update_time_display(self, frame_index: int):
        """Updates time label from frame index."""
        if self.video_session and hasattr(self, 'time_label'):
            # Calculate time in seconds
            time_seconds = frame_index / self.video_session.fps
            # Convert to minutes:seconds format
            minutes = int(time_seconds // 60)
            seconds = int(time_seconds % 60)
            self.time_label.setText(f"{minutes}:{seconds:02d}")
    
    def display_frame_with_slider_update(self, frame_index: int) -> bool:
        """Frame display + slider sync - wrapper for display_frame that also updates the frame slider."""
        result = self.keyboard_canvas.display_frame(frame_index)
        if result:
            self.update_frame_slider_position()
        return result
    
    def update_current_frame_display(self):
        """Reprocess current frame for live updates - triggers immediate visual feedback."""
        if not self.video_session or not hasattr(self, 'keyboard_canvas'):
            return False
            
        current_frame = self.app_state.video.current_frame_index
        
        # Reprocess current frame with updated parameters
        result = self.keyboard_canvas.display_frame(current_frame)
        
        # Ensure overlays are redrawn with current parameters
        if result:
            self.keyboard_canvas.draw_overlays()
            
        logging.debug(f"Live update: reprocessed frame {current_frame}")
        return result
