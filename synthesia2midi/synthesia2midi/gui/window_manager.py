"""
Window management functions for Synthesia2MIDI application.

This module handles window resize events, show events, and window positioning/sizing,
including frame redraw behavior when the window size changes.
"""
# Standard library imports
import logging

# Third-party imports
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication

# Local imports
from ..core.app_state import AppState


class WindowManager:
    """Manages window resize, show events, and positioning functionality."""
    
    def __init__(self, app_state: AppState, main_window):
        """
        Initialize WindowManager.
        
        Args:
            app_state: The application state object
            main_window: Reference to the main window widget
        """
        self.app_state = app_state
        self.main_window = main_window
        self._is_resizing = False  # Flag to prevent recursive resize handling
        self._resize_timer = None  # Timer for batching resize updates
    
    def handle_resize_event(self, event):
        """Handles resizing of the main window."""
        # Call the parent class resize event first
        super(self.main_window.__class__, self.main_window).resizeEvent(event)
        
        # If a video is loaded, batch resize updates to avoid lag
        if (hasattr(self.main_window, 'video_session') and 
            self.main_window.video_session and 
            hasattr(self.main_window, 'keyboard_canvas')):
            
            # Cancel any pending resize update
            if self._resize_timer:
                self._resize_timer.stop()
                self._resize_timer = None
            
            # Create new timer for batched update with minimal delay
            from PySide6.QtCore import QTimer
            self._resize_timer = QTimer()
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self._perform_resize_update)
            
            # Use very short delay (10ms) just to batch rapid resize events
            # This reduces lag while still preventing excessive updates
            self._resize_timer.start(10)
    
    def _perform_resize_update(self):
        """Perform the actual resize update after batching."""
        try:
            # Clear the timer reference
            self._resize_timer = None
            
            # Force a redraw of the current frame with new dimensions
            self.main_window.keyboard_canvas.display_frame(self.app_state.video.current_frame_index)
            # Ensure overlays are redrawn with correct positions
            self.main_window.keyboard_canvas.draw_overlays()
            # Update frame slider position via video_controls
            if hasattr(self.main_window, 'video_controls'):
                self.main_window.video_controls.update_frame_slider_position()
        except Exception as e:
            logging.error(f"Error during resize update: {e}")
    
    def handle_show_event(self, event):
        """Ensure window is properly positioned when shown."""
        # Call the parent class show event first
        super(self.main_window.__class__, self.main_window).showEvent(event)
        
        # If we have a video loaded, ensure window stays at top-left
        if hasattr(self.main_window, 'video_session') and self.main_window.video_session:
            screen = QApplication.primaryScreen()
            screen_rect = screen.availableGeometry()
            self.main_window.move(screen_rect.left(), screen_rect.top())
    
    def resize_and_position_window(self):
        """Resize and position window to ensure full visibility without scrolling."""
        screen = QApplication.primaryScreen()
        screen_rect = screen.availableGeometry()
        max_width = screen_rect.width() - 20  # Small margin
        max_height = screen_rect.height() - 40  # Leave space for menu/dock
        
        # Width: use the available display so the settings pane can open at a
        # readable default width instead of starting squeezed.
        optimal_width = max_width
        optimal_height = max_height
        
        # Ensure we don't exceed screen bounds
        if optimal_width > max_width:
            optimal_width = max_width
        if optimal_height > max_height:
            optimal_height = max_height
        
        self.main_window.resize(optimal_width, optimal_height)
        
        # Position window at exact top-left of screen
        self.main_window.move(screen_rect.left(), screen_rect.top())
        
        # Force layout update to ensure everything is positioned correctly
        QApplication.processEvents()
        
        settings_rail_width = getattr(getattr(self.main_window, "settings_rail_button", None), "width", lambda: 0)()
        logging.info(f"Window positioned at top-left ({screen_rect.left()}, {screen_rect.top()}) "
                     f"with size {optimal_width}x{optimal_height}, "
                     f"settings rail width: {settings_rail_width}px")
        
        # Update controls and redraw frame if available
        if hasattr(self.main_window, 'control_panel'):
            self.main_window.control_panel.update_controls_from_state()
        if hasattr(self.main_window, 'settings_tool_window'):
            self.main_window.settings_tool_window.fit_to_available_screen()
        
        # Redraw the frame to show the keyboard area outline if video is loaded
        if (hasattr(self.main_window, 'video_session') and 
            self.main_window.video_session and 
            hasattr(self.main_window, 'keyboard_canvas')):
            self.main_window.keyboard_canvas.display_frame(self.app_state.video.current_frame_index)
