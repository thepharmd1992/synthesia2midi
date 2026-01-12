"""
Mouse and keyboard interaction handling for the video canvas.

Separates user interaction logic from display rendering, making both
easier to understand and modify. Handles overlay selection, dragging,
resizing, and keyboard area selection operations.
"""
# Standard library imports
import logging
from typing import Any, Callable, Dict, Optional, Tuple

# Third-party imports
from PySide6.QtCore import QObject, QPoint, QRect, Qt, Signal
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QRubberBand, QWidget

# Local imports
from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.core.app_state import AppState
from synthesia2midi.gui.canvas.coordinates import CoordinateManager

# Debug logging is intentionally minimal in this module.


class CanvasInteraction(QObject):
    """
    Handles all mouse and keyboard interactions with the video canvas.
    
    Separated from rendering logic so interaction behavior can be
    modified without affecting display code and vice versa.
    """
    
    # Signals for communicating with other components
    overlay_selected = Signal(int)  # overlay_index
    overlay_moved = Signal(int, float, float)  # overlay_index, new_x, new_y
    overlay_resized = Signal(int, float, float, float, float)  # index, x, y, w, h
    color_picked = Signal(int, int, int, int, int)  # r, g, b, image_x, image_y from Ctrl+click
    request_repaint = Signal()  # Request canvas repaint after interaction
    spark_roi_selected = Signal(int, int)  # top_y, bottom_y coordinates for spark ROI
    keyboard_region_selected = Signal(int, int, int, int)  # x, y, width, height in image coordinates
    
    def __init__(self, canvas_widget: QWidget, coord_manager: CoordinateManager, app_state: AppState):
        super().__init__()
        self.canvas = canvas_widget
        self.coord_manager = coord_manager
        self.app_state = app_state
        self.logger = logging.getLogger(f"{__name__}.CanvasInteraction")
        
        # Interaction state
        self._dragging = False
        self._drag_data = {
            "x": 0, "y": 0, 
            "item": None, 
            "overlay_idx": -1, 
            "mode": None  # "drag" or "resize"
        }
        self._resize_pivot = {"x": 0, "y": 0}  # For resize from opposite corner
        
        # Performance optimization: throttle repaint requests during drag
        self._last_repaint_request = 0
        self._repaint_throttle_ms = 16  # ~60 FPS max
        
        # Panning is not supported.
        
        # ROI selection state (for spark detection regions)
        self._roi_selection_mode = False
        self._roi_selection_type = "spark"  # "spark"
        self._roi_selecting = False
        self._roi_start_pos = QPoint()
        self._roi_rubber_band = None
        
        # Keyboard region selection state
        self._keyboard_region_selection_mode = False
        self._keyboard_region_selecting = False
        self._keyboard_region_start_pos = QPoint()
        self._keyboard_region_rubber_band = None
        
        # Callbacks for accessing canvas state (to avoid tight coupling)
        self._get_overlays_callback: Optional[Callable] = None
        self._get_pixel_color_callback: Optional[Callable] = None
        self._get_current_frame_callback: Optional[Callable] = None
        
        self.logger.debug("CanvasInteraction initialized")
        
    def _request_throttled_repaint(self):
        """Request repaint with throttling to improve performance during drag operations."""
        import time
        current_time = int(time.time() * 1000)  # milliseconds
        
        if current_time - self._last_repaint_request >= self._repaint_throttle_ms:
            self.request_repaint.emit()
            self._last_repaint_request = current_time
    
    def set_callbacks(self, get_overlays: Callable, get_pixel_color: Callable, 
                     get_current_frame: Callable):
        """Set callback functions to access canvas state without tight coupling."""
        self._get_overlays_callback = get_overlays
        self._get_pixel_color_callback = get_pixel_color
        self._get_current_frame_callback = get_current_frame
    
    def enter_spark_roi_selection_mode(self):
        """Enter spark ROI selection mode."""
        self._roi_selection_mode = True
        self._roi_selection_type = "spark"
        self.logger.info("Entered spark ROI selection mode - click and drag to select region")
    
        
    def exit_spark_roi_selection_mode(self):
        """Exit spark ROI selection mode."""
        self._roi_selection_mode = False
        self._roi_selecting = False
        if self._roi_rubber_band:
            self._roi_rubber_band.hide()
            self._roi_rubber_band = None
        self.logger.info("Exited spark ROI selection mode")
    
        
    def is_in_roi_selection_mode(self) -> bool:
        """Check if currently in ROI selection mode."""
        return self._roi_selection_mode
    
    def enter_keyboard_region_selection_mode(self):
        """Enter keyboard region selection mode."""
        self._keyboard_region_selection_mode = True
        self.logger.info("Entered keyboard region selection mode - click and drag to select keyboard area")
        
    def exit_keyboard_region_selection_mode(self):
        """Exit keyboard region selection mode."""
        self._keyboard_region_selection_mode = False
        self._keyboard_region_selecting = False
        if self._keyboard_region_rubber_band:
            self._keyboard_region_rubber_band.hide()
            self._keyboard_region_rubber_band = None
        self.logger.info("Exited keyboard region selection mode")
        
    def is_in_keyboard_region_selection_mode(self) -> bool:
        """Check if currently in keyboard region selection mode."""
        return self._keyboard_region_selection_mode
    
    def handle_mouse_press(self, event: QMouseEvent) -> bool:
        """
        Handle mouse press events.
        
        Returns:
            True if event was handled, False to pass to default handler
        """
        # Check if in special selection modes first
        if self._keyboard_region_selection_mode:
            return self._handle_keyboard_region_selection_press(event)
        elif self._roi_selection_mode:
            return self._handle_roi_selection_press(event)
        elif event.modifiers() & Qt.ControlModifier:
            return self._handle_ctrl_press(event)
        else:
            return self._handle_normal_press(event)
    
    def handle_mouse_move(self, event: QMouseEvent) -> bool:
        """Handle mouse move events."""
        if self._keyboard_region_selecting:
            self._handle_keyboard_region_selection_move(event)
            return True
        elif self._roi_selecting:
            self._handle_roi_selection_move(event)
            return True
        elif self._dragging:
            self._handle_drag_motion(event)
            return True
        return False
    
    def handle_mouse_release(self, event: QMouseEvent) -> bool:
        """Handle mouse release events."""
        self.logger.debug(f"handle_mouse_release called: keyboard_selecting={self._keyboard_region_selecting}, "
                         f"roi_selecting={self._roi_selecting}, dragging={self._dragging}, "
                         f"button={event.button()}, pos=({event.x()}, {event.y()})")
        
        if self._keyboard_region_selecting:
            self._handle_keyboard_region_selection_release(event)
            return True
        elif self._roi_selecting:
            self._handle_roi_selection_release(event)
            return True
        elif self._dragging:
            self._finish_drag_operation(event)
            return True
        return False
    
    # Keyboard-region selection is currently unused.
    
    def _handle_normal_press(self, event: QMouseEvent) -> bool:
        """Handle normal mouse press (no modifiers)."""
        canvas_x, canvas_y = event.x(), event.y()
        
        # Check if clicking on an overlay
        overlay_info = self._find_overlay_at_position(canvas_x, canvas_y)
        
        if overlay_info is not None:
            overlay_idx, overlay, click_type = overlay_info
            
            if click_type == "center":
                # Start drag operation
                self._start_drag_operation(overlay_idx, overlay, canvas_x, canvas_y, "drag")
                self.overlay_selected.emit(overlay.key_id)  # Emit key_id, not index
            elif click_type == "corner":
                # Start resize operation
                self._start_resize_operation(overlay_idx, overlay, canvas_x, canvas_y)
                self.overlay_selected.emit(overlay.key_id)  # Emit key_id, not index
            
            return True
        else:
            # Clicked on empty area - always clear selection first
            self.app_state.ui.selected_overlay_id = None
            self.overlay_selected.emit(-1)  # Signal no selection
            
            return True
    
    def _handle_ctrl_press(self, event: QMouseEvent) -> bool:
        """Handle Ctrl+click for color picking and resize mode."""
        canvas_x, canvas_y = event.x(), event.y()
        
        # Check if clicking on an overlay for resize mode
        overlay_info = self._find_overlay_at_position(canvas_x, canvas_y)
        
        if overlay_info is not None:
            overlay_idx, overlay, _ = overlay_info
            # Force resize mode for Ctrl+click on overlay
            self._start_resize_operation(overlay_idx, overlay, canvas_x, canvas_y)
            self.overlay_selected.emit(overlay.key_id)  # Emit key_id, not index
        else:
            # Ctrl+click on empty area - color picking
            self._perform_color_picking(canvas_x, canvas_y)
        
        return True
    
    def _start_drag_operation(self, overlay_idx: int, overlay: OverlayConfig, 
                            canvas_x: float, canvas_y: float, mode: str):
        """Start dragging an overlay."""
        self._dragging = True
        self._drag_data = {
            "x": canvas_x,
            "y": canvas_y,
            "item": overlay,
            "overlay_idx": overlay_idx,
            "mode": mode,
            "initial_click_canvas": (canvas_x, canvas_y),
            "initial_overlay_pos": (overlay.x, overlay.y)
        }
        self.logger.debug(f"Started {mode} operation on overlay {overlay_idx}")
    
    def _start_resize_operation(self, overlay_idx: int, overlay: OverlayConfig,
                              canvas_x: float, canvas_y: float):
        """Start resizing an overlay."""
        self._start_drag_operation(overlay_idx, overlay, canvas_x, canvas_y, "resize")
        
        # Calculate resize pivot point (opposite corner)
        canvas_x_rect, canvas_y_rect, canvas_w, canvas_h = self.coord_manager.image_rect_to_canvas(
            overlay.x, overlay.y, overlay.width, overlay.height
        )
        x1_c, y1_c = canvas_x_rect, canvas_y_rect
        x2_c, y2_c = canvas_x_rect + canvas_w, canvas_y_rect + canvas_h
        
        # Find closest corner to click point to determine resize direction
        corners = [
            (x1_c, y1_c, "top_left"),
            (x2_c, y1_c, "top_right"), 
            (x1_c, y2_c, "bottom_left"),
            (x2_c, y2_c, "bottom_right")
        ]
        
        closest_corner = min(corners, key=lambda c: 
            (c[0] - canvas_x)**2 + (c[1] - canvas_y)**2)
        
        # Set pivot to opposite corner
        corner_type = closest_corner[2]
        if corner_type == "top_left":
            self._resize_pivot = {"x": x2_c, "y": y2_c}
        elif corner_type == "top_right":
            self._resize_pivot = {"x": x1_c, "y": y2_c}
        elif corner_type == "bottom_left":
            self._resize_pivot = {"x": x2_c, "y": y1_c}
        else:  # bottom_right
            self._resize_pivot = {"x": x1_c, "y": y1_c}
        
        
        self.logger.debug(f"Resize pivot set to ({self._resize_pivot['x']:.1f}, {self._resize_pivot['y']:.1f})")
    
    # Panning is not supported.
    
    def _handle_drag_motion(self, event: QMouseEvent):
        """Handle mouse motion during drag operations."""
        if not self._dragging:
            return
            
        canvas_x, canvas_y = event.x(), event.y()
        
        # For drag and resize operations, we need an overlay
        if self._drag_data["item"] is None:
            return
            
        overlay = self._drag_data["item"]
        
        if self._drag_data["mode"] == "drag":
            # Calculate movement delta in image coordinates
            canvas_delta_x = canvas_x - self._drag_data["x"]
            canvas_delta_y = canvas_y - self._drag_data["y"]
            
            image_delta_x, image_delta_y = self.coord_manager.scale_delta(
                canvas_delta_x, canvas_delta_y
            )
            
            # Calculate desired new position
            desired_x = overlay.x + image_delta_x
            desired_y = overlay.y + image_delta_y
            
            # Apply boundary constraints to keep overlay within image bounds
            constrained_x = max(0, min(desired_x, self.coord_manager.image_width - overlay.width))
            constrained_y = max(0, min(desired_y, self.coord_manager.image_height - overlay.height))
            
            # Calculate how much the overlay actually moved
            actual_delta_x = constrained_x - overlay.x
            actual_delta_y = constrained_y - overlay.y
            
            # Only update drag reference point by the amount the overlay actually moved
            # This prevents mouse-overlay desync when hitting boundaries
            if actual_delta_x != 0 or actual_delta_y != 0:
                # Convert actual movement back to canvas coordinates
                canvas_actual_delta_x = actual_delta_x * self.coord_manager.image_scale_factor
                canvas_actual_delta_y = actual_delta_y * self.coord_manager.image_scale_factor
                
                # Update drag reference by actual movement only
                self._drag_data["x"] += canvas_actual_delta_x
                self._drag_data["y"] += canvas_actual_delta_y
                
                # Emit position change
                self.overlay_moved.emit(self._drag_data["overlay_idx"], constrained_x, constrained_y)
            
        elif self._drag_data["mode"] == "resize":
            # Calculate new size based on current position and pivot
            pivot_x, pivot_y = self._resize_pivot["x"], self._resize_pivot["y"]
            
            # Convert canvas coordinates to image coordinates for resize calculation
            current_img_pos = self.coord_manager.canvas_to_image(canvas_x, canvas_y)
            pivot_img_pos = self.coord_manager.canvas_to_image(pivot_x, pivot_y)
            
            if current_img_pos and pivot_img_pos:
                # Calculate new rectangle from pivot to current position
                new_x = min(current_img_pos[0], pivot_img_pos[0])
                new_y = min(current_img_pos[1], pivot_img_pos[1])
                new_width = abs(current_img_pos[0] - pivot_img_pos[0])
                new_height = abs(current_img_pos[1] - pivot_img_pos[1])
                
                # Add bounds checking and minimum size constraints
                min_width, min_height = 1, 1  # Allow single-pixel overlays
                max_width = self.coord_manager.image_width - new_x
                max_height = self.coord_manager.image_height - new_y
                
                # Clamp to reasonable bounds
                new_width = max(min_width, min(new_width, max_width))
                new_height = max(min_height, min(new_height, max_height))
                
                # Ensure overlay stays within image bounds
                if new_x + new_width > self.coord_manager.image_width:
                    new_x = max(0, self.coord_manager.image_width - new_width)
                if new_y + new_height > self.coord_manager.image_height:
                    new_y = max(0, self.coord_manager.image_height - new_height)
                
                new_x = max(0, new_x)
                new_y = max(0, new_y)
                
                # Emit resize change
                self.overlay_resized.emit(
                    self._drag_data["overlay_idx"], 
                    new_x, new_y, new_width, new_height
                )
            else:
                self.logger.warning("Coordinate conversion failed during resize operation")
        
        # Request repaint to show updated overlay (throttled for performance)
        self._request_throttled_repaint()
    
    def _finish_drag_operation(self, event: QMouseEvent):
        """Finish drag or resize operation."""
        if self._dragging:
            self.logger.debug(f"Finished {self._drag_data['mode']} operation")
            self._dragging = False
            self._drag_data = {"x": 0, "y": 0, "item": None, "overlay_idx": -1, "mode": None}
            self.request_repaint.emit()
    
    def _perform_color_picking(self, canvas_x: float, canvas_y: float):
        """Perform color picking at the specified canvas position."""
        if not self._get_pixel_color_callback:
            self.logger.warning("Color picking callback not available")
            return
            
        image_pos = self.coord_manager.canvas_to_image(canvas_x, canvas_y)
        if image_pos:
            color = self._get_pixel_color_callback(int(image_pos[0]), int(image_pos[1]))
            if color:
                self.color_picked.emit(color[0], color[1], color[2], int(image_pos[0]), int(image_pos[1]))
                self.logger.info(f"Color picked at image({int(image_pos[0])}, {int(image_pos[1])}): RGB({color[0]}, {color[1]}, {color[2]})")
            else:
                self.logger.warning(f"No color data at image position ({int(image_pos[0])}, {int(image_pos[1])})")
        else:
            self.logger.warning(f"Canvas position ({canvas_x}, {canvas_y}) is outside image bounds")
    
    def _find_overlay_at_position(self, canvas_x: float, canvas_y: float) -> Optional[Tuple[int, OverlayConfig, str]]:
        """
        Find overlay at canvas position.
        
        Returns:
            (overlay_index, overlay, click_type) where click_type is "center" or "corner"
        """
        if not self._get_overlays_callback:
            return None
            
        overlays = self._get_overlays_callback()
        if not overlays:
            return None
        
        # Check overlays in reverse order (top-most first)
        for i in range(len(overlays) - 1, -1, -1):
            overlay = overlays[i]
            
            # Convert overlay rectangle to canvas coordinates
            canvas_rect = self.coord_manager.image_rect_to_canvas(
                overlay.x, overlay.y, overlay.width, overlay.height
            )
            x1, y1, w, h = canvas_rect
            x2, y2 = x1 + w, y1 + h
            
            # Check if point is inside overlay rectangle
            if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                # Determine if click is near a corner (for resize) or center (for drag)
                corner_threshold = 10  # pixels
                
                is_near_corner = (
                    (abs(canvas_x - x1) <= corner_threshold or abs(canvas_x - x2) <= corner_threshold) and
                    (abs(canvas_y - y1) <= corner_threshold or abs(canvas_y - y2) <= corner_threshold)
                )
                
                click_type = "corner" if is_near_corner else "center"
                
                self.logger.debug(f"Overlay {i} hit: canvas=({canvas_x:.1f}, {canvas_y:.1f}), click_type: {click_type}")
                
                return (i, overlay, click_type)
        
        return None
    
    # ROI selection helpers
    
    def _handle_roi_selection_press(self, event: QMouseEvent) -> bool:
        """Handle mouse press during ROI selection mode."""
        if event.button() == Qt.LeftButton:
            # Start ROI selection
            self._roi_selecting = True
            self._roi_start_pos = QPoint(event.x(), event.y())
            
            # Create rubber band for visual feedback
            if not self._roi_rubber_band:
                try:
                    self._roi_rubber_band = QRubberBand(QRubberBand.Rectangle, self.canvas)
                except (TypeError, AttributeError):
                    # Canvas might not be a real QWidget (e.g., during testing)
                    self.logger.debug("Cannot create QRubberBand - canvas not a valid QWidget")
            
            if self._roi_rubber_band:
                self._roi_rubber_band.setGeometry(QRect(self._roi_start_pos, self._roi_start_pos))
                self._roi_rubber_band.show()
                # Ensure rubber band doesn't interfere with mouse events
                self._roi_rubber_band.setAttribute(Qt.WA_TransparentForMouseEvents)
            
            self.logger.debug(f"Started ROI selection at ({event.x()}, {event.y()})")
            return True
        elif event.button() == Qt.RightButton:
            # Right click to cancel ROI selection mode
            self.exit_spark_roi_selection_mode()
            return True
        return False
    
    def _handle_roi_selection_move(self, event: QMouseEvent):
        """Handle mouse move during ROI selection."""
        if self._roi_selecting and self._roi_rubber_band:
            # Update rubber band to show current selection area
            current_pos = QPoint(event.x(), event.y())
            selection_rect = QRect(self._roi_start_pos, current_pos).normalized()
            self._roi_rubber_band.setGeometry(selection_rect)
    
    def _handle_roi_selection_release(self, event: QMouseEvent):
        """Handle mouse release to complete ROI selection."""
        if self._roi_selecting and event.button() == Qt.LeftButton:
            # Calculate final selection rectangle in image coordinates
            end_pos = QPoint(event.x(), event.y())
            selection_rect = QRect(self._roi_start_pos, end_pos).normalized()
            
            # Convert to image coordinates with clamping to handle selections beyond image bounds
            start_img = self.coord_manager.canvas_to_image(selection_rect.x(), selection_rect.y(), clamp_to_bounds=True)
            end_img = self.coord_manager.canvas_to_image(
                selection_rect.x() + selection_rect.width(), 
                selection_rect.y() + selection_rect.height(),
                clamp_to_bounds=True
            )
            
            if start_img and end_img:
                # Spark ROI is a horizontal band, so we only care about Y coordinates
                top_y = min(int(start_img[1]), int(end_img[1]))
                bottom_y = max(int(start_img[1]), int(end_img[1]))
                
                # Ensure minimum height of 1 pixel
                if bottom_y - top_y < 1:
                    bottom_y = top_y + 1
                
                # Clamp to image bounds
                top_y = max(0, top_y)
                bottom_y = min(self.coord_manager.image_height, bottom_y)
                
                self.logger.info(f"ROI selected: Y range {top_y} to {bottom_y}")
                
                # Emit signal with selected ROI
                self.spark_roi_selected.emit(top_y, bottom_y)
            else:
                self.logger.warning("Failed to convert ROI selection to image coordinates")
            
            # Clean up and exit ROI selection mode
            self.exit_spark_roi_selection_mode()
    
    def _handle_keyboard_region_selection_press(self, event: QMouseEvent) -> bool:
        """Handle mouse press during keyboard region selection mode."""
        if event.button() == Qt.LeftButton:
            # Start keyboard region selection
            self._keyboard_region_selecting = True
            self._keyboard_region_start_pos = QPoint(event.x(), event.y())
            
            # Create rubber band for visual feedback
            if not self._keyboard_region_rubber_band:
                try:
                    self._keyboard_region_rubber_band = QRubberBand(QRubberBand.Rectangle, self.canvas)
                except (TypeError, AttributeError):
                    # Canvas might not be a real QWidget (e.g., during testing)
                    self.logger.debug("Cannot create QRubberBand - canvas not a valid QWidget")
            
            if self._keyboard_region_rubber_band:
                self._keyboard_region_rubber_band.setGeometry(QRect(self._keyboard_region_start_pos, self._keyboard_region_start_pos))
                self._keyboard_region_rubber_band.show()
                # Ensure rubber band doesn't interfere with mouse events
                self._keyboard_region_rubber_band.setAttribute(Qt.WA_TransparentForMouseEvents)
            
            self.logger.debug(f"Started keyboard region selection at ({event.x()}, {event.y()})")
            return True
        elif event.button() == Qt.RightButton:
            # Right click to cancel keyboard region selection mode
            self.exit_keyboard_region_selection_mode()
            return True
        return False
    
    def _handle_keyboard_region_selection_move(self, event: QMouseEvent):
        """Handle mouse move during keyboard region selection."""
        if self._keyboard_region_selecting and self._keyboard_region_rubber_band:
            # Update rubber band to show current selection area
            current_pos = QPoint(event.x(), event.y())
            selection_rect = QRect(self._keyboard_region_start_pos, current_pos).normalized()
            self._keyboard_region_rubber_band.setGeometry(selection_rect)
    
    def _handle_keyboard_region_selection_release(self, event: QMouseEvent):
        """Handle mouse release to complete keyboard region selection."""
        self.logger.info("=== KEYBOARD REGION SELECTION RELEASE ===")
        
        if self._keyboard_region_selecting and event.button() == Qt.LeftButton:
            # Calculate final selection rectangle in image coordinates
            end_pos = QPoint(event.x(), event.y())
            selection_rect = QRect(self._keyboard_region_start_pos, end_pos).normalized()
            
            self.logger.debug(f"Canvas selection rect: x={selection_rect.x()}, y={selection_rect.y()}, "
                            f"width={selection_rect.width()}, height={selection_rect.height()}")
            
            # Convert to image coordinates with clamping to handle selections beyond image bounds
            start_img = self.coord_manager.canvas_to_image(selection_rect.x(), selection_rect.y(), clamp_to_bounds=True)
            end_img = self.coord_manager.canvas_to_image(
                selection_rect.x() + selection_rect.width(), 
                selection_rect.y() + selection_rect.height(),
                clamp_to_bounds=True
            )
            
            self.logger.debug(f"Start image coords: {start_img}")
            self.logger.debug(f"End image coords: {end_img}")
            
            if start_img and end_img:
                # Calculate keyboard region bounds
                x = min(int(start_img[0]), int(end_img[0]))
                y = min(int(start_img[1]), int(end_img[1]))
                width = abs(int(end_img[0]) - int(start_img[0]))
                height = abs(int(end_img[1]) - int(start_img[1]))
                
                self.logger.info(f"Image region bounds: x={x}, y={y}, width={width}, height={height}")
                
                # Ensure minimum size
                if width < 50 or height < 20:
                    self.logger.warning(f"Selected keyboard region too small ({width}x{height}), please select a larger area")
                    self.exit_keyboard_region_selection_mode()
                    return
                
                # Clamp to image bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, self.coord_manager.image_width - x)
                height = min(height, self.coord_manager.image_height - y)
                
                self.logger.info(f"=== EMITTING KEYBOARD REGION SIGNAL ===")
                self.logger.info(f"Final region: x={x}, y={y}, width={width}, height={height}")
                
                # Emit signal with selected region
                self.keyboard_region_selected.emit(x, y, width, height)
                self.logger.info("Signal emitted successfully")
            else:
                self.logger.warning("Failed to convert keyboard region selection to image coordinates")
                self.logger.warning(f"coord_manager dimensions: image={self.coord_manager.image_width}x{self.coord_manager.image_height}")
            
            # Clean up and exit keyboard region selection mode
            self.logger.info("Exiting keyboard region selection mode")
            self.exit_keyboard_region_selection_mode()
        else:
            self.logger.debug(f"Ignoring release: selecting={self._keyboard_region_selecting}, button={event.button()}")

    def get_interaction_state(self) -> Dict[str, Any]:
        """Get current interaction state for debugging."""
        return {
            "dragging": self._dragging,
            "drag_mode": self._drag_data.get("mode"),
            "selected_overlay": self._drag_data.get("overlay_idx", -1),
            "roi_selection_mode": self._roi_selection_mode,
            "roi_selecting": self._roi_selecting,
            "keyboard_region_selection_mode": self._keyboard_region_selection_mode,
            "keyboard_region_selecting": self._keyboard_region_selecting
        }
