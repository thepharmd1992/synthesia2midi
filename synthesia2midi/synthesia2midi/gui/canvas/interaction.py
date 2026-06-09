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
    manual_fit_group_moved = Signal(float, float)  # image-space dx, dy
    manual_fit_local_group_moved = Signal(float, float)  # image-space dx, dy
    manual_fit_region_selected = Signal(str, float, float)  # region_type, top_y, bottom_y
    manual_fit_local_selection_selected = Signal(float, float, float, float)  # left, top, right, bottom
    manual_fit_keyboard_box_selected = Signal(float, float, float, float)  # left, top, right, bottom
    manual_fit_keyboard_box_edge_changed = Signal(str, float)  # edge, image_x
    manual_fit_guide_line_changed = Signal(str, float)  # line_type, image_y
    manual_fit_guide_line_selected = Signal(str, float)  # line_type, image_y
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
        self._manual_fit_mode = "off"
        self._manual_fit_group_dragging = False
        self._manual_fit_group_drag_start = QPoint()
        self._manual_fit_local_dragging = False
        self._manual_fit_local_drag_start = QPoint()
        self._manual_fit_region_selecting = False
        self._manual_fit_region_start_pos = QPoint()
        self._manual_fit_region_rubber_band = None
        self._manual_fit_line_dragging = False
        self._manual_fit_keyboard_box_edge_dragging = False
        self._manual_fit_keyboard_box_edge = ""
        self._manual_fit_keyboard_box_edge_image_offset = 0.0
        
        # Performance optimization: throttle repaint requests during drag
        self._last_repaint_request = 0
        self._repaint_throttle_ms = 16  # ~60 FPS max
        
        # Panning is not supported.
        self.overlay_drawing_type = getattr(self.app_state.ui, "overlay_drawing_type", "key")
        
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
        self._get_manual_fit_local_key_ids_callback: Optional[Callable] = None
        self._get_manual_fit_keyboard_box_callback: Optional[Callable] = None
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
    
    def set_callbacks(
        self,
        get_overlays: Callable,
        get_pixel_color: Callable,
        get_current_frame: Callable,
        get_manual_fit_local_key_ids: Optional[Callable] = None,
        get_manual_fit_keyboard_box: Optional[Callable] = None,
    ):
        """Set callback functions to access canvas state without tight coupling."""
        self._get_overlays_callback = get_overlays
        self._get_pixel_color_callback = get_pixel_color
        self._get_current_frame_callback = get_current_frame
        self._get_manual_fit_local_key_ids_callback = get_manual_fit_local_key_ids
        self._get_manual_fit_keyboard_box_callback = get_manual_fit_keyboard_box
    
    def enter_spark_roi_selection_mode(self):
        """Enter spark ROI selection mode."""
        self._enter_roi_selection_mode("spark")

    def enter_shadow_roi_selection_mode(self):
        """Enter dormant/general shadow ROI selection mode."""
        self._enter_roi_selection_mode("shadow")

    def enter_shadow_white_roi_selection_mode(self):
        """Enter dormant white-key shadow ROI selection mode."""
        self._enter_roi_selection_mode("shadow_white")

    def enter_shadow_black_roi_selection_mode(self):
        """Enter dormant black-key shadow ROI selection mode."""
        self._enter_roi_selection_mode("shadow_black")

    def _enter_roi_selection_mode(self, roi_type: str):
        self._roi_selection_mode = True
        self._roi_selection_type = roi_type
        self.logger.info("Entered %s ROI selection mode - click and drag to select region", roi_type)
    
        
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

    def set_overlay_drawing_type(self, overlay_type: str) -> None:
        """Update the interaction drawing mode through an explicit API."""
        self.overlay_drawing_type = overlay_type

    def set_manual_fit_mode(self, mode: str) -> None:
        """Set manual keyboard fit interaction mode."""
        if mode not in {
            "off",
            "manual_fit_group",
            "manual_fit_single",
            "manual_fit_local_select",
            "manual_fit_keyboard_box",
            "manual_fit_keyboard_box_edges",
            "manual_fit_black_bottom",
            "manual_fit_white_start",
            "manual_fit_black_region",
            "manual_fit_white_region",
        }:
            raise ValueError(f"Unknown manual fit mode: {mode}")
        self._manual_fit_mode = mode
        self._manual_fit_group_dragging = False
        self._manual_fit_local_dragging = False
        self._manual_fit_region_selecting = False
        self._manual_fit_line_dragging = False
        self._manual_fit_keyboard_box_edge_dragging = False
        self._manual_fit_keyboard_box_edge = ""
        self._manual_fit_keyboard_box_edge_image_offset = 0.0

    def manual_fit_mode(self) -> str:
        return self._manual_fit_mode
    
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
        elif self._manual_fit_mode == "manual_fit_keyboard_box":
            return self._handle_manual_fit_keyboard_box_press(event)
        elif self._manual_fit_mode == "manual_fit_keyboard_box_edges":
            return self._handle_manual_fit_keyboard_box_edge_press(event)
        elif self._manual_fit_mode in {"manual_fit_black_bottom", "manual_fit_white_start"}:
            return self._handle_manual_fit_line_press(event)
        elif self._manual_fit_mode in {"manual_fit_black_region", "manual_fit_white_region"}:
            return self._handle_manual_fit_region_press(event)
        elif self._manual_fit_mode == "manual_fit_local_select":
            return self._handle_manual_fit_local_selection_press(event)
        elif self._manual_fit_mode == "manual_fit_group":
            return self._handle_manual_fit_group_press(event)
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
        elif self._manual_fit_region_selecting:
            self._handle_manual_fit_region_move(event)
            return True
        elif self._manual_fit_line_dragging:
            self._emit_manual_fit_line_y(event.x(), event.y(), completed=False)
            self._request_throttled_repaint()
            return True
        elif self._manual_fit_keyboard_box_edge_dragging:
            self._emit_manual_fit_keyboard_box_edge_x(event.x(), event.y())
            self._request_throttled_repaint()
            return True
        elif self._manual_fit_local_dragging:
            self._handle_manual_fit_local_group_motion(event)
            return True
        elif self._manual_fit_group_dragging:
            self._handle_manual_fit_group_motion(event)
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
        elif self._manual_fit_region_selecting:
            if self._manual_fit_mode in {"manual_fit_keyboard_box", "manual_fit_keyboard_box_edges"}:
                self._handle_manual_fit_keyboard_box_release(event)
            elif self._manual_fit_mode == "manual_fit_local_select":
                self._handle_manual_fit_local_selection_release(event)
            else:
                self._handle_manual_fit_region_release(event)
            return True
        elif self._manual_fit_line_dragging:
            self._handle_manual_fit_line_release(event)
            return True
        elif self._manual_fit_keyboard_box_edge_dragging:
            self._handle_manual_fit_keyboard_box_edge_release(event)
            return True
        elif self._manual_fit_local_dragging:
            self._finish_manual_fit_local_group_drag()
            return True
        elif self._manual_fit_group_dragging:
            self._finish_manual_fit_group_drag()
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
            # Clicked on empty area - emit no selection and let the canvas/controller own state.
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

    def _handle_manual_fit_group_press(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.LeftButton:
            return False
        canvas_x, canvas_y = event.x(), event.y()
        if self._is_point_inside_overlay_bounds(canvas_x, canvas_y):
            self._manual_fit_group_dragging = True
            self._manual_fit_group_drag_start = QPoint(canvas_x, canvas_y)
            return True
        self.overlay_selected.emit(-1)
        return True

    def _handle_manual_fit_group_motion(self, event: QMouseEvent) -> None:
        current = QPoint(event.x(), event.y())
        canvas_delta_x = current.x() - self._manual_fit_group_drag_start.x()
        canvas_delta_y = current.y() - self._manual_fit_group_drag_start.y()
        image_delta_x, image_delta_y = self.coord_manager.scale_delta(canvas_delta_x, canvas_delta_y)
        if image_delta_x or image_delta_y:
            self.manual_fit_group_moved.emit(image_delta_x, image_delta_y)
            self._manual_fit_group_drag_start = current
            self._request_throttled_repaint()

    def _finish_manual_fit_group_drag(self) -> None:
        self._manual_fit_group_dragging = False
        self.request_repaint.emit()

    def _handle_manual_fit_keyboard_box_press(self, event: QMouseEvent) -> bool:
        return self._start_manual_fit_rectangle_selection(event)

    def _handle_manual_fit_keyboard_box_edge_press(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.LeftButton:
            return False
        edge = self._manual_fit_keyboard_box_edge_at_position(event.x(), event.y())
        if edge is None:
            return self._start_manual_fit_rectangle_selection(event)
        self._manual_fit_keyboard_box_edge_dragging = True
        self._manual_fit_keyboard_box_edge = edge
        self._manual_fit_keyboard_box_edge_image_offset = self._manual_fit_keyboard_box_edge_press_offset(
            edge,
            event.x(),
            event.y(),
        )
        return True

    def _handle_manual_fit_keyboard_box_edge_release(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._emit_manual_fit_keyboard_box_edge_x(event.x(), event.y())
        self._manual_fit_keyboard_box_edge_dragging = False
        self._manual_fit_keyboard_box_edge = ""
        self._manual_fit_keyboard_box_edge_image_offset = 0.0
        self.request_repaint.emit()

    def _emit_manual_fit_keyboard_box_edge_x(self, canvas_x: float, canvas_y: float) -> None:
        image_pos = self.coord_manager.canvas_to_image(
            canvas_x,
            canvas_y,
            clamp_to_bounds=True,
        )
        if not image_pos or not self._manual_fit_keyboard_box_edge:
            return
        self.manual_fit_keyboard_box_edge_changed.emit(
            self._manual_fit_keyboard_box_edge,
            float(image_pos[0]) - self._manual_fit_keyboard_box_edge_image_offset,
        )

    def _handle_manual_fit_local_selection_press(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.LeftButton:
            return False
        canvas_x, canvas_y = event.x(), event.y()
        if self._is_point_inside_manual_fit_local_overlay(canvas_x, canvas_y):
            self._manual_fit_local_dragging = True
            self._manual_fit_local_drag_start = QPoint(canvas_x, canvas_y)
            return True
        return self._start_manual_fit_rectangle_selection(event)

    def _handle_manual_fit_local_group_motion(self, event: QMouseEvent) -> None:
        current = QPoint(event.x(), event.y())
        canvas_delta_x = current.x() - self._manual_fit_local_drag_start.x()
        canvas_delta_y = current.y() - self._manual_fit_local_drag_start.y()
        image_delta_x, image_delta_y = self.coord_manager.scale_delta(canvas_delta_x, canvas_delta_y)
        if image_delta_x or image_delta_y:
            self.manual_fit_local_group_moved.emit(image_delta_x, image_delta_y)
            self._manual_fit_local_drag_start = current
            self._request_throttled_repaint()

    def _finish_manual_fit_local_group_drag(self) -> None:
        self._manual_fit_local_dragging = False
        self.request_repaint.emit()

    def _handle_manual_fit_keyboard_box_release(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            return
        end_pos = QPoint(event.x(), event.y())
        selection_rect = QRect(self._manual_fit_region_start_pos, end_pos).normalized()
        start_img = self.coord_manager.canvas_to_image(
            self._manual_fit_region_start_pos.x(),
            self._manual_fit_region_start_pos.y(),
            clamp_to_bounds=True,
        )
        end_img = self.coord_manager.canvas_to_image(
            end_pos.x(),
            end_pos.y(),
            clamp_to_bounds=True,
        )
        self._finish_manual_fit_region_selection()
        if not start_img or not end_img:
            return
        left = min(float(start_img[0]), float(end_img[0]))
        right = max(float(start_img[0]), float(end_img[0]))
        top = min(float(start_img[1]), float(end_img[1]))
        bottom = max(float(start_img[1]), float(end_img[1]))
        if right <= left or bottom <= top:
            return
        self.manual_fit_keyboard_box_selected.emit(left, top, right, bottom)
        self.request_repaint.emit()

    def _manual_fit_keyboard_box_edge_at_position(self, canvas_x: float, canvas_y: float) -> Optional[str]:
        box = self._current_manual_fit_keyboard_box()
        if box is None:
            return None
        left = float(getattr(box, "left", 0.0))
        right = float(getattr(box, "right", left + 1.0))
        top = float(getattr(box, "top", 0.0))
        bottom = float(getattr(box, "bottom", top + 1.0))
        box_height = max(1.0, bottom - top)
        left_x, _top_y = self.coord_manager.image_to_canvas(left, top)
        right_x, bottom_y = self.coord_manager.image_to_canvas(right, bottom)
        _unused, protrusion_top_y = self.coord_manager.image_to_canvas(left, top - (box_height * 2.0))
        tolerance = max(8.0, 6.0 * self.coord_manager.image_scale_factor)
        if self._is_point_on_manual_fit_keyboard_box_edge_handle(
            canvas_x,
            canvas_y,
            edge_x=left_x,
            handle_top_y=protrusion_top_y,
            handle_bottom_y=bottom_y,
            inward_direction=1.0,
            tolerance=tolerance,
        ):
            return "left"
        if self._is_point_on_manual_fit_keyboard_box_edge_handle(
            canvas_x,
            canvas_y,
            edge_x=right_x,
            handle_top_y=protrusion_top_y,
            handle_bottom_y=bottom_y,
            inward_direction=-1.0,
            tolerance=tolerance,
        ):
            return "right"
        return None

    def _is_point_on_manual_fit_keyboard_box_edge_handle(
        self,
        canvas_x: float,
        canvas_y: float,
        *,
        edge_x: float,
        handle_top_y: float,
        handle_bottom_y: float,
        inward_direction: float,
        tolerance: float,
    ) -> bool:
        hit_top = min(handle_top_y, handle_bottom_y)
        hit_bottom = max(handle_top_y, handle_bottom_y)
        if hit_top <= canvas_y <= hit_bottom and abs(canvas_x - edge_x) <= tolerance:
            return True

        arm_length = abs(handle_bottom_y - handle_top_y) * 0.5
        arm_end_x = edge_x + (arm_length * inward_direction)
        arm_left = min(edge_x, arm_end_x) - tolerance
        arm_right = max(edge_x, arm_end_x) + tolerance
        return arm_left <= canvas_x <= arm_right and abs(canvas_y - handle_top_y) <= tolerance

    def _manual_fit_keyboard_box_edge_press_offset(self, edge: str, canvas_x: float, canvas_y: float) -> float:
        box = self._current_manual_fit_keyboard_box()
        image_pos = self.coord_manager.canvas_to_image(
            canvas_x,
            canvas_y,
            clamp_to_bounds=True,
        )
        if box is None or image_pos is None:
            return 0.0
        if edge == "left":
            edge_x = float(getattr(box, "left", 0.0))
        else:
            edge_x = float(getattr(box, "right", 0.0))
        return float(image_pos[0]) - edge_x

    def _current_manual_fit_keyboard_box(self):
        if self._get_manual_fit_keyboard_box_callback is None:
            return None
        try:
            return self._get_manual_fit_keyboard_box_callback()
        except Exception:
            self.logger.debug("Manual Fit keyboard box callback failed", exc_info=True)
            return None

    def _handle_manual_fit_local_selection_release(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            return
        end_pos = QPoint(event.x(), event.y())
        selection_rect = QRect(self._manual_fit_region_start_pos, end_pos).normalized()
        start_img = self.coord_manager.canvas_to_image(
            self._manual_fit_region_start_pos.x(),
            self._manual_fit_region_start_pos.y(),
            clamp_to_bounds=True,
        )
        end_img = self.coord_manager.canvas_to_image(
            end_pos.x(),
            end_pos.y(),
            clamp_to_bounds=True,
        )
        self._finish_manual_fit_region_selection()
        if not start_img or not end_img:
            return
        left = min(float(start_img[0]), float(end_img[0]))
        right = max(float(start_img[0]), float(end_img[0]))
        top = min(float(start_img[1]), float(end_img[1]))
        bottom = max(float(start_img[1]), float(end_img[1]))
        if right <= left or bottom <= top:
            return
        self.manual_fit_local_selection_selected.emit(left, top, right, bottom)
        self.request_repaint.emit()

    def _handle_manual_fit_line_press(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.LeftButton:
            return False
        self._manual_fit_line_dragging = True
        return True

    def _handle_manual_fit_line_release(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            return
        self._emit_manual_fit_line_y(event.x(), event.y(), completed=True)
        self._manual_fit_line_dragging = False
        self.request_repaint.emit()

    def _emit_manual_fit_line_y(self, canvas_x: float, canvas_y: float, *, completed: bool) -> None:
        image_pos = self.coord_manager.canvas_to_image(
            canvas_x,
            canvas_y,
            clamp_to_bounds=True,
        )
        if not image_pos:
            return
        line_type = "black_bottom" if self._manual_fit_mode == "manual_fit_black_bottom" else "white_start"
        if completed:
            self.manual_fit_guide_line_selected.emit(line_type, float(image_pos[1]))
        else:
            self.manual_fit_guide_line_changed.emit(line_type, float(image_pos[1]))

    def _handle_manual_fit_region_press(self, event: QMouseEvent) -> bool:
        if event.button() == Qt.RightButton:
            self._finish_manual_fit_region_selection()
            self.set_manual_fit_mode("manual_fit_group")
            return True
        if event.button() != Qt.LeftButton:
            return False
        return self._start_manual_fit_rectangle_selection(event)

    def _start_manual_fit_rectangle_selection(self, event: QMouseEvent) -> bool:
        self._manual_fit_region_selecting = True
        self._manual_fit_region_start_pos = QPoint(event.x(), event.y())
        if not self._manual_fit_region_rubber_band and isinstance(self.canvas, QWidget):
            try:
                self._manual_fit_region_rubber_band = QRubberBand(QRubberBand.Rectangle, self.canvas)
            except (TypeError, AttributeError):
                self.logger.debug("Cannot create Manual Fit QRubberBand - canvas not a valid QWidget")
        if self._manual_fit_region_rubber_band:
            self._manual_fit_region_rubber_band.setGeometry(
                QRect(self._manual_fit_region_start_pos, self._manual_fit_region_start_pos)
            )
            self._manual_fit_region_rubber_band.show()
            self._manual_fit_region_rubber_band.setAttribute(Qt.WA_TransparentForMouseEvents)
        return True

    def _handle_manual_fit_region_move(self, event: QMouseEvent) -> None:
        if not self._manual_fit_region_rubber_band:
            return
        current_pos = QPoint(event.x(), event.y())
        selection_rect = QRect(self._manual_fit_region_start_pos, current_pos).normalized()
        self._manual_fit_region_rubber_band.setGeometry(selection_rect)

    def _handle_manual_fit_region_release(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            return
        end_pos = QPoint(event.x(), event.y())
        selection_rect = QRect(self._manual_fit_region_start_pos, end_pos).normalized()
        start_img = self.coord_manager.canvas_to_image(
            selection_rect.x(),
            selection_rect.y(),
            clamp_to_bounds=True,
        )
        end_img = self.coord_manager.canvas_to_image(
            selection_rect.right(),
            selection_rect.bottom(),
            clamp_to_bounds=True,
        )
        region_type = "black" if self._manual_fit_mode == "manual_fit_black_region" else "white"
        self._finish_manual_fit_region_selection()
        self.set_manual_fit_mode("manual_fit_group")
        if not start_img or not end_img:
            return
        top = min(float(start_img[1]), float(end_img[1]))
        bottom = max(float(start_img[1]), float(end_img[1]))
        if bottom <= top:
            return
        self.manual_fit_region_selected.emit(region_type, top, bottom)
        self.request_repaint.emit()

    def _finish_manual_fit_region_selection(self) -> None:
        self._manual_fit_region_selecting = False
        if self._manual_fit_region_rubber_band:
            self._manual_fit_region_rubber_band.hide()

    def _is_point_inside_overlay_bounds(self, canvas_x: float, canvas_y: float) -> bool:
        if not self._get_overlays_callback:
            return False
        overlays = self._get_overlays_callback()
        if not overlays:
            return False

        rects = [
            self.coord_manager.image_rect_to_canvas(
                overlay.x,
                overlay.y,
                overlay.width,
                overlay.height,
            )
            for overlay in overlays
        ]
        left = min(rect[0] for rect in rects)
        top = min(rect[1] for rect in rects)
        right = max(rect[0] + rect[2] for rect in rects)
        bottom = max(rect[1] + rect[3] for rect in rects)
        return left <= canvas_x <= right and top <= canvas_y <= bottom

    def _is_point_inside_manual_fit_local_overlay(self, canvas_x: float, canvas_y: float) -> bool:
        if not self._get_overlays_callback or not self._get_manual_fit_local_key_ids_callback:
            return False
        local_key_ids = set(self._get_manual_fit_local_key_ids_callback() or set())
        if not local_key_ids:
            return False
        for overlay in self._get_overlays_callback() or []:
            if overlay.key_id not in local_key_ids:
                continue
            x1, y1, width, height = self.coord_manager.image_rect_to_canvas(
                overlay.x,
                overlay.y,
                overlay.width,
                overlay.height,
            )
            if x1 <= canvas_x <= x1 + width and y1 <= canvas_y <= y1 + height:
                return True
        return False
    
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
                
                # Emit signal with selected ROI when the active feature has a consumer.
                if self._roi_selection_type == "spark":
                    self.spark_roi_selected.emit(top_y, bottom_y)
                else:
                    self.logger.warning(
                        "%s ROI selection completed but shadow ROI persistence is not wired; selection ignored",
                        self._roi_selection_type,
                    )
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

                # Clean up selection visuals BEFORE emitting. The connected handler may open
                # a modal dialog, and synchronous signal delivery would otherwise keep the
                # rubber-band visible until that dialog closes.
                self.logger.info("Exiting keyboard region selection mode before signal emission")
                self.exit_keyboard_region_selection_mode()

                # Emit signal with selected region
                self.keyboard_region_selected.emit(x, y, width, height)
                self.logger.info("Signal emitted successfully")
            else:
                self.logger.warning("Failed to convert keyboard region selection to image coordinates")
                self.logger.warning(f"coord_manager dimensions: image={self.coord_manager.image_width}x{self.coord_manager.image_height}")

            # Ensure selection mode is exited even if coordinate conversion failed.
            if self._keyboard_region_selection_mode or self._keyboard_region_selecting:
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
