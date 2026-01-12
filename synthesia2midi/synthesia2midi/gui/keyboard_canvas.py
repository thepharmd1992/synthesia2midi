"""
PySide6 widget for displaying video frames and interactive key overlays.
"""
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image  # type: ignore
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QMouseEvent, QPaintEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QWidget

# PERFORMANCE FIX: Disable debug logging in paint methods to prevent progressive slowdown
# Paint methods are called frequently and logging causes cumulative file I/O overhead
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.core.app_state import AppState
from synthesia2midi.detection.factory import DetectionFactory
from synthesia2midi.detection.roi_utils import (
    euclidean_distance, 
    get_average_color_from_roi, 
    get_average_color_from_roi_with_offset,
    get_hist_feature, 
    hist_distance
)
from synthesia2midi.detection.detection_utils import calculate_detection_parameters, calculate_delta_value
from synthesia2midi.gui.canvas.coordinates import CoordinateManager
from synthesia2midi.gui.canvas.interaction import CanvasInteraction
from synthesia2midi.video_loader import VideoSession


class ThresholdDebugBox(QWidget):
    """Draggable debug box showing threshold parameters for the selected overlay."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(300, 240)  # Made it taller to accommodate frame info and navigation
        self.setStyleSheet("""
            background-color: rgba(0, 0, 0, 200);
            border-radius: 5px;
            color: white;
        """)
        # Initialize with default values
        self.overlay_id = None
        self.parameters = {}
        self.current_frame = 0
        # Make it draggable
        self._drag_start_position = None
        
    def set_debug_info(self, overlay_id: int, parameters: Dict[str, Any], current_frame: int = 0):
        """Update the debug information to display."""
        self.overlay_id = overlay_id
        self.parameters = parameters
        self.current_frame = current_frame
        self.update()  # Trigger repaint
        
    def paintEvent(self, event):
        """Paint the debug information."""
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set font using cross-platform helper
        from synthesia2midi.utils.font_helper import create_scaled_font
        font = create_scaled_font(size=9)
        painter.setFont(font)
        
        if self.overlay_id is None:
            painter.setPen(QColor("lightgray"))
            painter.drawText(10, 20, "No overlay selected")
            return
            
        if not self.parameters:
            painter.setPen(QColor("lightgray"))
            painter.drawText(10, 20, f"Overlay {self.overlay_id}")
            painter.drawText(10, 40, "No parameters available")
            return
            
        y_pos = 15
        line_height = 15
        
        # Title with frame info
        painter.setPen(QColor("lightgray"))
        painter.drawText(10, y_pos, f"Overlay {self.overlay_id} - Frame {self.current_frame}")
        y_pos += line_height + 5
        
        # Parameters
        for param_name, param_data in self.parameters.items():
            if param_name == 'overall_status':
                continue  # Skip overall_status, handled separately
                
            value = param_data.get('value', 0)
            threshold = param_data.get('threshold', 0)
            met_threshold = param_data.get('met_threshold', False)
            
            # Choose color based on whether threshold is met
            color = QColor("lime") if met_threshold else QColor("lightgray")
            painter.setPen(color)
            
            # Format the text
            if isinstance(value, float):
                if isinstance(threshold, (int, float)):
                    text = f"{param_name}: {value:.3f} >= {threshold:.3f}"
                else:
                    text = f"{param_name}: {value:.3f} ({threshold})"
            else:
                if isinstance(threshold, (int, float)):
                    text = f"{param_name}: {value} >= {threshold}"
                else:
                    text = f"{param_name}: {value} ({threshold})"
            
            painter.drawText(10, y_pos, text)
            y_pos += line_height
            
        # Overall status
        y_pos += 5
        overall_status = self.parameters.get('overall_status', 'OFF')
        status_color = QColor("lime") if overall_status == 'ON' else QColor("lightgray")
        painter.setPen(status_color)
        painter.drawText(10, y_pos, f"Status: {overall_status}")
        
        # Navigation instructions are shown next to the frame slider.
        
    def mousePressEvent(self, event):
        """Handle mouse press to start dragging."""
        if event.button() == Qt.LeftButton:
            self._drag_start_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
            
    def mouseMoveEvent(self, event):
        """Handle mouse move to drag the widget."""
        if event.buttons() == Qt.LeftButton and self._drag_start_position is not None:
            self.move(event.globalPos() - self._drag_start_position)
            event.accept()


class KeyboardCanvas(QWidget):
    """Manages display of video frames and draggable/resizable overlays."""

    def __init__(self, app_state: AppState, width: int, height: int,
                 on_color_pick_callback: Callable[[Tuple[int,int,int], Tuple[int,int]], None],
                 on_overlay_select_callback: Callable[[Optional[int]], None],
                 detect_pressed_func: Optional[Callable] = None): # Add detect_pressed_func
        super().__init__()
        self.app_state = app_state
        self.video_session: Optional[VideoSession] = None
        self.current_frame_rgb: Optional[np.ndarray] = None  # RGB NumPy array
        self.original_frame_size: Optional[Tuple[int, int]] = None  # Store original frame size
        self.detect_pressed_func = detect_pressed_func # Store the function
        self.last_detection_results: Set[int] = set()  # Store results from main detection run
        
        # Enable keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)

        # Frames are fetched on demand from the current video session.

        self.on_color_pick_callback = on_color_pick_callback
        self.on_overlay_select_callback = on_overlay_select_callback

        # Set widget properties
        self.setMinimumSize(width, height)
        self.setStyleSheet("background-color: black;")
        
        # Initialize coordinate management
        self.coord_manager = CoordinateManager()
        
        # Initialize interaction handling
        self.interaction = CanvasInteraction(self, self.coord_manager, app_state)
        self._setup_interaction_callbacks()
        self._connect_interaction_signals()
        
        # Enable mouse tracking to capture all mouse events
        self.setMouseTracking(True)
        
        # Display transform properties (coord_manager is the source of truth).
        self.image_x_offset = 0
        self.image_y_offset = 0
        self.image_scale_factor = 1.0
        
        # Store base pixmap without overlays for efficient partial repainting
        self._base_pixmap: Optional[QPixmap] = None
        
        # Cropping state for keyboard area optimization
        self._crop_offset_x = 0  # X offset of crop in original frame
        self._crop_offset_y = 0  # Y offset of crop in original frame
        self._is_frame_cropped = False  # Whether current frame is cropped
        
        # Advanced Visual Threshold Monitor
        self._debug_box = None
        self._debug_box_last_position = None  # Store last position when hiding

    def set_video_session(self, video_session: Optional[VideoSession]):
        self.video_session = video_session
    
    
    def _crop_frame_to_keyboard_area(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Return the frame used for detection/display.

        Cropping is currently disabled, so this returns the original frame.
        """
        self._is_frame_cropped = False
        self._crop_offset_x = 0
        self._crop_offset_y = 0
        return frame_rgb
    
    def get_processing_frame_bgr(self) -> Optional[np.ndarray]:
        """
        Return the current frame in BGR format for detection.

        Cropping hooks are present but currently return the full frame.
        """
        if self.current_frame_rgb is None:
            return None
        
        # Get cropped RGB frame
        cropped_rgb = self._crop_frame_to_keyboard_area(self.current_frame_rgb)
        
        # Convert to BGR for detection
        return cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
    
    def _adjust_overlay_for_crop(self, overlay: OverlayConfig) -> Optional[OverlayConfig]:
        """
        Returns an overlay with coordinates adjusted for the cropped frame.
        Returns None if overlay is outside the crop area.
        """
        if not self._is_frame_cropped:
            return overlay
        
        # Adjust overlay coordinates relative to crop
        adjusted_x = overlay.x - self._crop_offset_x
        adjusted_y = overlay.y - self._crop_offset_y
        
        # Check if overlay intersects with crop area
        if (adjusted_x + overlay.width <= 0 or adjusted_x >= self.current_frame_rgb.shape[1] - self._crop_offset_x or
            adjusted_y + overlay.height <= 0 or adjusted_y >= self.current_frame_rgb.shape[0] - self._crop_offset_y):
            return None  # Overlay is outside crop area
        
        # Create adjusted overlay (shallow copy with modified coordinates)
        adjusted_overlay = OverlayConfig(
            key_id=overlay.key_id,
            note_octave=overlay.note_octave,
            note_name_in_octave=overlay.note_name_in_octave,
            x=adjusted_x,
            y=adjusted_y,
            width=overlay.width,
            height=overlay.height,
            unlit_reference_color=overlay.unlit_reference_color,
            key_type=overlay.key_type,
            unlit_hist=overlay.unlit_hist,
            lit_hist=overlay.lit_hist
        )
        
        # Copy runtime state
        adjusted_overlay.prev_progression_ratio = overlay.prev_progression_ratio
        adjusted_overlay.last_progression_ratio = overlay.last_progression_ratio
        adjusted_overlay.last_is_lit = overlay.last_is_lit
        
        return adjusted_overlay

    def display_frame(self, frame_index: int) -> bool:
        """Load, display, and process a single video frame with live detection."""
        
        if not self.video_session:
            logging.warning("display_frame called with no video session.")
            return False

        # Load frame directly from video (no caching to prevent progressive slowdown)
        success, frame_bgr = self.video_session.get_frame(frame_index)
        if not success or frame_bgr is None:
            logging.warning(f"display_frame: Failed to get frame {frame_index}")
            self.current_frame_rgb = None 
            self._base_pixmap = None # Ensure base_pixmap is also cleared
            self.update() # Clear canvas
            return False
        
        # Convert BGR to RGB
        # Delete old frame data to prevent memory accumulation
        if hasattr(self, 'current_frame_rgb') and self.current_frame_rgb is not None:
            self.current_frame_rgb = None  # Release old frame
        self.current_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        self.app_state.video.current_frame_index = frame_index
        self.current_frame_index = frame_index  # Store current frame index for reference
        
        # Update coordinate transformations using CoordinateManager
        if self.original_frame_size is None: # Store original frame size if not already stored
            self.original_frame_size = (self.current_frame_rgb.shape[1], self.current_frame_rgb.shape[0])
        
        canvas_w, canvas_h = self.width(), self.height()
        img_h, img_w = self.current_frame_rgb.shape[:2]

        if img_w == 0 or img_h == 0 or canvas_w == 0 or canvas_h == 0:
            self._base_pixmap = None # Cannot create pixmap
            self.update()
            return True # Or False, debatable if this is a success

        # Update coordinate manager with current dimensions
        self.coord_manager.update_canvas_size(canvas_w, canvas_h)
        self.coord_manager.update_image_size(img_w, img_h)
        self.coord_manager.update_crop_settings(
            self._crop_offset_x, self._crop_offset_y, self._is_frame_cropped
        )
        
        # Sync compatibility properties for callers that still reference these fields.
        self.image_x_offset = self.coord_manager.image_x_offset
        self.image_y_offset = self.coord_manager.image_y_offset
        self.image_scale_factor = self.coord_manager.image_scale_factor
        
        # --- Update _base_pixmap (simplified to prevent progressive slowdown) ---
        # Create new pixmap each time to avoid Qt object reuse issues
        target_w = int(img_w * self.image_scale_factor)
        target_h = int(img_h * self.image_scale_factor)
        
        resized_rgb = cv2.resize(self.current_frame_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        if not resized_rgb.flags['C_CONTIGUOUS']:
            resized_rgb = np.ascontiguousarray(resized_rgb)
        
        r_height, r_width, _ = resized_rgb.shape
        r_bytes_per_line = 3 * r_width
        q_image_for_pixmap = QImage(resized_rgb.data, r_width, r_height, r_bytes_per_line, QImage.Format_RGB888)
        # Make a deep copy to prevent Qt from holding references to numpy data
        q_image_for_pixmap = q_image_for_pixmap.copy()
        
        # Delete old pixmap explicitly before creating new one
        if self._base_pixmap is not None:
            self._base_pixmap = None  # Release the old pixmap
        
        # Create new pixmap each time (no reuse optimization)
        self._base_pixmap = QPixmap(canvas_w, canvas_h)
        self._base_pixmap.fill(Qt.black) # Start with black background
        painter_for_pixmap = QPainter(self._base_pixmap)
        try:
            painter_for_pixmap.drawImage(self.image_x_offset, self.image_y_offset, q_image_for_pixmap)
        finally:
            painter_for_pixmap.end()
        # --- End update _base_pixmap ---

        # --- Main detection run for display (remains the same) --- 
        pressed_key_ids_for_display: Set[int] = set()
        adjusted_overlays_for_display: List[OverlayConfig] = [] # For drawing and monitor
        overlay_index_map: Dict[int, int] = {}  # Maps adjusted overlay key_id to original index
        
        if self.app_state.overlays and self.detect_pressed_func:
            # Prepare overlays, accounting for potential cropping if active
            # This adjusted list is used for both detection and drawing
            for idx, ov_cfg in enumerate(self.app_state.overlays):
                adjusted_ov = self._adjust_overlay_for_crop(ov_cfg)
                if adjusted_ov:
                    adjusted_overlays_for_display.append(adjusted_ov)
                    overlay_index_map[adjusted_ov.key_id] = idx
            
            if adjusted_overlays_for_display:
                # Use the potentially cropped frame for detection
                # frame_bgr_for_detection = self._get_frame_for_detection(frame_bgr)
                # No, detect_pressed expects original frame and does its own ROI extraction based on original coords (or adjusted if crop involved)

                # Use full detection parameters when live detection feedback is enabled
                # Otherwise use lightweight detection for performance during navigation
                use_full_detection = self.app_state.ui.live_detection_feedback
                
                pressed_key_ids_for_display = self.detect_pressed_func(
                    frame_bgr, # Pass original BGR frame
                    adjusted_overlays_for_display, # Pass overlays potentially adjusted for crop
                    self.app_state.detection.exemplar_lit_colors,
                    self.app_state.detection.exemplar_lit_histograms,
                    self.app_state.detection.detection_threshold,
                    self.app_state.detection.hist_ratio_threshold,
                    self.app_state.detection.rise_delta_threshold, 
                    self.app_state.detection.fall_delta_threshold, 
                    use_histogram_detection=self.app_state.detection.use_histogram_detection if use_full_detection else False,
                    use_delta_detection=self.app_state.detection.use_delta_detection if use_full_detection else False,
                    similarity_ratio=self.app_state.detection.similarity_ratio,
                    apply_black_filter=self.app_state.detection.winner_takes_black_enabled,
                )
                
                # Sync runtime state changes back to original overlays
                for adjusted_ov in adjusted_overlays_for_display:
                    if adjusted_ov.key_id in overlay_index_map:
                        orig_idx = overlay_index_map[adjusted_ov.key_id]
                        orig_overlay = self.app_state.overlays[orig_idx]
                        # Copy back the runtime state that detection may have modified
                        orig_overlay.prev_progression_ratio = adjusted_ov.prev_progression_ratio
                        orig_overlay.last_progression_ratio = adjusted_ov.last_progression_ratio
                        orig_overlay.last_is_lit = adjusted_ov.last_is_lit
                        orig_overlay.in_forced_delta_off_state = adjusted_ov.in_forced_delta_off_state
                
                # Store detection results for reuse in live detection overlay coloring
                self.last_detection_results = pressed_key_ids_for_display
        # --- End main detection run ---

        # Convert to PIL Image for drawing - This line is actually not used for drawing if _base_pixmap is used.
        # It was for self.current_frame_pil, which isn't directly used in paintEvent with _base_pixmap logic.
        # height, width, channel = self.current_frame_rgb.shape
        # bytes_per_line = 3 * width
        # q_image = QImage(self.current_frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        # self.current_frame_pil = Image.fromqimage(q_image) # Keep PIL version if needed elsewhere
        
        self.update() # Trigger repaint, which calls paintEvent
        
        # Handle visual threshold monitor after main detection and canvas update
        if self.app_state.ui.visual_threshold_monitor_enabled and self.app_state.ui.selected_overlay_id is not None:
            selected_overlay_config = next((o for o in adjusted_overlays_for_display 
                                            if o.key_id == self.app_state.ui.selected_overlay_id), None)
            if selected_overlay_config:
                # Determine if the selected overlay was lit in *this* detection run
                is_selected_overlay_lit = selected_overlay_config.key_id in pressed_key_ids_for_display
                # Pass this status to _get_debug_info_text
                debug_text = self._get_debug_info_text(selected_overlay_config, 
                                                       is_selected_overlay_lit_for_display=is_selected_overlay_lit)
                self._update_debug_box(debug_text, selected_overlay_config)
            else:
                self._hide_debug_box() # Selected overlay might be outside crop or not found
        else:
            self._hide_debug_box()
            
        return True
    

    def _display_image(self):
        """Displays the current frame with overlays."""
        if self.current_frame_rgb is None:
            return
            
        # Resize image to fit widget
        canvas_w, canvas_h = self.width(), self.height()
        img_h, img_w = self.current_frame_rgb.shape[:2]
        
        
        new_w = int(img_w * self.image_scale_factor)
        new_h = int(img_h * self.image_scale_factor)
        
        # Use OpenCV for resizing - it's more efficient than PIL
        if new_w != img_w or new_h != img_h:
            resized_rgb = cv2.resize(self.current_frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized_rgb = self.current_frame_rgb
        
        # Create QImage directly from NumPy array
        height, width, channel = resized_rgb.shape
        bytes_per_line = 3 * width
        # Ensure data is C-contiguous for QImage
        if not resized_rgb.flags['C_CONTIGUOUS']:
            resized_rgb = np.ascontiguousarray(resized_rgb)
        q_image = QImage(resized_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        # Make a deep copy to prevent Qt from holding references to numpy data
        q_image = q_image.copy()
        
        # Delete old pixmap explicitly before creating new one
        if self._base_pixmap is not None:
            self._base_pixmap = None  # Release the old pixmap
        
        # Create new pixmap each time (no reuse optimization to prevent progressive slowdown)
        self._base_pixmap = QPixmap(canvas_w, canvas_h)
        self._base_pixmap.fill(Qt.black)
        
        # Draw the image centered on base pixmap
        painter = QPainter(self._base_pixmap)
        try:
            painter.drawImage(self.image_x_offset, self.image_y_offset, q_image)
            
            # Draw keyboard area outline on base pixmap (since it doesn't change often)
            self._draw_keyboard_area_outline(painter)
        finally:
            painter.end()
        
        # Trigger a full repaint
        self.update()

    def draw_overlays(self):
        """Redraws only the overlay regions without recreating the base pixmap."""
        # Check if interaction module is currently dragging
        interaction_state = self.interaction.get_interaction_state()
        if not interaction_state.get("dragging", False):
            # If not dragging, do a full update
            self.update()
        else:
            # During drag, only update overlay regions
            for overlay in self.app_state.overlays:
                x1_c, y1_c, x2_c, y2_c = self._map_image_to_canvas_coords(
                    float(overlay.x), float(overlay.y), 
                    float(overlay.width), float(overlay.height)
                )
                # Add padding for overlay borders
                self.update(QRect(x1_c - 5, y1_c - 5, x2_c - x1_c + 10, y2_c - y1_c + 10))
    
    def paintEvent(self, event: QPaintEvent):
        """Custom paint event to handle partial updates efficiently."""
        
        painter = QPainter(self)
        if not painter.isActive():
            return
        
        try:
            if not self._base_pixmap:
                # If no base pixmap, just fill with black
                painter.fillRect(self.rect(), Qt.black)
            else:
                # Get the region that needs updating
                update_region = event.region()
                
                # Get the bounding rectangle of the update region
                rect = update_region.boundingRect()
                
                # Clip painting to this rectangle for efficiency
                painter.setClipRect(rect)
                
                # Draw the portion of base pixmap in this rect
                painter.drawPixmap(rect, self._base_pixmap, rect)
                
                # Draw overlays if enabled, but only those intersecting this rect
                if self.app_state.ui.show_overlays:
                    self._draw_overlays_on_painter_in_rect(painter, rect)
                
                # Draw spark ROI visualization and zones (if visible)
                if self.app_state.detection.spark_roi_visible:
                    self._draw_spark_roi_on_painter(painter)
                    self._draw_spark_zones_on_painter(painter)
                
            
            # Always draw frame number in upper-left corner (even with no video)
            # Remove clipping for frame number to ensure it's always visible
            painter.setClipRect(QRect())
            painter.setClipping(False)  # Explicitly disable clipping
            
            # Draw frame number
            self._draw_frame_number_simple(painter)
                
        except Exception as e:
            logging.error(f"Exception in paintEvent: {e}", exc_info=True)
        finally:
            painter.end()
            
        event.accept()
    
    def _draw_frame_number_simple(self, painter: QPainter):
        """Draw frame number in upper-left corner."""
        frame_text = str(self.app_state.video.current_frame_index)
        
        # Set up font for frame number
        from synthesia2midi.utils.font_helper import create_scaled_font
        font = create_scaled_font(size=24)
        font.setBold(True)
        painter.setFont(font)
        
        # Measure text dimensions
        fm = QFontMetrics(font)
        text_width = fm.horizontalAdvance(frame_text)
        text_height = fm.height()
        
        # Calculate position (upper-left with 10px margin)
        text_x = 10
        text_y = 10 + text_height
        
        # Draw black background rectangle for better visibility
        bg_rect = QRect(text_x - 5, text_y - text_height - 2, text_width + 10, text_height + 4)
        painter.fillRect(bg_rect, QColor(0, 0, 0, 200))  # Semi-transparent black
        
        # Draw white text
        painter.setPen(QColor("white"))
        painter.drawText(text_x, text_y, frame_text)
    
    def _draw_spark_roi_on_painter(self, painter: QPainter):
        """Draw spark ROI visualization overlay."""
        # Only draw if ROI is set (both values > 0)
        if (self.app_state.detection.spark_roi_top > 0 and 
            self.app_state.detection.spark_roi_bottom > 0 and
            self.app_state.detection.spark_roi_bottom > self.app_state.detection.spark_roi_top):
            
            # Convert image coordinates to canvas coordinates
            roi_top_canvas = self.coord_manager.image_rect_to_canvas(
                0, self.app_state.detection.spark_roi_top, 
                self.coord_manager.image_width, 1
            )
            roi_bottom_canvas = self.coord_manager.image_rect_to_canvas(
                0, self.app_state.detection.spark_roi_bottom,
                self.coord_manager.image_width, 1  
            )
            
            if roi_top_canvas and roi_bottom_canvas:
                # Get canvas coordinates
                x_start, y_top, canvas_width, _ = roi_top_canvas
                _, y_bottom, _, _ = roi_bottom_canvas
                
                # Use full canvas width for ROI visualization
                x_start = 0
                canvas_width = self.width()
                
                # Draw semi-transparent orange overlay for ROI region
                roi_rect = QRect(int(x_start), int(y_top), int(canvas_width), int(y_bottom - y_top))
                painter.fillRect(roi_rect, QColor(255, 165, 0, 60))  # Orange with transparency
                
                # Draw orange border lines for top and bottom
                painter.setPen(QPen(QColor(255, 165, 0), 2, Qt.SolidLine))  # Orange, 2px thick
                painter.drawLine(int(x_start), int(y_top), int(x_start + canvas_width), int(y_top))
                painter.drawLine(int(x_start), int(y_bottom), int(x_start + canvas_width), int(y_bottom))
                
                # Intentionally omit a text label to keep the ROI overlay unobtrusive.
    
    
    def _draw_spark_zones_on_painter(self, painter: QPainter):
        """Draw spark zone mappings showing key overlay to ROI connections."""
        try:
            from synthesia2midi.detection.spark_mapper import get_spark_zones
            
            # Only draw if spark ROI is set and we have overlays
            if (self.app_state.detection.spark_roi_top > 0 and 
                self.app_state.detection.spark_roi_bottom > 0 and
                self.app_state.overlays):
                
                # Get spark zones from mapper
                spark_zones = get_spark_zones(self.app_state)
                
                if spark_zones:
                    # Draw extension lines and zone outlines
                    self._draw_zone_extensions(painter, spark_zones)
                    self._draw_zone_outlines(painter, spark_zones)
                    
        except ImportError:
            # Spark mapper not available
            pass
        except Exception as e:
            # Log error but don't crash painting
            logging.warning(f"Error drawing spark zones: {e}")
    
    def _draw_zone_extensions(self, painter: QPainter, spark_zones):
        """Draw lines showing how key overlays extend to spark zones."""
        try:
            # Set up pen for extension lines (thin, dashed, cyan)
            painter.setPen(QPen(QColor(0, 255, 255, 150), 1, Qt.DashLine))  # Cyan, 1px, dashed
            
            for zone in spark_zones:
                # Find corresponding overlay
                overlay = None
                for ov in self.app_state.overlays:
                    if ov.key_id == zone.key_id:
                        overlay = ov
                        break
                
                if overlay:
                    # Convert overlay and zone coordinates to canvas
                    overlay_canvas = self.coord_manager.image_rect_to_canvas(
                        overlay.x, overlay.y, overlay.width, overlay.height
                    )
                    zone_canvas = self.coord_manager.image_rect_to_canvas(
                        zone.x, zone.y, zone.width, zone.height
                    )
                    
                    if overlay_canvas and zone_canvas:
                        # Get overlay bounds
                        ov_x, ov_y, ov_w, ov_h = overlay_canvas
                        ov_bottom = ov_y + ov_h
                        ov_center_x = ov_x + ov_w // 2
                        
                        # Get zone bounds  
                        zone_x, zone_y, zone_w, zone_h = zone_canvas
                        zone_center_x = zone_x + zone_w // 2
                        
                        # Draw extension lines from overlay bottom to zone top
                        if zone_y > ov_bottom:  # Zone is below overlay
                            # Vertical line from overlay bottom to zone top
                            painter.drawLine(int(ov_center_x), int(ov_bottom), 
                                           int(zone_center_x), int(zone_y))
                        elif zone_y + zone_h < ov_y:  # Zone is above overlay
                            # Vertical line from zone bottom to overlay top
                            painter.drawLine(int(zone_center_x), int(zone_y + zone_h),
                                           int(ov_center_x), int(ov_y))
        except Exception as e:
            pass  # Ignore drawing errors
    
    def _draw_zone_outlines(self, painter: QPainter, spark_zones):
        """Draw outlines of spark zones."""
        try:
            # Set up pen for zone outlines (thin, solid, cyan)
            painter.setPen(QPen(QColor(0, 255, 255, 200), 1, Qt.SolidLine))  # Cyan, 1px, solid
            
            for zone in spark_zones:
                # Convert zone coordinates to canvas
                zone_canvas = self.coord_manager.image_rect_to_canvas(
                    zone.x, zone.y, zone.width, zone.height
                )
                
                if zone_canvas:
                    zone_x, zone_y, zone_w, zone_h = zone_canvas
                    
                    # Draw zone rectangle outline
                    zone_rect = QRect(int(zone_x), int(zone_y), int(zone_w), int(zone_h))
                    painter.drawRect(zone_rect)
                    
                    # Draw small label with key info (if zone is big enough)
                    if zone_w > 30 and zone_h > 15:
                        painter.setPen(QColor("cyan"))
                        font = QFont()
                        font.setPointSize(16)
                        painter.setFont(font)
                        
                        # Draw key name in top-left of zone
                        text_x = int(zone_x + 2)
                        text_y = int(zone_y + 12)
                        painter.drawText(text_x, text_y, zone.note_name)
        except Exception as e:
            pass  # Ignore drawing errors
    
    def display_frame_no_detection(self, frame_index: int) -> bool:
        """Loads and displays a frame without running expensive detection algorithms."""
        if not self.video_session:
            return False

        # Load frame directly from video (no caching to prevent progressive slowdown)
        success, frame_bgr = self.video_session.get_frame(frame_index)
        if success and frame_bgr is not None:
            # Convert BGR to RGB
            self.current_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        if success and self.current_frame_rgb is not None:
            self.app_state.video.current_frame_index = frame_index
            
            # Store original frame size if not already stored
            if self.original_frame_size is None:
                self.original_frame_size = (self.current_frame_rgb.shape[1], self.current_frame_rgb.shape[0])
            
            # Get widget dimensions
            canvas_w, canvas_h = self.width(), self.height()
            img_h, img_w = self.current_frame_rgb.shape[:2]

            if img_w == 0 or img_h == 0 or canvas_w == 0 or canvas_h == 0:
                self._display_image_no_detection()
                return True

            # Update coordinate manager with current dimensions
            self.coord_manager.update_canvas_size(canvas_w, canvas_h)
            self.coord_manager.update_image_size(img_w, img_h)
            self.coord_manager.update_crop_settings(
                self._crop_offset_x, self._crop_offset_y, self._is_frame_cropped
            )
            
            # Sync compatibility properties for callers that still reference these fields.
            self.image_x_offset = self.coord_manager.image_x_offset
            self.image_y_offset = self.coord_manager.image_y_offset
            self.image_scale_factor = self.coord_manager.image_scale_factor


            self._display_image_no_detection()
            self._update_debug_box()  # Update debug box when frame changes
            return True
        return False
    
    def _display_image_no_detection(self):
        """Displays the current frame with overlays but without running detection."""
        # Always update the base image when frame content changes
        # The optimization to avoid recreation during drag is handled elsewhere
        self._display_image()
    
    def start_keyboard_area_selection(self):
        """Compatibility no-op: keyboard area selection is not supported."""
        logging.warning("Keyboard area selection is not supported.")
    
    def stop_keyboard_area_selection(self):
        """Compatibility no-op: keyboard area selection is not supported."""
        logging.warning("Keyboard area selection is not supported.")
    
    def _draw_keyboard_area_outline(self, painter: QPainter):
        """Compatibility no-op: keyboard area outline is not used."""
        return
    
    # Keyboard area selection helpers are not used (kept for compatibility).
    
    def _draw_overlays_no_detection(self, painter: QPainter):
        """Draws overlays without running expensive detection algorithms."""
        for overlay in self.app_state.overlays:
            # Map overlay image coordinates to canvas coordinates
            x1_c, y1_c, x2_c, y2_c = self._map_image_to_canvas_coords(
                float(overlay.x), float(overlay.y), 
                float(overlay.width), 
                float(overlay.height)
            )
            
            # Determine color and pen width (without live detection highlighting)
            # Check overlay type first for special colors
            if hasattr(overlay, 'overlay_type'):
                if overlay.overlay_type == 'spark':
                    overlay_color = "cyan"
                else:
                    overlay_color = self.app_state.ui.overlay_color if hasattr(self.app_state.ui, 'overlay_color') else "red"
            else:
                overlay_color = self.app_state.ui.overlay_color if hasattr(self.app_state.ui, 'overlay_color') else "red"
            
            color = QColor(overlay_color)
            pen_width = 2
            if self.app_state.ui.selected_overlay_id == overlay.key_id:
                color = QColor("blue")
            
            # Draw rectangle
            pen = QPen(color, pen_width)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(x1_c, y1_c, x2_c - x1_c, y2_c - y1_c)
            
            # Add note label
            # Apply the current octave transpose for display
            note_name = overlay.get_full_note_name(self.app_state.midi.octave_transpose)
            # Calculate center position for text
            text_x = (x1_c + x2_c) // 2
            text_y = (y1_c + y2_c) // 2
            
            # Measure text to center it properly
            from PySide6.QtGui import QFontMetrics
            from synthesia2midi.utils.font_helper import create_scaled_font
            font = create_scaled_font(size=10)
            painter.setFont(font)
            fm = QFontMetrics(font)
            text_width = fm.horizontalAdvance(note_name)
            text_height = fm.height()
            
            # Draw text centered with same color as overlay
            text_color = QColor(overlay_color) if self.app_state.ui.selected_overlay_id != overlay.key_id else QColor("blue")
            painter.setPen(text_color)
            painter.drawText(QPoint(text_x - text_width // 2, text_y + text_height // 4), note_name)

    def _draw_overlays_on_painter(self, painter: QPainter):
        """Draws all configured overlays on the painter."""
        # For live feedback, use the stored detection results from the main detection run
        pressed_key_ids_live = set()
        if self.app_state.ui.live_detection_feedback:
            # Reuse detection results from main detection run instead of running detection again
            pressed_key_ids_live = self.last_detection_results.copy()

        for overlay in self.app_state.overlays:
            # Map overlay image coordinates to canvas coordinates
            x1_c, y1_c, x2_c, y2_c = self._map_image_to_canvas_coords(
                float(overlay.x), float(overlay.y), 
                float(overlay.width), 
                float(overlay.height)
            )
            
            # Determine color and pen width - green (detection) takes priority over blue (selection)
            # Check overlay type first for special colors
            if hasattr(overlay, 'overlay_type'):
                if overlay.overlay_type == 'spark':
                    overlay_color = "cyan"
                else:
                    overlay_color = self.app_state.ui.overlay_color if hasattr(self.app_state.ui, 'overlay_color') else "red"
            else:
                overlay_color = self.app_state.ui.overlay_color if hasattr(self.app_state.ui, 'overlay_color') else "red"
            
            color = QColor(overlay_color)
            pen_width = 2
            if self.app_state.ui.live_detection_feedback and overlay.key_id in pressed_key_ids_live:
                color = QColor("green") # Highlight color for live detected keys (highest priority)
                pen_width = 6  # 3x thickness for active keys
            elif self.app_state.ui.selected_overlay_id == overlay.key_id:
                color = QColor("blue")
            else:
                pass  # Keep overlay_color
            
            # Draw normal rectangle for overlays
            pen = QPen(color, pen_width)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(x1_c, y1_c, x2_c - x1_c, y2_c - y1_c)
            
            # Add note label
            note_name = overlay.get_full_note_name(self.app_state.midi.octave_transpose)
            # Calculate center position for text
            text_x = (x1_c + x2_c) // 2
            text_y = (y1_c + y2_c) // 2
            
            # Measure text to center it properly
            from PySide6.QtGui import QFontMetrics
            from synthesia2midi.utils.font_helper import create_scaled_font
            font = create_scaled_font(size=10)
            painter.setFont(font)
            fm = QFontMetrics(font)
            text_width = fm.horizontalAdvance(note_name)
            text_height = fm.height()
            
            # Draw text centered with same color as overlay
            text_color = QColor(overlay_color) if self.app_state.ui.selected_overlay_id != overlay.key_id else QColor("blue")
            painter.setPen(text_color)
            painter.drawText(QPoint(text_x - text_width // 2, text_y + text_height // 4), note_name)
    
    def _draw_overlays_on_painter_in_rect(self, painter: QPainter, clip_rect: QRect):
        """Draws only overlays that intersect with the given clip rectangle."""
        # For live feedback, use the stored detection results from the main detection run
        pressed_key_ids_live = set()
        if self.app_state.ui.live_detection_feedback:
            # Reuse detection results from main detection run instead of running detection again
            pressed_key_ids_live = self.last_detection_results.copy()

        # Log critical info about overlays
        overlays_drawn = 0
        overlays_skipped = 0
        
        # Log first overlay position for debugging
        if self.app_state.overlays:
            first_overlay = self.app_state.overlays[0]
            x1_c, y1_c, x2_c, y2_c = self._map_image_to_canvas_coords(
                float(first_overlay.x), float(first_overlay.y), 
                float(first_overlay.width), float(first_overlay.height)
            )
        
        for overlay in self.app_state.overlays:
            # Map overlay image coordinates to canvas coordinates
            x1_c, y1_c, x2_c, y2_c = self._map_image_to_canvas_coords(
                float(overlay.x), float(overlay.y), 
                float(overlay.width), 
                float(overlay.height)
            )
            
            # Check if overlay intersects with clip rectangle
            overlay_rect = QRect(x1_c, y1_c, x2_c - x1_c, y2_c - y1_c)
            if not overlay_rect.intersects(clip_rect):
                overlays_skipped += 1
                continue  # Skip overlays outside the update region
            
            overlays_drawn += 1
            
            # Determine color and pen width - green (detection) takes priority over blue (selection)
            # Check overlay type first for special colors
            if hasattr(overlay, 'overlay_type'):
                if overlay.overlay_type == 'spark':
                    overlay_color = "cyan"
                else:
                    overlay_color = self.app_state.ui.overlay_color if hasattr(self.app_state.ui, 'overlay_color') else "red"
            else:
                overlay_color = self.app_state.ui.overlay_color if hasattr(self.app_state.ui, 'overlay_color') else "red"
            
            color = QColor(overlay_color)
            pen_width = 2
            if self.app_state.ui.live_detection_feedback and overlay.key_id in pressed_key_ids_live:
                color = QColor("green") # Highlight color for live detected keys (highest priority)
                pen_width = 6  # 3x thickness for active keys
            elif self.app_state.ui.selected_overlay_id == overlay.key_id:
                color = QColor("blue")
            else:
                pass  # Keep overlay_color
            
            # Draw normal rectangle for overlays
            pen = QPen(color, pen_width)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(x1_c, y1_c, x2_c - x1_c, y2_c - y1_c)
            
            # Add note label
            note_name = overlay.get_full_note_name(self.app_state.midi.octave_transpose)
            text_x = (x1_c + x2_c) // 2
            text_y = (y1_c + y2_c) // 2
            
            # Measure text to center it properly
            from PySide6.QtGui import QFontMetrics
            from synthesia2midi.utils.font_helper import create_scaled_font
            font = create_scaled_font(size=10)
            painter.setFont(font)
            fm = QFontMetrics(font)
            text_width = fm.horizontalAdvance(note_name)
            text_height = fm.height()
            
            # Draw text centered with same color as overlay
            text_color = QColor(overlay_color) if self.app_state.ui.selected_overlay_id != overlay.key_id else QColor("blue")
            painter.setPen(text_color)
            painter.drawText(QPoint(text_x - text_width // 2, text_y + text_height // 4), note_name)
        
        # Previously logged debug info about overlays drawn/skipped

    def _map_image_to_canvas_coords(self, img_x: float, img_y: float, img_width: float, img_height: float) -> Tuple[int, int, int, int]:
        """Maps image coordinates to canvas coordinates."""
        canvas_x, canvas_y, canvas_w, canvas_h = self.coord_manager.image_rect_to_canvas(
            img_x, img_y, img_width, img_height
        )
        # Return format: (x1, y1, x2, y2)
        x1, y1 = int(canvas_x), int(canvas_y)
        x2, y2 = int(canvas_x + canvas_w), int(canvas_y + canvas_h)
        
        return (x1, y1, x2, y2)

    def get_roi_bgr(self, overlay: OverlayConfig) -> Optional[np.ndarray]:
        """Extracts the BGR ROI for the given overlay from the current frame."""
        if self.current_frame_rgb is None:
            return None

        # Convert current_frame_rgb to BGR
        frame_bgr = cv2.cvtColor(self.current_frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Determine the correct frame and overlay coordinates to use based on cropping
        if self._is_frame_cropped:
            # Adjust overlay coordinates relative to crop for ROI extraction
            # These offsets are from the original full frame to the top-left of the crop area
            # The overlay_config coordinates are also relative to the original full frame
            
            # Calculate ROI within the original frame based on overlay, then adjust for crop offsets
            x_orig, y_orig = int(overlay.x), int(overlay.y)
            w_orig, h_orig = int(overlay.width), int(overlay.height)

            # ROI coordinates relative to the cropped frame
            x_roi_in_crop = x_orig - self._crop_offset_x
            y_roi_in_crop = y_orig - self._crop_offset_y
            
            # Get the cropped frame
            cropped_frame_bgr = frame_bgr[
                self._crop_offset_y : self._crop_offset_y + self._cropped_frame_height,
                self._crop_offset_x : self._crop_offset_x + self._cropped_frame_width
            ]
            
            # Define ROI boundaries within the cropped frame
            img_h_crop, img_w_crop = cropped_frame_bgr.shape[:2]
            x1, y1 = max(0, x_roi_in_crop), max(0, y_roi_in_crop)
            x2, y2 = min(img_w_crop, x_roi_in_crop + w_orig), min(img_h_crop, y_roi_in_crop + h_orig)

            if x1 >= x2 or y1 >= y2:
                return None
            return cropped_frame_bgr[y1:y2, x1:x2]
        else:
            # Use full frame and original overlay coordinates
            x, y, w, h = int(overlay.x), int(overlay.y), int(overlay.width), int(overlay.height)
            img_h, img_w = frame_bgr.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + w), min(img_h, y + h)

            if x1 >= x2 or y1 >= y2:
                return None
            return frame_bgr[y1:y2, x1:x2]

    def get_average_color_for_overlay(self, frame_rgb: np.ndarray, overlay_config: OverlayConfig) -> Optional[Tuple[int, int, int]]:
        """
        Calculates average RGB color for the given overlay from the provided frame_rgb.
        Adjusts for cropping if keyboard area is defined.
        """
        if frame_rgb is None:
            return None

        # OverlayConfig coordinates (x, y, width, height) are in original image space.
        x_orig_overlay, y_orig_overlay = int(overlay_config.x), int(overlay_config.y)
        w_overlay, h_overlay = int(overlay_config.width), int(overlay_config.height)

        # Determine the source frame and ROI coordinates based on cropping
        source_frame_for_roi = frame_rgb
        roi_x, roi_y, roi_w, roi_h = x_orig_overlay, y_orig_overlay, w_overlay, h_overlay

        if self._is_frame_cropped:
            # The frame_rgb provided to this function might be the *full* original frame,
            # or a pre-cropped one if called from somewhere else.
            # For consistency and to correctly use crop_offsets, we should work with
            # coordinates relative to the full frame if the input frame_rgb is the full one.
            # However, the current structure of _is_frame_cropped implies self.current_frame_rgb
            # is the one being referred to for cropping.
            # For this function, if _is_frame_cropped is true, we assume frame_rgb IS the *full* frame.
            
            # Adjust ROI coordinates for the crop area
            # These (roi_x, roi_y) will be relative to the top-left of the cropped area
            roi_x = x_orig_overlay - self._crop_offset_x
            roi_y = y_orig_overlay - self._crop_offset_y
            
            # The source_frame_for_roi becomes the cropped portion of the input frame_rgb
            source_frame_for_roi = frame_rgb[
                self._crop_offset_y : self._crop_offset_y + self._cropped_frame_height,
                self._crop_offset_x : self._crop_offset_x + self._cropped_frame_width
            ]
            # Ensure ROI dimensions are clamped to the cropped frame's dimensions
            # (roi_w, roi_h) remain overlay's original width/height, slicing handles boundaries.


        # Now, extract the ROI from the (potentially cropped) source_frame_for_roi
        # The roi_x, roi_y are relative to source_frame_for_roi if cropped,
        # or relative to frame_rgb if not cropped.
        
        img_h, img_w = source_frame_for_roi.shape[:2]
        
        # Define slice boundaries, clamping to the dimensions of source_frame_for_roi
        x1 = max(0, roi_x)
        y1 = max(0, roi_y)
        x2 = min(img_w, roi_x + roi_w) # roi_w is overlay's width
        y2 = min(img_h, roi_y + roi_h) # roi_h is overlay's height

        if x1 >= x2 or y1 >= y2:
            return None

        roi_rgb_values = source_frame_for_roi[y1:y2, x1:x2]

        if roi_rgb_values.size == 0:
            return None

        # Calculate average color of the ROI (RGB)
        avg_rgb_color = np.mean(roi_rgb_values, axis=(0, 1))
        return int(avg_rgb_color[0]), int(avg_rgb_color[1]), int(avg_rgb_color[2])

    def _get_progression_ratio(self, frame_rgb: np.ndarray, overlay: OverlayConfig) -> float:
        """
        Calculates the progression ratio for a given overlay on a specific frame_rgb.
        Uses shared detection logic to ensure consistency with StandardDetection.
        Returns 0.0 if calculation is not possible.
        """
        if frame_rgb is None or overlay.unlit_reference_color is None or overlay.key_type is None:
            return 0.0

        current_avg_rgb_color = self.get_average_color_for_overlay(frame_rgb, overlay)
        if current_avg_rgb_color is None:
            return 0.0

        # Use shared detection utility for consistent calculations
        detection_params = calculate_detection_parameters(
            overlay=overlay,
            current_avg_rgb_color=current_avg_rgb_color,
            unlit_ref_color=overlay.unlit_reference_color,
            exemplar_lit_colors=self.app_state.detection.exemplar_lit_colors,
            detection_threshold=self.app_state.detection.detection_threshold,
            hist_ratio_threshold=self.app_state.detection.hist_ratio_threshold,
            use_histogram_detection=False,  # Simple progression ratio doesn't use histogram
            hist_rule_hit=False,
            allow_delta_override_sanity=self.app_state.detection.use_delta_detection  # Match main detection behavior
        )
        
        return detection_params['current_max_progression_ratio']

    def mousePressEvent(self, event: QMouseEvent):
        """Delegate mouse press events to interaction module."""
        if event.button() == Qt.LeftButton:
            handled = self.interaction.handle_mouse_press(event)
            if not handled:
                super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Delegate mouse move events to interaction module."""
        if event.buttons() & Qt.LeftButton:
            handled = self.interaction.handle_mouse_move(event)
            if not handled:
                super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Delegate mouse release events to interaction module."""
        if event.button() == Qt.LeftButton:
            handled = self.interaction.handle_mouse_release(event)
            if not handled:
                super().mouseReleaseEvent(event)
        else:
            super().mouseReleaseEvent(event)
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        # Pass to parent for default handling
        super().keyPressEvent(event)
    
    def wheelEvent(self, event):
        """Mouse wheel event.

        Zoom is not supported; forward the event to the parent implementation.
        """
        super().wheelEvent(event)

    def update_overlay_display(self):
        """Force redraw of overlays, e.g., after external modification."""
        self.display_frame(self.app_state.video.current_frame_index)
        
    def _calculate_threshold_parameters(self, overlay: OverlayConfig, 
                                        is_lit_from_prior_detection: Optional[bool] = None) -> Dict[str, Any]:
        """
        Calculates various detection parameters for a given overlay on the current frame.
        Uses shared detection logic to ensure consistency with StandardDetection.
        If is_lit_from_prior_detection is provided, it uses that for overall_status,
        otherwise it runs detection.
        """
        # Initialize parameters with default values
        parameters = {
            "Progression Ratio": {"value": 0.0, "threshold": self.app_state.detection.detection_threshold, "met_threshold": False},
            "Sanity Check": {"value": 0.0, "threshold": self.app_state.detection.detection_threshold * 0.3, "met_threshold": False},
            "Histogram": {"value": False, "threshold": self.app_state.detection.hist_ratio_threshold, "met_threshold": False},
            "Base Lit": {"value": False, "threshold": "Combined", "met_threshold": False},
            "Delta": {"value": 0.0, "threshold": "Variable", "met_threshold": False},
            "overall_status": "OFF"
        }
        
        if not overlay or overlay.unlit_reference_color is None or overlay.key_type is None:
            return parameters

        unlit_ref_color = overlay.unlit_reference_color
        
        # Ensure current_frame_rgb is available
        if self.current_frame_rgb is None:
            logging.warning("_calculate_threshold_parameters: current_frame_rgb is None. Cannot get average color.")
            parameters["overall_status"] = "N/A (No Frame)"
            return parameters
            
        current_avg_rgb_color = self.get_average_color_for_overlay(self.current_frame_rgb, overlay)

        if current_avg_rgb_color is None:
            parameters["overall_status"] = "N/A (No Color)"
            return parameters

        # TODO: Implement histogram detection for visual monitor
        # For now, histogram is always False until properly implemented
        hist_rule_hit = False
        
        # Use shared detection utility for consistent calculations
        detection_params = calculate_detection_parameters(
            overlay=overlay,
            current_avg_rgb_color=current_avg_rgb_color,
            unlit_ref_color=unlit_ref_color,
            exemplar_lit_colors=self.app_state.detection.exemplar_lit_colors,
            detection_threshold=self.app_state.detection.detection_threshold,
            hist_ratio_threshold=self.app_state.detection.hist_ratio_threshold,
            use_histogram_detection=self.app_state.detection.use_histogram_detection,
            hist_rule_hit=hist_rule_hit,
            allow_delta_override_sanity=self.app_state.detection.use_delta_detection  # Match main detection behavior
        )
        
        # Update parameters with values from shared calculation
        parameters["Progression Ratio"]["value"] = detection_params['current_max_progression_ratio']
        parameters["Progression Ratio"]["met_threshold"] = detection_params['is_key_lit_by_color']
        
        parameters["Sanity Check"]["value"] = detection_params['current_max_progression_ratio']
        parameters["Sanity Check"]["threshold"] = detection_params['min_sanity_threshold']
        parameters["Sanity Check"]["met_threshold"] = detection_params['progression_passes_sanity']
        
        parameters["Histogram"]["value"] = hist_rule_hit
        parameters["Histogram"]["met_threshold"] = hist_rule_hit
        
        parameters["Base Lit"]["value"] = detection_params['base_lit']
        parameters["Base Lit"]["met_threshold"] = detection_params['base_lit']

        # Delta calculation
        delta = calculate_delta_value(
            current_progression_ratio=detection_params['current_max_progression_ratio'],
            prev_progression_ratio=overlay.prev_progression_ratio
        )
        parameters["Delta"]["value"] = delta
        
        # Delta threshold calculation (matching StandardDetection adaptive logic)
        if self.app_state.detection.use_delta_detection:
            # Apply same adaptive threshold logic as StandardDetection (lines 500-511)
            current_progression_ratio = detection_params['current_max_progression_ratio']
            
            if current_progression_ratio < self.app_state.detection.detection_threshold * 0.8:
                # Low progression ratio - require much higher delta to filter noise/spill
                required_rise_delta = self.app_state.detection.rise_delta_threshold * 3.0
                required_fall_delta = self.app_state.detection.fall_delta_threshold * 2.0
            else:
                # High progression ratio - use standard deltas
                required_rise_delta = self.app_state.detection.rise_delta_threshold
                required_fall_delta = self.app_state.detection.fall_delta_threshold
            
            # Convert thresholds to internal scale (divide by 10 for backward compatibility)
            rise_threshold_scaled = required_rise_delta / 10.0
            fall_threshold_scaled = required_fall_delta / 10.0
            
            # Display and check appropriate threshold based on delta direction
            if delta > 0:
                parameters["Delta"]["threshold"] = rise_threshold_scaled
                parameters["Delta"]["met_threshold"] = delta >= rise_threshold_scaled
            elif delta < 0:
                parameters["Delta"]["threshold"] = -fall_threshold_scaled
                parameters["Delta"]["met_threshold"] = delta <= -fall_threshold_scaled
            else:
                parameters["Delta"]["threshold"] = 0.0
                parameters["Delta"]["met_threshold"] = False
        else:
            # Delta detection disabled
            parameters["Delta"]["threshold"] = None
            parameters["Delta"]["met_threshold"] = None

        # Determine overall_status using delta detection state machine logic (matching StandardDetection)
        if is_lit_from_prior_detection is not None:
            # Apply delta detection state machine if enabled
            if self.app_state.detection.use_delta_detection:
                # Use the thresholds already calculated above
                delta_allows_on = False
                delta_forces_off = False
                
                if delta > 0:
                    # Positive delta - check if it meets the rise threshold
                    delta_allows_on = parameters["Delta"]["met_threshold"]
                elif delta < 0:
                    # Negative delta - check if it meets the fall threshold  
                    delta_forces_off = parameters["Delta"]["met_threshold"]
                
                # Apply state machine logic matching StandardDetection lines 227-242
                if delta_forces_off:
                    # Delta forces off regardless of base detection
                    final_lit_state = False
                elif not overlay.last_is_lit:
                    # Was off - need delta approval AND base detection to turn on
                    final_lit_state = detection_params['base_lit'] and delta_allows_on
                else:
                    # Was on - stay on unless delta forces off (already checked above)
                    final_lit_state = detection_params['base_lit']
                
                parameters["overall_status"] = "ON" if final_lit_state else "OFF"
            else:
                # No delta detection - use detection result directly
                parameters["overall_status"] = "ON" if is_lit_from_prior_detection else "OFF"
        else:
            # Fallback: Run full detection if status isn't provided
            # This part ensures _calculate_threshold_parameters can still work standalone if needed,
            # but ideally it gets the status from display_frame's detection run.
            if self.current_frame_rgb is not None and self.app_state.overlays and self.detect_pressed_func:
                frame_bgr = cv2.cvtColor(self.current_frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Need to handle potential cropping for accurate detection context
                # We are checking a single overlay, so we create a list with one adjusted overlay
                single_adjusted_overlay_list = []
                adj_ov = self._adjust_overlay_for_crop(overlay)
                if adj_ov:
                    single_adjusted_overlay_list.append(adj_ov)

                if single_adjusted_overlay_list:
                    # Create a temporary list of all overlays for context, but only the selected one is truly adjusted here
                    # for the purpose of this specific parameter calculation if we run detection.
                    # The detect_pressed_func expects a list of all overlays.
                    # We must be careful here: if we pass only one overlay, filters like black key spill won't work.
                    # This path (re-running detection) should be the exception for the monitor.
                    
                    # For accurate status if re-detecting, we need all overlays adjusted for crop
                    all_adjusted_overlays_for_fallback_detection = []
                    for ov_item in self.app_state.overlays:
                        adj_ov_item = self._adjust_overlay_for_crop(ov_item)
                        if adj_ov_item:
                            all_adjusted_overlays_for_fallback_detection.append(adj_ov_item)
                            
                    if all_adjusted_overlays_for_fallback_detection:
                        # Use actual detection settings when running threshold monitor
                        pressed_key_ids = self.detect_pressed_func(
                            frame_bgr, # This should be the potentially cropped frame if crop is active
                            all_adjusted_overlays_for_fallback_detection,
                            self.app_state.detection.exemplar_lit_colors,
                            self.app_state.detection.exemplar_lit_histograms,
                            self.app_state.detection.detection_threshold,
                            self.app_state.detection.hist_ratio_threshold,
                            self.app_state.detection.rise_delta_threshold,
                            self.app_state.detection.fall_delta_threshold,
                            use_histogram_detection=self.app_state.detection.use_histogram_detection,
                            use_delta_detection=self.app_state.detection.use_delta_detection,
                            similarity_ratio=self.app_state.detection.similarity_ratio,
                            apply_black_filter=self.app_state.detection.winner_takes_black_enabled,
                                )
                        lit_now_fallback = overlay.key_id in pressed_key_ids
                        parameters["overall_status"] = "ON" if lit_now_fallback else "OFF"
                    else: # No overlays after adjustment
                         parameters["overall_status"] = "OFF" # Default
                else: # Selected overlay is outside crop
                    parameters["overall_status"] = "OFF" # Default
            else:
                # Fallback to simpler color/hist check if full detection can't run
                # Note: Histogram logic not fully re-implemented here for brevity as it's complex.
                # This path is highly unlikely if called from display_frame.
                temp_lit_now = is_key_lit_by_color # or hist_rule_hit (hist part omitted for this simplified fallback)
                parameters["overall_status"] = "ON" if temp_lit_now else "OFF"
        
                
        return parameters
    
        
    def _update_debug_box(self, debug_text: Optional[str] = None, overlay: Optional[OverlayConfig] = None):
        """Update the debug box with current threshold parameters."""
        
        if not self.app_state.ui.visual_threshold_monitor_enabled:
            self._hide_debug_box()
            return

        # If overlay is not provided, try to get it from the selected_overlay_id
        if overlay is None:
            if self.app_state.ui.selected_overlay_id is not None:
                overlay = next((o for o in self.app_state.overlays if o.key_id == self.app_state.ui.selected_overlay_id), None)
            
        if overlay is None: # Still no overlay (none selected or selection is invalid)
            self._hide_debug_box()
            return

        # If debug_text is not provided, generate it using the determined overlay
        if debug_text is None:
            # Use the overlay's last_is_lit state as a proxy for its current display status
            # This is primarily for calls from mouse events where a full re-detection isn't performed.
            is_lit_for_monitor = overlay.last_is_lit 
            debug_text = self._get_debug_info_text(overlay, 
                                                   is_selected_overlay_lit_for_display=is_lit_for_monitor)
            
        # Create debug box if it doesn't exist
        if self._debug_box is None:
            self._debug_box = ThresholdDebugBox(self)
            
            # Determine position for the debug box
            if self._debug_box_last_position is not None:
                # Use last saved position, but ensure it's within canvas bounds
                pos = self._debug_box_last_position
                
                # Get canvas bounds
                canvas_width = self.width()
                canvas_height = self.height()
                box_width = self._debug_box.width()
                box_height = self._debug_box.height()
                
                # Clamp position to ensure box is at least partially visible
                margin = 50  # Minimum pixels of box that must be visible
                max_x = canvas_width - margin
                max_y = canvas_height - margin
                min_x = margin - box_width
                min_y = margin - box_height
                
                # Clamp the position
                x = max(min_x, min(pos.x(), max_x))
                y = max(min_y, min(pos.y(), max_y))
                
                self._debug_box.move(x, y)
            else:
                # Default position for first time
                self._debug_box.move(10, 10)
                
            self._debug_box.setVisible(True)
            
        # Update the ThresholdDebugBox internal data and then set its text content
        # _calculate_threshold_parameters is called inside set_debug_info if needed by ThresholdDebugBox's paintEvent
        # However, ThresholdDebugBox.set_debug_info itself expects the parameters directly to avoid re-calculation.
        # Let's ensure parameters are correctly passed here.
        
        # Recalculate params for the ThresholdDebugBox based on the overlay and its determined lit status for consistency
        is_lit_status_for_params = overlay.last_is_lit
        if 'Status: ON' in debug_text: # Infer from debug_text if possible
            is_lit_status_for_params = True
        elif 'Status: OFF' in debug_text:
            is_lit_status_for_params = False
            
        calculated_params = self._calculate_threshold_parameters(overlay, is_lit_from_prior_detection=is_lit_status_for_params)

        self._debug_box.set_debug_info(overlay.key_id, calculated_params, self.app_state.video.current_frame_index)
        # The debug_text from _get_debug_info_text is already formatted, so we can just set it if ThresholdDebugBox uses a simple QLabel
        # If ThresholdDebugBox does its own complex painting based on parameters, then set_debug_info is primary.
        # Based on ThresholdDebugBox.paintEvent, it uses self.parameters. So set_debug_info is primary.
        # The set_text in the original _update_debug_box of the widget was for the QWidget's own text, not the child debug box directly.
        # The ThresholdDebugBox itself has a paintEvent that uses the parameters set by set_debug_info.

        # The debug_text generated by _get_debug_info_text is what should be displayed.
        # The ThresholdDebugBox has its own paintEvent. Let's ensure its internal text variable gets set for simple text display if it has one,
        # or rely on its paintEvent drawing from self.parameters.
        # The current ThresholdDebugBox directly paints from self.parameters, so set_debug_info is the key.

        self._debug_box.show()
        self._debug_box.raise_()  # Bring to front
        
        self._debug_box.update() # Force immediate repaint of the debug box itself

    def _get_debug_info_text(self, overlay: OverlayConfig, 
                               is_selected_overlay_lit_for_display: Optional[bool] = None) -> str:
        """Generate debug information text for the overlay's threshold monitor box."""
        if not overlay or self.current_frame_rgb is None:
            return "No overlay or frame data."

        # Use the current_video_frame_index from app_state for the title
        frame_title_info = f" - Frame {self.app_state.video.current_frame_index}"
        title = f"Overlay {overlay.key_id}{frame_title_info}"

        # Pass the lit status from the main display_frame detection run
        params = self._calculate_threshold_parameters(overlay, is_lit_from_prior_detection=is_selected_overlay_lit_for_display)

        lines = [title]
        for name, data in params.items():
            if isinstance(data, dict): # Expected format: {"value": X, "threshold": Y, "met_threshold": Z}
                value_str = f"{data.get('value', 'N/A'):.3f}" if isinstance(data.get('value'), float) else str(data.get('value', 'N/A'))
                thresh_str = str(data.get('threshold', 'N/A'))
                met_str = "MET" if data.get('met_threshold', False) else "NOT MET"
                if name == "overall_status": # Special handling for overall_status
                    lines.append(f"Status: {value_str}") # value_str will be ON or OFF
                else:
                    lines.append(f"{name}: {value_str} (Th: {thresh_str}) [{met_str}]")
            else: # Fallback for unexpected format
                lines.append(f"{name}: {data}")
        
        return "\n".join(lines)

    def _hide_debug_box(self):
        """Hide and delete the debug box."""
        if self._debug_box:
            # Save the current position before deleting
            self._debug_box_last_position = self._debug_box.pos()
            self._debug_box.hide()
            self._debug_box.deleteLater()  # Properly delete the widget
            self._debug_box = None
    
    def resizeEvent(self, event):
        """Handle widget resize events to update coordinate transformations."""
        super().resizeEvent(event)
        
        # Update coordinate manager with new canvas size
        new_size = event.size()
        self.coord_manager.update_canvas_size(new_size.width(), new_size.height())
        
        # Sync compatibility properties for callers that still reference these fields.
        self.image_x_offset = self.coord_manager.image_x_offset
        self.image_y_offset = self.coord_manager.image_y_offset
        self.image_scale_factor = self.coord_manager.image_scale_factor
        
        # Update debug box position if visible, ensuring it stays within bounds
        if self._debug_box and self._debug_box.isVisible():
            # Check if debug box is still within bounds after resize
            canvas_width = new_size.width()
            canvas_height = new_size.height()
            box_width = self._debug_box.width()
            box_height = self._debug_box.height()
            current_pos = self._debug_box.pos()
            
            # Clamp position to ensure box remains at least partially visible
            margin = 50  # Minimum pixels of box that must be visible
            max_x = canvas_width - margin
            max_y = canvas_height - margin
            min_x = margin - box_width
            min_y = margin - box_height
            
            # Check if we need to reposition
            x = current_pos.x()
            y = current_pos.y()
            needs_reposition = False
            
            if x > max_x:
                x = max_x
                needs_reposition = True
            elif x < min_x:
                x = min_x
                needs_reposition = True
                
            if y > max_y:
                y = max_y
                needs_reposition = True
            elif y < min_y:
                y = min_y
                needs_reposition = True
                
            if needs_reposition:
                self._debug_box.move(x, y)
                
        self._update_debug_box()
    
    def _setup_interaction_callbacks(self):
        """Setup callback functions for interaction module to access canvas state."""
        self.interaction.set_callbacks(
            get_overlays=lambda: self.app_state.overlays,
            get_pixel_color=self._get_pixel_color_at_position,
            get_current_frame=lambda: self.current_frame_rgb
        )
    
    def _connect_interaction_signals(self):
        """Connect interaction signals to canvas methods."""
        self.interaction.overlay_selected.connect(self._handle_overlay_selected)
        self.interaction.overlay_moved.connect(self._handle_overlay_moved)
        self.interaction.overlay_resized.connect(self._handle_overlay_resized)
        self.interaction.color_picked.connect(self._handle_color_picked)
        self.interaction.request_repaint.connect(self._handle_repaint_request)
        self.interaction.spark_roi_selected.connect(self._handle_spark_roi_selected)
    
    def cleanup(self):
        """Clean up resources and disconnect signals before deletion."""
        # Disconnect all interaction signals
        try:
            self.interaction.overlay_selected.disconnect()
            self.interaction.overlay_moved.disconnect()
            self.interaction.overlay_resized.disconnect()
            self.interaction.color_picked.disconnect()
            self.interaction.request_repaint.disconnect()
            self.interaction.spark_roi_selected.disconnect()
        except:
            pass  # Ignore errors if already disconnected
        
        # Clean up debug box
        self._hide_debug_box()
        
        # Clean up base pixmap
        if self._base_pixmap:
            self._base_pixmap = None
        
        # Clear references
        self.current_frame_rgb = None
        self.video_session = None
    
    def _get_pixel_color_at_position(self, image_x: int, image_y: int) -> Optional[tuple]:
        """Get RGB color at image coordinates."""
        if self.current_frame_rgb is None:
            return None
        
        height, width = self.current_frame_rgb.shape[:2]
        if 0 <= image_x < width and 0 <= image_y < height:
            # current_frame_rgb is in RGB format
            color = self.current_frame_rgb[image_y, image_x]
            return (int(color[0]), int(color[1]), int(color[2]))
        return None
    
    def _handle_overlay_selected(self, overlay_key_id: int):
        """Handle overlay selection from interaction module."""
        if overlay_key_id >= 0:
            self.app_state.ui.selected_overlay_id = overlay_key_id
        else:
            self.app_state.ui.selected_overlay_id = None
        
        # Call the overlay selection callback to notify main window
        if self.on_overlay_select_callback:
            self.on_overlay_select_callback(overlay_key_id if overlay_key_id >= 0 else None)
            
        self.update()  # Repaint to show selection
    
    def _handle_overlay_moved(self, overlay_index: int, new_x: float, new_y: float):
        """Handle overlay movement from interaction module."""
        if 0 <= overlay_index < len(self.app_state.overlays):
            overlay = self.app_state.overlays[overlay_index]
            overlay.x = new_x
            overlay.y = new_y
            self.app_state.unsaved_changes = True
    
    def _handle_overlay_resized(self, overlay_index: int, new_x: float, new_y: float, 
                              new_width: float, new_height: float):
        """Handle overlay resizing from interaction module."""
        if 0 <= overlay_index < len(self.app_state.overlays):
            overlay = self.app_state.overlays[overlay_index]
            overlay.x = new_x
            overlay.y = new_y
            overlay.width = new_width
            overlay.height = new_height
            self.app_state.unsaved_changes = True
    
    def _handle_color_picked(self, r: int, g: int, b: int, image_x: int, image_y: int):
        """Handle color picking from interaction module."""
        logging.info(f"Color picked: RGB({r}, {g}, {b}) at image coordinates ({image_x}, {image_y})")
        
        # Pass both color and coordinates to the callback
        if self.on_color_pick_callback:
            self.on_color_pick_callback((r, g, b), (image_x, image_y))
    
    def _handle_spark_roi_selected(self, top_y: int, bottom_y: int):
        """Handle spark ROI selection from interaction module."""
        logging.info(f"Spark ROI selected: Y range {top_y} to {bottom_y}")
        
        # Update app state with new spark ROI values
        self.app_state.detection.spark_roi_top = top_y
        self.app_state.detection.spark_roi_bottom = bottom_y
        self.app_state.detection.spark_roi_visible = True  # Show ROI immediately after calibration
        self.app_state.unsaved_changes = True
        
        # Invalidate spark zone cache since ROI changed
        try:
            from synthesia2midi.detection.spark_mapper import get_spark_mapper
            get_spark_mapper().invalidate_cache()
        except ImportError:
            pass
        
        # Trigger repaint to show ROI and zone visualization
        self.update()
        
        # Update control panel to reflect new values
        if hasattr(self, 'on_spark_roi_callback') and self.on_spark_roi_callback:
            self.on_spark_roi_callback(top_y, bottom_y)
    
    
    def _handle_repaint_request(self):
        """Handle repaint request from interaction module."""
        # Always redraw the current frame (no zoom/pan state).
        self.display_frame(self.app_state.video.current_frame_index)
    
    def refresh_spark_roi_visualization(self):
        """Refresh spark ROI and zone visualization after manual changes."""
        # Invalidate spark zone cache to force recalculation
        try:
            from synthesia2midi.detection.spark_mapper import get_spark_mapper
            get_spark_mapper().invalidate_cache()
        except ImportError:
            pass
        self.update()
    
    # Keyboard area selection handler is not used.
