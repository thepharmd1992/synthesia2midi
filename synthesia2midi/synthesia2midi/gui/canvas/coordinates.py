"""
Coordinate transformation management for the video canvas.

Centralizes all coordinate space transformations between:
- Canvas coordinates (widget pixel space)
- Image coordinates (video frame pixel space)
- Crop-adjusted coordinates (when frame is cropped)

This eliminates the scattered coordinate math throughout keyboard_canvas.py
and provides a single source of truth for all transformations.
"""
# Standard library imports
import logging
from typing import Optional, Tuple


class CoordinateManager:
    """
    Handles all coordinate space transformations for the video canvas.
    
    Manages transformations between canvas space (widget pixels) and 
    image space (video frame pixels), with support for scaling, centering,
    and cropping operations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.CoordinateManager")
        
        # Canvas dimensions (widget size)
        self.canvas_width = 0
        self.canvas_height = 0
        
        # Image dimensions (video frame size)
        self.image_width = 0
        self.image_height = 0
        
        # Transform parameters
        self.image_scale_factor = 1.0
        self.image_x_offset = 0
        self.image_y_offset = 0
        
        # Zoom and pan are not user-controllable; transforms are derived from canvas/image size.
        
        # Crop parameters (for cropped video frames)
        self._crop_offset_x = 0
        self._crop_offset_y = 0
        self._is_frame_cropped = False
        
        self.logger.debug("CoordinateManager initialized")
    
    def update_canvas_size(self, width: int, height: int):
        """Update canvas dimensions and recalculate transforms."""
        self.canvas_width = width
        self.canvas_height = height
        if self.image_width > 0 and self.image_height > 0:
            self._calculate_transform()
        self.logger.debug(f"Canvas size updated: {width}x{height}")
    
    def update_image_size(self, width: int, height: int):
        """Update image dimensions and recalculate transforms."""
        self.image_width = width
        self.image_height = height
        if self.canvas_width > 0 and self.canvas_height > 0:
            self._calculate_transform()
        self.logger.debug(f"Image size updated: {width}x{height}")
    
    def update_crop_settings(self, crop_offset_x: int = 0, crop_offset_y: int = 0, 
                           is_cropped: bool = False):
        """Update crop settings for coordinate adjustments."""
        self._crop_offset_x = crop_offset_x
        self._crop_offset_y = crop_offset_y
        self._is_frame_cropped = is_cropped
        self.logger.debug(f"Crop settings: offset=({crop_offset_x}, {crop_offset_y}), cropped={is_cropped}")
    
    def _calculate_transform(self):
        """Calculate scale factor and alignment offsets to fit image in canvas."""
        if self.canvas_width <= 0 or self.canvas_height <= 0:
            return
        if self.image_width <= 0 or self.image_height <= 0:
            return
            
        # Calculate scale to fit image in canvas while maintaining aspect ratio
        width_scale = self.canvas_width / self.image_width
        height_scale = self.canvas_height / self.image_height
        
        # Use the smaller scale to ensure the entire image fits
        self.image_scale_factor = min(width_scale, height_scale)
        
        # Calculate alignment offsets - center the image
        scaled_width = int(self.image_width * self.image_scale_factor)
        scaled_height = int(self.image_height * self.image_scale_factor)
        
        # Align image to the right (no padding on right side) and center vertically
        self.image_x_offset = self.canvas_width - scaled_width  # Right-align the video
        self.image_y_offset = (self.canvas_height - scaled_height) // 2  # Center vertically
        
        self.logger.debug(f"Transform calculated: scale={self.image_scale_factor:.3f}, "
                         f"offset=({self.image_x_offset}, {self.image_y_offset})")
    
    def canvas_to_image(self, canvas_x: float, canvas_y: float, clamp_to_bounds: bool = False) -> Optional[Tuple[float, float]]:
        """
        Convert canvas coordinates to image coordinates.
        
        Args:
            canvas_x, canvas_y: Position in canvas (widget) space
            clamp_to_bounds: If True, clamp coordinates to image bounds instead of returning None
            
        Returns:
            (image_x, image_y) tuple or None if outside image bounds (unless clamp_to_bounds is True)
        """
        if self.image_scale_factor <= 0:
            return None
            
        # Remove canvas offset and scale down to image space
        image_x = (canvas_x - self.image_x_offset) / self.image_scale_factor
        image_y = (canvas_y - self.image_y_offset) / self.image_scale_factor
        
        # Adjust for crop if applicable
        if self._is_frame_cropped:
            image_x += self._crop_offset_x
            image_y += self._crop_offset_y
        
        # Check bounds
        if clamp_to_bounds:
            # Clamp to valid image bounds
            image_x = max(0, min(image_x, self.image_width - 1))
            image_y = max(0, min(image_y, self.image_height - 1))
            return (image_x, image_y)
        else:
            if (0 <= image_x < self.image_width and 0 <= image_y < self.image_height):
                return (image_x, image_y)
            else:
                return None
    
    def image_to_canvas(self, image_x: float, image_y: float) -> Tuple[float, float]:
        """
        Convert image coordinates to canvas coordinates.
        
        Args:
            image_x, image_y: Position in image (video frame) space
            
        Returns:
            (canvas_x, canvas_y) tuple in canvas space
        """
        # Adjust for crop if applicable
        adj_image_x = image_x
        adj_image_y = image_y
        if self._is_frame_cropped:
            adj_image_x -= self._crop_offset_x
            adj_image_y -= self._crop_offset_y
        
        # Scale up and add canvas offset
        canvas_x = adj_image_x * self.image_scale_factor + self.image_x_offset
        canvas_y = adj_image_y * self.image_scale_factor + self.image_y_offset
        
        return (canvas_x, canvas_y)
    
    def image_rect_to_canvas(self, image_x: float, image_y: float, 
                           image_w: float, image_h: float) -> Tuple[float, float, float, float]:
        """
        Convert image rectangle to canvas rectangle.
        
        Args:
            image_x, image_y: Top-left corner in image space
            image_w, image_h: Width and height in image space
            
        Returns:
            (canvas_x, canvas_y, canvas_w, canvas_h) tuple
        """
        canvas_x, canvas_y = self.image_to_canvas(image_x, image_y)
        canvas_w = image_w * self.image_scale_factor
        canvas_h = image_h * self.image_scale_factor
        
        
        return (canvas_x, canvas_y, canvas_w, canvas_h)
    
    def canvas_rect_to_image(self, canvas_x: float, canvas_y: float,
                           canvas_w: float, canvas_h: float, clamp_to_bounds: bool = False) -> Optional[Tuple[float, float, float, float]]:
        """
        Convert canvas rectangle to image rectangle.
        
        Args:
            canvas_x, canvas_y: Top-left corner in canvas space
            canvas_w, canvas_h: Width and height in canvas space
            clamp_to_bounds: If True, clamp rectangle to image bounds instead of returning None
            
        Returns:
            (image_x, image_y, image_w, image_h) tuple or None if outside bounds (unless clamp_to_bounds is True)
        """
        if self.image_scale_factor <= 0:
            return None
            
        if clamp_to_bounds:
            # Convert both corners and clamp to bounds
            top_left = self.canvas_to_image(canvas_x, canvas_y, clamp_to_bounds=True)
            bottom_right = self.canvas_to_image(canvas_x + canvas_w, canvas_y + canvas_h, clamp_to_bounds=True)
            
            if top_left is None or bottom_right is None:
                return None
                
            image_x, image_y = top_left
            image_x2, image_y2 = bottom_right
            
            # Calculate clamped width and height
            image_w = max(1, image_x2 - image_x)
            image_h = max(1, image_y2 - image_y)
            
            return (image_x, image_y, image_w, image_h)
        else:
            top_left = self.canvas_to_image(canvas_x, canvas_y)
            if top_left is None:
                return None
                
            image_x, image_y = top_left
            image_w = canvas_w / self.image_scale_factor
            image_h = canvas_h / self.image_scale_factor
            
            return (image_x, image_y, image_w, image_h)
    
    def scale_delta(self, canvas_delta_x: float, canvas_delta_y: float) -> Tuple[float, float]:
        """
        Convert canvas space deltas to image space deltas.
        
        Useful for dragging operations where you need to translate
        canvas movement to image coordinate changes.
        
        Args:
            canvas_delta_x, canvas_delta_y: Movement in canvas space
            
        Returns:
            (image_delta_x, image_delta_y) tuple in image space
        """
        if self.image_scale_factor <= 0:
            return (0.0, 0.0)
            
        image_delta_x = canvas_delta_x / self.image_scale_factor
        image_delta_y = canvas_delta_y / self.image_scale_factor
        
        return (image_delta_x, image_delta_y)
    
    def get_visible_image_rect(self) -> Tuple[float, float, float, float]:
        """
        Get the portion of the image that's currently visible in the canvas.
        
        Returns:
            (image_x, image_y, image_w, image_h) of visible area in image space
        """
        # Canvas viewport in image coordinates
        top_left = self.canvas_to_image(0, 0)
        bottom_right = self.canvas_to_image(self.canvas_width, self.canvas_height)
        
        if top_left is None or bottom_right is None:
            return (0, 0, self.image_width, self.image_height)
        
        image_x = max(0, top_left[0])
        image_y = max(0, top_left[1])
        image_w = min(self.image_width, bottom_right[0]) - image_x
        image_h = min(self.image_height, bottom_right[1]) - image_y
        
        return (image_x, image_y, image_w, image_h)
    
    def is_point_in_image(self, canvas_x: float, canvas_y: float) -> bool:
        """Check if a canvas point falls within the displayed image area."""
        return self.canvas_to_image(canvas_x, canvas_y) is not None
    
    def get_transform_info(self) -> dict:
        """Get current transformation parameters for debugging."""
        return {
            "canvas_size": (self.canvas_width, self.canvas_height),
            "image_size": (self.image_width, self.image_height),
            "scale_factor": self.image_scale_factor,
            "offset": (self.image_x_offset, self.image_y_offset),
            "crop_offset": (self._crop_offset_x, self._crop_offset_y),
            "is_cropped": self._is_frame_cropped,
            "zoom_level": 1.0,
            "pan_offset": (0, 0)
        }
    
    # Zoom and pan controls are not implemented; `get_transform_info()` reports fixed values.
