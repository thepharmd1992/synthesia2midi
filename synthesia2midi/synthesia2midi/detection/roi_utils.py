"""
Shared utilities for Region of Interest (ROI) extraction and processing.

Common functions used across different detection methods for extracting
and processing key regions from video frames.
"""
import logging
import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from synthesia2midi.app_config import OverlayConfig

# Global reference to app_state for accessing ROI downsampling configuration
_app_state = None

def set_app_state_reference(app_state):
    """Set the global app_state reference for ROI downsampling configuration."""
    global _app_state
    _app_state = app_state


def extract_roi_bgr(image: np.ndarray, overlay: OverlayConfig) -> Optional[np.ndarray]:
    """
    Extract BGR ROI for an overlay from an image with optional downsampling.
    
    Args:
        image: BGR image array
        overlay: Overlay configuration defining the region
        
    Returns:
        BGR ROI array (potentially downsampled) or None if extraction fails
    """
    try:
        x1, y1 = int(overlay.x), int(overlay.y)
        x2, y2 = x1 + int(overlay.width), y1 + int(overlay.height)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        return roi
        
    except Exception as e:
        logging.error(f"ROI extraction failed for overlay {overlay.key_id}: {e}")
        return None



def extract_roi_with_offset(image: np.ndarray, overlay: OverlayConfig,
                          x_offset: int, y_offset: int) -> Optional[np.ndarray]:
    """
    Extract ROI with offset for cropped frames.
    
    Args:
        image: BGR image array
        overlay: Overlay configuration
        x_offset: X offset to apply to overlay coordinates
        y_offset: Y offset to apply to overlay coordinates
        
    Returns:
        BGR ROI array or None if extraction fails
    """
    try:
        x1, y1 = int(overlay.x) + x_offset, int(overlay.y) + y_offset
        x2, y2 = x1 + int(overlay.width), y1 + int(overlay.height)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        roi = image[y1:y2, x1:x2]
        return roi if roi.size > 0 else None
        
    except Exception as e:
        logging.error(f"ROI extraction with offset failed for overlay {overlay.key_id}: {e}")
        return None






def calculate_brightness_percentile(roi_hsv: np.ndarray, percentile: float = 95.0) -> Optional[float]:
    """
    Calculate brightness percentile from HSV ROI.
    
    Args:
        roi_hsv: HSV ROI array
        percentile: Percentile to calculate (0-100)
        
    Returns:
        Brightness value or None if calculation fails
    """
    if roi_hsv is None or roi_hsv.size == 0:
        return None
        
    try:
        v_channel = roi_hsv[:, :, 2]  # HSV Value channel
        brightness = np.percentile(v_channel, percentile)
        return float(brightness)
        
    except Exception as e:
        logging.error(f"Brightness percentile calculation failed: {e}")
        return None


def validate_overlay_in_frame(overlay: OverlayConfig, frame_shape: Tuple[int, int]) -> bool:
    """
    Validate that overlay coordinates are within frame bounds.
    
    Args:
        overlay: Overlay configuration
        frame_shape: (height, width) of frame
        
    Returns:
        True if overlay is valid for this frame
    """
    try:
        x1, y1 = int(overlay.x), int(overlay.y)
        x2, y2 = x1 + int(overlay.width), y1 + int(overlay.height)
        
        frame_height, frame_width = frame_shape
        
        # Check if overlay is completely outside frame
        if x2 <= 0 or y2 <= 0 or x1 >= frame_width or y1 >= frame_height:
            return False
            
        # Check if overlay has some valid area within frame
        valid_x1 = max(0, x1)
        valid_y1 = max(0, y1)
        valid_x2 = min(frame_width, x2)
        valid_y2 = min(frame_height, y2)
        
        return valid_x2 > valid_x1 and valid_y2 > valid_y1
        
    except Exception:
        return False


def adjust_overlay_for_crop(overlay: OverlayConfig, 
                          crop_offset_x: int, crop_offset_y: int) -> Optional[OverlayConfig]:
    """
    Create adjusted overlay for cropped frame coordinates.
    
    Args:
        overlay: Original overlay configuration
        crop_offset_x: X offset of crop in original frame
        crop_offset_y: Y offset of crop in original frame
        
    Returns:
        New OverlayConfig with adjusted coordinates or None if outside crop
    """
    try:
        # Create copy of overlay with adjusted coordinates
        adjusted = OverlayConfig(
            key_id=overlay.key_id,
            x=overlay.x - crop_offset_x,
            y=overlay.y - crop_offset_y,
            width=overlay.width,
            height=overlay.height,
            key_type=overlay.key_type,
            unlit_reference_color=overlay.unlit_reference_color,
            # Copy other relevant attributes
            prev_progression_ratio=overlay.prev_progression_ratio,
            last_progression_ratio=overlay.last_progression_ratio,
            last_is_lit=overlay.last_is_lit
        )
        
        # Check if adjusted overlay has valid coordinates (at least partially in view)
        if adjusted.x + adjusted.width <= 0 or adjusted.y + adjusted.height <= 0:
            return None  # Completely outside cropped area
            
        return adjusted
        
    except Exception as e:
        logging.error(f"Overlay adjustment failed for key {overlay.key_id}: {e}")
        return None


# Color and histogram utility functions

def euclidean_distance(color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
    """Calculates the Euclidean distance between two RGB colors."""
    return math.sqrt((color1[0] - color2[0])**2 + (color1[1] - color2[1])**2 + (color1[2] - color2[2])**2)


def rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert RGB color to LAB color space for better color discrimination.
    
    Args:
        rgb: RGB color tuple (0-255 range)
        
    Returns:
        LAB color tuple (L: 0-100, a: -128 to 127, b: -128 to 127)
    """
    # Normalize RGB to 0-1 range
    r, g, b = [x / 255.0 for x in rgb]
    
    # Apply gamma correction
    def gamma_correct(c):
        if c > 0.04045:
            return ((c + 0.055) / 1.055) ** 2.4
        else:
            return c / 12.92
    
    r = gamma_correct(r)
    g = gamma_correct(g)
    b = gamma_correct(b)
    
    # Convert to XYZ color space (using sRGB matrix)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    
    # Normalize by D65 illuminant
    x = x / 0.95047
    y = y / 1.00000
    z = z / 1.08883
    
    # Apply LAB transformation
    def lab_transform(t):
        if t > 0.008856:
            return t ** (1/3)
        else:
            return (7.787 * t) + (16/116)
    
    fx = lab_transform(x)
    fy = lab_transform(y)
    fz = lab_transform(z)
    
    # Calculate LAB values
    l = (116 * fy) - 16
    a = 500 * (fx - fy)
    b_lab = 200 * (fy - fz)
    
    return (l, a, b_lab)


def lab_delta_e(color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
    """Calculate Delta E distance between two RGB colors using LAB color space.
    
    This provides better perceptual color distance than RGB Euclidean distance,
    especially for distinguishing blue from green when both are bright/lit.
    
    Args:
        color1: First RGB color (0-255 range)
        color2: Second RGB color (0-255 range)
        
    Returns:
        Delta E distance (lower = more similar, typical threshold ~10 for distinguishable)
    """
    lab1 = rgb_to_lab(color1)
    lab2 = rgb_to_lab(color2)
    
    # Calculate Delta E (CIE76 formula)
    delta_l = lab1[0] - lab2[0]
    delta_a = lab1[1] - lab2[1]
    delta_b = lab1[2] - lab2[2]
    
    delta_e = math.sqrt(delta_l**2 + delta_a**2 + delta_b**2)
    return delta_e


def get_hist_feature(roi_bgr: np.ndarray,
                     bins: tuple[int,int,int] = (8,4,4)) -> np.ndarray:
    """
    Returns a 24-dim L1-normalised HSV histogram for the ROI.
    bins=(H,S,V)  ->  8*4*4 = 24 bins.
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None,
                        bins,  # number of bins
                        [0,180, 0,256, 0,256])  # ranges
    hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
    return hist.flatten()


def get_hist_feature_from_hsv(roi_hsv: np.ndarray,
                              bins: tuple[int,int,int] = (8,4,4)) -> np.ndarray:
    """
    Returns a 24-dim L1-normalised HSV histogram for the ROI.
    This version accepts pre-converted HSV data to avoid redundant conversions.
    bins=(H,S,V)  ->  8*4*4 = 24 bins.
    """
    hist = cv2.calcHist([roi_hsv], [0,1,2], None,
                        bins,  # number of bins
                        [0,180, 0,256, 0,256])  # ranges
    hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
    return hist.flatten()


def hist_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Bhattacharyya distance; 0 = identical, 1 = no overlap."""
    return cv2.compareHist(h1.astype("float32"),
                           h2.astype("float32"),
                           cv2.HISTCMP_BHATTACHARYYA)


def get_average_color_from_roi(image: np.ndarray, overlay: OverlayConfig) -> Optional[Tuple[int, int, int]]:
    """Extracts ROI from image based on overlay and calculates average BGR color, then converts to RGB."""
    # Ensure overlay coordinates are integers and define ROI boundaries
    x, y, w, h = int(overlay.x), int(overlay.y), int(overlay.width), int(overlay.height)
    
    img_h, img_w = image.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)

    if x1 >= x2 or y1 >= y2: # Overlay is outside or has no area
        logging.debug(f"Overlay {overlay.key_id} ROI is outside image or has no area.")
        return None

    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        logging.debug(f"Overlay {overlay.key_id} ROI is empty.")
        return None

    # Calculate average color of the ROI (BGR)
    avg_bgr_color = np.mean(roi, axis=(0, 1))
    # Convert average BGR to average RGB
    avg_rgb_color = (int(avg_bgr_color[2]), int(avg_bgr_color[1]), int(avg_bgr_color[0]))
    return avg_rgb_color


def get_average_color_from_roi_with_offset(image: np.ndarray, overlay: OverlayConfig, x_off: int, y_off: int) -> Optional[Tuple[int, int, int]]:
    """Extracts ROI from image based on overlay and calculates average BGR color, then converts to RGB with coordinate offset."""
    # Ensure overlay coordinates are integers and define ROI boundaries
    x, y, w, h = int(overlay.x), int(overlay.y), int(overlay.width), int(overlay.height)
    # Adjust coordinates for offset
    x_adj, y_adj = x - x_off, y - y_off
    
    img_h, img_w = image.shape[:2]
    x1, y1 = max(0, x_adj), max(0, y_adj)
    x2, y2 = min(img_w, x_adj + w), min(img_h, y_adj + h)

    if x1 >= x2 or y1 >= y2: # Overlay is outside or has no area
        logging.debug(f"Overlay {overlay.key_id} ROI is outside image or has no area.")
        return None

    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        logging.debug(f"Overlay {overlay.key_id} ROI is empty.")
        return None

    # Calculate average color of the ROI (BGR)
    avg_bgr_color = np.mean(roi, axis=(0, 1))
    # Convert average BGR to average RGB
    avg_rgb_color = (int(avg_bgr_color[2]), int(avg_bgr_color[1]), int(avg_bgr_color[0]))
    return avg_rgb_color
