"""
ROI (Region of Interest) caching system for detection optimization.

This module provides caching functionality to eliminate redundant ROI extractions
and color space conversions during frame processing, significantly improving
performance by reusing previously computed values within the same frame.
"""

import logging
from typing import Dict, Optional

import cv2
import numpy as np

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.detection.roi_utils import extract_roi_bgr


class ROICache:
    """
    Cache for extracted ROIs to avoid redundant extractions.
    
    This cache stores BGR and HSV versions of ROIs extracted from a frame,
    ensuring each ROI is only extracted and converted once per frame.
    """
    
    def __init__(self):
        """Initialize the ROI cache."""
        self.bgr_cache: Dict[int, np.ndarray] = {}
        self.hsv_cache: Dict[int, np.ndarray] = {}
        self.rgb_cache: Dict[int, np.ndarray] = {}
        self.frame_id = -1
        self.logger = logging.getLogger(__name__)
        
    def clear(self):
        """
        Clear cache for new frame.
        
        This should be called at the beginning of each frame processing
        to ensure stale data from previous frames is not used.
        """
        self.bgr_cache.clear()
        self.hsv_cache.clear()
        self.rgb_cache.clear()
        self.frame_id += 1
        
    def get_roi_bgr(self, frame_bgr: np.ndarray, overlay: OverlayConfig) -> Optional[np.ndarray]:
        """
        Get BGR ROI, extracting only if not cached.
        
        Args:
            frame_bgr: The source BGR frame
            overlay: Overlay configuration defining the ROI
            
        Returns:
            BGR ROI array or None if extraction fails
        """
        key = overlay.key_id
        
        if key not in self.bgr_cache:
            roi = extract_roi_bgr(frame_bgr, overlay)
            if roi is not None:
                self.bgr_cache[key] = roi
            else:
                self.logger.warning(f"ROI extraction failed for overlay {overlay.key_id}")
                return None
                
        return self.bgr_cache[key]
    
    def get_roi_hsv(self, frame_bgr: np.ndarray, overlay: OverlayConfig) -> Optional[np.ndarray]:
        """
        Get HSV ROI, converting only if not cached.
        
        Uses ROI-level conversion which is optimal for sparse detection patterns
        like piano keys (0.1% frame coverage).
        
        Args:
            frame_bgr: The source BGR frame
            overlay: Overlay configuration defining the ROI
            
        Returns:
            HSV ROI array or None if extraction/conversion fails
        """
        key = overlay.key_id
        
        if key not in self.hsv_cache:
            # Get BGR ROI (will use cache if available)
            bgr_roi = self.get_roi_bgr(frame_bgr, overlay)
            if bgr_roi is None:
                return None
                
            # Convert to HSV and cache
            try:
                hsv_roi = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
                self.hsv_cache[key] = hsv_roi
            except Exception as e:
                self.logger.error(f"HSV conversion failed for overlay {overlay.key_id}: {e}")
                return None
                
        return self.hsv_cache[key]
    
    def get_roi_rgb(self, frame_bgr: np.ndarray, overlay: OverlayConfig) -> Optional[np.ndarray]:
        """
        Get RGB ROI, converting only if not cached.
        
        Args:
            frame_bgr: The source BGR frame
            overlay: Overlay configuration defining the ROI
            
        Returns:
            RGB ROI array or None if extraction/conversion fails
        """
        key = overlay.key_id
        
        if key not in self.rgb_cache:
            # First get BGR ROI (will use cache if available)
            bgr_roi = self.get_roi_bgr(frame_bgr, overlay)
            if bgr_roi is None:
                return None
                
            # Convert to RGB and cache
            try:
                rgb_roi = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
                self.rgb_cache[key] = rgb_roi
            except Exception as e:
                self.logger.error(f"RGB conversion failed for overlay {overlay.key_id}: {e}")
                return None
                
        return self.rgb_cache[key]
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics for debugging and monitoring.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'frame_id': self.frame_id,
            'bgr_cached': len(self.bgr_cache),
            'hsv_cached': len(self.hsv_cache),
            'rgb_cached': len(self.rgb_cache),
            'total_memory_kb': self._estimate_memory_usage() // 1024
        }
    
    def _estimate_memory_usage(self) -> int:
        """
        Estimate memory usage of cached arrays in bytes.
        
        Returns:
            Estimated memory usage in bytes
        """
        total = 0
        
        for roi in self.bgr_cache.values():
            total += roi.nbytes
            
        for roi in self.hsv_cache.values():
            total += roi.nbytes
            
        for roi in self.rgb_cache.values():
            total += roi.nbytes
            
        return total


