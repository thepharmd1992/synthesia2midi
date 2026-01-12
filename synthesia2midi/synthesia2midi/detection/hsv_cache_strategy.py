"""
HSV Caching Strategy for advanced performance optimization.

This module implements a persistent HSV caching strategy that provides
additional performance gains by caching HSV conversions across multiple frames
when processing the same regions repeatedly (e.g., during calibration or
parameter tuning).
"""

import logging
from typing import Dict, Optional, Tuple, Set
from collections import OrderedDict

import cv2
import numpy as np

from synthesia2midi.app_config import OverlayConfig


class PersistentHSVCache:
    """
    Persistent HSV cache that maintains conversions across multiple frames.
    
    This cache provides additional optimization by:
    1. Caching HSV ROIs across frames based on content hash
    2. LRU eviction policy to manage memory
    3. Smart invalidation based on frame content changes
    """
    
    def __init__(self, max_cache_size: int = 1000, similarity_threshold: float = 0.95):
        """
        Initialize the persistent HSV cache.
        
        Args:
            max_cache_size: Maximum number of cached HSV ROIs
            similarity_threshold: Threshold for considering ROIs identical (0-1)
        """
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.max_cache_size = max_cache_size
        self.similarity_threshold = similarity_threshold
        self.hit_count = 0
        self.miss_count = 0
        self.logger = logging.getLogger(__name__)
        
    def get_roi_hsv(self, bgr_roi: np.ndarray, overlay_id: int) -> np.ndarray:
        """
        Get HSV ROI from cache or convert and cache it.
        
        Args:
            bgr_roi: BGR ROI to convert
            overlay_id: Overlay identifier for cache key generation
            
        Returns:
            HSV converted ROI
        """
        # Generate cache key based on ROI content
        cache_key = self._generate_cache_key(bgr_roi, overlay_id)
        
        if cache_key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(cache_key)
            self.hit_count += 1
            return self.cache[cache_key]
        
        # Convert and cache
        hsv_roi = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
        self._add_to_cache(cache_key, hsv_roi)
        self.miss_count += 1
        
        return hsv_roi
    
    def _generate_cache_key(self, bgr_roi: np.ndarray, overlay_id: int) -> str:
        """
        Generate a cache key based on ROI content hash.
        
        Uses an optimized hash for calibration workflows where ROIs change slowly.
        
        Args:
            bgr_roi: BGR ROI array
            overlay_id: Overlay identifier
            
        Returns:
            Cache key string
        """
        # For calibration workflows, use faster hashing with larger sampling
        # Take every 8th pixel instead of 4th for better performance in calibration
        subset = bgr_roi[::8, ::8, :].flatten()
        # Use faster hash algorithm for better performance
        return f"{overlay_id}_{hash(subset.tobytes())}"
    
    def _add_to_cache(self, key: str, hsv_roi: np.ndarray):
        """
        Add HSV ROI to cache with LRU eviction.
        
        Args:
            key: Cache key
            hsv_roi: HSV converted ROI
        """
        # Evict least recently used if at capacity
        if len(self.cache) >= self.max_cache_size:
            self.cache.popitem(last=False)  # Remove oldest
        
        self.cache[key] = hsv_roi
    
    def get_hit_rate(self) -> float:
        """
        Get cache hit rate for performance monitoring.
        
        Returns:
            Hit rate as percentage (0-100)
        """
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        return (self.hit_count / total) * 100
    
    def clear(self):
        """Clear the entire cache."""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'cache_size': len(self.cache),
            'max_size': self.max_cache_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': self.get_hit_rate(),
            'memory_mb': self._estimate_memory_usage() / (1024 * 1024)
        }
    
    def _estimate_memory_usage(self) -> int:
        """
        Estimate memory usage of cached HSV arrays.
        
        Returns:
            Estimated memory usage in bytes
        """
        total = 0
        for hsv_roi in self.cache.values():
            total += hsv_roi.nbytes
        return total


class FrameSequenceHSVCache:
    """
    Optimized HSV cache for sequential frame processing.
    
    This cache is optimized for video processing where frames are
    processed sequentially and ROIs change gradually.
    """
    
    def __init__(self, window_size: int = 10):
        """
        Initialize frame sequence cache.
        
        Args:
            window_size: Number of frames to keep in sliding window
        """
        self.window_size = window_size
        self.frame_cache: OrderedDict[int, Dict[int, np.ndarray]] = OrderedDict()
        self.current_frame_idx = -1
        self.logger = logging.getLogger(__name__)
    
    def process_frame(self, frame_idx: int, frame_bgr: np.ndarray, 
                      overlays: list) -> Dict[int, np.ndarray]:
        """
        Process a frame and return HSV ROIs for all overlays.
        
        Args:
            frame_idx: Frame index in video
            frame_bgr: BGR frame
            overlays: List of overlay configurations
            
        Returns:
            Dictionary mapping overlay IDs to HSV ROIs
        """
        self.current_frame_idx = frame_idx
        
        # Check if we can reuse previous frame's HSV data
        if frame_idx - 1 in self.frame_cache:
            prev_hsv = self.frame_cache[frame_idx - 1]
            # For gradual changes, we could reuse with delta updates
            # This is a placeholder for more advanced temporal caching
        
        # Process current frame
        hsv_rois = {}
        for overlay in overlays:
            bgr_roi = self._extract_roi(frame_bgr, overlay)
            hsv_roi = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
            hsv_rois[overlay.key_id] = hsv_roi
        
        # Add to cache and maintain window
        self.frame_cache[frame_idx] = hsv_rois
        self._maintain_window()
        
        return hsv_rois
    
    def _extract_roi(self, frame: np.ndarray, overlay: OverlayConfig) -> np.ndarray:
        """
        Extract ROI from frame.
        
        Args:
            frame: Source frame
            overlay: Overlay configuration
            
        Returns:
            Extracted ROI
        """
        x, y = int(overlay.x), int(overlay.y)
        w, h = int(overlay.width), int(overlay.height)
        return frame[y:y+h, x:x+w]
    
    def _maintain_window(self):
        """Maintain sliding window of cached frames."""
        while len(self.frame_cache) > self.window_size:
            self.frame_cache.popitem(last=False)
    
    def get_temporal_hsv(self, frame_idx: int, overlay_id: int, 
                         temporal_range: int = 3) -> Optional[np.ndarray]:
        """
        Get temporally smoothed HSV using nearby frames.
        
        Args:
            frame_idx: Current frame index
            overlay_id: Overlay identifier
            temporal_range: Number of frames to average
            
        Returns:
            Temporally smoothed HSV ROI or None
        """
        hsv_values = []
        
        for i in range(frame_idx - temporal_range, frame_idx + temporal_range + 1):
            if i in self.frame_cache and overlay_id in self.frame_cache[i]:
                hsv_values.append(self.frame_cache[i][overlay_id])
        
        if len(hsv_values) >= 3:  # Need at least 3 frames for smoothing
            # Median filter for temporal smoothing
            return np.median(hsv_values, axis=0).astype(np.uint8)
        
        return None


class AdaptiveHSVCacheManager:
    """
    Manages multiple HSV caching strategies and selects the optimal one
    based on usage patterns.
    """
    
    def __init__(self):
        """Initialize the adaptive cache manager."""
        self.persistent_cache = PersistentHSVCache()
        self.sequence_cache = FrameSequenceHSVCache()
        self.usage_pattern = 'unknown'
        self.frame_count = 0
        self.repeat_count = 0
        self.logger = logging.getLogger(__name__)
    
    def process_roi(self, frame_idx: int, bgr_roi: np.ndarray, 
                   overlay: OverlayConfig, is_sequential: bool = True) -> np.ndarray:
        """
        Process ROI using the optimal caching strategy.
        
        Args:
            frame_idx: Frame index
            bgr_roi: BGR ROI
            overlay: Overlay configuration
            is_sequential: Whether frames are being processed sequentially
            
        Returns:
            HSV converted ROI
        """
        self.frame_count += 1
        
        # Detect usage pattern
        if is_sequential and self.frame_count > 10:
            self.usage_pattern = 'sequential'
        elif self.persistent_cache.get_hit_rate() > 50:
            self.usage_pattern = 'repetitive'
        
        # Use appropriate cache based on pattern
        if self.usage_pattern == 'repetitive':
            # Use persistent cache for repetitive access (calibration, tuning)
            return self.persistent_cache.get_roi_hsv(bgr_roi, overlay.key_id)
        else:
            # Direct conversion for unknown or sequential patterns
            return cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    
    def get_combined_stats(self) -> Dict:
        """
        Get combined statistics from all caching strategies.
        
        Returns:
            Dictionary with combined cache statistics
        """
        return {
            'usage_pattern': self.usage_pattern,
            'frame_count': self.frame_count,
            'persistent_cache': self.persistent_cache.get_stats(),
            'optimal_strategy': self._get_optimal_strategy()
        }
    
    def _get_optimal_strategy(self) -> str:
        """
        Determine the optimal caching strategy based on metrics.
        
        Returns:
            Name of optimal strategy
        """
        hit_rate = self.persistent_cache.get_hit_rate()
        
        if hit_rate > 70:
            return 'persistent_cache'
        elif self.usage_pattern == 'sequential':
            return 'sequence_cache'
        else:
            return 'direct_conversion'
    
    def clear_all(self):
        """Clear all caches."""
        self.persistent_cache.clear()
        self.sequence_cache = FrameSequenceHSVCache()
        self.frame_count = 0
        self.repeat_count = 0
        self.usage_pattern = 'unknown'