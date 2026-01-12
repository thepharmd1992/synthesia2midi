"""Frame cache to work around OpenCV seeking performance degradation."""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging


class FrameCache:
    """
    LRU cache for video frames to avoid repeated OpenCV seeks.
    
    This works around the OpenCV performance issue where repeated calls to
    cv2.VideoCapture.set(cv2.CAP_PROP_POS_FRAMES) get progressively slower.
    """
    
    def __init__(self, max_frames: int = 50):
        """
        Initialize frame cache.
        
        Args:
            max_frames: Maximum number of frames to cache (default 50)
        """
        self.cache: Dict[int, np.ndarray] = {}
        self.access_order: List[int] = []
        self.max_frames = max_frames
        self.hits = 0
        self.misses = 0
        
    def get_frame(self, video_cap: cv2.VideoCapture, frame_index: int) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get frame from cache or read from video.
        
        Args:
            video_cap: OpenCV VideoCapture object
            frame_index: Frame number to retrieve
            
        Returns:
            Tuple of (success, frame_data)
        """
        # Check cache first
        if frame_index in self.cache:
            self.hits += 1
            # Move to end (LRU)
            self.access_order.remove(frame_index)
            self.access_order.append(frame_index)
            # Return a copy to prevent modifications affecting cache
            return True, self.cache[frame_index].copy()
            
        # Cache miss - read from video
        self.misses += 1
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = video_cap.read()
        
        if ret and frame is not None:
            # Add to cache (store a copy)
            self.cache[frame_index] = frame.copy()
            self.access_order.append(frame_index)
            
            # Evict old frames if cache is full
            while len(self.cache) > self.max_frames:
                oldest = self.access_order.pop(0)
                del self.cache[oldest]
                
        return ret, frame
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()
        self.hits = 0
        self.misses = 0
        
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'total': total,
            'hit_rate': hit_rate
        }
        
    def preload_range(self, video_cap: cv2.VideoCapture, start_frame: int, end_frame: int):
        """
        Preload a range of frames into cache.
        
        Useful for smooth playback or when user is likely to navigate
        within a specific range.
        """
        frames_to_load = min(end_frame - start_frame + 1, self.max_frames)
        
        for i in range(frames_to_load):
            frame_idx = start_frame + i
            if frame_idx not in self.cache:
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = video_cap.read()
                if ret and frame is not None:
                    self.cache[frame_idx] = frame.copy()
                    self.access_order.append(frame_idx)
                    
                    # Maintain cache size
                    while len(self.cache) > self.max_frames:
                        oldest = self.access_order.pop(0)
                        del self.cache[oldest]

