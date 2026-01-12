"""
Video loading and frame extraction system.

Provides unified interface for video file handling and frame extraction using OpenCV.
Supports both video files (.mp4, .avi, etc.) and pre-converted image sequences for
improved performance. Includes frame caching to optimize sequential access patterns
common in video processing workflows.

Key Features:
- Automatic format detection (video files vs image sequences)
- Frame caching for better performance
- Consistent interface regardless of source format
- OpenCV version compatibility handling
"""
import logging
import os
import sys
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from synthesia2midi.frame_cache import FrameCache
from synthesia2midi.image_sequence_loader import ImageSequenceSession

class VideoSession:
    """Manages an OpenCV video capture session."""

    def __init__(self, filepath: str):
        """
        Initializes the video session.
        Args:
            filepath: Path to the video file.
        Raises:
            FileNotFoundError: If the video file cannot be opened.
        """
        self.filepath = filepath
        self.cap = cv2.VideoCapture(filepath)

        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {filepath}")

        # Determine OpenCV version for CAP_PROP constants
        cv_version = cv2.__version__.split('.')[0]
        if cv_version == '2':
            self._cap_prop_frame_count = cv2.cv.CV_CAP_PROP_FRAME_COUNT
            self._cap_prop_pos_frames = cv2.cv.CV_CAP_PROP_POS_FRAMES
            self._cap_prop_fps = cv2.cv.CV_CAP_PROP_FPS
            self._cap_prop_frame_width = cv2.cv.CV_CAP_PROP_FRAME_WIDTH
            self._cap_prop_frame_height = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
        else: # OpenCV 3, 4+
            self._cap_prop_frame_count = cv2.CAP_PROP_FRAME_COUNT
            self._cap_prop_pos_frames = cv2.CAP_PROP_POS_FRAMES
            self._cap_prop_fps = cv2.CAP_PROP_FPS
            self._cap_prop_frame_width = cv2.CAP_PROP_FRAME_WIDTH
            self._cap_prop_frame_height = cv2.CAP_PROP_FRAME_HEIGHT
        
        self.total_frames: int = int(self.cap.get(self._cap_prop_frame_count))
        # Try to detect FPS from video, default to 30 if detection fails
        detected_fps = self.cap.get(self._cap_prop_fps)
        if detected_fps > 0 and detected_fps < 1000:  # Sanity check for valid FPS
            self.fps: float = float(detected_fps)
        else:
            self.fps: float = 30.0  # Default fallback
            logging.warning(f"Could not detect valid FPS from video (got {detected_fps}), using default 30 FPS")
        self.width: int = int(self.cap.get(self._cap_prop_frame_width))
        self.height: int = int(self.cap.get(self._cap_prop_frame_height))
            
        # Initialize frame cache to work around OpenCV seeking performance issue
        self.frame_cache = FrameCache(max_frames=50)


    def get_frame(self, frame_num: int) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retrieves a specific frame from the video using frame cache.
        Args:
            frame_num: The 0-indexed frame number to retrieve.
        Returns:
            A tuple (success, frame_array).
            frame_array is a BGR NumPy array if successful, None otherwise.
        """
        if not self.cap.isOpened() or frame_num < 0 or frame_num >= self.total_frames:
            return False, None
        
        # Use frame cache to avoid OpenCV seeking performance degradation
        return self.frame_cache.get_frame(self.cap, frame_num)

    def get_frame_sequential(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retrieves the next frame in sequence without seeking.
        Much faster than get_frame() for sequential access.
        
        Returns:
            A tuple (success, frame_array).
            frame_array is a BGR NumPy array if successful, None otherwise.
        """
        if not self.cap.isOpened():
            return False, None
        
        # Get the 0-based index of the frame to be decoded/returned next according to OpenCV.
        frame_idx_before_read = int(self.cap.get(self._cap_prop_pos_frames))

        success, frame = self.cap.read()

        if success:
            # After a successful read, CAP_PROP_POS_FRAMES points to the *next* frame.
            # So, the frame we just read is CAP_PROP_POS_FRAMES - 1.
            frame_idx_actually_read = int(self.cap.get(self._cap_prop_pos_frames)) - 1
            
            # Log the frame index OpenCV believes it just read (disabled for performance)
            # logging.info(f"VideoSession: get_frame_sequential - OpenCV CAP_PROP_POS_FRAMES before read: {frame_idx_before_read}, actual frame index read: {frame_idx_actually_read}")
        else:
            # Log failure to read, including the position OpenCV was at before the failed attempt.
            logging.warning(f"VideoSession: get_frame_sequential - OpenCV cap.read() failed. CAP_PROP_POS_FRAMES before failed read attempt was: {frame_idx_before_read}")
            
        return success, frame
    
    def seek_to_frame(self, frame_num: int) -> bool:
        """
        Seeks to a specific frame number.
        Use this before get_frame_sequential() for batch processing.
        
        Args:
            frame_num: The 0-indexed frame number to seek to.
            
        Returns:
            True if seek was successful, False otherwise.
        """
        if not self.cap.isOpened() or frame_num < 0 or frame_num >= self.total_frames:
            return False
            
        self.cap.set(self._cap_prop_pos_frames, frame_num)
        return True

    def release(self) -> None:
        """Releases the video capture object and clears frame cache."""
        if self.cap.isOpened():
            self.cap.release()
        # Clear frame cache
        if hasattr(self, 'frame_cache'):
            self.frame_cache.clear()

    def __del__(self):
        self.release()


def create_video_session(filepath: str, fps: Optional[float] = None) -> Union[VideoSession, ImageSequenceSession]:
    """
    Factory function that creates appropriate session based on input.
    
    Args:
        filepath: Either a video file path or image sequence pattern
        
    Returns:
        VideoSession for video files, ImageSequenceSession for image patterns
        
    Examples:
        - create_video_session("video.mp4") -> VideoSession
        - create_video_session("frames/frame_%06d.jpg") -> ImageSequenceSession
        - create_video_session("frames/*.png") -> ImageSequenceSession
    """
    # Handle Windows WSL paths
    if filepath.startswith('\\\\wsl.localhost\\') or filepath.startswith('\\\\wsl$\\'):
        # Convert Windows WSL path to Linux path
        parts = filepath.replace('\\', '/').split('/')
        for i, part in enumerate(parts):
            if part in ['Ubuntu', 'Debian', 'openSUSE', 'kali-linux']:
                filepath = '/' + '/'.join(parts[i+1:])
                logging.info(f"[VIDEO-SESSION] Converted WSL path to Linux: {filepath}")
                break
    
    # Check if it's an image sequence pattern
    if '%' in filepath or '*' in filepath:
        logging.info(f"Detected image sequence pattern: {filepath}")
        return ImageSequenceSession(filepath, fps=fps)
    
    # Check if it's a directory (assume it contains frames)
    if os.path.isdir(filepath):
        # Look for common image patterns in the directory
        import glob
        patterns = [
            os.path.join(filepath, "frame_*.jpg"),
            os.path.join(filepath, "frame_*.png"),
            os.path.join(filepath, "*.jpg"),
            os.path.join(filepath, "*.png")
        ]
        for pattern in patterns:
            if glob.glob(pattern):
                logging.info(f"Found image sequence in directory: {pattern}")
                return ImageSequenceSession(pattern, fps=fps)
        raise FileNotFoundError(f"No image sequence found in directory: {filepath}")
    
    # Otherwise assume it's a video file
    logging.info(f"Loading video file: {filepath}")
    return VideoSession(filepath)
