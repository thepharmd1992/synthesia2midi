"""
Image sequence loader that provides the same interface as VideoSession.

This allows using pre-decoded image sequences for instant frame access,
solving the video codec seeking performance issue.
"""

import cv2
import os
import glob
import numpy as np
from typing import Tuple, Optional, List
import logging


class ImageSequenceSession:
    """
    Image-sequence-backed session that matches the `VideoSession` interface.
    
    Maintains the same interface so no changes are needed to detection logic.
    """
    
    def __init__(self, image_pattern: str, fps: Optional[float] = None):
        """
        Initialize image sequence session.
        
        Args:
            image_pattern: Pattern for image files, e.g.:
                - "/path/to/frames/frame_%06d.jpg"
                - "/path/to/frames/frame_*.png"
                - "/path/to/frames/*.jpg"
            fps: Optional FPS metadata to use for timing
        """
        self.image_pattern = image_pattern
        
        # Determine if pattern uses printf-style or glob-style
        if '%' in image_pattern:
            # Printf-style pattern (frame_%06d.jpg)
            self._use_printf = True
            self._load_images_printf()
        else:
            # Glob-style pattern (*.jpg or frame_*.png)
            self._use_printf = False
            self._load_images_glob()
            
        if self.total_frames == 0:
            raise FileNotFoundError(f"No images found matching pattern: {image_pattern}")
            
        # Load first frame to get dimensions
        success, first_frame = self.get_frame(0)
        if not success or first_frame is None:
            raise ValueError("Could not load first frame")
            
        self.height, self.width = first_frame.shape[:2]
        self.fps = float(fps) if fps else 30.0  # Default FPS for image sequences
        
        logging.info(
            f"Loaded image sequence: {self.total_frames} frames, "
            f"{self.width}x{self.height}, fps={self.fps}"
        )
        
    def _load_images_printf(self):
        """Load images using printf-style pattern."""
        # Extract directory and pattern
        self.base_dir = os.path.dirname(self.image_pattern)
        base_pattern = os.path.basename(self.image_pattern)
        
        # Find all matching files by trying sequential numbers
        self.frame_files = []
        frame_num = 0
        max_missing = 1000  # Stop after 1000 missing frames (more robust)
        missing_count = 0
        
        while missing_count < max_missing:
            filepath = os.path.join(self.base_dir, base_pattern % frame_num)
            if os.path.exists(filepath):
                self.frame_files.append(filepath)
                missing_count = 0
            else:
                missing_count += 1
            frame_num += 1
            
        self.total_frames = len(self.frame_files)
        
        # Log if we stopped due to missing frames
        if missing_count >= max_missing:
            logging.warning(f"Frame sequence loading stopped after {max_missing} consecutive missing frames. "
                          f"Last attempted frame: {frame_num-1}. Total frames loaded: {self.total_frames}")
        
    def _load_images_glob(self):
        """Load images using glob pattern."""
        self.frame_files = sorted(glob.glob(self.image_pattern))
        self.total_frames = len(self.frame_files)
        
    def get_frame(self, frame_num: int) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get a specific frame from the image sequence.
        
        Returns frame in BGR format to match VideoCapture behavior.
        
        Args:
            frame_num: 0-indexed frame number
            
        Returns:
            Tuple of (success, frame_array in BGR format)
        """
        if frame_num < 0 or frame_num >= self.total_frames:
            return False, None
            
        try:
            # Load image (cv2.imread returns BGR by default)
            frame = cv2.imread(self.frame_files[frame_num])
            if frame is None:
                logging.error(f"Failed to load frame {frame_num}: {self.frame_files[frame_num]}")
                return False, None
            return True, frame
        except Exception as e:
            logging.error(f"Error loading frame {frame_num}: {e}")
            return False, None
            
    def get_frame_sequential(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Sequential reading (for compatibility).
        
        Note: Image sequences don't benefit from sequential reading,
        but we provide this for API compatibility.
        """
        if not hasattr(self, '_sequential_position'):
            self._sequential_position = 0
            
        if self._sequential_position >= self.total_frames:
            return False, None
            
        success, frame = self.get_frame(self._sequential_position)
        if success:
            self._sequential_position += 1
        return success, frame
        
    def seek_to_frame(self, frame_num: int) -> bool:
        """Seek to frame (for API compatibility)."""
        if 0 <= frame_num < self.total_frames:
            self._sequential_position = frame_num
            return True
        return False
        
    def release(self):
        """Release resources (no-op for image sequences)."""
        pass
        
    def __del__(self):
        """Cleanup (no-op for image sequences)."""
        pass
        
    def isOpened(self) -> bool:
        """Check if sequence is valid."""
        return self.total_frames > 0
        
    def get(self, prop_id: int):
        """Get property (for VideoCapture compatibility)."""
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.total_frames)
        elif prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        elif prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        elif prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(getattr(self, '_sequential_position', 0))
        else:
            return 0.0


def convert_video_to_images(video_path: str, output_dir: str, quality: int = 95) -> str:
    """
    Convert video to image sequence using ffmpeg.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        quality: JPEG quality (1-100, higher is better)
        
    Returns:
        Pattern string for loading the images
    """
    from synthesia2midi.utils.ffmpeg_helper import run_ffmpeg_command
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine output format based on quality needs
    if quality >= 95:
        # Use PNG for lossless
        pattern = os.path.join(output_dir, "frame_%06d.png")
        args = [
            '-i', video_path,
            '-vf', 'format=bgr24',  # Ensure BGR format
            pattern
        ]
    else:
        # Use JPEG with specified quality
        pattern = os.path.join(output_dir, "frame_%06d.jpg")
        args = [
            '-i', video_path,
            '-q:v', str(100 - quality),  # ffmpeg uses inverse scale
            '-vf', 'format=bgr24',  # Ensure BGR format
            pattern
        ]
    
    # Use logging instead of print for conversion messages
    logger = logging.getLogger(__name__)
    logger.info(f"Converting {video_path} to image sequence...")
    logger.debug(f"FFmpeg args: {' '.join(args)}")
    
    try:
        run_ffmpeg_command(args, capture_output=False)
        logger.info(f"Successfully converted to {pattern}")
        return pattern
    except Exception as e:
        logger.error(f"Error converting video: {e}")
        raise
        

if __name__ == "__main__":
    # Test/demo script
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        output_dir = "frames_test"
        
        # Convert video to images
        pattern = convert_video_to_images(video_path, output_dir, quality=90)
        
        # Test loading
        session = ImageSequenceSession(pattern)
        logging.info(f"Loaded {session.total_frames} frames")
        
        # Test frame access
        import time
        for i in [0, 100, 1000, 2000, 3000, 4000]:
            if i < session.total_frames:
                start = time.time()
                success, frame = session.get_frame(i)
                elapsed = (time.time() - start) * 1000
                logging.info(f"Frame {i}: {elapsed:.1f}ms")
    else:
        logging.info("Usage: python image_sequence_loader.py <video_file>")
