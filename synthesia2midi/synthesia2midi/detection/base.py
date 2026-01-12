"""
Base classes and interfaces for all detection methods.

Provides clear interfaces that make it easy to add new detection algorithms
and test them independently. Each detection method implements the 
DetectionMethod interface.
"""
from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from synthesia2midi.app_config import OverlayConfig


class DetectionMethod(ABC):
    """
    Abstract base class for all detection methods.
    
    Provides a consistent interface for different detection algorithms,
    making it easy to swap methods and test them independently.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug(f"Initialized {name} detection method")
    
    def get_name(self) -> str:
        """Get the human-readable name of this detection method."""
        return self.name
    
    @abstractmethod
    def detect_frame(self, 
                    frame_bgr: np.ndarray, 
                    overlays: List[OverlayConfig],
                    exemplar_lit_colors: Dict[str, Optional[Tuple[int, int, int]]],
                    detection_threshold: float,
                    **kwargs) -> Set[int]:
        """
        Detect pressed keys in a single frame.
        
        Args:
            frame_bgr: Video frame in BGR format (OpenCV standard)
            overlays: List of overlay configurations defining key regions
            exemplar_lit_colors: Reference colors for lit keys by type
            detection_threshold: Primary threshold for detection
            **kwargs: Additional method-specific parameters
            
        Returns:
            Set of key_ids for pressed keys detected in this frame
        """
        pass
    
    @abstractmethod
    def reset_state(self):
        """
        Reset any internal state (called when switching videos or restarting).
        
        Each detection method may maintain frame-to-frame state that needs
        to be cleared when starting with a new video or configuration.
        """
        pass
    
    @abstractmethod
    def get_method_info(self) -> Dict[str, any]:
        """
        Get information about this detection method.
        
        Returns:
            Dict with method name, description, and parameter info
        """
        pass
    
    def validate_parameters(self, **kwargs) -> List[str]:
        """
        Validate method-specific parameters.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Common parameter validation
        if 'detection_threshold' in kwargs:
            threshold = kwargs['detection_threshold']
            if not 0.1 <= threshold <= 0.99:
                errors.append(f"Detection threshold {threshold} must be between 0.1 and 0.99")
        
        return errors


class DetectionResult:
    """
    Container for detection results with metadata.
    
    Provides additional information about the detection process
    beyond just the set of pressed keys.
    """
    
    def __init__(self, 
                 pressed_keys: Set[int],
                 method_name: str,
                 frame_index: Optional[int] = None,
                 processing_time_ms: Optional[float] = None,
                 debug_info: Optional[Dict] = None):
        self.pressed_keys = pressed_keys
        self.method_name = method_name
        self.frame_index = frame_index
        self.processing_time_ms = processing_time_ms
        self.debug_info = debug_info or {}
    
    def __len__(self):
        """Number of pressed keys detected."""
        return len(self.pressed_keys)
    
    def __contains__(self, key_id: int):
        """Check if a key was detected as pressed."""
        return key_id in self.pressed_keys
    
    def __iter__(self):
        """Iterate over pressed key IDs."""
        return iter(self.pressed_keys)


class DetectionError(Exception):
    """Base exception for detection-related errors."""
    pass


class DetectionConfigError(DetectionError):
    """Exception raised for detection configuration errors."""
    pass


class DetectionProcessingError(DetectionError):
    """Exception raised during detection processing."""
    pass