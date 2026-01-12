"""
Factory for creating detection methods based on configuration.

Provides a clean interface for selecting and configuring detection methods
without tight coupling to specific implementations.
"""
import logging
from typing import Dict, Optional, List, Tuple, Any

from .base import DetectionMethod, DetectionConfigError
from .standard import StandardDetection
from .spark_integrated import SparkIntegratedDetection
from synthesia2midi.app_config import OverlayConfig


class DetectionFactory:
    """
    Factory for creating appropriate detection methods based on configuration.
    
    Makes it easy to switch between detection methods and handles method-specific
    configuration and validation.
    """
    
    # Registry of available detection methods
    _methods = {
        'standard': StandardDetection,
        'spark_integrated': SparkIntegratedDetection
    }
    
    @classmethod
    def get_available_methods(cls) -> List[str]:
        """Get list of available detection method names."""
        return list(cls._methods.keys())
    
    @classmethod
    def create_detector(cls, 
                       method_name: str,
                       overlays: Optional[List[OverlayConfig]] = None,
                       **kwargs) -> DetectionMethod:
        """
        Create detector based on method name and configuration.
        
        Args:
            method_name: Name of detection method ('standard')
            overlays: List of overlay configurations
            **kwargs: Method-specific configuration parameters
            
        Returns:
            Configured detection method instance
            
        Raises:
            DetectionConfigError: If method unknown or configuration invalid
        """
        if method_name not in cls._methods:
            available = ', '.join(cls.get_available_methods())
            raise DetectionConfigError(
                f"Unknown detection method '{method_name}'. Available: {available}"
            )
        
        method_class = cls._methods[method_name]
        
        try:
            if method_name == 'standard':
                return method_class()
            elif method_name == 'spark_integrated':
                # Spark integrated requires app_state
                app_state = kwargs.get('app_state')
                if app_state is None:
                    raise DetectionConfigError("SparkIntegratedDetection requires app_state parameter")
                return method_class(app_state)
            else:
                # Generic creation for future methods
                return method_class(**kwargs)
                
        except Exception as e:
            raise DetectionConfigError(f"Failed to create {method_name} detector: {e}")
    
    @classmethod
    def create_from_app_state(cls, app_state, overlays: List[OverlayConfig], 
                            navigation_mode: bool = False) -> DetectionMethod:
        """
        Create detector based on current app state configuration.
        
        Args:
            app_state: Application state with detection settings
            overlays: List of overlay configurations
            navigation_mode: If True, creates lightweight detector for navigation
            
        Returns:
            Configured detection method based on app state
        """
        logger = logging.getLogger(f"{__name__}.DetectionFactory")
        
        logger.debug("[DETECTION-FACTORY] Creating detector:")
        logger.debug(f"  - Navigation mode: {navigation_mode}")
        logger.debug(f"  - Spark detection enabled: {app_state.detection.spark_detection_enabled}")
        
        # Check if spark detection is configured and calibrated
        detection_state = app_state.detection
        # Check if we have any complete calibration pair for key-type-specific calibration
        has_lw_cal = (detection_state.spark_calibration_lw_bar_only is not None and
                      detection_state.spark_calibration_lw_brightest_sparks is not None)
        has_lb_cal = (detection_state.spark_calibration_lb_bar_only is not None and
                      detection_state.spark_calibration_lb_brightest_sparks is not None)
        has_rw_cal = (detection_state.spark_calibration_rw_bar_only is not None and
                      detection_state.spark_calibration_rw_brightest_sparks is not None)
        has_rb_cal = (detection_state.spark_calibration_rb_bar_only is not None and
                      detection_state.spark_calibration_rb_brightest_sparks is not None)
        
        has_any_calibration = has_lw_cal or has_lb_cal or has_rw_cal or has_rb_cal
        
        spark_available = (
            detection_state.spark_detection_enabled and  # User has enabled spark detection
            detection_state.spark_roi_top > 0 and
            detection_state.spark_roi_bottom > detection_state.spark_roi_top and
            has_any_calibration and
            detection_state.spark_brightness_threshold > 0
        )
        
        # Force standard detection in navigation mode to avoid memory issues
        if navigation_mode:
            spark_available = False
            logger.debug("[DETECTION-FACTORY] Navigation mode - forcing standard detection")
        
        # Priority: Spark > Standard
        if spark_available:
            # Use spark-integrated detection
            logger.debug("[DETECTION-FACTORY] Creating SPARK INTEGRATED detection method")
            detector = cls.create_detector('spark_integrated', app_state=app_state)
            logger.debug("[DETECTION-FACTORY] Created SPARK-INTEGRATED detector")
        else:
            # Fall back to standard detection
            reasons = []
            if not detection_state.spark_detection_enabled:
                reasons.append("spark detection disabled")
            if not has_any_calibration:
                reasons.append("no calibration data")
            if detection_state.spark_roi_top <= 0 or detection_state.spark_roi_bottom <= detection_state.spark_roi_top:
                reasons.append("invalid ROI")
            logger.debug(f"[DETECTION-FACTORY] Creating STANDARD detection method (reasons: {', '.join(reasons)})")
            detector = cls.create_detector('standard')
            logger.debug(f"[DETECTION-FACTORY] Created STANDARD detector (reasons: {', '.join(reasons)})")
        
        return detector
    
    @classmethod
    def validate_method_config(cls, method_name: str, **kwargs) -> List[str]:
        """
        Validate configuration for a specific detection method.
        
        Args:
            method_name: Name of detection method
            **kwargs: Configuration parameters to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        if method_name not in cls._methods:
            return [f"Unknown detection method: {method_name}"]
        
        try:
            # Create temporary instance for validation
            if method_name == 'standard':
                detector = StandardDetection()
            elif method_name == 'spark_integrated':
                # Need mock app_state for validation
                from synthesia2midi.core.app_state import AppState
                mock_app_state = AppState()
                detector = SparkIntegratedDetection(mock_app_state)
            else:
                return []  # No validation for unknown methods
            
            return detector.validate_parameters(**kwargs)
            
        except Exception as e:
            return [f"Validation error for {method_name}: {e}"]
    
    @classmethod
    def get_method_info(cls, method_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific detection method.
        
        Args:
            method_name: Name of detection method
            
        Returns:
            Method information dictionary or None if method unknown
        """
        if method_name not in cls._methods:
            return None
        
        try:
            if method_name == 'standard':
                detector = StandardDetection()
            elif method_name == 'spark_integrated':
                # Need mock app_state for info
                from synthesia2midi.core.app_state import AppState
                mock_app_state = AppState()
                detector = SparkIntegratedDetection(mock_app_state)
            else:
                return None
            
            return detector.get_method_info()
            
        except Exception as e:
            logging.error(f"Error getting info for {method_name}: {e}")
            return None