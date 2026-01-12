"""
Centralized logging configuration for synthesia2midi.

This module provides a unified logging setup that can be configured
based on the user's requirements (per user instructions: WARNING and ERROR levels).
"""
import logging
import os
import datetime
from typing import Optional, Dict, Any

# Default log directory from app_config
from synthesia2midi.app_config import LOG_DIR


class LoggingConfig:
    """Centralized logging configuration manager."""
    
    # Default logging level per user requirements
    DEFAULT_LEVEL = logging.WARNING
    
    # Log format
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    
    # Module-specific log levels can be configured here
    MODULE_LEVELS: Dict[str, int] = {
        # Critical modules that might need more verbose logging
        'synthesia2midi.detection': logging.WARNING,
        'synthesia2midi.workflows': logging.WARNING,
        'synthesia2midi.workflows.video_loading': logging.INFO,  # Enable INFO for frame conversion debugging
        'synthesia2midi.video_loader': logging.INFO,  # Enable INFO for video loader debugging
        'synthesia2midi.utils.ffmpeg_helper': logging.INFO,  # Enable INFO for ffmpeg debugging
        'synthesia2midi.gui': logging.ERROR,  # GUI errors only
        
        # Third-party libraries - suppress most of their logs
        'PIL': logging.ERROR,
        'matplotlib': logging.ERROR,
        'numpy': logging.ERROR,
        'cv2': logging.ERROR,
    }
    
    @classmethod
    def setup_logging(cls, 
                     log_to_file: bool = True,
                     log_to_console: bool = False,
                     log_level: Optional[int] = None,
                     log_dir: Optional[str] = None) -> str:
        """
        Setup centralized logging configuration.
        
        Args:
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
            log_level: Override default log level
            log_dir: Override default log directory
            
        Returns:
            Path to log file if logging to file, empty string otherwise
        """
        # Use provided level or default
        root_level = log_level or cls.DEFAULT_LEVEL
        
        # Create handlers list
        handlers = []
        log_filename = ""
        
        # File handler
        if log_to_file:
            log_dir = log_dir or LOG_DIR
            os.makedirs(log_dir, exist_ok=True)
            log_filename = os.path.join(
                log_dir, 
                f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            file_handler = logging.FileHandler(log_filename, mode='w')
            file_handler.setFormatter(logging.Formatter(cls.LOG_FORMAT))
            handlers.append(file_handler)
        
        # Console handler (useful for development/debugging)
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(cls.LOG_FORMAT))
            handlers.append(console_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=root_level,
            format=cls.LOG_FORMAT,
            handlers=handlers,
            force=True  # Force reconfiguration if already configured
        )
        
        # Apply module-specific log levels
        for module_name, level in cls.MODULE_LEVELS.items():
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(level)
        
        # Suppress debug/info for most synthesia2midi modules, but allow INFO for video optimization
        synthesia2midi_logger = logging.getLogger('synthesia2midi')
        synthesia2midi_logger.setLevel(logging.WARNING)
        
        # Allow INFO level for video optimization logging
        video_opt_modules = [
            'synthesia2midi.workflows.video_optimization',
            'synthesia2midi.workflows.video_loading',
            'synthesia2midi.detection.frame_optimizer',  # Enable frame optimizer logging
            'synthesia2midi.workflows.conversion'  # Enable conversion timing logs
        ]
        for module in video_opt_modules:
            logging.getLogger(module).setLevel(logging.INFO)
        
        # Log that video optimization logging is enabled
        logger = logging.getLogger(__name__)
        logger.warning("[LOGGING-CONFIG] Video optimization logging ENABLED for modules:")
        for module in video_opt_modules:
            logger.warning(f"[LOGGING-CONFIG]   - {module}: INFO level")
        
        # Ensure specific noisy modules are suppressed
        noisy_modules = [
            'synthesia2midi.detection.monolithic_detector',
            'synthesia2midi.detection.factory', 
            'synthesia2midi.detection.auto_detect_adapter'
        ]
        for module in noisy_modules:
            logging.getLogger(module).setLevel(logging.WARNING)
        
        # Log the configuration
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured: level={logging.getLevelName(root_level)}, "
                   f"file={'yes' if log_to_file else 'no'}, "
                   f"console={'yes' if log_to_console else 'no'}")
        if log_to_file:
            logger.info(f"Log file: {log_filename}")
            
        return log_filename
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger instance with the appropriate configuration.
        
        Args:
            name: Logger name (typically __name__)
            
        Returns:
            Configured logger instance
        """
        return logging.getLogger(name)
    
    @classmethod
    def set_module_level(cls, module_name: str, level: int):
        """
        Set logging level for a specific module.
        
        Args:
            module_name: Module name (e.g., 'synthesia2midi.detection')
            level: Logging level (e.g., logging.WARNING)
        """
        logger = logging.getLogger(module_name)
        logger.setLevel(level)
        cls.MODULE_LEVELS[module_name] = level
    
    @classmethod
    def suppress_verbose_libraries(cls):
        """Suppress verbose logging from third-party libraries."""
        verbose_libs = [
            'PIL', 'PIL.Image', 'PIL.PngImagePlugin',
            'matplotlib', 'matplotlib.pyplot', 'matplotlib.font_manager',
            'numpy', 'scipy', 'cv2', 'midiutil',
            'urllib3', 'requests', 'h5py',
        ]
        
        for lib in verbose_libs:
            logging.getLogger(lib).setLevel(logging.ERROR)


# Convenience function for backward compatibility
def setup_logging(**kwargs) -> str:
    """Setup logging with default configuration."""
    return LoggingConfig.setup_logging(**kwargs)