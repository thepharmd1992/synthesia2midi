"""
Cross-platform font detection and selection utilities.

Provides reliable font selection across Windows, macOS, and Linux.
"""
import sys
from typing import List, Optional

from PySide6.QtGui import QFont, QFontDatabase


def get_system_default_font() -> str:
    """
    Get the system's default font family.
    
    Returns:
        Font family name
    """
    return QFont().defaultFamily()


def find_available_font(preferred_fonts: List[str]) -> str:
    """
    Find the first available font from a list of preferences.
    
    Args:
        preferred_fonts: List of font names in order of preference
        
    Returns:
        The first available font, or system default if none found
    """
    db = QFontDatabase()
    available_families = db.families()
    
    for font in preferred_fonts:
        if font in available_families:
            return font
    
    # If none found, return system default
    return get_system_default_font()


def get_monospace_font() -> str:
    """
    Get a suitable monospace font for the current platform.
    
    Returns:
        Monospace font family name
    """
    # Platform-specific preferences
    if sys.platform == "win32":
        preferences = ["Consolas", "Courier New", "Lucida Console"]
    elif sys.platform == "darwin":  # macOS
        preferences = ["SF Mono", "Monaco", "Menlo", "Courier New"]
    else:  # Linux and others
        preferences = ["DejaVu Sans Mono", "Ubuntu Mono", "Liberation Mono", 
                      "Monospace", "Courier New"]
    
    return find_available_font(preferences)


def get_sans_serif_font() -> str:
    """
    Get a suitable sans-serif font for the current platform.
    
    Returns:
        Sans-serif font family name
    """
    # Platform-specific preferences
    if sys.platform == "win32":
        preferences = ["Segoe UI", "Arial", "Calibri", "Tahoma"]
    elif sys.platform == "darwin":  # macOS
        preferences = ["SF Pro Display", "Helvetica Neue", "Helvetica", "Arial"]
    else:  # Linux and others
        preferences = ["Ubuntu", "DejaVu Sans", "Liberation Sans", 
                      "Sans", "Arial", "Helvetica"]
    
    return find_available_font(preferences)


def get_display_font() -> str:
    """
    Get a suitable display/UI font for overlays and annotations.
    This is what should replace hardcoded "Arial" in the codebase.
    
    Returns:
        Display font family name
    """
    return get_sans_serif_font()


def create_scaled_font(family: Optional[str] = None, size: int = 10, 
                      bold: bool = False) -> QFont:
    """
    Create a QFont with proper scaling for the current platform.
    
    Args:
        family: Font family (None for default)
        size: Base font size
        bold: Whether font should be bold
        
    Returns:
        Configured QFont object
    """
    if family is None:
        family = get_display_font()
    
    font = QFont(family)
    font.setPointSize(size)
    font.setBold(bold)
    
    # Platform-specific adjustments
    if sys.platform == "darwin":  # macOS
        # macOS typically renders fonts slightly smaller
        font.setPointSize(int(size * 1.1))
    
    return font