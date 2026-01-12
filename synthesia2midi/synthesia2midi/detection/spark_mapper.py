"""
Spark zone mapping utility for connecting key overlays to spark detection regions.

This module handles the crucial logic of mapping each piano key overlay to its
corresponding spark detection zone within the spark ROI. The approach is:

1. Each key overlay defines a horizontal region (x, width)
2. Spark ROI defines a vertical band (top_y, bottom_y) 
3. Intersection creates a "spark zone" for each key
4. Spark zones are used for targeted brightness detection

This avoids the complexity of manually positioning 88 individual spark overlays
while ensuring precise key-to-spark mapping.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.core.app_state import AppState


class SparkZone(NamedTuple):
    """
    Represents a spark detection zone for a specific key.
    
    This is the intersection of a key overlay's horizontal extent
    with the global spark ROI's vertical extent.
    """
    key_id: int
    x: int           # Left edge in image coordinates
    y: int           # Top edge (spark ROI top)
    width: int       # Width (from key overlay)
    height: int      # Height (spark ROI height)
    note_name: str   # For debugging/display


@dataclass
class SparkMappingConfig:
    """Configuration parameters for spark zone mapping."""
    
    # Minimum zone dimensions (prevent tiny zones)
    min_zone_width: int = 5
    min_zone_height: int = 10
    
    # Zone padding (extend zones slightly beyond overlay bounds)
    horizontal_padding: int = 2
    vertical_padding: int = 0  # Usually 0 since ROI defines exact vertical bounds
    
    # Overlap handling
    allow_zone_overlap: bool = True  # Whether adjacent zones can overlap
    overlap_resolve_mode: str = "split"  # "split", "priority", "larger_wins"


class SparkZoneMapper:
    """
    Handles mapping between key overlays and spark detection zones.
    
    This class provides the core logic for creating spark zones from
    the intersection of key overlays and the spark ROI region.
    """
    
    def __init__(self, config: Optional[SparkMappingConfig] = None):
        self.config = config or SparkMappingConfig()
        self.logger = logging.getLogger(f"{__name__}.SparkZoneMapper")
        
        # Cached mappings
        self._cached_zones: Optional[List[SparkZone]] = None
        self._cache_state_hash: Optional[str] = None
        
    def get_spark_zones(self, app_state: AppState) -> List[SparkZone]:
        """
        Get spark zones for all key overlays.
        
        Returns cached zones if app state hasn't changed, otherwise
        recalculates zones from current overlays and spark ROI.
        
        Args:
            app_state: Application state with overlays and spark ROI
            
        Returns:
            List of spark zones, one per valid key overlay
        """
        # Check if we can use cached zones
        current_hash = self._calculate_state_hash(app_state)
        if (self._cached_zones is not None and 
            self._cache_state_hash == current_hash):
            return self._cached_zones
        
        # Recalculate zones
        zones = self._calculate_spark_zones(app_state)
        
        # Update cache
        self._cached_zones = zones
        self._cache_state_hash = current_hash
        
        self.logger.debug(f"Calculated {len(zones)} spark zones")
        return zones
    
    def get_spark_zone_for_key(self, key_id: int, app_state: AppState) -> Optional[SparkZone]:
        """
        Get the spark zone for a specific key.
        
        Args:
            key_id: Key identifier
            app_state: Application state
            
        Returns:
            SparkZone for the key, or None if key not found
        """
        zones = self.get_spark_zones(app_state)
        for zone in zones:
            if zone.key_id == key_id:
                return zone
        return None
    
    def is_spark_roi_valid(self, app_state: AppState) -> bool:
        """
        Check if spark ROI is properly configured.
        
        Args:
            app_state: Application state
            
        Returns:
            True if spark ROI is valid for zone mapping
        """
        roi_top = app_state.detection.spark_roi_top
        roi_bottom = app_state.detection.spark_roi_bottom
        
        return (roi_top > 0 and 
                roi_bottom > roi_top and
                roi_bottom - roi_top >= self.config.min_zone_height)
    
    def get_mapping_stats(self, app_state: AppState) -> Dict[str, int]:
        """
        Get statistics about current spark zone mapping.
        
        Args:
            app_state: Application state
            
        Returns:
            Dictionary with mapping statistics
        """
        if not self.is_spark_roi_valid(app_state):
            return {
                "total_overlays": len(app_state.overlays),
                "valid_zones": 0,
                "roi_valid": False,
                "average_zone_width": 0,
                "average_zone_height": 0
            }
        
        zones = self.get_spark_zones(app_state)
        
        if not zones:
            avg_width = avg_height = 0
        else:
            avg_width = sum(z.width for z in zones) // len(zones)
            avg_height = sum(z.height for z in zones) // len(zones)
        
        return {
            "total_overlays": len(app_state.overlays),
            "valid_zones": len(zones),
            "roi_valid": True,
            "average_zone_width": avg_width,
            "average_zone_height": avg_height
        }
    
    def _calculate_spark_zones(self, app_state: AppState) -> List[SparkZone]:
        """Calculate spark zones from current app state."""
        if not self.is_spark_roi_valid(app_state):
            self.logger.warning("Cannot calculate spark zones - invalid ROI")
            return []
        
        zones = []
        roi_top = app_state.detection.spark_roi_top
        roi_bottom = app_state.detection.spark_roi_bottom
        roi_height = roi_bottom - roi_top
        
        for overlay in app_state.overlays:
            # Create spark zone from overlay horizontal extent + ROI vertical extent
            zone_x = max(0, overlay.x - self.config.horizontal_padding)
            zone_width = overlay.width + (2 * self.config.horizontal_padding)
            zone_y = roi_top
            zone_height = roi_height
            
            # Apply minimum size constraints
            if (zone_width >= self.config.min_zone_width and 
                zone_height >= self.config.min_zone_height):
                
                zone = SparkZone(
                    key_id=overlay.key_id,
                    x=zone_x,
                    y=zone_y,
                    width=zone_width,
                    height=zone_height,
                    note_name=overlay.get_full_note_name()
                )
                zones.append(zone)
                
                self.logger.debug(f"Created spark zone for {zone.note_name}: "
                                f"({zone.x}, {zone.y}, {zone.width}, {zone.height})")
        
        return zones
    
    def _calculate_state_hash(self, app_state: AppState) -> str:
        """
        Calculate hash representing current state for caching.
        
        Includes overlay positions/sizes and spark ROI bounds.
        """
        # Create hash from overlay positions and ROI bounds
        overlay_data = []
        for overlay in app_state.overlays:
            overlay_data.append(f"{overlay.key_id}:{overlay.x}:{overlay.y}:{overlay.width}:{overlay.height}")
        
        roi_data = f"{app_state.detection.spark_roi_top}:{app_state.detection.spark_roi_bottom}"
        
        combined = "|".join(overlay_data) + f"|ROI:{roi_data}"
        return str(hash(combined))
    
    def invalidate_cache(self):
        """Force recalculation of spark zones on next access."""
        self._cached_zones = None
        self._cache_state_hash = None
        self.logger.debug("Spark zone cache invalidated")


# Global mapper instance
_spark_mapper_instance: Optional[SparkZoneMapper] = None


def get_spark_mapper() -> SparkZoneMapper:
    """Get the global spark zone mapper instance."""
    global _spark_mapper_instance
    if _spark_mapper_instance is None:
        _spark_mapper_instance = SparkZoneMapper()
    return _spark_mapper_instance


def get_spark_zones(app_state: AppState) -> List[SparkZone]:
    """Convenience function to get spark zones using global mapper."""
    return get_spark_mapper().get_spark_zones(app_state)


def get_spark_zone_for_key(key_id: int, app_state: AppState) -> Optional[SparkZone]:
    """Convenience function to get spark zone for specific key."""
    return get_spark_mapper().get_spark_zone_for_key(key_id, app_state)