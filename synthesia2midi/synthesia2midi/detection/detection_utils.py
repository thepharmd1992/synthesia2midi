"""
Shared detection utilities for color progression and histogram calculations.

This module provides pure functions used by both StandardDetection and the Visual Threshold Monitor
to ensure identical calculations and prevent drift between the two systems.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Any
from ..core.app_state import OverlayConfig
from .roi_utils import euclidean_distance


def calculate_detection_parameters(
    overlay: OverlayConfig,
    current_avg_rgb_color: Tuple[int, int, int],
    unlit_ref_color: Tuple[int, int, int],
    exemplar_lit_colors: Dict[str, Optional[Tuple[int, int, int]]],
    detection_threshold: float,
    hist_ratio_threshold: float,
    use_histogram_detection: bool,
    hist_rule_hit: bool = False,
    allow_delta_override_sanity: bool = False
) -> Dict[str, Any]:
    """
    Calculate all detection parameters for an overlay.
    
    This is the source of truth for color progression and histogram calculations,
    used by both StandardDetection and the Visual Threshold Monitor.
    
    Args:
        overlay: The overlay being analyzed
        current_avg_rgb_color: Current average RGB color of the ROI
        unlit_ref_color: Reference color when key is unlit
        exemplar_lit_colors: Dictionary of exemplar lit colors (LW, LB, RW, RB, etc.)
        detection_threshold: Threshold for progression ratio
        hist_ratio_threshold: Threshold for histogram ratio
        use_histogram_detection: Whether histogram detection is enabled
        hist_rule_hit: Whether histogram rule was triggered (passed from caller)
        allow_delta_override_sanity: Whether delta detection can override sanity check
    
    Returns:
        Dictionary containing:
            - current_max_progression_ratio: Maximum progression ratio across valid exemplars
            - is_key_lit_by_color: Whether key is lit based on color progression
            - min_sanity_threshold: Minimum threshold for sanity check
            - progression_passes_sanity: Whether progression passes sanity check
            - base_lit: Base detection result (before delta)
            - base_lit_ignore_sanity: Base detection ignoring sanity (for delta override)
            - exemplars_checked: List of exemplar types that were checked
    """
    base_color_type = overlay.key_type[-1]  # "W" or "B"
    
    # Determine which exemplars to check based on key color
    # CRITICAL: This matches StandardDetection logic - only check matching color exemplars
    exemplar_types_to_check = []
    if base_color_type == "W":
        exemplar_types_to_check.extend(["LW", "RW"])
    elif base_color_type == "B":
        exemplar_types_to_check.extend(["LB", "RB"])
    else:
        # Invalid key type
        return {
            'current_max_progression_ratio': 0.0,
            'is_key_lit_by_color': False,
            'min_sanity_threshold': detection_threshold * 0.3,
            'progression_passes_sanity': False,
            'base_lit': False,
            'exemplars_checked': []
        }
    
    # Also check any additional COLOR_N exemplars that match the key type
    for key_type, color in exemplar_lit_colors.items():
        if key_type.startswith("COLOR_") and color is not None:
            if key_type.endswith(f"_{base_color_type}"):
                exemplar_types_to_check.append(key_type)
    
    # Calculate progression ratios for valid exemplars
    is_key_lit_by_color = False
    current_max_progression_ratio = 0.0
    
    for exemplar_key_type in exemplar_types_to_check:
        lit_ref_color = exemplar_lit_colors.get(exemplar_key_type)
        if lit_ref_color is None:
            continue
        
        try:
            d_unlit_to_lit_ref = euclidean_distance(unlit_ref_color, lit_ref_color)
            d_unlit_to_current = euclidean_distance(unlit_ref_color, current_avg_rgb_color)
            
            progression_ratio = 0.0
            if d_unlit_to_lit_ref > 1e-6:
                progression_ratio = d_unlit_to_current / d_unlit_to_lit_ref
            elif d_unlit_to_current < 1e-6:
                progression_ratio = 1.0
            
            current_max_progression_ratio = max(current_max_progression_ratio, progression_ratio)
            
            if progression_ratio >= detection_threshold:
                is_key_lit_by_color = True
        
        except Exception:
            continue
    
    # Sanity check - mirrors StandardDetection logic
    min_sanity_threshold = detection_threshold * 0.3
    progression_passes_sanity = current_max_progression_ratio >= min_sanity_threshold
    
    # Compute base_lit decision with sanity check
    if use_histogram_detection:
        base_lit_with_sanity = (is_key_lit_by_color or hist_rule_hit) and progression_passes_sanity
        base_lit_ignore_sanity = (is_key_lit_by_color or hist_rule_hit)
    else:
        base_lit_with_sanity = is_key_lit_by_color and progression_passes_sanity
        base_lit_ignore_sanity = is_key_lit_by_color
    
    # Choose which base_lit to use based on delta override setting
    if allow_delta_override_sanity:
        base_lit = base_lit_ignore_sanity
    else:
        base_lit = base_lit_with_sanity
    
    return {
        'current_max_progression_ratio': current_max_progression_ratio,
        'is_key_lit_by_color': is_key_lit_by_color,
        'min_sanity_threshold': min_sanity_threshold,
        'progression_passes_sanity': progression_passes_sanity,
        'base_lit': base_lit,
        'base_lit_ignore_sanity': base_lit_ignore_sanity,
        'exemplars_checked': exemplar_types_to_check
    }


def calculate_delta_value(
    current_progression_ratio: float,
    prev_progression_ratio: float
) -> float:
    """
    Calculate delta between current and previous progression ratios.
    
    Args:
        current_progression_ratio: Current progression ratio
        prev_progression_ratio: Previous progression ratio
        
    Returns:
        Delta value (current - previous)
    """
    return current_progression_ratio - prev_progression_ratio