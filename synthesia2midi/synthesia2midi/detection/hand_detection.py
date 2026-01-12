"""
Hand detection logic based on hue analysis.

Uses established hue thresholds to classify individual note instances 
as left-hand or right-hand played based on color analysis.
"""
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

# Established hue centers from analysis
LEFT_HAND_HUE_CENTER = 159.3   # Blue-Green family
RIGHT_HAND_HUE_CENTER = 100.8  # Yellow-Green family


def classify_hand_by_hue(average_hue: float) -> Tuple[str, float]:
    """
    Classify hand (left or right) based on average hue value.
    
    Args:
        average_hue: Average hue value (0-360 degrees) from the overlay region
        
    Returns:
        Tuple of (hand_classification, confidence_score)
        hand_classification: "LEFT" or "RIGHT"
        confidence_score: 0.0-1.0 where 1.0 is highest confidence
    """
    # Calculate distance to each hue center
    dist_to_left = abs(average_hue - LEFT_HAND_HUE_CENTER)
    dist_to_right = abs(average_hue - RIGHT_HAND_HUE_CENTER)
    
    # Handle wraparound for hue values (360° = 0°)
    if dist_to_left > 180:
        dist_to_left = 360 - dist_to_left
    if dist_to_right > 180:
        dist_to_right = 360 - dist_to_right
    
    # Classify based on closest center
    if dist_to_left < dist_to_right:
        hand = "LEFT"
        confidence = calculate_confidence(dist_to_left)
    else:
        hand = "RIGHT"
        confidence = calculate_confidence(dist_to_right)
    
    return hand, confidence


def calculate_confidence(distance_to_center: float) -> float:
    """
    Calculate confidence score based on distance from hue center.
    
    Args:
        distance_to_center: Angular distance from the nearest hue center
        
    Returns:
        Confidence score 0.0-1.0 where 1.0 is highest confidence
    """
    # Maximum meaningful distance (90 degrees = quarter of color wheel)
    max_distance = 90.0
    
    # Normalize distance to 0-1 range, then invert for confidence
    normalized_distance = min(distance_to_center / max_distance, 1.0)
    confidence = 1.0 - normalized_distance
    
    return confidence


def extract_hue_from_roi(roi_bgr: np.ndarray) -> Optional[float]:
    """
    Extract average hue value from a BGR image region.
    
    Args:
        roi_bgr: BGR image region (numpy array)
        
    Returns:
        Average hue value in degrees (0-360), or None if extraction fails
    """
    try:
        if roi_bgr is None or roi_bgr.size == 0:
            return None
            
        # Convert BGR to HSV
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        
        # Extract hue channel (0-179 in OpenCV, need to convert to 0-360)
        hue_channel = hsv[:, :, 0]
        
        # Calculate mean hue and convert to 0-360 degrees
        mean_hue_cv = np.mean(hue_channel)
        mean_hue_degrees = mean_hue_cv * 2.0  # OpenCV hue is 0-179, convert to 0-360
        
        return mean_hue_degrees
        
    except Exception as e:
        logging.warning(f"Failed to extract hue from ROI: {e}")
        return None


def detect_hand_for_overlay_frame(roi_bgr: np.ndarray) -> Tuple[Optional[str], float]:
    """
    Detect hand classification for a specific overlay at a specific frame.
    
    Args:
        roi_bgr: BGR image region from the overlay
        
    Returns:
        Tuple of (hand_classification, confidence_score)
        hand_classification: "LEFT", "RIGHT", or None if detection failed
        confidence_score: 0.0-1.0 confidence level
    """
    # Extract hue from the region
    average_hue = extract_hue_from_roi(roi_bgr)
    
    if average_hue is None:
        return None, 0.0
    
    # Classify based on hue
    hand, confidence = classify_hand_by_hue(average_hue)
    
    logging.debug(f"Hand detection: hue={average_hue:.1f}°, hand={hand}, confidence={confidence:.2f}")
    
    return hand, confidence


def get_key_type_for_overlay(overlay_config, detected_hand: str) -> str:
    """
    Determine key type (LW/LB/RW/RB) based on overlay properties and detected hand.
    
    Args:
        overlay_config: OverlayConfig object with note information
        detected_hand: "LEFT" or "RIGHT" from hand detection
        
    Returns:
        Key type string: "LW", "LB", "RW", or "RB"
    """
    # Get existing key type from overlay
    existing_key_type = overlay_config.key_type or ""
    note_name = overlay_config.note_name_in_octave
    
    logging.debug(f"[HAND-DETECT] get_key_type_for_overlay called:")
    logging.debug(f"[HAND-DETECT]   overlay_config.key_id: {overlay_config.key_id}")
    logging.debug(f"[HAND-DETECT]   overlay_config.note_name_in_octave: '{note_name}'")
    logging.debug(f"[HAND-DETECT]   overlay_config.key_type: '{existing_key_type}'")
    logging.debug(f"[HAND-DETECT]   detected_hand: {detected_hand}")
    
    # Strategy 1: Use existing key_type's color but update hand based on detection
    if existing_key_type and len(existing_key_type) == 2:
        existing_color = existing_key_type[1]  # W or B
        hand_prefix = "L" if detected_hand == "LEFT" else "R"
        key_type_from_existing = f"{hand_prefix}{existing_color}"
        logging.debug(f"[HAND-DETECT]   key_type_from_existing: {key_type_from_existing}")
    else:
        key_type_from_existing = None
        logging.debug(f"[HAND-DETECT]   key_type_from_existing: None (invalid existing key_type)")
    
    # Strategy 2: Determine from note name (original logic)
    is_white_key = "♯" not in note_name and "♭" not in note_name
    hand_prefix = "L" if detected_hand == "LEFT" else "R"
    color_suffix = "W" if is_white_key else "B"
    key_type_from_note = f"{hand_prefix}{color_suffix}"
    logging.debug(f"[HAND-DETECT]   key_type_from_note: {key_type_from_note}")
    
    # Use existing key_type's color if available, otherwise fall back to note analysis
    final_key_type = key_type_from_existing if key_type_from_existing else key_type_from_note
    logging.debug(f"[HAND-DETECT]   final_key_type: {final_key_type}")
    
    return final_key_type


def log_hand_detection_stats(detections: list):
    """
    Log summary statistics for hand detection results.
    
    Args:
        detections: List of (hand, confidence) tuples from detection results
    """
    if not detections:
        logging.info("No hand detections to analyze")
        return
    
    left_count = sum(1 for hand, _ in detections if hand == "LEFT")
    right_count = sum(1 for hand, _ in detections if hand == "RIGHT")
    failed_count = sum(1 for hand, _ in detections if hand is None)
    
    total_confidence = sum(conf for _, conf in detections if conf > 0)
    avg_confidence = total_confidence / len([c for _, c in detections if c > 0]) if total_confidence > 0 else 0
    
    logging.info(f"Hand detection summary: LEFT={left_count}, RIGHT={right_count}, "
                f"FAILED={failed_count}, avg_confidence={avg_confidence:.2f}")