"""
Standard detection method: color progression with optional histogram and delta rules.

This detector compares each key ROI to calibrated unlit/lit exemplars and returns
the set of pressed key IDs for the current frame.
"""
import logging
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from synthesia2midi.app_config import OverlayConfig

from .base import DetectionMethod
from .roi_cache import ROICache
from .roi_utils import (
    euclidean_distance, 
    get_average_color_from_roi, 
    get_hist_feature, 
    get_hist_feature_from_hsv,
    hist_distance
)
from .detection_utils import calculate_detection_parameters

SIMILARITY_RATIO_DEFAULT = 0.60


class StandardDetection(DetectionMethod):
    """
    Standard key-press detection using ROI color progression.
    
    Algorithm:
    1. Color progression: Compare current color to unlit→lit reference progression
    2. Optional histogram: Compare current histogram to unlit→lit histogram progression
    3. Optional black filter: Filter out black-key spill between adjacent keys
    4. Optional delta: Apply frame-to-frame change thresholds for press/release timing
    """
    
    def __init__(self):
        super().__init__("Standard Detection")
        self.roi_cache = ROICache()
        self._warned_missing_unlit_calibration = False
        
    def detect_frame(self, 
                    frame_bgr: np.ndarray, 
                    overlays: List[OverlayConfig],
                    exemplar_lit_colors: Dict[str, Optional[Tuple[int, int, int]]],
                    exemplar_lit_histograms: Dict[str, Optional[np.ndarray]],
                    detection_threshold: float,
                    hist_ratio_threshold: float = 0.8,
                    rise_delta_threshold: float = 0.15,
                    fall_delta_threshold: float = 0.15,
                    use_histogram_detection: bool = False,
                    use_delta_detection: bool = True,
                    similarity_ratio: float = SIMILARITY_RATIO_DEFAULT,
                    apply_black_filter: bool = True,
                    **kwargs) -> Set[int]:
        """
        Detect pressed keys using standard color progression algorithm.
        
        Args:
            frame_bgr: Video frame in BGR format (OpenCV standard)
            overlays: List of overlay configurations defining key regions
            exemplar_lit_colors: Reference colors for lit keys by type (LW/LB/RW/RB)
            exemplar_lit_histograms: Reference histograms for lit keys by type
            detection_threshold: Primary threshold for color-based detection (0.1-0.99)
            hist_ratio_threshold: Threshold for histogram-based detection (0.0-1.0)
            rise_delta_threshold: Frame-to-frame delta for key press detection
            fall_delta_threshold: Frame-to-frame delta for key release detection
            use_histogram_detection: Enable histogram comparison method
            use_delta_detection: Enable frame-to-frame delta filtering
            similarity_ratio: Color similarity ratio for detection
            apply_black_filter: Filter adjacent black key false positives
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Set of key_ids for pressed keys detected in this frame
            
        Note:
            Black key filtering is applied before delta detection to prevent
            adjacent black keys from lighting up when main keys are released.
        """
        # Clear ROI cache for new frame
        self.roi_cache.clear()
        
        pressed_key_ids: Set[int] = set()
        base_detected_key_ids: Set[int] = set()  # Keys detected before delta filtering

        # EARLY EXIT 1: No overlays to process
        if not overlays:
            return pressed_key_ids
            
        # EARLY EXIT 2: Quick check if frame is mostly black
        # DISABLED: This check was causing false positives on piano videos where
        # the background is black but piano areas have valid data
        # if self._is_frame_black(frame_bgr):
        #     self.logger.error(f"UNIT-TEST-DEBUG: EARLY EXIT - Frame is black")
        #     # Update all overlay states to off
        #     for overlay in overlays:
        #         overlay.last_progression_ratio = 0.0
        #         overlay.last_is_lit = False
        #     return pressed_key_ids
        
        # First pass: Calculate base detection (color + histogram) for all overlays
        overlay_detection_data = {}
        missing_unlit_calibration = 0

        for overlay in overlays:
            # Update progression ratio history before calculations
            overlay.prev_progression_ratio = overlay.last_progression_ratio

            if overlay.key_type is None or overlay.unlit_reference_color is None:
                # Not calibrated - skip but maintain state consistency    
                missing_unlit_calibration += 1
                overlay.last_progression_ratio = 0.0
                overlay.last_is_lit = False
                continue

            unlit_ref_color = overlay.unlit_reference_color
            
            # Use cached ROI extraction
            roi_bgr = self.roi_cache.get_roi_bgr(frame_bgr, overlay)
            if roi_bgr is None:
                # Could not extract ROI - skip
                overlay.last_progression_ratio = 0.0
                overlay.last_is_lit = False
                continue
                
            # Calculate average color from cached ROI (more efficient without reshape)
            avg_bgr = np.mean(roi_bgr, axis=(0, 1))
            current_avg_rgb_color = (int(round(avg_bgr[2])), int(round(avg_bgr[1])), int(round(avg_bgr[0])))  # Convert BGR to RGB

            # First calculate detection parameters without histogram
            # Allow delta override of sanity when delta detection is enabled
            detection_params = calculate_detection_parameters(
                overlay=overlay,
                current_avg_rgb_color=current_avg_rgb_color,
                unlit_ref_color=unlit_ref_color,
                exemplar_lit_colors=exemplar_lit_colors,
                detection_threshold=detection_threshold,
                hist_ratio_threshold=hist_ratio_threshold,
                use_histogram_detection=False,  # First pass without histogram
                hist_rule_hit=False,
                allow_delta_override_sanity=use_delta_detection
            )
            
            # Optional histogram detection with early exit
            hist_rule_hit = False
            if use_histogram_detection and self._should_check_histogram(detection_params['current_max_progression_ratio']):
                hist_rule_hit = self._apply_histogram_detection(
                    overlay, frame_bgr, hist_ratio_threshold, exemplar_lit_histograms, kwargs
                )
                
                # Recalculate with histogram result
                detection_params = calculate_detection_parameters(
                    overlay=overlay,
                    current_avg_rgb_color=current_avg_rgb_color,
                    unlit_ref_color=unlit_ref_color,
                    exemplar_lit_colors=exemplar_lit_colors,
                    detection_threshold=detection_threshold,
                    hist_ratio_threshold=hist_ratio_threshold,
                    use_histogram_detection=use_histogram_detection,
                    hist_rule_hit=hist_rule_hit,
                    allow_delta_override_sanity=use_delta_detection
                )
            
            # Extract values from shared calculation
            is_key_lit_by_color = detection_params['is_key_lit_by_color']
            current_max_progression_ratio = detection_params['current_max_progression_ratio']
            min_sanity_threshold = detection_params['min_sanity_threshold']
            progression_passes_sanity = detection_params['progression_passes_sanity']
            base_lit = detection_params['base_lit']
            
            # TARGETED DEBUG: Log any activity on problematic overlays
            if overlay.key_id in [36, 43, 48] and self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "DEBUG-KEY-%s: ratio=%.6f, is_lit_by_color=%s, sanity_pass=%s, threshold=%s",
                    overlay.key_id,
                    current_max_progression_ratio,
                    is_key_lit_by_color,
                    progression_passes_sanity,
                    detection_threshold,
                )
                
            # TARGETED DEBUG: Log base_lit result
            if overlay.key_id in [36, 43, 48] and self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("DEBUG-KEY-%s: base_lit=%s", overlay.key_id, base_lit)

            # Store detection data for this overlay
            overlay_detection_data[overlay.key_id] = {
                'is_key_lit_by_color': is_key_lit_by_color,
                'hist_rule_hit': hist_rule_hit,
                'current_max_progression_ratio': current_max_progression_ratio,
                'base_lit': base_lit
            }


            # Add to base detected keys if lit by base detection
            if base_lit:
                base_detected_key_ids.add(overlay.key_id)

        # Apply black key filtering to base detection results BEFORE delta      
        if apply_black_filter:
            base_detected_key_ids = self._apply_black_key_filter(
                base_detected_key_ids, overlays, similarity_ratio
            )

        if missing_unlit_calibration and not self._warned_missing_unlit_calibration:
            self.logger.warning(
                "Skipping %d overlays without unlit calibration; run Unlit Key Calibration to enable pressed-key detection.",
                missing_unlit_calibration,
            )
            self._warned_missing_unlit_calibration = True

        # Second pass: Apply delta detection to the filtered base results
        for overlay in overlays:
            if overlay.key_id not in overlay_detection_data:
                continue
                
            detection_data = overlay_detection_data[overlay.key_id]
            
            # Check if this key survived base detection + black filtering
            key_lit_after_base_filtering = overlay.key_id in base_detected_key_ids
            
            # TARGETED DEBUG: Log filtering results
            if overlay.key_id in [36, 43, 48] and self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "DEBUG-KEY-%s: after_filtering=%s, base_keys=%s",
                    overlay.key_id,
                    key_lit_after_base_filtering,
                    sorted(base_detected_key_ids),
                )
            
            
            # Apply delta detection for timing
            delta_allows_on, delta_forces_off = False, False
            if use_delta_detection:
                delta_allows_on, delta_forces_off = self._apply_delta_detection(
                    overlay, detection_data['current_max_progression_ratio'], 
                    rise_delta_threshold, fall_delta_threshold, detection_threshold
                )

            # Final decision: combine base detection (post-filtering) with delta
            if use_delta_detection:
                if delta_forces_off:
                    lit_now = False  # Delta forces off regardless
                elif not overlay.last_is_lit:
                    # Was off - need delta approval to turn on
                    # Sanity check now applied at base detection level
                    if key_lit_after_base_filtering and delta_allows_on:
                        lit_now = True
                    else:
                        lit_now = False
                else:
                    # Was on - stay on unless delta forces off (already checked above)
                    lit_now = key_lit_after_base_filtering
            else:
                # No delta detection - use filtered base result
                lit_now = key_lit_after_base_filtering
                
            # TARGETED DEBUG: Log final decision
            if overlay.key_id in [36, 43, 48] and self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("DEBUG-KEY-%s: FINAL lit_now=%s", overlay.key_id, lit_now)

            # Update overlay state (CRITICAL: progression ratio update AFTER delta detection)
            overlay.last_progression_ratio = detection_data['current_max_progression_ratio']
            overlay.last_is_lit = lit_now

            if lit_now:
                pressed_key_ids.add(overlay.key_id)

        return pressed_key_ids
    
    def _calculate_color_progression(self, overlay: OverlayConfig, 
                                   current_color: Tuple[int, int, int],
                                   unlit_ref_color: Tuple[int, int, int],
                                   exemplar_lit_colors: Dict[str, Optional[Tuple[int, int, int]]],
                                   detection_threshold: float,
                                   frame_bgr: np.ndarray,
                                   kwargs: dict) -> Tuple[bool, float]:
        """Calculate color progression ratio and check threshold.
        
        Now uses hand detection to determine which exemplar to compare against,
        preventing false positives from cross-hand comparisons.
        """
        
        base_color_type = overlay.key_type[-1]  # "W" or "B"
        
        # Determine which hand is playing this key if hand detection is available
        hand_assignment_enabled = kwargs.get('hand_assignment_enabled', False)
        hand_detection_calibrated = kwargs.get('hand_detection_calibrated', False)
        left_hand_hue_mean = kwargs.get('left_hand_hue_mean', 0.0)
        right_hand_hue_mean = kwargs.get('right_hand_hue_mean', 0.0)
        
        # Determine which exemplar to use
        exemplar_types_to_check = []
        
        if hand_assignment_enabled and hand_detection_calibrated:
            # Check if hue values are different enough for reliable hand detection
            hue_diff = abs(left_hand_hue_mean - right_hand_hue_mean)
            
            if hue_diff >= 5.0:  # Reliable hand detection threshold
                # Use hue-based hand detection to select appropriate exemplar
                hand_type = self._determine_hand_from_hue(
                    overlay, frame_bgr, left_hand_hue_mean, right_hand_hue_mean
                )
                
                if hand_type == 'L':
                    # Left hand - only check left exemplars
                    exemplar_types_to_check.append('L' + base_color_type)
                elif hand_type == 'R':
                    # Right hand - only check right exemplars
                    exemplar_types_to_check.append('R' + base_color_type)
                else:
                    # Could not determine - fall back to checking both
                    if base_color_type == "W":
                        exemplar_types_to_check.extend(["LW", "RW"])
                    elif base_color_type == "B":
                        exemplar_types_to_check.extend(["LB", "RB"])
            else:
                # Hue values too similar - fall back to checking all exemplars
                # Any key can be lit by hands of any color, so check all combinations
                exemplar_types_to_check.extend(["LW", "LB", "RW", "RB"])
        else:
            # Hand detection not available - check all exemplars
            if base_color_type == "W":
                exemplar_types_to_check.extend(["LW", "RW"])
            elif base_color_type == "B":
                exemplar_types_to_check.extend(["LB", "RB"])
            else:
                self.logger.warning(f"Overlay {overlay.key_id} has unexpected key_type: {overlay.key_type}")
                return False, 0.0
        
        # Also check any additional COLOR_N exemplars that are calibrated
        # Match based on white/black key type
        for key_type, color in exemplar_lit_colors.items():
            if key_type.startswith("COLOR_") and color is not None:
                # Check if this additional color matches the current key type (W or B)
                if key_type.endswith(f"_{base_color_type}"):
                    exemplar_types_to_check.append(key_type)

        
        is_key_lit_by_color = False
        current_max_progression_ratio = 0.0

        for exemplar_key_type in exemplar_types_to_check:
            lit_ref_color = exemplar_lit_colors.get(exemplar_key_type)
            if lit_ref_color is None:
                continue

            try:
                d_unlit_to_lit_ref = euclidean_distance(unlit_ref_color, lit_ref_color)
                d_unlit_to_current = euclidean_distance(unlit_ref_color, current_color)

                progression_ratio = 0.0
                if d_unlit_to_lit_ref > 1e-6:
                    progression_ratio = d_unlit_to_current / d_unlit_to_lit_ref
                elif d_unlit_to_current < 1e-6:
                    progression_ratio = 1.0

                current_max_progression_ratio = max(current_max_progression_ratio, progression_ratio)

                if progression_ratio >= detection_threshold:
                    is_key_lit_by_color = True

            except Exception as e:
                self.logger.error(f"Error processing overlay {overlay.key_id} against exemplar {exemplar_key_type}: {e}")
                continue

        return is_key_lit_by_color, current_max_progression_ratio
    
    def _determine_hand_from_hue(self, overlay: OverlayConfig, frame_bgr: np.ndarray,
                                left_hand_hue_mean: float, right_hand_hue_mean: float) -> str:
        """Determine which hand (L or R) is playing the key based on hue analysis.
        
        Returns:
            'L' for left hand
            'R' for right hand
            'U' for unknown/undetermined
        """
        try:
            # Get HSV ROI using cache (optimal for sparse detection)
            hsv = self.roi_cache.get_roi_hsv(frame_bgr, overlay)
            
            if hsv is not None:
                # Extract average hue from cached HSV
                avg_hue = np.mean(hsv[:, :, 0])
                
                # Calculate distance to left and right hand hue means
                left_dist = abs(avg_hue - left_hand_hue_mean)
                right_dist = abs(avg_hue - right_hand_hue_mean)
                
                # Classify based on closest distance
                if left_dist < right_dist:
                    return 'L'  # Left hand
                elif right_dist < left_dist:
                    return 'R'  # Right hand
                else:
                    # Exact tie - default to right hand
                    return 'R'
        except Exception as e:
            self.logger.debug(f"Could not determine hand for overlay {overlay.key_id}: {e}")
        
        return 'U'  # Unknown
    
    def _apply_histogram_detection(self, overlay: OverlayConfig, frame_bgr: np.ndarray, 
                                 hist_ratio_threshold: float, 
                                 exemplar_lit_histograms: Dict[str, Optional[np.ndarray]],
                                 kwargs: dict) -> bool:
        """Apply histogram comparison detection using exemplar histograms.
        
        Now uses hand detection to select appropriate exemplar, consistent with
        color progression detection.
        """
        if overlay.unlit_hist is None:
            return False
            
        # Determine which exemplar histograms to check based on key color
        base_color_type = overlay.key_type[-1]  # "W" or "B"
        
        # Get hand detection parameters
        hand_assignment_enabled = kwargs.get('hand_assignment_enabled', False)
        hand_detection_calibrated = kwargs.get('hand_detection_calibrated', False)
        left_hand_hue_mean = kwargs.get('left_hand_hue_mean', 0.0)
        right_hand_hue_mean = kwargs.get('right_hand_hue_mean', 0.0)
        
        exemplar_types_to_check = []
        
        if hand_assignment_enabled and hand_detection_calibrated:
            # Check if hue values are different enough for reliable hand detection
            hue_diff = abs(left_hand_hue_mean - right_hand_hue_mean)
            
            if hue_diff >= 5.0:  # Reliable hand detection threshold
                # Use hue-based hand detection to select appropriate exemplar
                hand_type = self._determine_hand_from_hue(
                    overlay, frame_bgr, left_hand_hue_mean, right_hand_hue_mean
                )
                
                if hand_type == 'L':
                    # Left hand - only check left exemplars
                    exemplar_types_to_check.append('L' + base_color_type)
                elif hand_type == 'R':
                    # Right hand - only check right exemplars
                    exemplar_types_to_check.append('R' + base_color_type)
                else:
                    # Could not determine - fall back to checking both
                    if base_color_type == "W":
                        exemplar_types_to_check.extend(["LW", "RW"])
                    elif base_color_type == "B":
                        exemplar_types_to_check.extend(["LB", "RB"])
            else:
                # Hue values too similar - fall back to checking all exemplars
                # Any key can be lit by hands of any color, so check all combinations
                exemplar_types_to_check.extend(["LW", "LB", "RW", "RB"])
        else:
            # Hand detection not available - check all exemplars
            if base_color_type == "W":
                exemplar_types_to_check.extend(["LW", "RW"])
            elif base_color_type == "B":
                exemplar_types_to_check.extend(["LB", "RB"])
            else:
                self.logger.warning(f"Overlay {overlay.key_id} has unexpected key_type: {overlay.key_type}")
                return False
        
        # Also check any additional COLOR_N exemplars that are calibrated
        # Match based on white/black key type
        for key_type, hist in exemplar_lit_histograms.items():
            if key_type.startswith("COLOR_") and hist is not None:
                # Check if this additional color matches the current key type (W or B)
                if key_type.endswith(f"_{base_color_type}"):
                    exemplar_types_to_check.append(key_type)
            
        try:
            # Get HSV ROI using cache (optimal for sparse detection)
            roi_hsv = self.roi_cache.get_roi_hsv(frame_bgr, overlay)
            if roi_hsv is None:
                return False
                
            # Get histogram from the cached HSV ROI (avoids redundant conversion)
            current_hist = get_hist_feature_from_hsv(roi_hsv)
            if current_hist is None:
                return False

            # Check against both left and right exemplars (similar to color detection)
            max_hist_progression_ratio = 0.0
            
            for exemplar_key_type in exemplar_types_to_check:
                exemplar_lit_hist = exemplar_lit_histograms.get(exemplar_key_type)
                if exemplar_lit_hist is None:
                    continue
                    
                d_unlit_to_lit_hist = hist_distance(overlay.unlit_hist, exemplar_lit_hist)
                d_unlit_to_current_hist = hist_distance(overlay.unlit_hist, current_hist)

                hist_progression_ratio = 0.0
                if d_unlit_to_lit_hist > 1e-6:
                    hist_progression_ratio = d_unlit_to_current_hist / d_unlit_to_lit_hist
                    
                max_hist_progression_ratio = max(max_hist_progression_ratio, hist_progression_ratio)

            return max_hist_progression_ratio >= hist_ratio_threshold

        except Exception as e:
            self.logger.error(f"Histogram detection error for overlay {overlay.key_id}: {e}")
            return False
    
    def _apply_delta_detection(self, overlay: OverlayConfig, current_progression_ratio: float,
                             rise_delta_threshold: float, fall_delta_threshold: float, 
                             detection_threshold: float = 0.7) -> Tuple[bool, bool]:
        """Apply delta detection with adaptive thresholds based on progression ratio."""
        delta = current_progression_ratio - overlay.prev_progression_ratio
        delta_allows_on = False
        delta_forces_off = False

        # Adaptive delta logic: lower progression ratios need higher deltas to prove legitimacy
        # Note: Delta thresholds are now scaled 10x higher for better user control
        # (e.g., rise_delta_threshold = 0.5 instead of 0.05)
        if current_progression_ratio < detection_threshold * 0.8:
            # Low progression ratio - require much higher delta to filter noise/spill
            required_rise_delta = rise_delta_threshold * 3.0  # 1.5 instead of 0.5
            required_fall_delta = fall_delta_threshold * 2.0  # 0.2 instead of 0.1
        else:
            # High progression ratio - use standard deltas
            required_rise_delta = rise_delta_threshold
            required_fall_delta = fall_delta_threshold
        
        # Convert to internal scale (divide by 10 for backward compatibility with progression ratios)
        required_rise_delta = required_rise_delta / 10.0
        required_fall_delta = required_fall_delta / 10.0

        if delta >= required_rise_delta:
            delta_allows_on = True
        elif delta <= -required_fall_delta:
            delta_forces_off = True

        return delta_allows_on, delta_forces_off
    
    def _apply_black_key_filter(self, pressed_key_ids: Set[int], overlays: List[OverlayConfig],
                               similarity_ratio: float) -> Set[int]:
        """Apply black key spill filter to remove false positives."""
        if not pressed_key_ids:
            return pressed_key_ids
            
        # Create overlay lookup map
        overlay_map = {ov.key_id: ov for ov in overlays}
        
        # Separate white and black keys - white keys are always confirmed
        confirmed_pressed_key_ids = set()
        black_keys_to_filter = set()
        
        for key_id in pressed_key_ids:
            overlay = overlay_map.get(key_id)
            if overlay and overlay.key_type and not overlay.key_type.endswith("B"):
                # White key - always confirmed
                confirmed_pressed_key_ids.add(key_id)
            else:
                # Black key - needs filtering
                black_keys_to_filter.add(key_id)
        
        # Process black keys in clusters
        processed_black_keys = set()
        
        for key_id in black_keys_to_filter:
            if key_id in processed_black_keys:
                continue
                
            # Find cluster of adjacent black keys that are also pressed
            cluster = self._find_black_key_cluster(key_id, black_keys_to_filter, overlay_map)
            processed_black_keys.update(cluster)
            
            # Apply winner-takes-all logic to the cluster
            winners = self._apply_winner_takes_all(cluster, overlay_map, similarity_ratio)
            confirmed_pressed_key_ids.update(winners)
        return confirmed_pressed_key_ids
    
    def _find_black_key_cluster(self, key_id: int, pressed_black_keys: Set[int], 
                               overlay_map: Dict[int, OverlayConfig]) -> Set[int]:
        """Find cluster of adjacent black keys that are currently pressed."""
        cluster = {key_id}
        
        # Black keys are spaced 2 semitones apart, so check ±2
        for offset in [-2, 2]:
            neighbor_id = key_id + offset
            if neighbor_id in pressed_black_keys:
                neighbor_overlay = overlay_map.get(neighbor_id)
                # Verify it's actually a black key
                if neighbor_overlay and neighbor_overlay.key_type and neighbor_overlay.key_type.endswith("B"):
                    cluster.add(neighbor_id)
        
        return cluster
    
    def _apply_winner_takes_all(self, cluster: Set[int], overlay_map: Dict[int, OverlayConfig],
                               similarity_ratio: float) -> Set[int]:
        """Apply winner-takes-all logic to a cluster of black keys."""
        if not cluster:
            return set()
        
        # Calculate jumps (frame-to-frame progression ratio changes)
        jumps = {}
        has_meaningful_history = False
        for key_id in cluster:
            overlay = overlay_map[key_id]
            jump = overlay.last_progression_ratio - overlay.prev_progression_ratio
            jumps[key_id] = jump
            # Check if we have meaningful frame-to-frame data
            if overlay.prev_progression_ratio > 0.01:
                has_meaningful_history = True
        
        # If no meaningful history, use simplified winner-takes-all (strongest progression only)
        if not has_meaningful_history:
            self.logger.debug(f"No meaningful progression history for cluster {cluster}, using strongest progression only")
            strongest_key_id = max(cluster, key=lambda k: overlay_map[k].last_progression_ratio)
            return {strongest_key_id}
        
        # Find the strongest jump (winner)
        strongest_key_id = max(cluster, key=lambda k: jumps.get(k, -float('inf')))
        strongest_jump = jumps.get(strongest_key_id, 0.0)
        
        winners = set()
        
        if strongest_jump <= 0:
            # No brightening detected - only keep keys that were already lit
            for key_id in cluster:
                overlay = overlay_map[key_id]
                if overlay.last_is_lit:
                    winners.add(key_id)
        else:
            # Active brightening - apply similarity ratio and sustained note logic
            for key_id in cluster:
                overlay = overlay_map[key_id]
                member_jump = jumps[key_id]
                
                # Condition 1: Key is brightening comparable to the winner
                is_comparable_jump = False
                if member_jump > 0 and strongest_jump > 1e-6:
                    jump_ratio = member_jump / strongest_jump
                    is_comparable_jump = jump_ratio >= similarity_ratio
                
                # Condition 2: Key is sustained (already lit, bright, and stable)
                is_sustained = (overlay.last_is_lit and 
                               overlay.last_progression_ratio > 0.60 and
                               member_jump >= -0.05)  # Not dimming significantly
                
                if is_comparable_jump or is_sustained:
                    winners.add(key_id)
        
        # Additional check: If all black keys in a cluster are being kept, 
        # apply more aggressive filtering by only keeping the strongest
        if len(winners) == len(cluster) and len(cluster) > 1:
            self.logger.debug(f"Aggressive filtering applied to cluster {cluster}: kept strongest key only")
            # Keep only the key with the strongest progression ratio
            strongest_key_id = max(cluster, key=lambda k: overlay_map[k].last_progression_ratio)
            winners = {strongest_key_id}
        
        return winners
    
    def _should_check_histogram(self, progression_ratio: float) -> bool:
        """
        Determine if histogram detection should be performed.
        
        Skip histogram detection when color progression is very conclusive
        to save processing time.
        
        Args:
            progression_ratio: Current color progression ratio
            
        Returns:
            True if histogram check should be performed
        """
        # Skip if color detection is very confident
        if progression_ratio > 0.95:  # Clearly pressed
            return False
        if progression_ratio < 0.05:  # Clearly not pressed
            return False
            
        # Check histogram for ambiguous cases
        return True
    
    def _is_frame_black(self, frame_bgr: np.ndarray, threshold: int = 10) -> bool:
        """
        Quick check if frame is mostly black.
        
        For piano videos, we need to be more conservative since backgrounds are often black
        but the piano area contains valid data. Sample more points and require ALL to be black.
        
        Args:
            frame_bgr: BGR frame to check
            threshold: Maximum pixel value to consider "black"
            
        Returns:
            True if frame appears to be completely black/empty
        """
        h, w = frame_bgr.shape[:2]
        
        # Sample more points across the frame, including piano area
        sample_points = [
            (h//4, w//4),      # Top-left quadrant
            (h//4, 3*w//4),    # Top-right quadrant
            (3*h//4, w//4),    # Bottom-left quadrant
            (3*h//4, 3*w//4),  # Bottom-right quadrant
            (h//2, w//2),      # Center
            (2*h//3, w//2),    # Lower center (piano keys area)
            (h//2, w//3),      # Left center
            (h//2, 2*w//3),    # Right center
        ]
        
        black_count = 0
        for y, x in sample_points:
            # Check if this point is black
            if np.all(frame_bgr[y, x] <= threshold):
                black_count += 1
        
        # Only consider frame black if ALL sample points are black
        # This prevents false positives on piano videos with dark backgrounds
        return black_count == len(sample_points)
    
    def reset_state(self):
        """Reset state for standard detection (overlay state is managed externally)."""
        self.logger.debug("Standard detection state reset")
    
    def get_method_info(self) -> Dict[str, any]:
        """Get information about standard detection method."""
        return {
            "name": self.name,
            "description": "Color progression with optional histogram and delta detection",
            "parameters": {
                "detection_threshold": "Primary color progression threshold (0.1-0.99)",
                "hist_ratio_threshold": "Histogram comparison threshold (0.1-1.0)",
                "rise_delta_threshold": "Minimum positive change for press detection",
                "fall_delta_threshold": "Minimum negative change for release detection",
                "use_histogram_detection": "Enable histogram comparison",
                "use_delta_detection": "Enable delta timing logic",
                "apply_black_filter": "Enable black key spill filtering",
            },
            "algorithm": "Color distance progression + optional histogram + delta detection"
        }
    
    def validate_parameters(self, **kwargs) -> List[str]:
        """Validate standard detection parameters."""
        errors = super().validate_parameters(**kwargs)
        
        if 'hist_ratio_threshold' in kwargs:
            threshold = kwargs['hist_ratio_threshold']
            if not 0.1 <= threshold <= 1.0:
                errors.append(f"Histogram ratio threshold {threshold} must be between 0.1 and 1.0")
        
        if 'rise_delta_threshold' in kwargs:
            threshold = kwargs['rise_delta_threshold']
            if not 0.01 <= threshold <= 0.5:
                errors.append(f"Rise delta threshold {threshold} must be between 0.01 and 0.5")
        
        if 'fall_delta_threshold' in kwargs:
            threshold = kwargs['fall_delta_threshold']
            if not 0.01 <= threshold <= 0.5:
                errors.append(f"Fall delta threshold {threshold} must be between 0.01 and 0.5")
        
        return errors
