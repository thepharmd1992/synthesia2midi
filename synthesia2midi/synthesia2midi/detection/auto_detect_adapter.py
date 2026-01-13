"""
Adapter module to bridge the monolithic piano detector with the calibration wizard.

This module provides a clean interface between the standalone monolithic detector
and the synthesia2midi application, handling coordinate transformations and data
conversion while preserving all existing overlay functionality.
"""
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from synthesia2midi.app_config import NOTE_NAMES_SHARP, OverlayConfig


@dataclass
class DetectedKey:
    """Represents a detected piano key from the monolithic detector."""
    x: int
    y: int
    width: int
    height: int
    note_name: str  # e.g., "C4", "F#3"
    key_type: str  # "white" or "black"
    

class AutoDetectAdapter:
    """
    Adapter for integrating the monolithic piano detector with the wizard.
    
    This adapter:
    1. Runs the monolithic detector on video frames
    2. Converts detector results to OverlayConfig objects
    3. Handles coordinate transformations
    4. Provides only position and note information (no calibration)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.AutoDetectAdapter")        
        self._detector = None
        self._temp_image_path = None
        self.last_failure_reason: Optional[str] = None
    
    def detect_from_frame(self, frame: np.ndarray, keyboard_region: Optional[Tuple[int, int, int, int]] = None) -> Optional[Dict]:
        """
        Run auto-detection on a video frame.
        
        Args:
            frame: Video frame as numpy array (RGB format)
            keyboard_region: Optional manual keyboard region (x, y, width, height)
            
        Returns:
            Dictionary with detection results or None if detection fails:
            {
                'total_keys': int,
                'leftmost_note': str,  # e.g., "A"
                'leftmost_octave': int,  # e.g., 0
                'detected_keys': List[DetectedKey],
                'keyboard_region': Tuple[int, int, int, int]  # top, bottom, left, right
            }
        """
        self.logger.info("=== AUTO DETECT ADAPTER - detect_from_frame called ===")
        self.logger.info(f"Frame shape: {frame.shape if frame is not None else 'None'}")
        self.logger.info(f"Keyboard region: {keyboard_region}")
        
        try:
            # Import here to avoid circular dependencies
            from .monolithic_detector import MonolithicPianoDetector
            
            # Save frame to temporary file (detector expects file path)
            self._save_frame_to_temp(frame)
            
            # Run detection pipeline
            self.logger.info("=== STARTING MONOLITHIC PIANO DETECTION ===")
            
            # A manual keyboard region is required.
            if not keyboard_region:
                self.logger.error("Manual keyboard region is required.")
                return None
                
            # When a manual region is provided, the frame has already been cropped
            # keyboard_region contains the ORIGINAL ROI coordinates: (roi_x, roi_y, roi_width, roi_height)
            roi_x, roi_y, roi_width, roi_height = keyboard_region
            
            self.logger.info("=== USER-DEFINED ROI RECTANGLE DEBUG ===")
            self.logger.info(f"Original ROI rectangle: x={roi_x}, y={roi_y}, width={roi_width}, height={roi_height}")
            self.logger.info(f"ROI rectangle bounds: left={roi_x}, right={roi_x + roi_width}, top={roi_y}, bottom={roi_y + roi_height}")
            
            # For the detector working on the cropped frame, use 0-based coordinates
            # since the cropped frame starts at (0,0)
            top_y = 0
            bottom_y = roi_height
            left_x = 0
            right_x = roi_width
            
            # Store original region for coordinate conversion back to absolute canvas coordinates
            detector_region = (roi_y, roi_y + roi_height, roi_x, roi_x + roi_width)
            
            self.logger.info(f"Using manual keyboard region (original): y={roi_y}-{roi_y + roi_height}, x={roi_x}-{roi_x + roi_width}")
            self.logger.info(f"Cropped frame coordinates: y={top_y}-{bottom_y}, x={left_x}-{right_x}")
            self.logger.info(f"Detector region for coordinate conversion: {detector_region}")
            
            self.last_failure_reason = None
            cropped_region = (top_y, bottom_y, left_x, right_x)

            detection_profiles = [
                {"name": "default", "params": {}},
                {
                    "name": "lenient_1",
                    "params": {
                        "black_threshold": 60,
                        "black_column_ratio": 0.08,
                        "black_min_width": 8,
                        "black_max_width": 120,
                        "white_edge_std_factor": 1.6,
                        "white_min_width": 12,
                    },
                },
                {
                    "name": "lenient_2",
                    "params": {
                        "black_threshold": 50,
                        "black_column_ratio": 0.06,
                        "black_min_width": 6,
                        "black_max_width": 140,
                        "white_edge_std_factor": 1.4,
                        "white_min_width": 10,
                        "white_initial_top_ratio": 0.65,
                        "white_initial_height_ratio": 0.35,
                    },
                },
            ]

            last_error: Optional[Exception] = None

            for profile in detection_profiles:
                try:
                    self.logger.info(f"Attempting detection with profile: {profile['name']}")
                    self._detector = MonolithicPianoDetector(
                        self._temp_image_path,
                        keyboard_region=cropped_region,
                        detection_profile=profile["params"],
                    )

                    self.logger.info(f"Initialized detector with keyboard region: {cropped_region}")

                    num_black, num_white = self._detector.detect_keys()
                    total_keys = num_black + num_white
                    self.logger.info(f"[{profile['name']}] Detected {num_white} white keys, {num_black} black keys")

                    key_notes = self._detector.assign_notes()
                    if not key_notes:
                        raise ValueError("Failed to assign notes to keys")

                    conversion_region = detector_region
                    detected_keys = self._convert_detector_results(key_notes, conversion_region)

                    leftmost_note, leftmost_octave = self._get_leftmost_note_info(detected_keys)

                    self.logger.info(f"[{profile['name']}] Detection complete: {total_keys} keys, leftmost: {leftmost_note}{leftmost_octave}")

                    return {
                        'total_keys': total_keys,
                        'leftmost_note': leftmost_note,
                        'leftmost_octave': leftmost_octave,
                        'detected_keys': detected_keys,
                        'keyboard_region': detector_region
                    }
                except Exception as e:
                    last_error = e
                    self.logger.warning(f"Detection attempt '{profile['name']}' failed: {e}", exc_info=True)
                    continue

            self.last_failure_reason = "low_quality"
            if last_error:
                self.logger.error(f"All detection attempts failed. Last error: {last_error}")
            return None

        except Exception as e:
            self.last_failure_reason = "error"
            self.logger.error(f"Auto-detection failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self._cleanup_temp_file()
    
    def create_overlays_from_detection(self, detection_results: Dict, 
                                     existing_overlays: List[OverlayConfig]) -> List[OverlayConfig]:
        """
        Create OverlayConfig objects from detection results.
        
        This method creates new overlays with detected positions while preserving
        the structure and properties expected by the existing system.
        
        Args:
            detection_results: Results from detect_from_frame()
            existing_overlays: Current overlays (used as reference for structure/properties)
            
        Returns:
            List of new OverlayConfig objects with detected positions
        """
        new_overlays = []
        detected_keys = detection_results['detected_keys']
        
        # Sort keys by x position to ensure consistent key_id assignment
        sorted_keys = sorted(detected_keys, key=lambda k: k.x)
        
        for key_id, detected_key in enumerate(sorted_keys):
            # Parse note name and octave
            note_full = detected_key.note_name
            if len(note_full) >= 2:
                # Find where the octave number starts (first digit or minus sign)
                octave_start = len(note_full)
                for i, char in enumerate(note_full):
                    if char.isdigit() or char == '-':
                        octave_start = i
                        break
                
                if octave_start < len(note_full):
                    note_name = note_full[:octave_start]
                    octave_str = note_full[octave_start:]
                    try:
                        octave = int(octave_str)
                    except ValueError:
                        self.logger.warning(f"Invalid octave in note: {note_full}")
                        continue
                else:
                    self.logger.warning(f"No octave found in note: {note_full}")
                    continue
            else:
                self.logger.warning(f"Invalid note format: {note_full}")
                continue
            
            # Determine key type (LW, LB, RW, RB) based on position and color
            # Simple split: left half = L, right half = R
            is_left_hand = key_id < (len(sorted_keys) / 2)
            hand_prefix = "L" if is_left_hand else "R"
            color_suffix = "W" if detected_key.key_type == "white" else "B"
            key_type = f"{hand_prefix}{color_suffix}"
            
            # Create overlay with detected position
            # NOTE: The detected key positions are already in absolute coordinates
            # (converted in _convert_detector_results), so we should NOT apply
            # the keyboard_region_offset here as it would double the offset
            x_pos = float(detected_key.x)
            y_pos = float(detected_key.y)
            
            # Note: detected positions are already in absolute coordinates.
            # Do not apply any additional keyboard-region offset here.
            
            overlay = OverlayConfig(
                key_id=key_id,
                note_octave=octave,
                note_name_in_octave=note_name,
                x=x_pos,
                y=y_pos,
                width=float(detected_key.width),
                height=float(detected_key.height),
                key_type=key_type,
                # Leave calibration data as None - will be set by existing systems
                unlit_reference_color=None,
                unlit_hist=None,
                lit_hist=None
            )
            
            # Log first few overlays for debugging
            if key_id < 3:
                self.logger.info(f"Created overlay {key_id}: x={x_pos}, y={y_pos}, w={detected_key.width}, h={detected_key.height}, type={key_type}")
            
            new_overlays.append(overlay)
        
        # Post-process: Adjust octaves so leftmost key is appropriate for piano type
        if new_overlays:
            # Find the leftmost overlay
            leftmost = min(new_overlays, key=lambda o: o.x)
            num_keys = len(new_overlays)
            
            # Determine target starting note based on number of keys
            # 88-key piano: A0 to C8
            # 76-key piano: E0 to G7  
            # 61-key piano: C1 to C6
            # But for simplicity, if we detect A as leftmost, assume it should be A0
            
            # Calculate octave adjustment needed
            if leftmost.note_name_in_octave == "A":
                # For pianos starting with A, use A0 as the target
                octave_adjustment = 0 - leftmost.note_octave
            elif leftmost.note_name_in_octave in ["A♯", "A#"]:
                # If leftmost is A#, then A would be in the previous octave
                octave_adjustment = -1 - leftmost.note_octave
            elif leftmost.note_name_in_octave == "B":
                # If leftmost is B, then A would be in the same octave  
                octave_adjustment = 0 - leftmost.note_octave
            elif leftmost.note_name_in_octave == "C":
                # For pianos starting with C
                if num_keys <= 61:
                    # 61-key piano starts at C1
                    octave_adjustment = 1 - leftmost.note_octave
                else:
                    # Larger pianos with C as leftmost, keep as is
                    octave_adjustment = 0
            else:
                # For other starting notes, try to make a reasonable guess
                self.logger.warning(f"Unexpected leftmost note: {leftmost.note_name_in_octave}")
                octave_adjustment = 0
            
            # Apply adjustment to all overlays
            if octave_adjustment != 0:
                self.logger.info(f"Adjusting all octaves by {octave_adjustment} (leftmost: {leftmost.note_name_in_octave}{leftmost.note_octave} -> {leftmost.note_name_in_octave}{leftmost.note_octave + octave_adjustment})")
                for overlay in new_overlays:
                    overlay.note_octave += octave_adjustment
        
        self.logger.info(f"Created {len(new_overlays)} overlays from detection results")
        return new_overlays
    
    def _save_frame_to_temp(self, frame: np.ndarray):
        """Save frame to temporary file for detector."""
        # Create temporary file
        fd, self._temp_image_path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
        
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Save to file
        cv2.imwrite(self._temp_image_path, bgr_frame)
        self.logger.debug(f"Saved frame to temporary file: {self._temp_image_path}")
    
    def _cleanup_temp_file(self):
        """Remove temporary image file."""
        if self._temp_image_path and os.path.exists(self._temp_image_path):
            os.unlink(self._temp_image_path)
            self._temp_image_path = None
    
    def _convert_detector_results(self, key_notes: Dict, keyboard_region: Tuple) -> List[DetectedKey]:
        """Convert monolithic detector results to DetectedKey objects."""
        detected_keys = []
        top_y, bottom_y, left_x, right_x = keyboard_region
        
        # Log keyboard region being used for conversion
        self.logger.info("=== WHITE KEY OVERLAY COORDINATE CONVERSION DEBUG ===")
        self.logger.info(f"Converting detector results with keyboard_region: top_y={top_y}, bottom_y={bottom_y}, left_x={left_x}, right_x={right_x}")
        self.logger.info(f"Keyboard region offset: left_x={left_x}, top_y={top_y}")
        
        white_key_count = 0
        for i, (center_x, note_info) in enumerate(key_notes.items()):
            box = note_info['box']
            x, y, w, h = box
            
            # Convert coordinates from keyboard-relative to absolute
            abs_x = left_x + x
            abs_y = top_y + y
            
            # Log detailed info for white keys and first few keys
            if note_info['type'] == 'white' or i < 5:
                is_white = note_info['type'] == 'white'
                key_type_str = "WHITE KEY" if is_white else "BLACK KEY"
                
                self.logger.info(f"{key_type_str} {i} ({note_info['note']}): detector_box=({x},{y},{w},{h}) -> absolute=({abs_x},{abs_y},{w},{h})")
                
                if is_white:
                    white_key_count += 1
                    # Compare with ROI rectangle for white keys
                    roi_left = left_x
                    roi_right = right_x  
                    roi_top = top_y
                    roi_bottom = bottom_y
                    
                    self.logger.info(f"  WHITE KEY {white_key_count} position relative to ROI:")
                    self.logger.info(f"    ROI bounds: left={roi_left}, right={roi_right}, top={roi_top}, bottom={roi_bottom}")
                    self.logger.info(f"    Key bounds: left={abs_x}, right={abs_x+w}, top={abs_y}, bottom={abs_y+h}")
                    self.logger.info(f"    Key position within ROI: x_offset={x}, y_offset={y}")
                    
                    # Check if key is positioned correctly relative to ROI
                    if abs_y < roi_top:
                        self.logger.warning(f"    WARNING: WHITE KEY {white_key_count} is ABOVE the ROI rectangle! (key_top={abs_y} < roi_top={roi_top})")
                    elif abs_y > roi_bottom:
                        self.logger.warning(f"    WARNING: WHITE KEY {white_key_count} is BELOW the ROI rectangle! (key_top={abs_y} > roi_bottom={roi_bottom})")
                    else:
                        self.logger.info(f"    OK: WHITE KEY {white_key_count} is within ROI vertical bounds")
            
            detected_key = DetectedKey(
                x=abs_x,
                y=abs_y,
                width=w,
                height=h,
                note_name=note_info['note'],
                key_type=note_info['type']
            )
            detected_keys.append(detected_key)
        
        self.logger.info(f"Total white keys processed: {white_key_count}")
        self.logger.info("=== END WHITE KEY OVERLAY COORDINATE DEBUG ===")
        
        return detected_keys
    
    def _get_leftmost_note_info(self, detected_keys: List[DetectedKey]) -> Tuple[str, int]:
        """Extract leftmost note name and octave from detected keys."""
        if not detected_keys:
            return "C", 4  # Default fallback
        
        # Find leftmost key
        leftmost_key = min(detected_keys, key=lambda k: k.x)
        
        # Parse note and octave
        note_full = leftmost_key.note_name
        if len(note_full) >= 2:
            # Find where the octave number starts (first digit or minus sign)
            octave_start = len(note_full)
            for i, char in enumerate(note_full):
                if char.isdigit() or char == '-':
                    octave_start = i
                    break
            
            if octave_start < len(note_full):
                note_name = note_full[:octave_start]
                octave_str = note_full[octave_start:]
                try:
                    octave = int(octave_str)
                except ValueError:
                    self.logger.warning(f"Invalid octave in leftmost note: {note_full}")
                    return "C", 4  # Default fallback
            else:
                self.logger.warning(f"No octave found in leftmost note: {note_full}")
                return "C", 4  # Default fallback
            
            # Ensure note name is in standard format
            if note_name not in NOTE_NAMES_SHARP:
                # Try to find closest match
                if '#' in note_name:
                    note_name = note_name.replace('#', '♯')
                if note_name not in NOTE_NAMES_SHARP:
                    self.logger.warning(f"Unknown note name: {note_name}, defaulting to C")
                    note_name = "C"
            
            return note_name, octave
        else:
            return "C", 4  # Default fallback
