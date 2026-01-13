#!/usr/bin/env python3
"""
Monolithic Piano Keyboard Auto-Detector.

Provides comprehensive piano keyboard detection functionality including:
- Manual ROI-based key detection (requires a user-specified keyboard region)
- Black and white key identification using computer vision
- Musical note assignment with chromatic scanning from F# anchor
- Edge key validation ensuring leftmost/rightmost keys are white
- Final visualization generation with overlay annotations

This detector requires manual ROI specification and focuses on accuracy
over automation for reliable key detection in various video conditions.
"""
import logging

import cv2
import numpy as np

DEFAULT_DETECTION_PARAMS = {
    "black_upper_ratio": 0.6,
    "black_threshold": 70,
    "black_column_ratio": 0.10,
    "black_min_width": 10,
    "black_max_width": 100,
    "white_bottom_ratio": 0.85,
    "white_edge_std_factor": 2.0,
    "white_min_width": 15,
    "white_initial_top_ratio": 0.7,
    "white_initial_height_ratio": 0.3,
    "edge_boundary_padding_px": 3,
    "padding_percent": 0.15,
    "trim_saturation_threshold": 45,
    "trim_gray_threshold": 140,
    "trim_row_height": 20,
}

class MonolithicPianoDetector:
    """
    Comprehensive piano keyboard detector for static images.
    
    Detects individual piano keys within a manually specified region and assigns
    musical notes using chromatic scanning from F# anchor points. Requires manual
    ROI specification for reliable detection across various video conditions.
    
    Args:
        image_path: Path to the image file to analyze
        keyboard_region: Tuple of (top_y, bottom_y, left_x, right_x) defining
                        the manual ROI for keyboard detection
    """
    
    def __init__(self, image_path, keyboard_region=None, detection_profile=None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.height, self.width = self.gray.shape
        self.logger.debug(f"Analyzing image: {self.width}x{self.height} pixels")
        
        # Detection results
        self.black_keys = []
        self.white_keys = []
        self.keyboard_region = keyboard_region  # Must be provided for manual ROI
        self.key_notes = {}
        # Detection parameters (allow overrides for low-quality fallbacks)
        self.params = {**DEFAULT_DETECTION_PARAMS, **(detection_profile or {})}
        
    def _add_overlay_padding(self, start_x, y, width, height, padding_percent=None):
        """Add padding to overlay by shrinking inward from left and right sides"""
        if padding_percent is None:
            padding_percent = self.params.get("padding_percent", 0.15)
        padding_pixels = int(width * padding_percent)
        new_start_x = start_x + padding_pixels
        new_width = width - (2 * padding_pixels)
        return new_start_x, y, new_width, height
        
    # ================== KEYBOARD REGION ==================
    # This detector requires a manually specified keyboard_region.
    
    # ================== KEY DETECTION ==================
    
    def detect_keys(self):
        """Detect individual piano keys within the keyboard region"""
        if not self.keyboard_region:
            raise ValueError("Must detect keyboard region first")
        
        top_y, bottom_y, left_x, right_x = self.keyboard_region
        keyboard_img = self.image[top_y:bottom_y, left_x:right_x]
        keyboard_gray = cv2.cvtColor(keyboard_img, cv2.COLOR_BGR2GRAY)
        
        self.logger.debug(f"\n=== Detecting Keys in Region {right_x-left_x}x{bottom_y-top_y} ===")
        
        # Detect black keys first (easier to identify)
        self.black_keys = self._detect_black_keys(keyboard_gray)
        self.logger.debug(f"Detected {len(self.black_keys)} black keys")
        
        self.logger.debug("First 5 black keys detected:")
        for i, (x, y, w, h) in enumerate(self.black_keys[:5]):
            self.logger.debug(f"  Black key {i}: x={x}, y={y}, w={w}, h={h} (absolute x={left_x + x})")
        
        # Detect white keys
        self.white_keys = self._detect_white_keys(keyboard_gray)
        self.logger.debug(f"Detected {len(self.white_keys)} white keys")
        
        self.logger.debug("First 5 white keys detected:")
        for i, (x, y, w, h) in enumerate(self.white_keys[:5]):
            self.logger.debug(f"  White key {i}: x={x}, y={y}, w={w}, h={h} (absolute x={left_x + x})")
        
        return len(self.black_keys), len(self.white_keys)
    
    def _detect_black_keys(self, gray_img):
        """Detect black keys using column scanning"""
        height, width = gray_img.shape
        
        # Focus on upper portion where black keys are
        upper_ratio = self.params["black_upper_ratio"]
        upper_region = gray_img[:int(height * upper_ratio), :]
        
        # Create binary image - black keys are dark (conservative threshold to avoid shadowed white keys)
        _, binary = cv2.threshold(
            upper_region,
            self.params["black_threshold"],
            255,
            cv2.THRESH_BINARY_INV,
        )
        
        # Scan columns to find black key regions
        column_sums = np.sum(binary, axis=0)
        
        # Find where columns have significant black pixels
        threshold = np.max(column_sums) * self.params["black_column_ratio"]  # Reduced threshold for better detection
        black_regions = column_sums > threshold
        
        # Find start and end of each black key
        black_keys = []
        in_key = False
        start_x = 0
        
        for x in range(len(black_regions)):
            if black_regions[x] and not in_key:
                start_x = x
                in_key = True
            elif not black_regions[x] and in_key:
                width = x - start_x
                if (
                    self.params["black_min_width"]
                    < width
                    < self.params["black_max_width"]
                ):  # Reasonable key width
                    padded_overlay = self._add_overlay_padding(start_x, 0, width, upper_region.shape[0])
                    black_keys.append(padded_overlay)
                in_key = False
        
        # Handle last key
        if in_key:
            width = len(black_regions) - start_x
            if self.params["black_min_width"] < width < self.params["black_max_width"]:
                padded_overlay = self._add_overlay_padding(start_x, 0, width, upper_region.shape[0])
                black_keys.append(padded_overlay)
        
        return black_keys
    
    def _detect_white_keys(self, gray_img):
        """Detect white keys by finding vertical separations"""
        height, width = gray_img.shape
        
        # Look at bottom portion where white keys are clearly separated
        bottom_y = int(height * self.params["white_bottom_ratio"])
        bottom_row = gray_img[bottom_y, :]
        
        # Apply smoothing
        bottom_smooth = cv2.GaussianBlur(bottom_row.reshape(1, -1), (1, 5), 0).flatten()
        
        # Calculate gradient to find edges
        gradient = np.gradient(bottom_smooth)
        
        # Find significant edges
        edges = []
        edge_threshold = np.std(gradient) * self.params["white_edge_std_factor"]
        
        for i in range(1, len(gradient) - 1):
            if abs(gradient[i]) > edge_threshold:
                # Local extremum
                if ((gradient[i-1] < gradient[i] > gradient[i+1]) or 
                    (gradient[i-1] > gradient[i] < gradient[i+1])):
                    edges.append(i)
        
        self.logger.debug(f"DEBUG: White key edge detection:")
        self.logger.debug(f"  Gradient std: {np.std(gradient):.2f}, edge_threshold: {edge_threshold:.2f}")
        self.logger.debug(f"  Found {len(edges)} edges: {edges[:10]}...")  # Show first 10 edges
        
        # MISSING KEY FIX: Add left and right boundaries if not present
        boundary_pad = self.params["edge_boundary_padding_px"]
        if not edges or edges[0] > boundary_pad:  # If first edge is far from left boundary (reduced from 20 to 3)
            edges.insert(0, 0)  # Add left boundary at x=0
            self.logger.debug(f"DEBUG: Added left boundary edge at x=0")

        if not edges or edges[-1] < width - boundary_pad:  # If last edge is far from right boundary (reduced from 20 to 3)
            edges.append(width - 1)  # Add right boundary
            self.logger.debug(f"DEBUG: Added right boundary edge at x={width-1}")
        
        # Convert edges to key boundaries
        white_keys = []
        min_key_width = self.params["white_min_width"]
        
        for i in range(len(edges) - 1):
            start_x = edges[i]
            end_x = edges[i + 1]
            key_width = end_x - start_x
            
            if key_width > min_key_width:
                # Initial white key region - start lower to avoid black keys
                initial_top = int(height * self.params["white_initial_top_ratio"])  # Start at 70% instead of 60%
                initial_height = int(height * self.params["white_initial_height_ratio"])  # 30% height instead of 40%
                
                # Trim top of white key if it dips into black key area
                trimmed_top, trimmed_height = self._trim_white_key_top(
                    gray_img, start_x, end_x, initial_top, initial_height)
                
                # Add padding to white key overlay
                padded_overlay = self._add_overlay_padding(start_x, trimmed_top, key_width, trimmed_height)
                white_keys.append(padded_overlay)
        
        return white_keys
    
    def _trim_white_key_top(self, gray_img, start_x, end_x, initial_top, initial_height):
        """Trim white key overlay top when it dips into black key area"""
        height, width = gray_img.shape
        
        # Get the key region from the full keyboard image
        full_keyboard_region = self.image[self.keyboard_region[0]:self.keyboard_region[1], 
                                         self.keyboard_region[2]:self.keyboard_region[3]]
        
        # Convert key region to HSV for saturation analysis  
        key_region = full_keyboard_region[initial_top:initial_top + initial_height, start_x:end_x]
        key_hsv = cv2.cvtColor(key_region, cv2.COLOR_BGR2HSV)
        
        # Scan upward in 20-pixel rows from bottom as requested
        row_height = self.params["trim_row_height"]
        trimmed_top = initial_top
        
        for y in range(key_region.shape[0] - row_height, 0, -row_height):
            if y + row_height <= key_region.shape[0]:
                row_hsv = key_hsv[y:y + row_height, :, :]
                avg_saturation = np.mean(row_hsv[:, :, 1])
                avg_gray = np.mean(cv2.cvtColor(key_region[y:y + row_height, :], cv2.COLOR_BGR2GRAY))
                
                # If saturation increases significantly from white key baseline, stop here
                # White keys typically have sat=15-18, but cream/beige keys can be ~38
                # Increased threshold to accommodate cream-colored white keys (like halo video)
                if avg_saturation > self.params["trim_saturation_threshold"] or avg_gray < self.params["trim_gray_threshold"]:  # Accommodate cream/beige white keys
                    trimmed_top = initial_top + y + row_height
                    break
        
        # Calculate new height
        trimmed_height = (initial_top + initial_height) - trimmed_top
        trimmed_height = max(30, trimmed_height)  # Minimum height for visibility
        
        return trimmed_top, trimmed_height
    
    # ================== NOTE ASSIGNMENT ==================
    
    def assign_notes(self):
        """Assign musical notes using unified chromatic scanning from F# anchor"""
        if not self.black_keys or not self.white_keys:
            raise ValueError("Must detect keys first")
        
        self.logger.debug(f"\n=== Assigning Notes to {len(self.black_keys)} black + {len(self.white_keys)} white keys ===")
        
        # Find F# anchor using confident LSSL pattern detection
        f_sharp_position = self._find_confident_f_sharp_anchor()
        
        if f_sharp_position is None:
            self.logger.debug("Could not find confident F# anchor - using fallback assignment")
            return self._fallback_note_assignment()
        
        # Unified chromatic assignment using pixel-by-pixel scanning
        self.key_notes = self._assign_notes_chromatically_from_anchor(f_sharp_position)
        
        self.logger.debug(f"DEBUG: Total assigned keys: {len(self.key_notes)}")
        
        if self.key_notes:
            self.logger.debug("First 10 chromatic note assignments:")
            sorted_notes = sorted(self.key_notes.items())
            for i, (center_x, note_info) in enumerate(sorted_notes[:10]):
                self.logger.debug(f"  Key {i}: center_x={center_x}, note={note_info['note']}, type={note_info['type']}")
        
        # UNIVERSAL VALIDATION: Leftmost and rightmost keys must ALWAYS be white
        self._validate_edge_keys()
        
        self.logger.debug(f"Assigned notes to {len(self.key_notes)} keys")
        return self.key_notes
    
    def _validate_edge_keys(self):
        """Validate and enforce absolute rule: leftmost and rightmost keys must be white"""
        if not self.key_notes:
            return
        
        # Get leftmost and rightmost keys
        sorted_positions = sorted(self.key_notes.keys())
        leftmost_pos = sorted_positions[0]
        rightmost_pos = sorted_positions[-1]
        
        leftmost_key = self.key_notes[leftmost_pos]
        rightmost_key = self.key_notes[rightmost_pos]
        
        self.logger.debug(f"\n=== EDGE KEY VALIDATION & ENFORCEMENT ===")
        self.logger.debug(f"Leftmost key: {leftmost_key['note']} (type: {leftmost_key['type']})")
        self.logger.debug(f"Rightmost key: {rightmost_key['note']} (type: {rightmost_key['type']})")
        
        # ABSOLUTE RULE ENFORCEMENT: Remove black keys from edges
        keys_removed = False
        
        self.logger.debug(f"DEBUG: All keys before edge validation ({len(sorted_positions)} total):")
        for i, pos in enumerate(sorted_positions[:10]):  # Show first 10
            key_info = self.key_notes[pos]
            self.logger.debug(f"  Position {i}: center_x={pos}, note={key_info['note']}, type={key_info['type']}")
        
        # Remove leftmost keys until we find a white key
        while sorted_positions and self.key_notes[sorted_positions[0]]['type'] != 'white':
            removed_pos = sorted_positions.pop(0)
            removed_key = self.key_notes.pop(removed_pos)
            self.logger.debug(f"🔧 REMOVED leftmost black key: {removed_key['note']} at position {removed_pos}")
            keys_removed = True
        
        # Remove rightmost keys until we find a white key
        while sorted_positions and self.key_notes[sorted_positions[-1]]['type'] != 'white':
            removed_pos = sorted_positions.pop(-1)
            removed_key = self.key_notes.pop(removed_pos)
            self.logger.debug(f"🔧 REMOVED rightmost black key: {removed_key['note']} at position {removed_pos}")
            keys_removed = True
        
        if keys_removed:
            self.logger.debug(f"✅ ABSOLUTE RULE ENFORCED: Removed edge black keys to ensure white keys at boundaries")
            
            # Update leftmost and rightmost after removal
            if sorted_positions:
                leftmost_key = self.key_notes[sorted_positions[0]]
                rightmost_key = self.key_notes[sorted_positions[-1]]
                self.logger.debug(f"NEW Leftmost key: {leftmost_key['note']} (type: {leftmost_key['type']})")
                self.logger.debug(f"NEW Rightmost key: {rightmost_key['note']} (type: {rightmost_key['type']})")
        
        # Final validation
        if sorted_positions:
            final_leftmost = self.key_notes[sorted_positions[0]]
            final_rightmost = self.key_notes[sorted_positions[-1]]
            
            if final_leftmost['type'] == 'white' and final_rightmost['type'] == 'white':
                self.logger.debug("✅ ABSOLUTE RULE SATISFIED: Both leftmost and rightmost are white keys")
            else:
                self.logger.debug("❌ ABSOLUTE RULE VIOLATION: Could not ensure white edge keys")
        else:
            self.logger.debug("❌ ERROR: No keys remaining after edge removal")
    
    def _find_confident_f_sharp_anchor(self):
        """Find F# anchor by locating confident LSSL pattern (3-black-key group)"""
        if len(self.black_keys) < 3:
            self.logger.debug("Not enough black keys to find F# anchor")
            return None
        
        self.logger.debug("Scanning left-to-right for confident LSSL patterns...")
        
        # Calculate gaps between consecutive black keys
        gaps = []
        for i in range(len(self.black_keys) - 1):
            gap = self.black_keys[i+1][0] - (self.black_keys[i][0] + self.black_keys[i][2])
            gaps.append(gap)
        
        # Find median gap to distinguish small from large gaps
        median_gap = sorted(gaps)[len(gaps)//2]
        gap_threshold = median_gap * 1.4
        
        self.logger.debug(f"Gap analysis: median={median_gap:.1f}, threshold={gap_threshold:.1f}")
        
        # Look for LSSL patterns (Large gap, then 3 black keys with Small-Small-Large gaps)
        for i in range(len(gaps) - 3):
            # Check for LSSL pattern starting at position i
            if (gaps[i] > gap_threshold and          # L: Large gap before group
                gaps[i+1] <= gap_threshold and       # S: Small gap (F# to G#)
                gaps[i+2] <= gap_threshold and       # S: Small gap (G# to A#)  
                gaps[i+3] > gap_threshold):          # L: Large gap after group
                
                # Found confident LSSL pattern
                f_sharp_key_idx = i + 1  # F# is first key after the large gap
                f_sharp_key = self.black_keys[f_sharp_key_idx]
                f_sharp_center_x = f_sharp_key[0] + f_sharp_key[2] // 2
                
                self.logger.debug(f"✅ Found confident LSSL pattern at black key index {f_sharp_key_idx}")
                self.logger.debug(f"   F# anchor: center_x={f_sharp_center_x}, box={f_sharp_key}")
                self.logger.debug(f"   Gap sequence: {gaps[i]:.1f}(L) {gaps[i+1]:.1f}(S) {gaps[i+2]:.1f}(S) {gaps[i+3]:.1f}(L)")
                
                return f_sharp_center_x
        
        # Fallback: look for any SSL pattern (3 consecutive black keys)
        self.logger.debug("No confident LSSL found, looking for any SSL pattern...")
        for i in range(len(gaps) - 2):
            if (gaps[i] <= gap_threshold and         # S: Small gap
                gaps[i+1] <= gap_threshold and       # S: Small gap
                gaps[i+2] > gap_threshold):          # L: Large gap after
                
                f_sharp_key_idx = i
                f_sharp_key = self.black_keys[f_sharp_key_idx]
                f_sharp_center_x = f_sharp_key[0] + f_sharp_key[2] // 2
                
                self.logger.debug(f"⚠️ Fallback SSL pattern at index {f_sharp_key_idx}")
                self.logger.debug(f"   F# anchor (fallback): center_x={f_sharp_center_x}")
                
                return f_sharp_center_x
        
        self.logger.debug("❌ Could not find any F# anchor pattern")
        return None
    
    def _assign_notes_chromatically_from_anchor(self, f_sharp_center_x):
        """Assign notes chromatically using pixel-by-pixel scanning from F# anchor"""
        self.logger.debug(f"Starting chromatic assignment from F# anchor at x={f_sharp_center_x}")
        
        # Create unified list of all key overlays (black + white) sorted by position
        all_overlays = []
        
        # Add black keys
        for black_key in self.black_keys:
            center_x = black_key[0] + black_key[2] // 2
            all_overlays.append({
                'center_x': center_x,
                'type': 'black',
                'box': black_key,
                'assigned': False
            })
        
        # Add white keys  
        for white_key in self.white_keys:
            center_x = white_key[0] + white_key[2] // 2
            all_overlays.append({
                'center_x': center_x,
                'type': 'white', 
                'box': white_key,
                'assigned': False
            })
        
        # Sort all overlays by center_x position
        all_overlays.sort(key=lambda k: k['center_x'])
        
        self.logger.debug(f"Total overlays to assign: {len(all_overlays)} (scanning from F# anchor)")
        
        # Find F# overlay in sorted list
        f_sharp_idx = None
        for i, overlay in enumerate(all_overlays):
            if abs(overlay['center_x'] - f_sharp_center_x) < 5:  # Close match to F# anchor
                f_sharp_idx = i
                break
        
        if f_sharp_idx is None:
            self.logger.debug("❌ Could not find F# overlay in sorted list")
            return {}
        
        self.logger.debug(f"F# anchor found at overlay index {f_sharp_idx}")
        
        # Chromatic note sequence (semitones)
        chromatic_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        f_sharp_note_idx = 6  # F# is at index 6 in chromatic sequence
        
        assigned_notes = {}
        
        # Assign F# anchor first
        octave = 0  # Start at octave 0, will adjust based on position
        all_overlays[f_sharp_idx]['assigned'] = True
        note_name = 'F#'
        
        # Determine if F# overlay is black or white to set correct type
        overlay_type = all_overlays[f_sharp_idx]['type']
        
        assigned_notes[f_sharp_center_x] = {
            'note': f'{note_name}{octave}',
            'type': overlay_type,
            'box': all_overlays[f_sharp_idx]['box']
        }
        
        self.logger.debug(f"✅ Assigned F# anchor: x={f_sharp_center_x}, note=F#{octave}, type={overlay_type}")
        
        # Scan rightward from F# anchor
        current_note_idx = f_sharp_note_idx
        current_octave = octave
        
        for i in range(f_sharp_idx + 1, len(all_overlays)):
            overlay = all_overlays[i]
            if overlay['assigned']:
                continue
                
            # Move to next chromatic note
            current_note_idx = (current_note_idx + 1) % 12
            if current_note_idx == 0:  # Wrapped around to C
                current_octave += 1
            
            note_name = chromatic_notes[current_note_idx]
            overlay['assigned'] = True
            
            assigned_notes[overlay['center_x']] = {
                'note': f'{note_name}{current_octave}',
                'type': overlay['type'],
                'box': overlay['box']
            }
            
            self.logger.debug(f"→ Right scan: x={overlay['center_x']}, note={note_name}{current_octave}, type={overlay['type']}")
        
        # Scan leftward from F# anchor
        current_note_idx = f_sharp_note_idx
        current_octave = octave
        
        for i in range(f_sharp_idx - 1, -1, -1):
            overlay = all_overlays[i]
            if overlay['assigned']:
                continue
                
            # Move to previous chromatic note
            current_note_idx = (current_note_idx - 1) % 12
            if current_note_idx == 11:  # Wrapped around to B from C
                current_octave -= 1
            
            note_name = chromatic_notes[current_note_idx]
            overlay['assigned'] = True
            
            assigned_notes[overlay['center_x']] = {
                'note': f'{note_name}{current_octave}',
                'type': overlay['type'],
                'box': overlay['box']
            }
            
            self.logger.debug(f"← Left scan: x={overlay['center_x']}, note={note_name}{current_octave}, type={overlay['type']}")
        
        self.logger.debug(f"✅ Chromatic assignment complete: {len(assigned_notes)} keys assigned")
        return assigned_notes
    
    def _assign_black_key_notes(self, f_sharp_idx):
        """Assign notes to black keys starting from F# anchor"""
        black_notes = {}
        
        # Black key pattern in chromatic sequence
        black_key_pattern = ['C#', 'D#', 'F#', 'G#', 'A#']
        
        # Calculate starting octave (A0 starts the 88-key piano)
        # F# is the 3rd black key in the pattern (index 2)
        pattern_position = 2  # F# position in pattern
        
        # Estimate octave based on position with edge key adjustment
        base_octave = max(0, (f_sharp_idx - pattern_position) // 5)
        
        # Check if leftmost key would be black with current octave
        leftmost_pattern_idx = (0 - f_sharp_idx + pattern_position) % 5
        leftmost_note = black_key_pattern[leftmost_pattern_idx]
        
        # If leftmost key would be C# or D#, we need to start with a white key instead
        # Adjust octave to ensure leftmost key is white (universal piano rule)
        if leftmost_note in ['C#', 'D#'] and base_octave == 0:
            # If we would start with C#0 or D#0, adjust so we start with C0 instead
            estimated_octave = 0  # Keep same octave but notes will be shifted appropriately
            self.logger.debug(f"Adjusted octave calculation: leftmost would be {leftmost_note}{base_octave}, ensuring white key start")
        else:
            estimated_octave = base_octave
        
        for i, black_key in enumerate(self.black_keys):
            # Calculate pattern index
            pattern_idx = (i - f_sharp_idx + pattern_position) % 5
            
            # Calculate octave
            octave = estimated_octave + ((i - f_sharp_idx + pattern_position) // 5)
            
            # Get note name
            note_name = black_key_pattern[pattern_idx]
            full_note = f"{note_name}{octave}"
            
            # Store with center position as key
            center_x = black_key[0] + black_key[2] // 2
            black_notes[center_x] = {
                'note': full_note,
                'type': 'black',
                'box': black_key
            }
        
        return black_notes
    
    def _assign_white_key_notes_by_scanning(self, black_notes):
        """Assign white key notes by scanning from F# anchor"""
        white_notes = {}
        
        # Find F# position
        f_sharp_center = None
        f_sharp_note = None
        
        for center, note_info in black_notes.items():
            if note_info['note'].startswith('F#'):
                f_sharp_center = center
                f_sharp_note = note_info['note']
                break
        
        if f_sharp_center is None:
            return self._fallback_white_assignment()
        
        # White key pattern starting from F (before F#)
        white_pattern = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        
        # Extract octave from F# note
        f_sharp_octave = int(f_sharp_note[2:])
        
        # F comes before F# in the same octave
        f_note = f'F{f_sharp_octave}'
        f_pattern_idx = 3  # F is at index 3 in white pattern
        
        # Sort white keys by position
        sorted_white_keys = sorted(self.white_keys, key=lambda k: k[0])
        
        # Find the white key closest to and left of F#
        f_key_idx = None
        min_distance = float('inf')
        
        for i, white_key in enumerate(sorted_white_keys):
            white_center = white_key[0] + white_key[2] // 2
            if white_center < f_sharp_center:
                distance = f_sharp_center - white_center
                if distance < min_distance:
                    min_distance = distance
                    f_key_idx = i
        
        if f_key_idx is None:
            return self._fallback_white_assignment()
        
        # Assign notes starting from F
        for i, white_key in enumerate(sorted_white_keys):
            # Calculate position relative to F
            relative_pos = i - f_key_idx
            
            # Calculate pattern index and octave
            pattern_idx = (f_pattern_idx + relative_pos) % 7
            octave_offset = (f_pattern_idx + relative_pos) // 7
            octave = f_sharp_octave + octave_offset
            
            # Get note name
            note_name = white_pattern[pattern_idx]
            full_note = f"{note_name}{octave}"
            
            # Store with center position as key
            center_x = white_key[0] + white_key[2] // 2
            white_notes[center_x] = {
                'note': full_note,
                'type': 'white',
                'box': white_key
            }
        
        return white_notes
    
    def _fallback_note_assignment(self):
        """Fallback note assignment when F# anchor fails"""
        notes = {}
        
        # Simple chromatic assignment starting from C4
        chromatic_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Combine all keys and sort by position
        all_keys = []
        for bk in self.black_keys:
            center_x = bk[0] + bk[2] // 2
            all_keys.append((center_x, 'black', bk))
        
        for wk in self.white_keys:
            center_x = wk[0] + wk[2] // 2
            all_keys.append((center_x, 'white', wk))
        
        all_keys.sort()
        
        # Assign notes starting from C4
        start_octave = 4
        for i, (center_x, key_type, box) in enumerate(all_keys):
            note_idx = i % 12
            octave = start_octave + (i // 12)
            
            note_name = chromatic_notes[note_idx]
            full_note = f"{note_name}{octave}"
            
            notes[center_x] = {
                'note': full_note,
                'type': key_type,
                'box': box
            }
        
        return notes
    
    def _fallback_white_assignment(self):
        """Fallback white key assignment"""
        white_notes = {}
        white_pattern = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        
        sorted_white_keys = sorted(self.white_keys, key=lambda k: k[0])
        
        for i, white_key in enumerate(sorted_white_keys):
            pattern_idx = i % 7
            octave = 4 + (i // 7)
            
            note_name = white_pattern[pattern_idx]
            full_note = f"{note_name}{octave}"
            
            center_x = white_key[0] + white_key[2] // 2
            white_notes[center_x] = {
                'note': full_note,
                'type': 'white',
                'box': white_key
            }
        
        return white_notes
    
    # ================== VISUALIZATION ==================
    
    def create_final_visualization(self):
        """Create the final detection visualization on the full image"""
        if not self.keyboard_region or not self.key_notes:
            raise ValueError("Must complete detection and note assignment first")
        
        self.logger.debug(f"\n=== Creating Final Visualization ===")
        
        top_y, bottom_y, left_x, right_x = self.keyboard_region
        
        # Create labeled keyboard region
        keyboard_img = self.image[top_y:bottom_y, left_x:right_x].copy()
        
        # Draw key overlays and labels
        for center_x, note_info in self.key_notes.items():
            box = note_info['box']
            note = note_info['note']
            key_type = note_info['type']
            
            x, y, w, h = box
            
            # Draw bounding box
            color = (0, 255, 0) if key_type == 'white' else (0, 0, 255)
            cv2.rectangle(keyboard_img, (x, y), (x + w, y + h), color, 2)
            
            # Add note label with better positioning and visibility
            if key_type == 'white':
                # Place label at bottom of white key area, within the key region
                label_y = y + h - 5  # Near bottom of white key
                label_x = x + w // 2 - 10  # Center horizontally
                text_color = (255, 0, 0)  # Red text for better visibility on white
            else:
                # Place label in middle of black key
                label_y = y + h // 2 + 5  # Middle of black key
                label_x = x + w // 2 - 8  # Center horizontally  
                text_color = (255, 255, 255)  # White text for visibility on black
            
            cv2.putText(keyboard_img, note, (max(0, label_x), label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # Create final full image
        final_image = self.image.copy()
        final_image[top_y:bottom_y, left_x:right_x] = keyboard_img
        
        # Add region boundary
        cv2.rectangle(final_image, (left_x, top_y), (right_x, bottom_y), (0, 255, 0), 3)
        
        # Add title and stats
        cv2.putText(final_image, "Piano Keyboard Auto-Detection", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        stats = f"Region: y={top_y}-{bottom_y}, x={left_x}-{right_x} | Keys: {len(self.black_keys)} black, {len(self.white_keys)} white"
        cv2.putText(final_image, stats, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Save final result
        import os
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(output_dir, 'final_detection_result.jpg')
        cv2.imwrite(output_path, final_image)
        self.logger.debug(f"Final detection saved to: {output_path}")
        
        return output_path
    
    # ================== MAIN PIPELINE ==================
    
    def run_complete_detection(self):
        """Run the complete detection pipeline"""
        self.logger.debug(f"\n{'='*60}")
        self.logger.debug(f"MONOLITHIC PIANO DETECTOR - Complete Analysis")
        self.logger.debug(f"{'='*60}")
        
        try:
            # Verify keyboard region was provided
            if not self.keyboard_region:
                raise ValueError("Keyboard region must be provided for manual ROI detection")
            
            self.logger.debug(f"Using provided keyboard region: {self.keyboard_region}")
            
            # Step 1: Detect individual keys
            num_black, num_white = self.detect_keys()
            
            # Step 2: Assign musical notes
            self.assign_notes()
            
            # Step 3: Create final visualization
            output_path = self.create_final_visualization()
            
            # Summary
            self.logger.debug(f"\n{'='*60}")
            self.logger.debug(f"DETECTION COMPLETE")
            self.logger.debug(f"{'='*60}")
            self.logger.debug(f"Keyboard region: {self.keyboard_region}")
            self.logger.debug(f"Black keys detected: {num_black}")
            self.logger.debug(f"White keys detected: {num_white}")
            self.logger.debug(f"Total keys: {num_black + num_white}")
            self.logger.debug(f"Notes assigned: {len(self.key_notes)}")
            self.logger.debug(f"Final result: {output_path}")
            
            return {
                'region': self.keyboard_region,
                'black_keys': num_black,
                'white_keys': num_white,
                'total_keys': num_black + num_white,
                'notes_assigned': len(self.key_notes),
                'output_path': output_path
            }
            
        except Exception as e:
            self.logger.debug(f"Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    # Example usage - update path as needed
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Usage: python monolithic_detector.py <image_path>")
        print("Note: This detector requires a manual keyboard region (ROI).")
        sys.exit(1)
    
    # Manual ROI required - example coordinates (adjust as needed)
    # Format: (top_y, bottom_y, left_x, right_x)
    manual_roi = (100, 300, 50, 1850)  # Example values
    
    detector = MonolithicPianoDetector(image_path, keyboard_region=manual_roi)
    results = detector.run_complete_detection()
