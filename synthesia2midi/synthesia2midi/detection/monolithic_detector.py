#!/usr/bin/env python3
"""
Monolithic Piano Keyboard Auto-Detector.

Provides comprehensive piano keyboard detection functionality including:
- Manual ROI-based key detection (requires a user-specified keyboard region)
- Black and white key identification using computer vision
- Musical note assignment with chromatic scanning from F# anchor
- Final visualization generation with overlay annotations

This detector requires manual ROI specification and focuses on accuracy
over automation for reliable key detection in various video conditions.
"""
import logging

import cv2
import numpy as np

DEFAULT_DETECTION_PARAMS = {
    "black_upper_ratio": 0.6,
    "black_bottom_ratio": 0.5,
    "black_threshold": 70,
    "black_threshold_method": "otsu",
    "black_adaptive_block_size": 31,
    "black_adaptive_c": 5,
    "black_column_ratio": 0.10,
    "black_min_width": 10,
    "black_max_width": 100,
    "white_bottom_ratio": 0.85,
    "white_edge_std_factor": 2.0,
    "white_min_width": 15,
    "white_initial_top_ratio": 0.8,
    "white_initial_height_ratio": 0.3,
    "edge_boundary_padding_px": 3,
    "padding_percent": 0.25,
    "trim_saturation_threshold": 45,
    "trim_gray_threshold": 140,
    "trim_row_height": 20,
    "white_strip_dark_threshold": 60,
    "white_strip_dark_fraction": 0.02,
    "white_strip_min_run": 8,
    "white_strip_allow_failures": 1,
    "white_edge_left_shift_ticks": 0,
    "white_edge_right_shift_ticks": 0,
    "white_sep_ratio": 0.55,
    "white_sep_dyn_min": 8,
    "white_sep_close_kernel": 5,
    "white_sep_open_kernel": 3,
    "white_sep_min_width": 1,
    "type_aware_assignment": True,
    "black_recovery_enabled": True,
    "black_recovery_ratio": 0.6,
    "black_recovery_column_ratio_scale": 0.6,
    "black_split_max_factor": 1.6,
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
            padding_percent = self.params.get("padding_percent", 0.25)
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
        
        # Detect white keys from black-key geometry by default.
        self.white_keys = self._detect_white_keys_from_black(keyboard_gray)
        # If geometry reconstruction under-detects badly, retry separator scan.
        if len(self.white_keys) < 4:
            self.white_keys = self._detect_white_keys(keyboard_gray)
        self.logger.debug(f"Detected {len(self.white_keys)} white keys")        
        
        self.logger.debug("First 5 white keys detected:")
        for i, (x, y, w, h) in enumerate(self.white_keys[:5]):
            self.logger.debug(f"  White key {i}: x={x}, y={y}, w={w}, h={h} (absolute x={left_x + x})")
        
        self._maybe_recover_black_keys(keyboard_gray)

        return len(self.black_keys), len(self.white_keys)
    
    def _detect_black_keys(self, gray_img, recovery=False):
        """Detect black keys using column scanning"""
        height, width = gray_img.shape

        # Focus on upper portion where black keys are
        upper_ratio = self.params["black_upper_ratio"]
        max_bottom_ratio = float(self.params.get("black_bottom_ratio", 1.0))
        max_bottom_ratio = max(0.05, min(1.0, max_bottom_ratio))
        max_bottom_y = max(1, min(height, int(round(height * max_bottom_ratio))))
        strip_start = self._find_white_strip_start(gray_img)
        fallback_bottom_y = max(1, min(height, int(round(height * upper_ratio))))
        if strip_start <= 0 or strip_start > height:
            upper_bottom_y = fallback_bottom_y
        else:
            upper_bottom_y = max(1, min(height, int(strip_start)))
        upper_bottom_y = max(1, min(upper_bottom_y, max_bottom_y))

        upper_region = gray_img[:upper_bottom_y, :]
        if upper_region.size == 0:
            upper_region = gray_img[:max(1, min(fallback_bottom_y, max_bottom_y)), :]

        binary = self._threshold_black_region(upper_region)
        
        # Scan columns to find black key regions
        column_sums = np.sum(binary, axis=0)
        
        # Find where columns have significant black pixels
        column_ratio = self.params["black_column_ratio"]
        if recovery:
            column_ratio *= self.params.get("black_recovery_column_ratio_scale", 0.6)
        column_ratio = max(0.01, column_ratio)
        threshold = np.max(column_sums) * column_ratio  # Reduced threshold for better detection
        black_regions = column_sums > threshold

        # Find start and end of each black key
        segments = []
        in_key = False
        start_x = 0
        
        for x in range(len(black_regions)):
            if black_regions[x] and not in_key:
                start_x = x
                in_key = True
            elif not black_regions[x] and in_key:
                width = x - start_x
                if width > 0:
                    segments.append((start_x, width))
                in_key = False
        
        # Handle last key
        if in_key:
            width = len(black_regions) - start_x
            if width > 0:
                segments.append((start_x, width))

        black_keys = []
        min_width = self.params["black_min_width"]
        max_width = self.params["black_max_width"]
        if recovery:
            max_width = max_width * 2

        widths = [w for _, w in segments if w > 0]
        median_width = None
        valid_widths = [w for w in widths if min_width < w < max_width]
        if valid_widths:
            median_width = float(np.median(valid_widths))
        elif widths:
            median_width = float(np.median(widths))

        split_factor = float(self.params.get("black_split_max_factor", 1.6))

        for start_x, width in segments:
            if width <= min_width:
                continue

            if recovery and median_width and width > (median_width * split_factor):
                splits = int(round(width / median_width))
                splits = max(2, splits)
                sub_width = float(width) / splits
                for i in range(splits):
                    sub_start = int(round(start_x + (i * sub_width)))
                    sub_end = int(round(start_x + ((i + 1) * sub_width))) - 1
                    sub_w = sub_end - sub_start + 1
                    if sub_w <= min_width:
                        continue
                    padded_overlay = self._add_overlay_padding(
                        sub_start,
                        0,
                        sub_w,
                        upper_region.shape[0],
                    )
                    black_keys.append(padded_overlay)
                continue

            if width > max_width and not recovery:
                continue

            padded_overlay = self._add_overlay_padding(start_x, 0, width, upper_region.shape[0])
            black_keys.append(padded_overlay)

        return black_keys

    def _maybe_recover_black_keys(self, gray_img):
        if not self.params.get("black_recovery_enabled", False):
            return
        if not self.white_keys:
            return

        white_count = len(self.white_keys)
        expected_black = int(round(white_count * 5 / 7))
        min_ratio = float(self.params.get("black_recovery_ratio", 0.6))
        min_black = max(1, int(expected_black * min_ratio))

        if len(self.black_keys) >= min_black:
            return

        recovered = self._detect_black_keys(gray_img, recovery=True)
        if len(recovered) > len(self.black_keys):
            self.black_keys = recovered

    def _threshold_black_region(self, upper_region):
        method = self.params.get("black_threshold_method", "fixed")
        if method == "adaptive":
            block_size = int(self.params.get("black_adaptive_block_size", 31))
            if block_size < 3:
                block_size = 3
            if block_size % 2 == 0:
                block_size += 1
            c = int(self.params.get("black_adaptive_c", 5))
            binary = cv2.adaptiveThreshold(
                upper_region,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                c,
            )
            return binary

        if method == "otsu":
            _, binary = cv2.threshold(
                upper_region,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )
            return binary

        threshold = self.params["black_threshold"]
        _, binary = cv2.threshold(
            upper_region,
            threshold,
            255,
            cv2.THRESH_BINARY_INV,
        )
        return binary

    def _find_white_strip_start(
        self,
        gray_img: np.ndarray,
        *,
        dark_thr: int = None,
        frac_thr: float = None,
        min_run: int = None,
        allow_failures: int = None,
    ) -> int:
        """Return y0 where rows y0..end are mostly free of dark pixels."""
        h, _ = gray_img.shape
        if dark_thr is None:
            dark_thr = int(self.params.get("white_strip_dark_threshold", 60))
        if frac_thr is None:
            frac_thr = float(self.params.get("white_strip_dark_fraction", 0.02))
        if min_run is None:
            min_run = int(self.params.get("white_strip_min_run", 8))
        if allow_failures is None:
            allow_failures = int(self.params.get("white_strip_allow_failures", 1))

        if min_run <= 0:
            return int(h * self.params.get("black_upper_ratio", 0.6))

        dark_frac = np.mean(gray_img < dark_thr, axis=1)

        for y in range(0, max(0, h - min_run)):
            window = dark_frac[y:y + min_run]
            ok = np.sum(window < frac_thr) >= (min_run - allow_failures)
            if ok:
                return y

        return int(h * self.params.get("black_upper_ratio", 0.6))

    def _runs_from_mask(self, mask: np.ndarray, *, min_width: int = 2):
        runs = []
        in_run = False
        start = 0
        for x, v in enumerate(mask):
            if v and not in_run:
                in_run = True
                start = x
            elif (not v) and in_run:
                end = x - 1
                if (end - start + 1) >= min_width:
                    runs.append((start, end))
                in_run = False
        if in_run:
            end = len(mask) - 1
            if (end - start + 1) >= min_width:
                runs.append((start, end))
        return runs

    def _estimate_white_key_width(self, keyboard_width, candidate_widths):
        """Estimate typical white-key width for recovery heuristics."""
        estimates = []

        if self.black_keys:
            approx_white_count = max(7, int(round(len(self.black_keys) * 7.0 / 5.0)))
            estimates.append(float(keyboard_width) / float(approx_white_count))

        if candidate_widths:
            sorted_widths = sorted(candidate_widths)
            lower_count = max(1, int(len(sorted_widths) * 0.6))
            estimates.append(float(np.median(sorted_widths[:lower_count])))

        if estimates:
            return max(8.0, min(estimates))

        return max(8.0, float(keyboard_width) / 26.0)

    def _find_white_valley_centers(self, col_profile, x_left, x_right, min_sep_width):
        """Find likely separator valleys inside a white span."""
        if x_left >= x_right:
            return []

        segment = col_profile[x_left:x_right + 1]
        if segment.size < 4:
            return []

        hi = float(np.percentile(segment, 85))
        lo = float(np.percentile(segment, 15))
        dyn = hi - lo
        if dyn < 6.0:
            return []

        valley_threshold = hi - (0.7 * dyn)
        valley_mask = segment < valley_threshold
        valley_runs = self._runs_from_mask(
            valley_mask,
            min_width=max(1, int(min_sep_width)),
        )
        if valley_runs:
            return [x_left + ((a + b) // 2) for a, b in valley_runs]

        # Fallback: find gentle local minima on a smoothed profile.
        smooth_window = max(5, min(31, (segment.size // 12) | 1))
        kernel = np.ones(smooth_window, dtype=np.float32) / float(smooth_window)
        smoothed = np.convolve(segment.astype(np.float32), kernel, mode="same")
        baseline = float(np.median(smoothed))
        min_depth = max(1.0, dyn * 0.08)

        minima = []
        for i in range(1, smoothed.size - 1):
            if smoothed[i] <= smoothed[i - 1] and smoothed[i] <= smoothed[i + 1]:
                depth = baseline - float(smoothed[i])
                if depth >= min_depth:
                    minima.append((depth, x_left + i))

        if not minima:
            return []

        minima.sort(reverse=True)
        capped = minima[:12]
        return sorted(x for _, x in capped)

    def _guided_split_white_span(
        self,
        col_profile,
        x_left,
        x_right,
        expected_width,
        min_key_width,
        min_sep_width,
    ):
        """Split oversized white spans by placing cuts on local separator valleys."""
        initial_width = x_right - x_left + 1
        if initial_width < min_key_width:
            return []

        split_trigger = max(
            int(round(expected_width * 2.2)),
            int(min_key_width * 2),
        )
        if initial_width <= split_trigger:
            return [(x_left, x_right)]

        spans = [(x_left, x_right)]
        max_iterations = 12

        for _ in range(max_iterations):
            widths = [b - a + 1 for a, b in spans]
            max_index = int(np.argmax(widths))
            span_left, span_right = spans[max_index]
            span_width = span_right - span_left + 1

            if span_width <= split_trigger:
                break

            margin = max(min_key_width, int(round(expected_width * 0.55)))
            valleys = self._find_white_valley_centers(
                col_profile,
                span_left,
                span_right,
                min_sep_width,
            )
            valleys = [
                v for v in valleys
                if (v - span_left) >= margin and (span_right - v) >= margin
            ]
            if not valleys:
                break

            midpoint = (span_left + span_right) // 2
            split_x = min(valleys, key=lambda v: abs(v - midpoint))

            left_span = (span_left, split_x)
            right_span = (split_x + 1, span_right)
            if (
                (left_span[1] - left_span[0] + 1) < min_key_width
                or (right_span[1] - right_span[0] + 1) < min_key_width
            ):
                break

            spans[max_index:max_index + 1] = [left_span, right_span]

        spans.sort(key=lambda span: span[0])
        return spans

    def _classify_large_center_gaps(self, center_diffs):
        """Classify center-to-center gaps as single-white or double-white spans."""
        if not center_diffs:
            return []

        diffs = np.asarray(center_diffs, dtype=np.float32)
        median_diff = float(np.median(diffs))
        if len(diffs) < 3:
            threshold = max(median_diff * 1.35, median_diff + 4.0)
            return [bool(val >= threshold) for val in diffs]

        sorted_diffs = np.sort(diffs)
        jumps = np.diff(sorted_diffs)
        largest_jump = float(np.max(jumps)) if jumps.size > 0 else 0.0

        if largest_jump >= max(2.0, median_diff * 0.12):
            jump_idx = int(np.argmax(jumps))
            threshold = float((sorted_diffs[jump_idx] + sorted_diffs[jump_idx + 1]) / 2.0)
        else:
            threshold = max(median_diff * 1.35, median_diff + 4.0)

        large_mask = [bool(val >= threshold) for val in diffs]
        if any(large_mask) and not all(large_mask):
            return large_mask

        fallback_threshold = max(median_diff * 1.30, median_diff + 3.0)
        return [bool(val >= fallback_threshold) for val in diffs]

    def _split_span_evenly(self, x_left, x_right, count, min_key_width):
        """Split an overly wide span into evenly sized key candidates."""
        if count <= 1:
            return [(x_left, x_right)]

        span_width = x_right - x_left + 1
        if span_width <= 0:
            return []

        count = max(1, int(count))
        spans = []
        for idx in range(count):
            seg_left = x_left + int(round((idx * span_width) / count))
            seg_right = x_left + int(round(((idx + 1) * span_width) / count)) - 1
            if seg_right < seg_left:
                continue
            if (seg_right - seg_left + 1) >= min_key_width:
                spans.append((seg_left, seg_right))
        return spans

    def _score_black_note_center_map(self, note_centers):
        """Score black-note maps by how many complete D-anchored octave segments they provide."""
        if not note_centers:
            return 0.0

        octaves = set()
        for note in note_centers.keys():
            octave = self._extract_note_octave(note)
            if octave is not None:
                octaves.add(octave)

        pair_count = 0
        full_segment_count = 0
        for octave in sorted(octaves):
            has_pair = f"C#{octave}" in note_centers and f"D#{octave}" in note_centers
            if has_pair:
                pair_count += 1
            has_full = (
                has_pair
                and f"F#{octave}" in note_centers
                and f"G#{octave}" in note_centers
                and f"A#{octave}" in note_centers
                and f"C#{octave + 1}" in note_centers
                and f"D#{octave + 1}" in note_centers
            )
            if has_full:
                full_segment_count += 1

        return (full_segment_count * 10.0) + pair_count + (len(note_centers) * 0.01)

    def _build_black_note_center_map(self):
        """Assign black-note labels for geometry solving and return note->center map."""
        if not self.black_keys:
            return {}

        self.black_keys = sorted(self.black_keys, key=lambda key: key[0])
        candidates = self._find_f_sharp_anchor_candidates()
        if not candidates:
            return {}

        best_map = {}
        best_score = -1.0
        for f_sharp_idx in candidates:
            try:
                black_notes = self._assign_black_key_notes(f_sharp_idx)
            except Exception:
                continue

            note_centers = {}
            for center_x, note_info in black_notes.items():
                note = str(note_info.get("note", "")).strip()
                if not note:
                    continue
                note_centers[note] = float(center_x)

            score = self._score_black_note_center_map(note_centers)
            if score > best_score:
                best_score = score
                best_map = note_centers

        return best_map

    def _estimate_black_residual_samples(self, black_note_centers, white_center_by_note):
        """Collect black-key residual samples: observed black center minus predicted center."""
        if not black_note_centers or not white_center_by_note:
            return []

        black_to_white_neighbors = {
            "C#": ("C", "D"),
            "D#": ("D", "E"),
            "F#": ("F", "G"),
            "G#": ("G", "A"),
            "A#": ("A", "B"),
        }

        raw_samples = []
        for black_note, observed_center in black_note_centers.items():
            black_name = self._extract_note_name(black_note)
            octave = self._extract_note_octave(black_note)
            if octave is None or black_name not in black_to_white_neighbors:
                continue

            left_white_name, right_white_name = black_to_white_neighbors[black_name]
            left_white_note = f"{left_white_name}{octave}"
            right_white_note = f"{right_white_name}{octave}"
            if left_white_note not in white_center_by_note or right_white_note not in white_center_by_note:
                continue

            predicted_center = (white_center_by_note[left_white_note] + white_center_by_note[right_white_note]) / 2.0
            residual = float(observed_center) - float(predicted_center)
            raw_samples.append((float(predicted_center), residual, black_name))

        if len(raw_samples) < 3:
            return []

        class_values = {}
        for _, residual, black_name in raw_samples:
            class_values.setdefault(black_name, []).append(residual)
        class_bias = {
            black_name: float(np.median(values))
            for black_name, values in class_values.items()
            if values
        }

        # Remove note-class bias so the fit captures global edge trend rather than
        # intrinsic keyboard asymmetry (e.g., C#/D# vs F#/G# midpoint offsets).
        samples = [
            (x_pos, residual - class_bias.get(black_name, 0.0))
            for x_pos, residual, black_name in raw_samples
        ]
        return samples

    def _apply_black_residual_edge_warp(
        self,
        centers,
        width,
        step_estimate,
        black_note_centers,
        white_center_by_note,
    ):
        """
        Apply a smooth edge warp learned from black-key residuals.

        The correction is forced near zero in the center and grows toward edges.
        """
        if not centers or width <= 2 or step_estimate <= 0:
            return centers

        samples = self._estimate_black_residual_samples(black_note_centers, white_center_by_note)
        if len(samples) < 3:
            return centers

        x_samples = np.asarray([pt[0] for pt in samples], dtype=np.float64)
        residual_samples = np.asarray([pt[1] for pt in samples], dtype=np.float64)

        x_mid = (float(width) - 1.0) / 2.0
        half_width = max(1.0, (float(width) - 1.0) / 2.0)
        x_norm = (x_samples - x_mid) / half_width

        if len(samples) >= 5:
            degree = 2
        else:
            degree = 1

        try:
            coeffs = np.polyfit(x_norm, residual_samples, degree)
        except Exception:
            return centers

        poly = np.poly1d(coeffs)
        center_bias = float(poly(0.0))
        max_correction = min(6.0, max(2.0, step_estimate * 0.22))

        corrected = []
        for center in centers:
            x_n = (float(center) - x_mid) / half_width
            raw_delta = float(poly(x_n) - center_bias)
            edge_gain = min(1.0, abs(x_n) ** 1.15)
            delta = raw_delta * edge_gain
            delta = float(np.clip(delta, -max_correction, max_correction))
            corrected.append(float(center) + delta)

        # Enforce strict ordering and a minimum spacing to avoid post-warp inversions.
        min_gap = max(3.0, step_estimate * 0.35)
        corrected = sorted(corrected)
        for idx in range(1, len(corrected)):
            corrected[idx] = max(corrected[idx], corrected[idx - 1] + min_gap)

        right_limit = float(width - 1)
        overflow = corrected[-1] - right_limit
        if overflow > 0:
            corrected = [val - overflow for val in corrected]

        if corrected[0] < 0:
            shift = -corrected[0]
            corrected = [val + shift for val in corrected]

        for idx in range(1, len(corrected)):
            corrected[idx] = max(corrected[idx], corrected[idx - 1] + min_gap)

        if corrected[-1] > right_limit:
            compression = corrected[-1] - right_limit
            corrected[-1] = right_limit
            if len(corrected) > 1:
                spread = len(corrected) - 1
                for idx in range(spread):
                    backshift = compression * ((spread - idx) / spread)
                    corrected[idx] = max(0.0, corrected[idx] - backshift)

        final_centers = [float(np.clip(val, 0.0, right_limit)) for val in corrected]
        self.logger.debug(
            "Applied black residual edge warp: samples=%d degree=%d max_corr=%.2f",
            len(samples),
            degree,
            max_correction,
        )
        return final_centers

    def _estimate_white_centers_from_d_lattice(self, black_note_centers, width):
        """
        Estimate white-key centers from D anchors.

        For each D(o)->D(o+1) span:
        - derive local 3-step D->G and A->D spacing from black geometry
        - place E/F/G and A/B/C as normalized local fractions of the span
        """
        c_sharp_by_octave = {}
        d_sharp_by_octave = {}
        f_sharp_by_octave = {}
        g_sharp_by_octave = {}
        a_sharp_by_octave = {}

        for note, center in black_note_centers.items():
            note_name = self._extract_note_name(note)
            octave = self._extract_note_octave(note)
            if octave is None:
                continue
            if note_name == "C#":
                c_sharp_by_octave[octave] = float(center)
            elif note_name == "D#":
                d_sharp_by_octave[octave] = float(center)
            elif note_name == "F#":
                f_sharp_by_octave[octave] = float(center)
            elif note_name == "G#":
                g_sharp_by_octave[octave] = float(center)
            elif note_name == "A#":
                a_sharp_by_octave[octave] = float(center)

        d_anchor_by_octave = {}
        for octave in sorted(set(c_sharp_by_octave.keys()) & set(d_sharp_by_octave.keys())):
            d_anchor_by_octave[octave] = (c_sharp_by_octave[octave] + d_sharp_by_octave[octave]) / 2.0

        segment_data = []
        for octave in sorted(d_anchor_by_octave.keys()):
            next_octave = octave + 1
            if next_octave not in d_anchor_by_octave:
                continue

            d0 = float(d_anchor_by_octave[octave])
            d1 = float(d_anchor_by_octave[next_octave])
            span = d1 - d0
            if span < 12.0:
                continue

            if octave in f_sharp_by_octave and octave in g_sharp_by_octave:
                g_anchor = (f_sharp_by_octave[octave] + g_sharp_by_octave[octave]) / 2.0
            else:
                g_anchor = d0 + (span * (3.0 / 7.0))

            if octave in g_sharp_by_octave and octave in a_sharp_by_octave:
                a_anchor = (g_sharp_by_octave[octave] + a_sharp_by_octave[octave]) / 2.0
            else:
                a_anchor = d0 + (span * (4.0 / 7.0))

            # Keep anchors in plausible octave-relative windows.
            g_anchor = float(np.clip(g_anchor, d0 + (span * 0.25), d0 + (span * 0.60)))
            a_anchor = float(np.clip(a_anchor, d0 + (span * 0.40), d0 + (span * 0.82)))
            if a_anchor <= g_anchor + (span * 0.08):
                a_anchor = min(d1 - (span * 0.10), g_anchor + (span * 0.16))

            left_step_raw = (g_anchor - d0) / 3.0
            right_step_raw = (d1 - a_anchor) / 3.0
            if left_step_raw <= 0:
                left_step_raw = span / 7.0
            if right_step_raw <= 0:
                right_step_raw = span / 7.0

            segment_data.append(
                {
                    "octave": octave,
                    "d0": d0,
                    "d1": d1,
                    "span": span,
                    "left_raw": left_step_raw,
                    "right_raw": right_step_raw,
                }
            )

        if not segment_data:
            return [], 0.0, {}

        left_raw_values = [seg["left_raw"] for seg in segment_data]
        right_raw_values = [seg["right_raw"] for seg in segment_data]
        left_median = float(np.median(left_raw_values)) if left_raw_values else 0.0
        right_median = float(np.median(right_raw_values)) if right_raw_values else 0.0

        white_center_by_note = {}

        def set_center(note_key, value):
            val = float(value)
            if note_key in white_center_by_note:
                white_center_by_note[note_key] = (white_center_by_note[note_key] + val) / 2.0
            else:
                white_center_by_note[note_key] = val

        for octave, d_center in d_anchor_by_octave.items():
            set_center(f"D{octave}", d_center)

        for seg in segment_data:
            octave = int(seg["octave"])
            span = float(seg["span"])
            d0 = float(seg["d0"])
            d1 = float(seg["d1"])

            left_step = float(seg["left_raw"])
            right_step = float(seg["right_raw"])
            if left_median > 0:
                left_step = float(np.clip(left_step, left_median * 0.78, left_median * 1.22))
            if right_median > 0:
                right_step = float(np.clip(right_step, right_median * 0.78, right_median * 1.22))

            # Physical guard rails for 7 white steps inside one D->D octave span.
            min_step = span * 0.10
            max_step = span * 0.20
            left_step = float(np.clip(left_step, min_step, max_step))
            right_step = float(np.clip(right_step, min_step, max_step))

            # If either side is too distorted, fall back to equal 1/7 spacing.
            if (left_step * 3.0) + (right_step * 3.0) > (span * 0.95):
                equal_step = span / 7.0
                left_step = equal_step
                right_step = equal_step

            e_center = d0 + left_step
            f_center = d0 + (left_step * 2.0)
            g_center = d0 + (left_step * 3.0)
            a_center = d1 - (right_step * 3.0)
            b_center = a_center + right_step
            c_next_center = a_center + (right_step * 2.0)

            if not (d0 < e_center < f_center < g_center < a_center < b_center < c_next_center < d1):
                equal_step = span / 7.0
                e_center = d0 + equal_step
                f_center = d0 + (equal_step * 2.0)
                g_center = d0 + (equal_step * 3.0)
                a_center = d0 + (equal_step * 4.0)
                b_center = d0 + (equal_step * 5.0)
                c_next_center = d0 + (equal_step * 6.0)

            set_center(f"E{octave}", e_center)
            set_center(f"F{octave}", f_center)
            set_center(f"G{octave}", g_center)
            set_center(f"A{octave}", a_center)
            set_center(f"B{octave}", b_center)
            set_center(f"C{octave + 1}", c_next_center)

        raw_centers = sorted(white_center_by_note.values())
        if len(raw_centers) < 4:
            return [], 0.0, {}

        center_diffs = [raw_centers[i + 1] - raw_centers[i] for i in range(len(raw_centers) - 1)]
        center_diffs = [diff for diff in center_diffs if diff > 1.0]
        step_estimate = float(np.median(center_diffs)) if center_diffs else 0.0
        if step_estimate <= 0:
            return [], 0.0, {}

        min_center_gap = max(3.0, step_estimate * 0.40)
        deduped_centers = []
        for center in raw_centers:
            if deduped_centers and abs(center - deduped_centers[-1]) < min_center_gap:
                deduped_centers[-1] = (deduped_centers[-1] + center) / 2.0
            else:
                deduped_centers.append(center)

        if len(deduped_centers) < 3:
            return [], 0.0, {}

        # Extend to ROI edges if D-anchored spans do not cover cropped ends.
        # Use ceil (not round) so edge gaps don't leave a large unsolved span,
        # which can trigger unstable guided splits and indexing drift.
        left_room = deduped_centers[0] - (step_estimate * 0.5)
        right_room = (float(width - 1) - deduped_centers[-1]) - (step_estimate * 0.5)
        extra_left = max(0, int(np.ceil(left_room / step_estimate)))
        extra_right = max(0, int(np.ceil(right_room / step_estimate)))
        max_extra = max(0, int(np.ceil(float(width) / max(step_estimate, 1.0))) + 2)
        extra_left = min(extra_left, max_extra)
        extra_right = min(extra_right, max_extra)

        for _ in range(extra_left):
            deduped_centers.insert(0, deduped_centers[0] - step_estimate)
        for _ in range(extra_right):
            deduped_centers.append(deduped_centers[-1] + step_estimate)

        final_centers = []
        for center in deduped_centers:
            clamped = float(np.clip(center, 0.0, float(width - 1)))
            if final_centers and abs(clamped - final_centers[-1]) < min_center_gap:
                continue
            final_centers.append(clamped)

        return final_centers, step_estimate, dict(white_center_by_note)

    def _build_white_spans_from_centers(
        self,
        centers,
        width,
        col_med,
        expected_white_width,
        min_key_width,
        min_sep_width,
    ):
        """Convert sorted white-key centers into key spans, splitting overly-wide regions."""
        if len(centers) < 2:
            return []

        centers = sorted(float(c) for c in centers if 0.0 <= c <= float(width - 1))
        if len(centers) < 2:
            return []

        boundaries = [0.0]
        for idx in range(len(centers) - 1):
            boundaries.append((centers[idx] + centers[idx + 1]) / 2.0)
        boundaries.append(float(width))

        spans = []
        split_trigger = max(expected_white_width * 1.75, float(min_key_width * 2))
        for idx in range(len(boundaries) - 1):
            x_left = int(round(boundaries[idx]))
            x_right = int(round(boundaries[idx + 1])) - 1
            if x_right < x_left:
                continue

            x_left = max(0, min(width - 1, x_left))
            x_right = max(0, min(width - 1, x_right))
            span_width = x_right - x_left + 1
            if span_width < min_key_width:
                continue

            if span_width <= split_trigger:
                spans.append((x_left, x_right))
                continue

            split_spans = self._guided_split_white_span(
                col_med,
                x_left,
                x_right,
                expected_white_width,
                min_key_width,
                min_sep_width,
            )
            if split_spans and len(split_spans) > 1:
                spans.extend(split_spans)
                continue

            split_count = max(2, int(round(span_width / max(1.0, expected_white_width))))
            split_count = min(6, split_count)
            spans.extend(
                self._split_span_evenly(
                    x_left,
                    x_right,
                    split_count,
                    min_key_width,
                )
            )

        if not spans:
            return []

        spans.sort(key=lambda span: span[0])
        deduped_spans = []
        for x_left, x_right in spans:
            if deduped_spans and x_left <= deduped_spans[-1][1]:
                prev_left, prev_right = deduped_spans[-1]
                deduped_spans[-1] = (prev_left, max(prev_right, x_right))
            else:
                deduped_spans.append((x_left, x_right))

        return deduped_spans

    def _detect_white_keys_from_black_d_lattice(self, gray_img):
        """Primary white-key solver: derive white centers from D anchors and local octave spacing."""
        height, width = gray_img.shape
        if not self.black_keys:
            return []

        black_note_centers = self._build_black_note_center_map()
        if not black_note_centers:
            return []

        white_centers, step_estimate, white_center_by_note = self._estimate_white_centers_from_d_lattice(
            black_note_centers,
            width,
        )
        if len(white_centers) < 4 or step_estimate <= 0:
            return []

        white_centers = self._apply_black_residual_edge_warp(
            white_centers,
            width,
            step_estimate,
            black_note_centers,
            white_center_by_note,
        )

        strip_start = self._find_white_strip_start(gray_img)
        if strip_start < 0 or strip_start >= height:
            strip_start = int(height * self.params.get("white_bottom_ratio", 0.85))
        y_top = max(
            strip_start,
            int(height * self.params.get("white_initial_top_ratio", 0.7)),
        )
        y_bottom = height - 1
        key_height = max(1, y_bottom - y_top + 1)

        strip = gray_img[strip_start:, :] if 0 <= strip_start < height else gray_img
        if strip.size == 0:
            strip = gray_img
        col_med = np.median(strip, axis=0).astype(np.float32)

        expected_white_width = max(8.0, float(step_estimate))
        min_key_width = max(6, int(round(expected_white_width * 0.45)))
        min_sep_width = int(self.params.get("white_sep_min_width", 1))

        white_spans = self._build_white_spans_from_centers(
            white_centers,
            width,
            col_med,
            expected_white_width,
            min_key_width,
            min_sep_width,
        )
        if not white_spans:
            return []

        white_keys = []
        for x_left, x_right in white_spans:
            key_width = x_right - x_left + 1
            if key_width < min_key_width:
                continue
            padded_overlay = self._add_overlay_padding(
                x_left,
                y_top,
                key_width,
                key_height,
            )
            if padded_overlay[2] > 1:
                white_keys.append(padded_overlay)

        self.logger.debug(
            "white-from-black D-lattice solve: centers=%d spans=%d step=%.2f",
            len(white_centers),
            len(white_keys),
            step_estimate,
        )

        return white_keys

    def _detect_white_keys_from_black_boundary_solver(self, gray_img):
        """Fallback white solver using boundary inference from black-key center gaps."""
        height, width = gray_img.shape
        if not self.black_keys:
            return []

        sorted_black = sorted(self.black_keys, key=lambda key: key[0])
        black_centers = [
            float(black_key[0] + (black_key[2] / 2.0))
            for black_key in sorted_black
            if black_key[2] > 0
        ]
        if len(black_centers) < 2:
            return []

        strip_start = self._find_white_strip_start(gray_img)
        if strip_start < 0 or strip_start >= height:
            strip_start = int(height * self.params.get("white_bottom_ratio", 0.85))
        y_top = max(
            strip_start,
            int(height * self.params.get("white_initial_top_ratio", 0.7)),
        )
        y_bottom = height - 1
        key_height = max(1, y_bottom - y_top + 1)

        center_diffs = [black_centers[i + 1] - black_centers[i] for i in range(len(black_centers) - 1)]
        if not center_diffs:
            return []

        large_gap_mask = self._classify_large_center_gaps(center_diffs)
        single_white_diffs = [
            gap
            for gap, is_large in zip(center_diffs, large_gap_mask)
            if not is_large
        ]
        if not single_white_diffs:
            single_white_diffs = center_diffs

        expected_white_width = max(8.0, float(np.median(single_white_diffs)))
        min_key_width = max(6, int(round(expected_white_width * 0.45)))
        min_sep_width = int(self.params.get("white_sep_min_width", 1))

        # Build a full white-key boundary set:
        # - all black-key centers (C|D, D|E, F|G, G|A, A|B)
        # - inferred inner split for each large center gap (E|F or B|C)
        boundary_positions = [black_centers[0]]
        inferred_boundary_count = 0
        for idx, gap in enumerate(center_diffs):
            if large_gap_mask[idx]:
                left_est = expected_white_width
                right_est = expected_white_width

                if idx > 0:
                    prev_gap = center_diffs[idx - 1]
                    left_est = prev_gap if not large_gap_mask[idx - 1] else min(prev_gap, expected_white_width)
                if idx + 1 < len(center_diffs):
                    next_gap = center_diffs[idx + 1]
                    right_est = next_gap if not large_gap_mask[idx + 1] else min(next_gap, expected_white_width)

                min_est = max(4.0, expected_white_width * 0.55)
                max_est = max(min_est + 1.0, expected_white_width * 1.45)
                left_est = float(np.clip(left_est, min_est, max_est))
                right_est = float(np.clip(right_est, min_est, max_est))

                if (left_est + right_est) <= 0:
                    split_ratio = 0.5
                else:
                    split_ratio = left_est / (left_est + right_est)
                inferred_boundary = black_centers[idx] + (gap * split_ratio)

                margin = max(2.0, expected_white_width * 0.20)
                inferred_boundary = float(np.clip(
                    inferred_boundary,
                    black_centers[idx] + margin,
                    black_centers[idx + 1] - margin,
                ))
                boundary_positions.append(inferred_boundary)
                inferred_boundary_count += 1

            boundary_positions.append(black_centers[idx + 1])

        boundary_positions.sort()
        deduped_boundaries = []
        min_boundary_gap = max(2, int(round(expected_white_width * 0.18)))
        for boundary in boundary_positions:
            boundary_x = int(round(boundary))
            boundary_x = max(1, min(width - 1, boundary_x))
            if deduped_boundaries and abs(boundary_x - deduped_boundaries[-1]) < min_boundary_gap:
                deduped_boundaries[-1] = int(round((deduped_boundaries[-1] + boundary_x) / 2.0))
            else:
                deduped_boundaries.append(boundary_x)

        if not deduped_boundaries:
            return []

        strip = gray_img[strip_start:, :] if 0 <= strip_start < height else gray_img
        if strip.size == 0:
            strip = gray_img
        col_med = np.median(strip, axis=0).astype(np.float32)

        white_spans = []
        walls = [0] + deduped_boundaries + [width]
        split_trigger = max(expected_white_width * 1.75, float(min_key_width * 2))
        for idx in range(len(walls) - 1):
            x_left = int(walls[idx])
            x_right = int(walls[idx + 1]) - 1
            span_width = x_right - x_left + 1
            if span_width < min_key_width:
                continue

            if span_width <= split_trigger:
                white_spans.append((x_left, x_right))
                continue

            split_spans = self._guided_split_white_span(
                col_med,
                x_left,
                x_right,
                expected_white_width,
                min_key_width,
                min_sep_width,
            )
            if split_spans and len(split_spans) > 1:
                white_spans.extend(split_spans)
                continue

            split_count = max(2, int(round(span_width / max(1.0, expected_white_width))))
            split_count = min(6, split_count)
            white_spans.extend(
                self._split_span_evenly(
                    x_left,
                    x_right,
                    split_count,
                    min_key_width,
                )
            )

        if not white_spans:
            return []

        white_spans.sort(key=lambda span: span[0])
        deduped_spans = []
        for x_left, x_right in white_spans:
            if deduped_spans and x_left <= deduped_spans[-1][1]:
                prev_left, prev_right = deduped_spans[-1]
                deduped_spans[-1] = (prev_left, max(prev_right, x_right))
            else:
                deduped_spans.append((x_left, x_right))

        white_keys = []
        for x_left, x_right in deduped_spans:
            key_width = x_right - x_left + 1
            if key_width < min_key_width:
                continue
            padded_overlay = self._add_overlay_padding(
                x_left,
                y_top,
                key_width,
                key_height,
            )
            if padded_overlay[2] > 1:
                white_keys.append(padded_overlay)

        self.logger.debug(
            "white-from-black boundary solve: black=%d inferred=%d boundaries=%d spans=%d",
            len(sorted_black),
            inferred_boundary_count,
            len(deduped_boundaries),
            len(white_keys),
        )

        return white_keys

    def _detect_white_keys_from_black(self, gray_img):
        """Reconstruct white keys from black geometry, preferring D-lattice normalization."""
        white_keys = self._detect_white_keys_from_black_d_lattice(gray_img)
        if len(white_keys) >= 4:
            return white_keys

        self.logger.debug(
            "white-from-black D-lattice solver under-detected (%d). Falling back to boundary solver.",
            len(white_keys),
        )
        return self._detect_white_keys_from_black_boundary_solver(gray_img)

    def _detect_white_keys(self, gray_img):
        """Detect white keys by finding vertical separations"""
        height, width = gray_img.shape

        strip_start = self._find_white_strip_start(gray_img)
        if strip_start < 0 or strip_start >= height:
            strip_start = int(height * self.params.get("white_bottom_ratio", 0.85))
        strip = gray_img[strip_start:, :] if 0 <= strip_start < height else gray_img

        col_med = np.median(strip, axis=0).astype(np.float32)
        col_med_u8 = np.clip(col_med, 0, 255).astype(np.uint8)

        # Ignore extreme edge columns for global separator stats.
        edge_trim = max(4, int(round(width * 0.02)))
        if width - (2 * edge_trim) < 40:
            edge_trim = 0

        if edge_trim > 0:
            stats_col_med = col_med[edge_trim:width - edge_trim]
            stats_col_med_u8 = col_med_u8[edge_trim:width - edge_trim]
        else:
            stats_col_med = col_med
            stats_col_med_u8 = col_med_u8

        white_level = float(np.percentile(stats_col_med, 90))
        dark_level = float(np.percentile(stats_col_med, 10))
        dyn = white_level - dark_level

        sep_cols = None
        dyn_min = float(self.params.get("white_sep_dyn_min", 8))
        otsu_threshold = None
        if stats_col_med_u8.size >= 2:
            otsu_threshold, _ = cv2.threshold(
                stats_col_med_u8.reshape(1, -1),
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            otsu_threshold = float(otsu_threshold)
        elif col_med_u8.size >= 2:
            otsu_threshold, _ = cv2.threshold(
                col_med_u8.reshape(1, -1),
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            otsu_threshold = float(otsu_threshold)

        if dyn >= dyn_min:
            ratio = float(self.params.get("white_sep_ratio", 0.55))
            thresh = white_level - (ratio * dyn)
            sep_cols = col_med < thresh
        else:
            if otsu_threshold is None:
                sep_cols = (col_med < np.median(col_med))
            else:
                sep_cols = (col_med < otsu_threshold)

        raw_sep_cols = sep_cols.copy()
        close_k = int(self.params.get("white_sep_close_kernel", 5))
        if close_k > 1:
            sep_u8 = (sep_cols.astype(np.uint8) * 255).reshape(1, -1)
            sep_u8 = cv2.morphologyEx(
                sep_u8,
                cv2.MORPH_CLOSE,
                np.ones((1, close_k), np.uint8),
            )
            sep_cols = (sep_u8.flatten() > 0)

        open_k = int(self.params.get("white_sep_open_kernel", 3))
        if open_k > 1:
            sep_u8 = (sep_cols.astype(np.uint8) * 255).reshape(1, -1)
            sep_u8 = cv2.morphologyEx(
                sep_u8,
                cv2.MORPH_OPEN,
                np.ones((1, open_k), np.uint8),
            )
            sep_cols = (sep_u8.flatten() > 0)

        min_sep_width = int(self.params.get("white_sep_min_width", 2))
        runs = self._runs_from_mask(sep_cols, min_width=min_sep_width)

        # If morphology over-merged separators, retry with a gentler pass.
        if len(runs) < 8:
            relaxed_sep_cols = raw_sep_cols.copy()
            relaxed_close_k = min(close_k, 3)
            if relaxed_close_k > 1:
                relaxed_u8 = (relaxed_sep_cols.astype(np.uint8) * 255).reshape(1, -1)
                relaxed_u8 = cv2.morphologyEx(
                    relaxed_u8,
                    cv2.MORPH_CLOSE,
                    np.ones((1, relaxed_close_k), np.uint8),
                )
                relaxed_sep_cols = (relaxed_u8.flatten() > 0)
            relaxed_runs = self._runs_from_mask(relaxed_sep_cols, min_width=min_sep_width)
            if len(relaxed_runs) > len(runs):
                sep_cols = relaxed_sep_cols
                runs = relaxed_runs

        if len(runs) < 5:
            if otsu_threshold is None:
                sep_cols = (col_med < np.median(col_med))
            else:
                sep_cols = (col_med < otsu_threshold)
            if close_k > 1:
                sep_u8 = (sep_cols.astype(np.uint8) * 255).reshape(1, -1)
                sep_u8 = cv2.morphologyEx(
                    sep_u8,
                    cv2.MORPH_CLOSE,
                    np.ones((1, close_k), np.uint8),
                )
                sep_cols = (sep_u8.flatten() > 0)
            runs = self._runs_from_mask(sep_cols, min_width=min_sep_width)

        gaps = []
        for i in range(len(runs) - 1):
            gap = runs[i + 1][0] - runs[i][1] - 1
            if gap > 0:
                gaps.append(gap)
        if gaps:
            med_gap = float(np.median(gaps))
            min_key_width = max(6, int(med_gap * 0.50))
        else:
            min_key_width = max(6, int(width / 150))

        candidate_spans = []
        walls = [(-1, -1)] + runs + [(width, width)]
        for i in range(len(walls) - 1):
            x_left = walls[i][1] + 1
            x_right = walls[i + 1][0] - 1
            key_width = x_right - x_left + 1
            if key_width >= min_key_width:
                candidate_spans.append((x_left, x_right))

        expected_width = self._estimate_white_key_width(
            width,
            [x_right - x_left + 1 for x_left, x_right in candidate_spans],
        )

        white_spans = []
        for x_left, x_right in candidate_spans:
            split_spans = self._guided_split_white_span(
                col_med,
                x_left,
                x_right,
                expected_width,
                min_key_width,
                min_sep_width,
            )
            if split_spans:
                white_spans.extend(split_spans)

        white_keys = []
        y_top = max(
            strip_start,
            int(height * self.params.get("white_initial_top_ratio", 0.7)),
        )
        y_bottom = height - 1
        key_height = y_bottom - y_top + 1
        for x_left, x_right in white_spans:
            key_width = x_right - x_left + 1
            if key_width < min_key_width:
                continue
            padded_overlay = self._add_overlay_padding(
                x_left,
                y_top,
                key_width,
                key_height,
            )
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
        
        if self.params.get("type_aware_assignment", False):
            self.key_notes = self._assign_notes_type_aware()
        else:
            # Find F# anchor using confident LSSL pattern detection
            f_sharp_position = self._find_confident_f_sharp_anchor()

            if f_sharp_position is None:
                self.logger.debug("Could not find confident F# anchor - using fallback assignment")
                self.key_notes = self._fallback_note_assignment()
            else:
                # Unified chromatic assignment using pixel-by-pixel scanning
                self.key_notes = self._assign_notes_chromatically_from_anchor(f_sharp_position)

        self._apply_white_post_assignment_adjustments()
        
        self.logger.debug(f"DEBUG: Total assigned keys: {len(self.key_notes)}")
        
        if self.key_notes:
            self.logger.debug("First 10 chromatic note assignments:")
            sorted_notes = sorted(self.key_notes.items())
            for i, (center_x, note_info) in enumerate(sorted_notes[:10]):
                self.logger.debug(f"  Key {i}: center_x={center_x}, note={note_info['note']}, type={note_info['type']}")
        
        self.logger.debug(f"Assigned notes to {len(self.key_notes)} keys")
        return self.key_notes

    def _extract_note_name(self, full_note):
        """Extract note class (e.g. C, F#, B) from note+octave string."""
        if not full_note:
            return ""
        idx = 0
        while idx < len(full_note) and not (full_note[idx].isdigit() or full_note[idx] == "-"):
            idx += 1
        return full_note[:idx]

    def _extract_note_octave(self, full_note):
        """Extract octave number from note+octave string."""
        if not full_note:
            return None
        idx = 0
        while idx < len(full_note) and not (full_note[idx].isdigit() or full_note[idx] == "-"):
            idx += 1
        if idx >= len(full_note):
            return None
        try:
            return int(full_note[idx:])
        except ValueError:
            return None

    def _apply_white_post_assignment_adjustments(self):
        """Apply post-assignment geometry adjustments for white-key overlays."""
        if not self.key_notes or not self.keyboard_region:
            return

        # Use current white key ordering to clamp shifts between neighboring keys.
        sorted_whites = sorted(
            (
                {
                    "center_x": center_x,
                    "note_info": note_info,
                    "x": int(note_info["box"][0]),
                    "y": int(note_info["box"][1]),
                    "w": int(note_info["box"][2]),
                    "h": int(note_info["box"][3]),
                    "note_name": self._extract_note_name(note_info.get("note", "")),
                }
                for center_x, note_info in self.key_notes.items()
                if note_info.get("type") == "white" and note_info.get("box")
            ),
            key=lambda item: item["x"],
        )

        if not sorted_whites:
            return

        _, _, roi_left_x, roi_right_x = self.keyboard_region
        roi_width = max(1, int(roi_right_x - roi_left_x))

        # Lock D/G/A onto midpoint between adjacent black keys.
        midpoint_pairs = {
            "D": ("C#", "D#"),
            "G": ("F#", "G#"),
            "A": ("G#", "A#"),
        }
        black_center_by_note = {
            str(note_info.get("note", "")): int(center_x)
            for center_x, note_info in self.key_notes.items()
            if note_info.get("type") == "black"
        }

        midpoint_count = 0
        for idx, item in enumerate(sorted_whites):
            note_name = item["note_name"]
            if note_name not in midpoint_pairs:
                continue

            octave = self._extract_note_octave(item["note_info"].get("note", ""))
            if octave is None:
                continue

            left_note, right_note = midpoint_pairs[note_name]
            left_note = f"{left_note}{octave}"
            right_note = f"{right_note}{octave}"
            if left_note not in black_center_by_note or right_note not in black_center_by_note:
                continue

            target_center = int(round((black_center_by_note[left_note] + black_center_by_note[right_note]) / 2.0))
            desired_x = target_center - (item["w"] // 2)

            min_x = 0
            max_x = max(0, roi_width - item["w"])
            if idx > 0:
                prev = sorted_whites[idx - 1]
                min_x = max(min_x, prev["x"] + prev["w"] + 1)
            if idx < (len(sorted_whites) - 1):
                nxt = sorted_whites[idx + 1]
                max_x = min(max_x, nxt["x"] - item["w"] - 1)

            if max_x < min_x:
                continue

            new_x = int(max(min_x, min(max_x, desired_x)))
            if new_x == item["x"]:
                continue

            note_info = item["note_info"]
            note_info["box"] = (new_x, item["y"], item["w"], item["h"])
            item["x"] = new_x
            midpoint_count += 1

        # Re-sort after midpoint locking before manual edge-tail shift.
        sorted_whites.sort(key=lambda item: item["x"])

        left_ticks = int(round(float(self.params.get("white_edge_left_shift_ticks", 0))))
        right_ticks = int(round(float(self.params.get("white_edge_right_shift_ticks", 0))))
        edge_shift_count = 0
        if (left_ticks != 0 or right_ticks != 0) and len(sorted_whites) >= 3:
            edge_count = max(1, int(round(len(sorted_whites) * 0.20)))
            if (2 * edge_count) >= len(sorted_whites):
                edge_count = max(1, (len(sorted_whites) - 1) // 2)

            median_width = float(np.median([item["w"] for item in sorted_whites]))
            px_per_tick = median_width * 0.05
            left_edge_delta_px = -float(left_ticks) * px_per_tick
            right_edge_delta_px = float(right_ticks) * px_per_tick
            inner_weight_floor = 0.01
            edge_falloff_power = 6.0

            def _edge_weight(edge_progress: float) -> float:
                """Map edge progress to shift weight with a 1% floor at inner tail edge."""
                base_min = 1.0 / float(max(1, edge_count))
                if edge_progress <= base_min:
                    return inner_weight_floor
                normalized = (edge_progress - base_min) / max(1e-6, 1.0 - base_min)
                normalized = float(np.clip(normalized, 0.0, 1.0))
                curved = normalized ** edge_falloff_power
                return inner_weight_floor + ((1.0 - inner_weight_floor) * curved)

            desired_positions = []
            for idx, item in enumerate(sorted_whites):
                shift_px = 0.0

                if idx < edge_count:
                    edge_progress = (edge_count - idx) / float(max(1, edge_count))
                    shift_px += left_edge_delta_px * _edge_weight(edge_progress)

                if idx >= len(sorted_whites) - edge_count:
                    right_idx = (len(sorted_whites) - 1) - idx
                    edge_progress = (edge_count - right_idx) / float(max(1, edge_count))
                    shift_px += right_edge_delta_px * _edge_weight(edge_progress)

                desired_x = int(round(item["x"] + shift_px))
                max_x = max(0, roi_width - item["w"])
                desired_x = max(0, min(max_x, desired_x))
                desired_positions.append(desired_x)

            adjusted_positions = list(desired_positions)
            for idx in range(1, len(sorted_whites)):
                prev_item = sorted_whites[idx - 1]
                min_x = adjusted_positions[idx - 1] + prev_item["w"] + 1
                if adjusted_positions[idx] < min_x:
                    adjusted_positions[idx] = min_x

            for idx in range(len(sorted_whites) - 2, -1, -1):
                cur_item = sorted_whites[idx]
                max_x = adjusted_positions[idx + 1] - cur_item["w"] - 1
                if adjusted_positions[idx] > max_x:
                    adjusted_positions[idx] = max_x

            for idx, item in enumerate(sorted_whites):
                max_x = max(0, roi_width - item["w"])
                adjusted_positions[idx] = max(0, min(max_x, adjusted_positions[idx]))

            for idx in range(1, len(sorted_whites)):
                prev_item = sorted_whites[idx - 1]
                min_x = adjusted_positions[idx - 1] + prev_item["w"] + 1
                if adjusted_positions[idx] < min_x:
                    adjusted_positions[idx] = min_x

            for idx in range(len(sorted_whites) - 2, -1, -1):
                cur_item = sorted_whites[idx]
                max_x = adjusted_positions[idx + 1] - cur_item["w"] - 1
                if adjusted_positions[idx] > max_x:
                    adjusted_positions[idx] = max_x

            for idx, item in enumerate(sorted_whites):
                new_x = adjusted_positions[idx]
                if new_x == item["x"]:
                    continue
                item["x"] = new_x
                item["note_info"]["box"] = (new_x, item["y"], item["w"], item["h"])
                edge_shift_count += 1

        if midpoint_count > 0 or edge_shift_count > 0:
            self.logger.debug(
                "Applied white post-adjustments: midpoint=%d edge_shift=%d left_ticks=%d right_ticks=%d",
                midpoint_count,
                edge_shift_count,
                left_ticks,
                right_ticks,
            )
    
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
    
    def _find_f_sharp_anchor_candidates(self):
        """Return candidate F# black-key indices, ordered by confidence."""
        if len(self.black_keys) < 3:
            return []

        gaps = []
        for i in range(len(self.black_keys) - 1):
            gap = self.black_keys[i + 1][0] - (self.black_keys[i][0] + self.black_keys[i][2])
            gaps.append(gap)

        median_gap = sorted(gaps)[len(gaps) // 2]
        gap_threshold = median_gap * 1.4

        candidates = []

        for i in range(len(gaps) - 3):
            if (
                gaps[i] > gap_threshold and
                gaps[i + 1] <= gap_threshold and
                gaps[i + 2] <= gap_threshold and
                gaps[i + 3] > gap_threshold
            ):
                candidates.append(i + 1)

        for i in range(len(gaps) - 2):
            if (
                gaps[i] <= gap_threshold and
                gaps[i + 1] <= gap_threshold and
                gaps[i + 2] > gap_threshold
            ):
                if i not in candidates:
                    candidates.append(i)

        return candidates

    def _assign_notes_type_aware(self):
        """Assign notes by key type, using black-key anchors and white-key scanning."""
        candidates = self._find_f_sharp_anchor_candidates()
        if not candidates:
            return self._fallback_note_assignment()

        fallback_notes = None

        for f_sharp_idx in candidates:
            black_notes = self._assign_black_key_notes(f_sharp_idx)
            white_notes, used_fallback = self._assign_white_key_notes_by_scanning(
                black_notes,
                return_fallback=True,
            )

            if not white_notes:
                continue

            combined = {**black_notes, **white_notes}
            if not used_fallback:
                return combined

            if fallback_notes is None:
                fallback_notes = combined

        if fallback_notes:
            return fallback_notes

        return self._fallback_note_assignment()

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
        
        # Estimate octave based on anchor position in black-key groups
        base_octave = max(0, (f_sharp_idx - pattern_position) // 5)
        
        for i, black_key in enumerate(self.black_keys):
            # Calculate pattern index
            pattern_idx = (i - f_sharp_idx + pattern_position) % 5
            
            # Calculate octave
            octave = base_octave + ((i - f_sharp_idx + pattern_position) // 5)
            
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
    
    def _assign_white_key_notes_by_scanning(self, black_notes, return_fallback=False):
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
            white_notes = self._fallback_white_assignment()
            if return_fallback:
                return white_notes, True
            return white_notes
        
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
            white_notes = self._fallback_white_assignment()
            if return_fallback:
                return white_notes, True
            return white_notes
        
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
        
        if return_fallback:
            return white_notes, False
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
