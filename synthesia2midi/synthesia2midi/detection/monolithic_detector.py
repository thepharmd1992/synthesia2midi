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
from pathlib import Path
from typing import Tuple

DEFAULT_DETECTION_PARAMS = {
    "preprocess_mode": "none",  # none | clahe | clahe_unsharp
    "preprocess_upscale": 1,  # 1 (off) or 2 (common)
    # Optionally decouple preprocessing used for white-edge detection from black-key detection.
    # - mode_white: "inherit" uses preprocess_mode; otherwise one of: none | clahe | clahe_unsharp
    # - upscale_white: 0 uses preprocess_upscale; otherwise an integer >= 1
    "preprocess_mode_white": "inherit",
    "preprocess_upscale_white": 0,
    "preprocess_clahe_clip": 2.0,
    "preprocess_clahe_tile": 8,
    "preprocess_unsharp_amount": 1.0,
    "preprocess_unsharp_sigma": 1.0,
    "black_upper_ratio": 0.6,
    "black_detection_method": "columns",  # "columns" | "components"
    "black_threshold": 70,
    "black_threshold_method": "fixed",
    "black_adaptive_block_size": 21,
    "black_adaptive_c": 7,
    "black_column_ratio": 0.10,
    "black_edge_column_ratio": 0.03,
    "black_min_width": 10,
    "black_max_width": 100,
    "black_min_height_ratio": 0.4,
    "black_max_height_ratio": 1.0,
    "black_min_area_ratio": 0.001,
    "black_min_count_for_valid": 3,
    "black_edge_fallback": False,
    "white_bottom_ratio": 0.85,
    "white_edge_band_ratio": 0.12,
    "white_edge_smooth_kernel": 5,
    "white_edge_std_factor": 2.0,
    # White key boundary detection method:
    # - "edge": vertical edge strength (Sobel) across a horizontal band
    # - "blackhat": emphasize dark vertical seams via morphological black-hat (more tolerant of blur)
    # - "dark_columns": detect dark separator columns by robust per-column brightness thresholding
    "white_seam_method": "dark_columns",
    "white_blackhat_kernel_width": 15,
    "white_dark_column_threshold": "otsu",  # "otsu" | "relative" | int(0-255)
    # How to summarize brightness per x-column for dark-column seam detection.
    # - "median": robust but can be polluted by dark note labels if they cover much of the column
    # - "p90": robust to sparse dark labels; separator columns stay dark across most of the column
    "white_dark_column_stat": "median",  # "median" | "p90" | "p95"
    # Exclude bottom part of the white strip from the column statistic (helps ignore note labels).
    "white_strip_exclude_bottom_ratio": 0.0,
    "white_dark_column_white_percentile": 90,  # for "relative": baseline "white" estimate
    "white_dark_column_dark_percentile": 20,  # for "relative": baseline "dark" estimate
    "white_dark_column_relative_ratio": 0.5,  # threshold = white - ratio*(white-dark)
    "white_separator_min_width": 2,
    "white_auto_strip": True,  # attempt to crop out black-key region for dark_columns
    "white_auto_strip_dark_threshold": 60,
    "white_auto_strip_frac_threshold": 0.02,
    "white_auto_strip_min_run": 6,
    "white_min_width": 15,
    "white_initial_top_ratio": 0.7,
    "white_initial_height_ratio": 0.3,
    "edge_boundary_padding_px": 3,
    "white_gap_fill": False,
    "white_gap_fill_max_ratio": 1.6,
    "white_gap_fill_min_edges": 6,
    "padding_percent": 0.15,
    "trim_saturation_threshold": 45,
    "trim_gray_threshold": 140,
    "trim_row_height": 20,
    # Optional debug: save the ROI + preprocessed grayscale images used for detection.
    # Set by AutoDetectAdapter when the user runs "Calibrate Key Overlays".
    "debug_save_preprocess_dir": None,  # directory path
    "debug_save_tag": "",  # profile name or other identifier
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
        self._white_strip_start = None
        self._white_separator_runs = None

    def _save_preprocess_debug_images(
        self,
        base_dir: str,
        tag: str,
        keyboard_img_bgr: np.ndarray,
        keyboard_gray: np.ndarray,
        black_gray: np.ndarray,
        white_gray: np.ndarray,
        black_scale: Tuple[float, float],
        white_scale: Tuple[float, float],
        mode_black: str,
        mode_white: str,
        upscale_black: int,
        upscale_white: int,
    ) -> None:
        try:
            base = Path(base_dir)
            base.mkdir(parents=True, exist_ok=True)

            safe_tag = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in (tag or ""))
            prefix = safe_tag + "_" if safe_tag else ""

            cv2.imwrite(
                str(base / f"{prefix}roi_bgr.png"),
                keyboard_img_bgr,
            )
            cv2.imwrite(
                str(base / f"{prefix}roi_gray.png"),
                keyboard_gray,
            )
            cv2.imwrite(
                str(base / f"{prefix}preprocessed_black_gray_{mode_black}_x{upscale_black}_sx{black_scale[0]:.2f}_sy{black_scale[1]:.2f}.png"),
                black_gray,
            )
            cv2.imwrite(
                str(base / f"{prefix}preprocessed_white_gray_{mode_white}_x{upscale_white}_sx{white_scale[0]:.2f}_sy{white_scale[1]:.2f}.png"),
                white_gray,
            )
            self.logger.info("Saved detector preprocess debug images to: %s", str(base))
        except Exception as e:
            self.logger.warning("Failed saving detector preprocess debug images: %s", e, exc_info=True)
        
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
        black_gray, black_x_scale, black_y_scale = self._preprocess_gray_for_detection(
            keyboard_gray
        )

        mode_white = (self.params.get("preprocess_mode_white") or "inherit").lower()
        if mode_white == "inherit":
            mode_white = (self.params.get("preprocess_mode") or "none").lower()
        upscale_white = int(self.params.get("preprocess_upscale_white", 0) or 0)
        if upscale_white <= 0:
            upscale_white = int(self.params.get("preprocess_upscale", 1) or 1)

        white_gray, white_x_scale, white_y_scale = self._preprocess_gray_for_detection(
            keyboard_gray,
            mode_override=mode_white,
            upscale_override=upscale_white,
        )

        debug_dir = self.params.get("debug_save_preprocess_dir")
        if debug_dir:
            self._save_preprocess_debug_images(
                base_dir=str(debug_dir),
                tag=str(self.params.get("debug_save_tag", "")),
                keyboard_img_bgr=keyboard_img,
                keyboard_gray=keyboard_gray,
                black_gray=black_gray,
                white_gray=white_gray,
                black_scale=(black_x_scale, black_y_scale),
                white_scale=(white_x_scale, white_y_scale),
                mode_black=str(self.params.get("preprocess_mode", "none")),
                mode_white=str(mode_white),
                upscale_black=int(self.params.get("preprocess_upscale", 1) or 1),
                upscale_white=int(upscale_white),
            )
        self.logger.info(
            "Detector preprocessing: mode=%s upscale=%s x_scale=%.3f y_scale=%.3f",
            self.params.get("preprocess_mode", "none"),
            self.params.get("preprocess_upscale", 1),
            black_x_scale,
            black_y_scale,
        )
        self.logger.info(
            "Detector preprocessing (white): mode=%s upscale=%s x_scale=%.3f y_scale=%.3f",
            mode_white,
            upscale_white,
            white_x_scale,
            white_y_scale,
        )
        self.logger.info(
            "Detector params: black_method=%s black_threshold=%s black_edge_fallback=%s white_gap_fill=%s",
            self.params.get("black_threshold_method", "fixed"),
            self.params.get("black_threshold", None),
            self.params.get("black_edge_fallback", False),
            self.params.get("white_gap_fill", False),
        )
        
        self.logger.debug(f"\n=== Detecting Keys in Region {right_x-left_x}x{bottom_y-top_y} ===")

        # Detect black keys first (easier to identify)
        self.black_keys = self._detect_black_keys(black_gray)
        if black_x_scale != 1.0 or black_y_scale != 1.0:
            self.black_keys = [
                self._scale_overlay(box, black_x_scale, black_y_scale)
                for box in self.black_keys
            ]
        self.logger.debug(f"Detected {len(self.black_keys)} black keys")
        
        self.logger.debug("First 5 black keys detected:")
        for i, (x, y, w, h) in enumerate(self.black_keys[:5]):
            self.logger.debug(f"  Black key {i}: x={x}, y={y}, w={w}, h={h} (absolute x={left_x + x})")
        
        # Detect white keys
        if white_x_scale != 1.0 or white_y_scale != 1.0:
            edge_profile = self._compute_white_edge_profile(white_gray)
            self.white_keys = self._detect_white_keys_from_profile(
                keyboard_gray,
                edge_profile=edge_profile,
                edge_profile_x_scale=white_x_scale,
                edge_profile_y_scale=white_y_scale,
            )
        else:
            self.white_keys = self._detect_white_keys(keyboard_gray)
        self.logger.debug(f"Detected {len(self.white_keys)} white keys")        
        
        self.logger.debug("First 5 white keys detected:")
        for i, (x, y, w, h) in enumerate(self.white_keys[:5]):
            self.logger.debug(f"  White key {i}: x={x}, y={y}, w={w}, h={h} (absolute x={left_x + x})")
        
        return len(self.black_keys), len(self.white_keys)

    def _preprocess_gray_for_detection(
        self,
        gray_img: np.ndarray,
        *,
        mode_override=None,
        upscale_override=None,
    ):
        """Preprocess ROI grayscale to improve detection on blurry footage."""
        height, width = gray_img.shape

        if upscale_override is None:
            upscale = int(self.params.get("preprocess_upscale", 1) or 1)
        else:
            upscale = int(upscale_override or 1)
        if upscale < 1:
            upscale = 1

        if mode_override is None:
            mode = (self.params.get("preprocess_mode") or "none").lower()
        else:
            mode = (mode_override or "none").lower()

        processed = gray_img
        if upscale != 1:
            processed = cv2.resize(
                processed,
                (width * upscale, height * upscale),
                interpolation=cv2.INTER_CUBIC,
            )

        if mode in {"clahe", "clahe_unsharp"}:
            tile = int(self.params.get("preprocess_clahe_tile", 8) or 8)
            tile = max(2, tile)
            clip = float(self.params.get("preprocess_clahe_clip", 2.0) or 2.0)
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
            processed = clahe.apply(processed)

        if mode == "clahe_unsharp":
            amount = float(self.params.get("preprocess_unsharp_amount", 1.0) or 1.0)
            sigma = float(self.params.get("preprocess_unsharp_sigma", 1.0) or 1.0)
            blurred = cv2.GaussianBlur(processed, (0, 0), sigma)
            processed = cv2.addWeighted(processed, 1.0 + amount, blurred, -amount, 0)
            processed = np.clip(processed, 0, 255).astype(np.uint8)

        x_scale = processed.shape[1] / float(width) if width else 1.0
        y_scale = processed.shape[0] / float(height) if height else 1.0
        return processed, x_scale, y_scale

    def _scale_overlay(self, box, x_scale: float, y_scale: float):
        """Map overlay coordinates from a scaled detection image back to original."""
        x, y, w, h = box
        x = int(round(x / x_scale)) if x_scale else x
        y = int(round(y / y_scale)) if y_scale else y
        w = int(round(w / x_scale)) if x_scale else w
        h = int(round(h / y_scale)) if y_scale else h
        w = max(1, w)
        h = max(1, h)
        x = max(0, x)
        y = max(0, y)
        return (x, y, w, h)

    def _extract_black_key_regions(self, regions, region_height):
        """Convert a boolean region mask into black key overlays."""
        black_keys = []
        in_key = False
        start_x = 0

        for x in range(len(regions)):
            if regions[x] and not in_key:
                start_x = x
                in_key = True
            elif not regions[x] and in_key:
                width = x - start_x
                if (
                    self.params["black_min_width"]
                    < width
                    < self.params["black_max_width"]
                ):
                    padded_overlay = self._add_overlay_padding(
                        start_x, 0, width, region_height
                    )
                    black_keys.append(padded_overlay)
                in_key = False

        if in_key:
            width = len(regions) - start_x
            if self.params["black_min_width"] < width < self.params["black_max_width"]:
                padded_overlay = self._add_overlay_padding(start_x, 0, width, region_height)
                black_keys.append(padded_overlay)

        return black_keys

    def _compute_white_edge_profile(self, gray_img):
        """Compute a robust boundary profile across a horizontal band."""
        height, width = gray_img.shape
        method = (self.params.get("white_seam_method") or "edge").lower()

        response = None
        self._white_strip_start = None
        self._white_separator_runs = None

        if method == "dark_columns":
            # Prefer the full "white-only" strip when possible; this is less sensitive
            # to blur creating double edges and works well when separator lines become wide/gray.
            strip = gray_img
            strip_start = 0
            if self.params.get("white_auto_strip"):
                dark_thr = int(self.params.get("white_auto_strip_dark_threshold", 60) or 60)
                frac_thr = float(self.params.get("white_auto_strip_frac_threshold", 0.02) or 0.02)
                min_run = int(self.params.get("white_auto_strip_min_run", 6) or 6)

                # Compute dark-pixel fraction per row, then find the first row after which the
                # fraction stays near zero for a small run (indicating we're below black keys).
                dark_frac = np.mean(strip < dark_thr, axis=1)
                win = 5
                if dark_frac.size >= win:
                    kernel = np.ones(win, dtype=np.float32) / float(win)
                    smooth = np.convolve(dark_frac.astype(np.float32), kernel, mode="same")
                else:
                    smooth = dark_frac.astype(np.float32)

                cut = None
                for y in range(0, max(0, len(smooth) - min_run)):
                    if np.all(smooth[y : y + min_run] < frac_thr):
                        cut = y
                        break
                if cut is not None and cut > 0 and cut < height:
                    strip_start = int(cut)
                    strip = gray_img[strip_start:, :]

            self._white_strip_start = int(strip_start)

            exclude_ratio = float(self.params.get("white_strip_exclude_bottom_ratio", 0.0) or 0.0)
            exclude_ratio = max(0.0, min(0.9, exclude_ratio))
            exclude_px = int(round(strip.shape[0] * exclude_ratio)) if strip.size else 0
            analysis_strip = strip[: max(1, strip.shape[0] - exclude_px), :] if strip.size else strip

            stat_mode = (self.params.get("white_dark_column_stat") or "median").lower()
            if stat_mode in {"p90", "p95"}:
                pct = int(stat_mode[1:])
                col_stat = np.percentile(analysis_strip.astype(np.uint8), pct, axis=0).astype(np.uint8)
                stat_label = stat_mode
            else:
                col_stat = np.median(analysis_strip.astype(np.uint8), axis=0).astype(np.uint8)
                stat_label = "median"
            # Return a "profile" where lower values indicate separators; edge finder will
            # interpret this based on method.
            edge_profile = col_stat.astype(np.float32)

            debug_dir = self.params.get("debug_save_preprocess_dir")
            if debug_dir:
                try:
                    base = Path(str(debug_dir))
                    base.mkdir(parents=True, exist_ok=True)
                    tag = str(self.params.get("debug_save_tag", ""))
                    safe_tag = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in (tag or ""))
                    prefix = safe_tag + "_" if safe_tag else ""

                    cv2.imwrite(str(base / f"{prefix}white_strip_gray_y{strip_start}.png"), strip)
                    if analysis_strip is not strip:
                        cv2.imwrite(
                            str(base / f"{prefix}white_strip_gray_analysis_excl_bottom_{exclude_px}px.png"),
                            analysis_strip,
                        )
                    col_img = np.repeat(col_stat.reshape(1, -1), 60, axis=0)
                    cv2.imwrite(str(base / f"{prefix}white_col_{stat_label}.png"), col_img)

                    # Also save a quick-look mask for separator columns based on the configured threshold mode.
                    thr = self.params.get("white_dark_column_threshold", "otsu")
                    thr_mode = thr.lower() if isinstance(thr, str) else None
                    if thr is None or thr_mode == "otsu":
                        _t, mask = cv2.threshold(
                            col_stat.reshape(1, -1),
                            0,
                            255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
                        )
                        mask_img = np.repeat(mask.astype(np.uint8), 60, axis=0)
                        cv2.imwrite(str(base / f"{prefix}white_separator_mask_otsu.png"), mask_img)
                    elif thr_mode == "relative":
                        white_p = int(self.params.get("white_dark_column_white_percentile", 90) or 90)
                        dark_p = int(self.params.get("white_dark_column_dark_percentile", 20) or 20)
                        ratio = float(self.params.get("white_dark_column_relative_ratio", 0.5) or 0.5)
                        white_level = float(np.percentile(col_stat, white_p))
                        dark_level = float(np.percentile(col_stat, dark_p))
                        if white_level < dark_level:
                            white_level, dark_level = dark_level, white_level
                        chosen_threshold = int(round(white_level - ratio * (white_level - dark_level)))
                        chosen_threshold = max(0, min(255, chosen_threshold))
                        mask = (col_stat < chosen_threshold).astype(np.uint8) * 255
                        mask_img = np.repeat(mask.reshape(1, -1), 60, axis=0)
                        cv2.imwrite(str(base / f"{prefix}white_separator_mask_relative_t{chosen_threshold}.png"), mask_img)
                    else:
                        chosen_threshold = int(thr)
                        mask = (col_stat < chosen_threshold).astype(np.uint8) * 255
                        mask_img = np.repeat(mask.reshape(1, -1), 60, axis=0)
                        cv2.imwrite(str(base / f"{prefix}white_separator_mask_t{chosen_threshold}.png"), mask_img)
                except Exception as e:
                    self.logger.warning("Failed saving dark-column debug images: %s", e, exc_info=True)

            return edge_profile

        band_ratio = self.params["white_edge_band_ratio"]
        center_ratio = self.params["white_bottom_ratio"]
        half_band = band_ratio / 2.0

        y_start = int(max(0, (center_ratio - half_band) * height))
        y_end = int(min(height, (center_ratio + half_band) * height))
        if y_end <= y_start:
            y_start = max(0, int(center_ratio * height) - 1)
            y_end = min(height, y_start + 2)

        band = gray_img[y_start:y_end, :]
        if band.shape[0] == 0:
            band = gray_img[int(height * center_ratio):int(height * center_ratio) + 1, :]

        band_blur = cv2.GaussianBlur(band, (3, 5), 0)

        if method == "blackhat":
            k = int(self.params.get("white_blackhat_kernel_width", 15) or 15)
            if k < 3:
                k = 3
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
            response = cv2.morphologyEx(band_blur, cv2.MORPH_BLACKHAT, kernel)
            edge_profile = np.mean(response.astype(np.float32), axis=0)
        else:
            grad_x = cv2.Sobel(band_blur, cv2.CV_64F, 1, 0, ksize=3)
            edge_profile = np.mean(np.abs(grad_x), axis=0)

        smooth_kernel = int(self.params["white_edge_smooth_kernel"])      
        if smooth_kernel % 2 == 0:
            smooth_kernel += 1
        if smooth_kernel > 1:
            edge_profile = cv2.GaussianBlur(
                edge_profile.reshape(1, -1), (smooth_kernel, 1), 0        
            ).flatten()

        debug_dir = self.params.get("debug_save_preprocess_dir")
        if debug_dir:
            try:
                base = Path(str(debug_dir))
                base.mkdir(parents=True, exist_ok=True)
                tag = str(self.params.get("debug_save_tag", ""))
                safe_tag = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in (tag or ""))
                prefix = safe_tag + "_" if safe_tag else ""

                cv2.imwrite(str(base / f"{prefix}white_band_gray.png"), band)
                if response is not None:
                    cv2.imwrite(
                        str(base / f"{prefix}white_band_blackhat_k{int(self.params.get('white_blackhat_kernel_width', 15) or 15)}.png"),
                        response,
                    )

                # Save a simple visualization of the 1D profile as an image for quick inspection.
                prof = edge_profile.astype(np.float32)
                pmin = float(np.min(prof)) if prof.size else 0.0
                pmax = float(np.max(prof)) if prof.size else 0.0
                if pmax > pmin:
                    norm = (prof - pmin) / (pmax - pmin)
                else:
                    norm = np.zeros_like(prof)
                profile_img = (norm * 255.0).clip(0, 255).astype(np.uint8)
                profile_img = np.repeat(profile_img.reshape(1, -1), 60, axis=0)
                cv2.imwrite(str(base / f"{prefix}white_profile_{method}.png"), profile_img)
            except Exception as e:
                self.logger.warning("Failed saving white-profile debug images: %s", e, exc_info=True)

        return edge_profile

    def _find_white_key_edges_from_profile(self, edge_profile):
        """Detect likely vertical boundaries from an edge-strength profile."""
        edges = []

        method = (self.params.get("white_seam_method") or "edge").lower()
        if method == "dark_columns":
            col_med = edge_profile.astype(np.uint8)
            thr = self.params.get("white_dark_column_threshold", "otsu")
            thr_mode = thr.lower() if isinstance(thr, str) else None

            if thr is None or thr_mode == "otsu":
                _t, mask = cv2.threshold(
                    col_med.reshape(1, -1),
                    0,
                    255,
                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
                )
                separator_cols = mask.flatten() > 0
                chosen_threshold = None
            elif thr_mode == "relative":
                white_p = int(self.params.get("white_dark_column_white_percentile", 90) or 90)
                dark_p = int(self.params.get("white_dark_column_dark_percentile", 20) or 20)
                ratio = float(self.params.get("white_dark_column_relative_ratio", 0.5) or 0.5)

                white_level = float(np.percentile(col_med, white_p))
                dark_level = float(np.percentile(col_med, dark_p))
                if white_level < dark_level:
                    white_level, dark_level = dark_level, white_level

                chosen_threshold = int(round(white_level - ratio * (white_level - dark_level)))
                chosen_threshold = max(0, min(255, chosen_threshold))
                separator_cols = col_med < chosen_threshold
            else:
                chosen_threshold = int(thr)
                separator_cols = col_med < chosen_threshold

            min_w = int(self.params.get("white_separator_min_width", 2) or 2)
            runs = []
            in_seg = False
            start = 0
            for x, is_sep in enumerate(separator_cols):
                if is_sep and not in_seg:
                    in_seg = True
                    start = x
                elif (not is_sep) and in_seg:
                    end = x - 1
                    width = end - start + 1
                    if width >= min_w:
                        runs.append((start, end))
                    in_seg = False
            if in_seg:
                end = len(separator_cols) - 1
                width = end - start + 1
                if width >= min_w:
                    runs.append((start, end))

            self.logger.debug(
                "DEBUG: White key separator (dark_columns): found %d runs (threshold=%s)",
                len(runs),
                chosen_threshold if chosen_threshold is not None else "otsu",
            )
            self._white_separator_runs = runs
            return runs

        edge_threshold = np.std(edge_profile) * self.params["white_edge_std_factor"]

        for i in range(1, len(edge_profile) - 1):
            if (
                edge_profile[i] > edge_threshold
                and edge_profile[i] > edge_profile[i - 1]
                and edge_profile[i] > edge_profile[i + 1]
            ):
                edges.append(i)

        self.logger.debug("DEBUG: White key edge detection:")
        self.logger.debug(
            f"  Edge profile std: {np.std(edge_profile):.2f}, edge_threshold: {edge_threshold:.2f}"
        )
        self.logger.debug(f"  Found {len(edges)} edges: {edges[:10]}...")
        return edges

    def _suppress_dense_edges(self, edges, edge_profile, min_sep_px: int):
        """Suppress overly-dense edge candidates by keeping stronger peaks."""
        if min_sep_px <= 1:
            return edges

        strengths = {int(e): float(edge_profile[int(e)]) for e in edges if 0 <= int(e) < len(edge_profile)}
        ranked = sorted(strengths.items(), key=lambda kv: kv[1], reverse=True)

        kept = []
        for pos, _strength in ranked:
            if all(abs(pos - k) >= min_sep_px for k in kept):
                kept.append(pos)

        return sorted(kept)

    def _fill_missing_white_edges(self, edges):
        """Fill missing edges using a spacing prior."""
        if len(edges) < self.params["white_gap_fill_min_edges"]:
            return edges

        edges = sorted(edges)
        gaps = np.diff(edges)
        if gaps.size == 0:
            return edges

        median_gap = np.median(gaps)
        if median_gap <= 0:
            return edges

        max_ratio = self.params["white_gap_fill_max_ratio"]
        filled_edges = [edges[0]]

        for i, gap in enumerate(gaps):
            if gap > median_gap * max_ratio:
                missing_count = int(round(gap / median_gap)) - 1
                for j in range(1, missing_count + 1):
                    candidate = int(edges[i] + j * median_gap)
                    if candidate < edges[i + 1]:
                        filled_edges.append(candidate)
            filled_edges.append(edges[i + 1])

        return sorted(set(filled_edges))

    def _detect_white_keys_from_profile(
        self,
        gray_img,
        edge_profile,
        edge_profile_x_scale: float = 1.0,
        edge_profile_y_scale: float = 1.0,
    ):
        """Detect white keys using an externally computed edge profile."""
        height, width = gray_img.shape
        method = (self.params.get("white_seam_method") or "edge").lower()
        edges_scaled = self._find_white_key_edges_from_profile(edge_profile)

        if method == "dark_columns":
            if not edges_scaled:
                self.logger.info("White separator runs: none detected (dark_columns)")
                return []
            if not isinstance(edges_scaled[0], tuple):
                self.logger.info("White separator runs: unexpected edge format; falling back to edge logic")
            else:
                runs_scaled = edges_scaled
                if edge_profile_x_scale and edge_profile_x_scale != 1.0:
                    runs = [
                        (
                            int(round(s / edge_profile_x_scale)),
                            int(round(e / edge_profile_x_scale)),
                        )
                        for s, e in runs_scaled
                    ]
                else:
                    runs = list(runs_scaled)

                runs = [
                    (max(0, min(width - 1, s)), max(0, min(width - 1, e)))
                    for s, e in runs
                    if width > 0
                ]
                runs = [(s, e) for s, e in runs if e >= s]
                runs.sort(key=lambda r: r[0])

                min_key_width = int(self.params.get("white_min_width", 1) or 1)
                strip_start = self._white_strip_start
                if strip_start is not None and edge_profile_y_scale and edge_profile_y_scale != 1.0:
                    strip_start = int(round(strip_start / edge_profile_y_scale))

                use_strip_top = strip_start is not None and self.params.get("white_auto_strip")
                if use_strip_top:
                    trimmed_top = max(0, min(height - 1, int(strip_start)))
                    trimmed_height = max(1, height - trimmed_top)

                white_keys = []
                segments_passing = 0
                gaps = []

                cursor = 0
                for start, end in runs:
                    gap_start = cursor
                    gap_end = start - 1
                    if gap_end >= gap_start:
                        gaps.append(gap_end - gap_start + 1)
                        if (gap_end - gap_start + 1) >= min_key_width:
                            segments_passing += 1
                            key_width = gap_end - gap_start + 1
                            if use_strip_top:
                                padded_overlay = self._add_overlay_padding(
                                    gap_start, trimmed_top, key_width, trimmed_height
                                )
                            else:
                                initial_top = int(height * self.params["white_initial_top_ratio"])
                                initial_height = int(height * self.params["white_initial_height_ratio"])
                                trimmed_top, trimmed_height = self._trim_white_key_top(
                                    gray_img,
                                    gap_start,
                                    gap_end,
                                    initial_top,
                                    initial_height,
                                )
                                padded_overlay = self._add_overlay_padding(
                                    gap_start, trimmed_top, key_width, trimmed_height
                                )
                            white_keys.append(padded_overlay)
                    cursor = end + 1

                if cursor <= width - 1:
                    gap_start = cursor
                    gap_end = width - 1
                    gaps.append(gap_end - gap_start + 1)
                    if (gap_end - gap_start + 1) >= min_key_width:
                        segments_passing += 1
                        key_width = gap_end - gap_start + 1
                        if use_strip_top:
                            padded_overlay = self._add_overlay_padding(
                                gap_start, trimmed_top, key_width, trimmed_height
                            )
                        else:
                            initial_top = int(height * self.params["white_initial_top_ratio"])
                            initial_height = int(height * self.params["white_initial_height_ratio"])
                            trimmed_top, trimmed_height = self._trim_white_key_top(
                                gray_img,
                                gap_start,
                                gap_end,
                                initial_top,
                                initial_height,
                            )
                            padded_overlay = self._add_overlay_padding(
                                gap_start, trimmed_top, key_width, trimmed_height
                            )
                        white_keys.append(padded_overlay)

                if gaps:
                    self.logger.info(
                        "White separator runs: runs=%d gaps(min/med/max)=%.1f/%.1f/%.1f min_key_width=%d",
                        len(runs),
                        float(np.min(gaps)),
                        float(np.median(gaps)),
                        float(np.max(gaps)),
                        int(min_key_width),
                    )
                else:
                    self.logger.info(
                        "White separator runs: runs=%d gaps(none) min_key_width=%d",
                        len(runs),
                        int(min_key_width),
                    )
                self.logger.info(
                    "White key result: segments_total=%d segments_passing=%d keys=%d",
                    max(0, len(gaps)),
                    segments_passing,
                    len(white_keys),
                )
                return white_keys

        if edge_profile_x_scale and edge_profile_x_scale != 1.0:
            edges = [
                int(round(e / edge_profile_x_scale))
                for e in edges_scaled
            ]
        else:
            edges = edges_scaled

        edges = [max(0, min(width - 1, e)) for e in edges] if width else []
        edges = sorted(set(edges))

        estimated_white_keys = None
        expected_edges = None
        min_sep_px = None
        try:
            black_count = len(self.black_keys or [])
            if black_count >= 5 and width > 0:
                estimated_white_keys = max(1, int(round((black_count * 7) / 5)))
                expected_edges = estimated_white_keys + 1
                expected_white_width = width / float(estimated_white_keys)
                min_sep_px = max(2, int(round(expected_white_width * 0.5)))
        except Exception:
            pass

        boundary_pad = self.params["edge_boundary_padding_px"]
        if not edges or edges[0] > boundary_pad:
            edges.insert(0, 0)
        if not edges or edges[-1] < width - boundary_pad:
            edges.append(width - 1)

        # If we have far too many edges (common on blurry/noisy footage), suppress dense peaks.
        if expected_edges and len(edges) > int(expected_edges * 1.5):
            self.logger.info(
                "White key edge suppression: edges=%d expected~%d (black=%s) min_sep=%s",
                len(edges),
                expected_edges,
                len(self.black_keys or []),
                min_sep_px,
            )
            fixed_boundaries = {0, max(0, width - 1)}
            inner = [e for e in edges if e not in fixed_boundaries]
            inner_suppressed = self._suppress_dense_edges(
                inner,
                edge_profile if edge_profile_x_scale == 1.0 else cv2.resize(edge_profile.reshape(1, -1), (width, 1), interpolation=cv2.INTER_LINEAR).flatten(),
                min_sep_px or self.params["white_min_width"],
            )
            edges = sorted(set(list(fixed_boundaries) + inner_suppressed))
            self.logger.info("White key edge suppression result: edges=%d", len(edges))

        if self.params["white_gap_fill"]:
            # Only gap-fill if edges are too sparse; if they are already dense,
            # gap-filling will make the result worse (tiny segments that get filtered out).
            if expected_edges and len(edges) >= expected_edges:
                self.logger.info(
                    "White key edge gap fill skipped (edges=%d >= expected~%d)",
                    len(edges),
                    expected_edges,
                )
            else:
                before = len(edges)
                edges = self._fill_missing_white_edges(edges)
                self.logger.info("White key edge gap fill: edges %d -> %d", before, len(edges))

        white_keys = []
        min_key_width = self.params["white_min_width"]
        gaps = np.diff(edges) if len(edges) > 1 else np.array([])
        if gaps.size:
            self.logger.info(
                "White edge stats: edges=%d gaps(min/med/max)=%.1f/%.1f/%.1f min_key_width=%d",
                len(edges),
                float(np.min(gaps)),
                float(np.median(gaps)),
                float(np.max(gaps)),
                int(min_key_width),
            )
        else:
            self.logger.info(
                "White edge stats: edges=%d gaps(none) min_key_width=%d",
                len(edges),
                int(min_key_width),
            )
        segments_passing = 0

        for i in range(len(edges) - 1):
            start_x = edges[i]
            end_x = edges[i + 1]
            key_width = end_x - start_x

            if key_width > min_key_width:
                segments_passing += 1
                initial_top = int(height * self.params["white_initial_top_ratio"])
                initial_height = int(height * self.params["white_initial_height_ratio"])

                trimmed_top, trimmed_height = self._trim_white_key_top(
                    gray_img,
                    start_x,
                    end_x,
                    initial_top,
                    initial_height,
                )

                padded_overlay = self._add_overlay_padding(
                    start_x, trimmed_top, key_width, trimmed_height
                )
                white_keys.append(padded_overlay)

        self.logger.info(
            "White key result: segments_total=%d segments_passing=%d keys=%d",
            max(0, len(edges) - 1),
            segments_passing,
            len(white_keys),
        )
        return white_keys

    def _detect_black_keys(self, gray_img):
        """Detect black keys using column scanning or connected components."""
        height, width = gray_img.shape

        # Focus on upper portion where black keys are
        upper_ratio = self.params["black_upper_ratio"]
        upper_region = gray_img[:int(height * upper_ratio), :]

        detection_method = (self.params.get("black_detection_method") or "columns").lower()
        if detection_method == "components":
            return self._detect_black_keys_components(upper_region)

        threshold_method = self.params.get("black_threshold_method", "fixed")
        self.logger.debug("Black key detection: threshold_method=%s", threshold_method)
        if threshold_method == "adaptive":
            block_size = int(self.params["black_adaptive_block_size"])
            if block_size % 2 == 0:
                block_size += 1
            if block_size < 3:
                block_size = 3
            binary = cv2.adaptiveThreshold(
                upper_region,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                self.params["black_adaptive_c"],
            )
        elif threshold_method == "otsu" or self.params["black_threshold"] is None:
            otsu_val, binary = cv2.threshold(
                upper_region,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )
            self.logger.debug("Black key detection: Otsu threshold=%.2f", otsu_val)
        else:
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

        black_keys = self._extract_black_key_regions(black_regions, upper_region.shape[0])
        self.logger.info(
            "Black key stats: method=%s fixed_threshold=%s column_ratio=%.3f column_thr=%.1f columns_over=%d keys=%d",
            threshold_method,
            self.params.get("black_threshold", None),
            float(self.params["black_column_ratio"]),
            float(threshold),
            int(np.sum(black_regions)) if hasattr(np, "sum") else 0,
            len(black_keys),
        )

        if (
            self.params["black_edge_fallback"]
            and len(black_keys) < self.params["black_min_count_for_valid"]
        ):
            self.logger.info(
                "Black key edge fallback enabled: base_detected=%d (<%d), attempting edge-based fallback",
                len(black_keys),
                self.params["black_min_count_for_valid"],
            )
            edges = cv2.Canny(upper_region, 50, 150)
            edge_sums = np.sum(edges > 0, axis=0)
            edge_threshold = np.max(edge_sums) * self.params["black_edge_column_ratio"]
            edge_regions = edge_sums > edge_threshold
            edge_keys = self._extract_black_key_regions(edge_regions, upper_region.shape[0])
            if len(edge_keys) > len(black_keys):
                black_keys = edge_keys
                self.logger.info("Black key edge fallback improved count to %d", len(black_keys))
            else:
                self.logger.info(
                    "Black key edge fallback did not improve count (edge=%d, base=%d)",
                    len(edge_keys),
                    len(black_keys),
                )

        return black_keys

    def _detect_black_keys_components(self, upper_region):
        """Detect black keys using connected components on a thresholded upper band."""
        height, width = upper_region.shape
        threshold_method = self.params.get("black_threshold_method", "fixed")

        if threshold_method == "adaptive":
            block_size = int(self.params["black_adaptive_block_size"])
            if block_size % 2 == 0:
                block_size += 1
            if block_size < 3:
                block_size = 3
            binary = cv2.adaptiveThreshold(
                upper_region,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                self.params["black_adaptive_c"],
            )
        elif threshold_method == "otsu" or self.params["black_threshold"] is None:
            _, binary = cv2.threshold(
                upper_region,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )
        else:
            _, binary = cv2.threshold(
                upper_region,
                self.params["black_threshold"],
                255,
                cv2.THRESH_BINARY_INV,
            )

        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        min_w = int(self.params["black_min_width"])
        max_w = int(self.params["black_max_width"])
        min_h = max(1, int(height * float(self.params.get("black_min_height_ratio", 0.4))))
        max_h = max(1, int(height * float(self.params.get("black_max_height_ratio", 1.0))))
        min_area = max(1, int(width * height * float(self.params.get("black_min_area_ratio", 0.001))))

        black_keys = []
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            if w < min_w or w > max_w:
                continue
            if h < min_h or h > max_h:
                continue
            if area < min_area:
                continue

            padded_overlay = self._add_overlay_padding(x, 0, w, height)
            black_keys.append(padded_overlay)

        return black_keys
    
    def _detect_white_keys(self, gray_img):
        """Detect white keys by finding vertical separations"""
        edge_profile = self._compute_white_edge_profile(gray_img)
        return self._detect_white_keys_from_profile(
            gray_img,
            edge_profile=edge_profile,
            edge_profile_x_scale=1.0,
            edge_profile_y_scale=1.0,
        )
    
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
            raise ValueError(
                f"Must detect keys first (black={len(self.black_keys)}, white={len(self.white_keys)})"
            )
        
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
