"""Black-key detection stage for the manual piano keyboard detector."""

import cv2
import numpy as np


class BlackKeyDetectionMixin:
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
