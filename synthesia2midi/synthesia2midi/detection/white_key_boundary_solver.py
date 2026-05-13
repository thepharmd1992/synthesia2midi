"""Boundary and separator white-key solvers for the manual piano keyboard detector."""

import cv2
import numpy as np


class WhiteKeyBoundarySolverMixin:
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
