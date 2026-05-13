"""White-key geometry primitives for the manual piano keyboard detector."""

import cv2
import numpy as np


class WhiteKeyGeometryMixin:
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
