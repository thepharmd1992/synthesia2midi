"""D-lattice white-key span solver."""

import numpy as np


class WhiteKeyLatticeSolverMixin:
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
