"""Manual keyboard fit session model."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Set, Tuple

import numpy as np
from synthesia2midi.app_config import NOTE_NAMES_SHARP


WHITE_NOTE_NAMES = {name for name in NOTE_NAMES_SHARP if "♯" not in name and "♭" not in name}
HORIZONTAL_SAFE_INSET_FRACTION = 0.10
VERTICAL_SAFE_INSET_FRACTION = 0.15


def keyboard_box_background_warnings(
    frame_rgb: np.ndarray | None,
    keyboard_box: Sequence[float] | KeyboardBox | None,
) -> list[str]:
    if frame_rgb is None or keyboard_box is None:
        return []
    box = _coerce_keyboard_box(keyboard_box)
    if box is None:
        return []
    height, width = frame_rgb.shape[:2]
    left = max(0, min(int(round(box.left)), width - 1))
    right = max(left + 1, min(int(round(box.right)), width))
    top = max(0, min(int(round(box.top)), height - 1))
    bottom = max(top + 1, min(int(round(box.bottom)), height))
    box_width = right - left
    box_height = bottom - top
    if box_width < 20 or box_height < 20:
        return []

    lower_top = top + int(box_height * 0.70)
    band_width = max(2, int(box_width * 0.04))
    lower_region = frame_rgb[lower_top:bottom, left:right, :3].astype(float)
    if lower_region.size == 0:
        return []

    center_left = max(band_width, int(box_width * 0.25))
    center_right = min(box_width - band_width, int(box_width * 0.75))
    if center_right <= center_left:
        return []

    center_luma = _luminance(lower_region[:, center_left:center_right, :])
    reference = float(np.median(center_luma))
    if reference <= 1.0:
        return []

    warnings: list[str] = []
    side_bands = {
        "left": lower_region[:, :band_width, :],
        "right": lower_region[:, box_width - band_width:, :],
    }
    for side, band in side_bands.items():
        band_luma = _luminance(band)
        dark_threshold = reference * 0.55
        dark_fraction = float(np.mean(band_luma < dark_threshold))
        band_median = float(np.median(band_luma))
        if dark_fraction >= 0.85 and band_median <= reference - 40.0:
            warnings.append(
                f"The keyboard box lower {side} edge looks like background, not white keys."
            )
    return warnings


@dataclass
class ManualFitParams:
    group_dx: float = 0.0
    group_dy: float = 0.0
    keyboard_width_delta: float = 0.0
    keyboard_top_delta: float = 0.0
    left_edge_drift: float = 0.0
    right_edge_drift: float = 0.0
    white_width_delta: float = 0.0
    black_width_delta: float = 0.0
    left_slant_delta: float = 0.0
    right_slant_delta: float = 0.0


@dataclass
class LocalFitParams:
    x_delta: float = 0.0
    y_delta: float = 0.0
    spread_delta: float = 0.0
    width_delta: float = 0.0
    slant_delta: float = 0.0


CONTROL_PARAM_NAMES = (
    "keyboard_width_delta",
    "keyboard_top_delta",
    "left_edge_drift",
    "right_edge_drift",
    "white_width_delta",
    "black_width_delta",
    "left_slant_delta",
    "right_slant_delta",
)

GROUP_FIT_SCOPES = {"all", "white", "black"}


@dataclass(frozen=True)
class OverlayGeometry:
    x: float
    y: float
    width: float
    height: float
    rotation_degrees: float = 0.0


@dataclass(frozen=True)
class DetectionRegion:
    top: float
    bottom: float


@dataclass(frozen=True)
class KeyboardBox:
    left: float
    top: float
    right: float
    bottom: float


def _coerce_keyboard_box(value: Sequence[float] | KeyboardBox | None) -> KeyboardBox | None:
    if value is None:
        return None
    if isinstance(value, KeyboardBox):
        left, top, right, bottom = value.left, value.top, value.right, value.bottom
    else:
        if len(value) != 4:
            return None
        left, top, right, bottom = (float(part) for part in value)
    if right <= left or bottom <= top:
        return None
    return KeyboardBox(float(left), float(top), float(right), float(bottom))


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[..., 0]) + (0.7152 * rgb[..., 1]) + (0.0722 * rgb[..., 2])


@dataclass
class OverlayOverride:
    x_delta: float = 0.0
    y_delta: float = 0.0
    width_delta: float = 0.0
    height_delta: float = 0.0


@dataclass
class LocalFitAdjustment:
    key_ids: Tuple[int, ...]
    params: LocalFitParams


class ManualKeyboardFitSession:
    def __init__(self, app_state):
        self.app_state = app_state
        self._group_params: Dict[str, ManualFitParams] = {
            "all": ManualFitParams(),
            "white": ManualFitParams(),
            "black": ManualFitParams(),
        }
        self._active_group_scope = "all"
        self.params = self._group_params["all"]
        self._previous_unsaved_changes = bool(app_state.unsaved_changes)
        self._previous_octave_transpose = int(getattr(app_state.midi, "octave_transpose", 0))
        self._cancel_baseline: Dict[int, OverlayGeometry] = {
            overlay.key_id: OverlayGeometry(
                float(overlay.x),
                float(overlay.y),
                float(overlay.width),
                float(overlay.height),
                float(getattr(overlay, "rotation_degrees", 0.0) or 0.0),
            )
            for overlay in app_state.overlays
        }
        self._baseline: Dict[int, OverlayGeometry] = dict(self._cancel_baseline)
        self._local_fits: list[LocalFitAdjustment] = []
        self._active_local_fit_index: int | None = None
        self._overrides: Dict[int, OverlayOverride] = {}
        self._bounds = self._calculate_bounds(self._baseline.values())
        self._center_bounds = self._calculate_center_bounds(self._baseline.values())
        self._default_regions = self._calculate_default_regions()
        self._custom_regions: Dict[str, DetectionRegion] = {}
        self._keyboard_box: KeyboardBox | None = _coerce_keyboard_box(
            getattr(app_state.calibration, "manual_keyboard_box", None)
        )
        self._setup_keyboard_box: KeyboardBox | None = None
        self._setup_black_bottom: float | None = None
        self._setup_white_start: float | None = None

    def update_params(self, params: ManualFitParams) -> None:
        self._group_params[self._active_group_scope] = params
        self.params = params
        self.apply_preview()

    def update_control_params(self, params: ManualFitParams) -> None:
        active_params = self.active_group_params()
        for name in CONTROL_PARAM_NAMES:
            setattr(active_params, name, getattr(params, name))
        self.apply_preview()

    def set_param(self, name: str, value: float) -> None:
        active_params = self.active_group_params()
        if not hasattr(active_params, name):
            raise AttributeError(f"Unknown manual fit parameter: {name}")
        setattr(active_params, name, float(value))
        self.apply_preview()

    def set_group_scope(self, scope: str) -> None:
        if scope not in GROUP_FIT_SCOPES:
            raise ValueError(f"Unknown manual fit group scope: {scope}")
        self._active_group_scope = scope
        self.params = self._group_params[scope]

    def active_group_scope(self) -> str:
        return self._active_group_scope

    def active_group_params(self) -> ManualFitParams:
        return self._group_params[self._active_group_scope]

    def translate_group(self, dx: float, dy: float) -> None:
        active_params = self.active_group_params()
        active_params.group_dx += float(dx)
        active_params.group_dy += float(dy)
        self.apply_preview()

    def reset_position(self) -> None:
        active_params = self.active_group_params()
        active_params.group_dx = 0.0
        active_params.group_dy = 0.0
        self.apply_preview()

    def set_detection_region(self, key_type: str, top: float, bottom: float) -> None:
        if key_type not in {"white", "black"}:
            raise ValueError(f"Unknown detection region type: {key_type}")
        all_params = self._group_params["all"]
        region_top = min(float(top), float(bottom)) - all_params.group_dy
        region_bottom = max(float(top), float(bottom)) - all_params.group_dy
        self._custom_regions[key_type] = DetectionRegion(region_top, region_bottom)
        self.apply_preview()

    def detection_region_guides(self) -> Dict[str, object]:
        guides: Dict[str, object] = {
            "white": self._region_for_type("white"),
            "black": self._region_for_type("black"),
        }
        if self._keyboard_box is not None:
            guides["keyboard_box"] = self._keyboard_box
        return guides

    def keyboard_box(self) -> KeyboardBox | None:
        return self._keyboard_box

    def set_keyboard_box(self, left: float, top: float, right: float, bottom: float) -> None:
        box = self._normalize_keyboard_box(left, top, right, bottom)
        self._keyboard_box = box
        self._persist_keyboard_box()
        self.apply_preview()

    def select_local_cluster(
        self,
        left: float,
        top: float,
        right: float,
        bottom: float,
        *,
        key_filter: str = "black",
    ) -> Set[int]:
        box_left = min(float(left), float(right))
        box_right = max(float(left), float(right))
        box_top = min(float(top), float(bottom))
        box_bottom = max(float(top), float(bottom))
        selected = {
            overlay.key_id
            for overlay in self.app_state.overlays
            if self._overlay_matches_local_filter(overlay, key_filter)
            and box_left <= float(overlay.x) + (float(overlay.width) / 2.0) <= box_right
            and box_top <= float(overlay.y) + (float(overlay.height) / 2.0) <= box_bottom
        }
        if not selected:
            self._active_local_fit_index = None
            return set()

        key_ids = tuple(sorted(selected))
        for index, local_fit in enumerate(self._local_fits):
            if local_fit.key_ids == key_ids:
                self._active_local_fit_index = index
                return selected

        self._local_fits.append(LocalFitAdjustment(key_ids, LocalFitParams()))
        self._active_local_fit_index = len(self._local_fits) - 1
        return selected

    def update_active_local_params(self, params: LocalFitParams) -> None:
        local_fit = self._active_local_fit()
        if local_fit is None:
            return
        local_fit.params = params
        self.apply_preview()

    def reset_active_local_fit(self) -> None:
        local_fit = self._active_local_fit()
        if local_fit is None:
            return
        local_fit.params = LocalFitParams()
        self.apply_preview()

    def translate_active_local_fit(self, dx: float, dy: float) -> None:
        local_fit = self._active_local_fit()
        if local_fit is None:
            return
        local_fit.params.x_delta += float(dx)
        local_fit.params.y_delta += float(dy)
        self.apply_preview()

    def active_local_key_ids(self) -> Set[int]:
        local_fit = self._active_local_fit()
        if local_fit is None:
            return set()
        return set(local_fit.key_ids)

    def active_local_params(self) -> LocalFitParams:
        local_fit = self._active_local_fit()
        if local_fit is None:
            return LocalFitParams()
        return local_fit.params

    def set_setup_keyboard_box(self, left: float, top: float, right: float, bottom: float) -> None:
        self._setup_keyboard_box = self._normalize_keyboard_box(left, top, right, bottom)
        self._setup_black_bottom = self.default_setup_black_bottom()
        self._setup_white_start = self.default_setup_white_start()

    def set_setup_black_bottom(self, y: float) -> None:
        box = self._require_setup_keyboard_box()
        self._setup_black_bottom = self._clamp(float(y), box.top + 1.0, box.bottom - 1.0)
        self._setup_white_start = self.default_setup_white_start()

    def set_setup_white_start(self, y: float) -> None:
        box = self._require_setup_keyboard_box()
        black_bottom = self._setup_black_bottom_or_default()
        self._setup_white_start = self._clamp(float(y), black_bottom, box.bottom - 1.0)

    def default_setup_black_bottom(self) -> float:
        box = self._require_setup_keyboard_box()
        return box.top + ((box.bottom - box.top) * 0.45)

    def default_setup_white_start(self) -> float:
        box = self._require_setup_keyboard_box()
        black_bottom = self._setup_black_bottom_or_default()
        return black_bottom + ((box.bottom - black_bottom) * 0.20)

    def setup_guides(self) -> Dict[str, object]:
        guides: Dict[str, object] = {}
        if self._setup_keyboard_box is not None:
            guides["keyboard_box"] = self._setup_keyboard_box
            guides["black"] = DetectionRegion(
                self._setup_keyboard_box.top,
                self._setup_black_bottom_or_default(),
            )
            guides["white"] = DetectionRegion(
                self._setup_white_start_or_default(),
                self._setup_keyboard_box.bottom,
            )
        return guides

    def setup_guides_for_step(self, step: str) -> Dict[str, object]:
        guides = self.setup_guides()
        if step == "keyboard_box_edit":
            if self._keyboard_box is not None:
                return {"keyboard_box": self._keyboard_box}
            return {}
        if step == "keyboard_box":
            return {
                key: value
                for key, value in guides.items()
                if key == "keyboard_box"
            }
        if step == "black_bottom":
            return {
                key: value
                for key, value in guides.items()
                if key in {"keyboard_box", "black"}
            }
        if step == "white_start":
            return {
                key: value
                for key, value in guides.items()
                if key in {"keyboard_box", "black", "white"}
            }
        return guides

    def finalize_setup_geometry(self) -> None:
        box = self._require_setup_keyboard_box()
        black_bottom = self._setup_black_bottom_or_default()
        white_start = self._setup_white_start_or_default()
        self._group_params = {
            "all": ManualFitParams(),
            "white": ManualFitParams(),
            "black": ManualFitParams(),
        }
        self._active_group_scope = "all"
        self.params = self._group_params["all"]
        self._overrides.clear()
        self._custom_regions.clear()
        self._local_fits.clear()
        self._active_local_fit_index = None
        self._baseline = self._generate_baseline_from_setup_box(box)
        self._bounds = self._calculate_bounds(self._baseline.values())
        self._center_bounds = self._calculate_center_bounds(self._baseline.values())
        self._keyboard_box = box
        self._persist_keyboard_box()
        self._default_regions = {
            "black": DetectionRegion(box.top, black_bottom),
            "white": DetectionRegion(white_start, box.bottom),
        }
        self.apply_preview()

    def set_octave_transpose(self, value: int) -> None:
        self.app_state.midi.octave_transpose = int(value)
        self.app_state.unsaved_changes = True

    def current_octave_transpose(self) -> int:
        return int(getattr(self.app_state.midi, "octave_transpose", 0))

    def move_single_overlay_by_index(self, overlay_index: int, new_x: float, new_y: float) -> bool:
        if not 0 <= overlay_index < len(self.app_state.overlays):
            return False

        overlay = self.app_state.overlays[overlay_index]
        base = self._transformed_rect_without_override(overlay.key_id)
        if base is None:
            return False

        self._overrides[overlay.key_id] = OverlayOverride(
            x_delta=float(new_x) - base.x,
            y_delta=float(new_y) - base.y,
            width_delta=float(overlay.width) - base.width,
            height_delta=float(overlay.height) - base.height,
        )
        self.apply_preview()
        return True

    def resize_single_overlay_by_index(
        self,
        overlay_index: int,
        new_x: float,
        new_y: float,
        new_width: float,
        new_height: float,
    ) -> bool:
        if not 0 <= overlay_index < len(self.app_state.overlays):
            return False

        overlay = self.app_state.overlays[overlay_index]
        base = self._transformed_rect_without_override(overlay.key_id)
        if base is None:
            return False

        self._overrides[overlay.key_id] = OverlayOverride(
            x_delta=float(new_x) - base.x,
            y_delta=float(new_y) - base.y,
            width_delta=float(new_width) - base.width,
            height_delta=float(new_height) - base.height,
        )
        self.apply_preview()
        return True

    def clear_override_for_key_id(self, key_id: int) -> bool:
        if key_id not in self._overrides:
            return False
        del self._overrides[key_id]
        self.apply_preview()
        return True

    def clear_selected_override(self) -> bool:
        selected_key_id = getattr(self.app_state.ui, "selected_overlay_id", None)
        if selected_key_id is None:
            return False
        return self.clear_override_for_key_id(int(selected_key_id))

    def reset_all(self) -> None:
        self._group_params = {
            "all": ManualFitParams(),
            "white": ManualFitParams(),
            "black": ManualFitParams(),
        }
        self.params = self._group_params[self._active_group_scope]
        self._overrides.clear()
        self._local_fits.clear()
        self._active_local_fit_index = None
        self._custom_regions.clear()
        self.app_state.midi.octave_transpose = self._previous_octave_transpose
        self.apply_preview()

    def cancel(self) -> None:
        self._restore_baseline()
        self.app_state.midi.octave_transpose = self._previous_octave_transpose
        self.app_state.unsaved_changes = self._previous_unsaved_changes

    def apply(self) -> None:
        self.app_state.unsaved_changes = True

    def overridden_key_ids(self) -> Set[int]:
        return set(self._overrides)

    def apply_preview(self) -> None:
        for overlay in self.app_state.overlays:
            rect = self._transformed_rect_without_override(overlay.key_id)
            if rect is None:
                continue
            override = self._overrides.get(overlay.key_id)
            if override is not None:
                rect = OverlayGeometry(
                    rect.x + override.x_delta,
                    rect.y + override.y_delta,
                    max(1.0, rect.width + override.width_delta),
                    max(1.0, rect.height + override.height_delta),
                    rect.rotation_degrees,
                )
            rect = self._constrain_rect_to_keyboard_box(rect)
            overlay.x = rect.x
            overlay.y = rect.y
            overlay.width = rect.width
            overlay.height = rect.height
            overlay.rotation_degrees = rect.rotation_degrees
        self.app_state.unsaved_changes = True

    def _restore_baseline(self) -> None:
        for overlay in self.app_state.overlays:
            baseline = self._cancel_baseline.get(overlay.key_id)
            if baseline is None:
                continue
            overlay.x = baseline.x
            overlay.y = baseline.y
            overlay.width = baseline.width
            overlay.height = baseline.height
            overlay.rotation_degrees = baseline.rotation_degrees

    def _transformed_rect_without_override(self, key_id: int) -> OverlayGeometry | None:
        rect = self._global_transformed_rect(key_id)
        if rect is None:
            return None
        return self._apply_local_fits(key_id, rect)

    def _global_transformed_rect(self, key_id: int) -> OverlayGeometry | None:
        rect = self._all_group_transformed_rect(key_id)
        if rect is None:
            return None

        scope = "white" if self._is_white_key_id(key_id) else "black"
        params = self._group_params[scope]
        if self._params_are_neutral(params):
            return rect

        bounds, center_bounds = self._scope_bounds_after_all_group(scope)
        return self._scoped_group_transformed_rect(key_id, rect, params, bounds, center_bounds)

    def _all_group_transformed_rect(self, key_id: int) -> OverlayGeometry | None:
        baseline = self._baseline.get(key_id)
        if baseline is None:
            return None

        params = self._group_params["all"]
        span = max(1.0, self._bounds[2])
        target_span = max(1.0, span + params.keyboard_width_delta)
        scale = target_span / span
        left, _right, _span, center = self._bounds

        baseline_center_x = baseline.x + baseline.width / 2
        norm = (baseline_center_x - left) / span
        scaled_width = max(1.0, baseline.width * scale)
        scaled_center_x = center + ((baseline_center_x - center) * scale)
        left_edge_weight, right_edge_weight = self._edge_drift_weights(baseline_center_x)
        edge_shift = (params.left_edge_drift * left_edge_weight) + (
            params.right_edge_drift * right_edge_weight
        )
        left_slant_weight = max(0.0, min(1.0, (0.5 - norm) / 0.5))
        right_slant_weight = max(0.0, min(1.0, (norm - 0.5) / 0.5))
        rotation_degrees = self._clamp(
            baseline.rotation_degrees
            + (params.left_slant_delta * left_slant_weight)
            + (params.right_slant_delta * right_slant_weight),
            -45.0,
            45.0,
        )

        x = scaled_center_x - scaled_width / 2 + params.group_dx + edge_shift
        width = scaled_width
        top = baseline.y + params.group_dy
        bottom = baseline.y + baseline.height + params.group_dy

        if self._is_white_key_id(key_id):
            top, bottom = self._safe_region_bounds("white")
            width = max(1.0, width + params.white_width_delta)
            x = (scaled_center_x + params.group_dx + edge_shift) - width / 2
        else:
            top, bottom = self._safe_region_bounds("black")
            width = max(1.0, width + params.black_width_delta)
            x = (scaled_center_x + params.group_dx + edge_shift) - width / 2

        x, width = self._apply_fractional_x_inset(x, width)

        if bottom < top + 1.0:
            bottom = top + 1.0

        return OverlayGeometry(x, top, width, bottom - top, rotation_degrees)

    def _scoped_group_transformed_rect(
        self,
        key_id: int,
        rect: OverlayGeometry,
        params: ManualFitParams,
        bounds: Tuple[float, float, float, float],
        center_bounds: Tuple[float, float, float],
    ) -> OverlayGeometry:
        span = max(1.0, bounds[2])
        target_span = max(1.0, span + params.keyboard_width_delta)
        scale = target_span / span
        left, _right, _span, center = bounds
        rect_center_x = rect.x + rect.width / 2.0
        norm = (rect_center_x - left) / span
        scaled_width = max(1.0, rect.width * scale)
        scaled_center_x = center + ((rect_center_x - center) * scale)
        left_edge_weight, right_edge_weight = self._edge_drift_weights_from_bounds(
            rect_center_x,
            center_bounds,
        )
        edge_shift = (params.left_edge_drift * left_edge_weight) + (
            params.right_edge_drift * right_edge_weight
        )
        left_slant_weight = max(0.0, min(1.0, (0.5 - norm) / 0.5))
        right_slant_weight = max(0.0, min(1.0, (norm - 0.5) / 0.5))
        rotation_degrees = self._clamp(
            rect.rotation_degrees
            + (params.left_slant_delta * left_slant_weight)
            + (params.right_slant_delta * right_slant_weight),
            -45.0,
            45.0,
        )

        if self._is_white_key_id(key_id):
            width = max(1.0, scaled_width + params.white_width_delta)
        else:
            width = max(1.0, scaled_width + params.black_width_delta)
        x = (scaled_center_x + params.group_dx + edge_shift) - (width / 2.0)
        top = rect.y + params.group_dy + params.keyboard_top_delta
        bottom = rect.y + rect.height + params.group_dy
        if bottom < top + 1.0:
            bottom = top + 1.0
        return OverlayGeometry(x, top, width, bottom - top, rotation_degrees)

    def _apply_local_fits(self, key_id: int, rect: OverlayGeometry) -> OverlayGeometry:
        adjusted = rect
        for local_fit in self._local_fits:
            if key_id not in local_fit.key_ids:
                continue
            adjusted = self._apply_local_fit(local_fit, key_id, adjusted)
        return adjusted

    def _apply_local_fit(
        self,
        local_fit: LocalFitAdjustment,
        key_id: int,
        rect: OverlayGeometry,
    ) -> OverlayGeometry:
        cluster_rects = [
            global_rect
            for member_key_id in local_fit.key_ids
            if (global_rect := self._global_transformed_rect(member_key_id)) is not None
        ]
        if not cluster_rects:
            return rect

        centers = [geometry.x + geometry.width / 2.0 for geometry in cluster_rects]
        left_center = min(centers)
        right_center = max(centers)
        span = max(1.0, right_center - left_center)
        target_span = max(1.0, span + local_fit.params.spread_delta)
        scale = target_span / span
        cluster_center = left_center + (span / 2.0)
        rect_center = rect.x + rect.width / 2.0
        scaled_center = cluster_center + ((rect_center - cluster_center) * scale)
        width = max(1.0, rect.width + local_fit.params.width_delta)
        return OverlayGeometry(
            scaled_center - (width / 2.0) + local_fit.params.x_delta,
            rect.y + local_fit.params.y_delta,
            width,
            rect.height,
            self._clamp(rect.rotation_degrees + local_fit.params.slant_delta, -45.0, 45.0),
        )

    def _active_local_fit(self) -> LocalFitAdjustment | None:
        if self._active_local_fit_index is None:
            return None
        if not 0 <= self._active_local_fit_index < len(self._local_fits):
            return None
        return self._local_fits[self._active_local_fit_index]

    @staticmethod
    def _overlay_matches_local_filter(overlay, key_filter: str) -> bool:
        normalized = key_filter.lower()
        is_white = overlay.note_name_in_octave in WHITE_NOTE_NAMES
        if normalized == "all":
            return True
        if normalized == "white":
            return is_white
        return not is_white

    def _safe_region_bounds(self, key_type: str) -> Tuple[float, float]:
        region = self._region_for_type(key_type)
        all_params = self._group_params["all"]
        top = region.top + all_params.group_dy + all_params.keyboard_top_delta
        bottom = region.bottom + all_params.group_dy
        region_height = max(1.0, bottom - top)
        vertical_inset = region_height * VERTICAL_SAFE_INSET_FRACTION
        return top + vertical_inset, bottom - vertical_inset

    def _region_for_type(self, key_type: str) -> DetectionRegion:
        return self._custom_regions.get(key_type) or self._default_regions[key_type]

    @staticmethod
    def _apply_fractional_x_inset(x: float, width: float) -> Tuple[float, float]:
        inset = width * HORIZONTAL_SAFE_INSET_FRACTION
        safe_inset = max(0.0, min(inset, (width - 1.0) / 2.0))
        return x + safe_inset, max(1.0, width - (2.0 * safe_inset))

    def _calculate_default_regions(self) -> Dict[str, DetectionRegion]:
        white_geometries = []
        black_geometries = []
        for overlay in self.app_state.overlays:
            geometry = self._baseline.get(overlay.key_id)
            if geometry is None:
                continue
            if overlay.note_name_in_octave in WHITE_NOTE_NAMES:
                white_geometries.append(geometry)
            else:
                black_geometries.append(geometry)
        return {
            "white": self._calculate_vertical_region(white_geometries or self._baseline.values()),
            "black": self._calculate_vertical_region(black_geometries or self._baseline.values()),
        }

    @staticmethod
    def _calculate_vertical_region(geometries: Iterable[OverlayGeometry]) -> DetectionRegion:
        geometries = list(geometries)
        if not geometries:
            return DetectionRegion(0.0, 1.0)
        top = min(geometry.y for geometry in geometries)
        bottom = max(geometry.y + geometry.height for geometry in geometries)
        return DetectionRegion(top, max(top + 1.0, bottom))

    @staticmethod
    def _calculate_bounds(geometries: Iterable[OverlayGeometry]) -> Tuple[float, float, float, float]:
        geometries = list(geometries)
        if not geometries:
            return 0.0, 1.0, 1.0, 0.5
        left = min(geometry.x for geometry in geometries)
        right = max(geometry.x + geometry.width for geometry in geometries)
        span = max(1.0, right - left)
        center = left + span / 2
        return left, right, span, center

    @staticmethod
    def _calculate_center_bounds(geometries: Iterable[OverlayGeometry]) -> Tuple[float, float, float]:
        centers = [geometry.x + geometry.width / 2.0 for geometry in geometries]
        if not centers:
            return 0.0, 1.0, 0.5
        left_center = min(centers)
        right_center = max(centers)
        midpoint = left_center + ((right_center - left_center) / 2.0)
        return left_center, right_center, midpoint

    def _edge_drift_weights(self, baseline_center_x: float) -> Tuple[float, float]:
        return self._edge_drift_weights_from_bounds(baseline_center_x, self._center_bounds)

    def _edge_drift_weights_from_bounds(
        self,
        baseline_center_x: float,
        center_bounds: Tuple[float, float, float],
    ) -> Tuple[float, float]:
        left_center, right_center, midpoint = center_bounds
        if right_center <= left_center:
            return 0.0, 0.0

        left_weight = 0.0
        if baseline_center_x < midpoint:
            left_weight = (midpoint - baseline_center_x) / max(1.0, midpoint - left_center)

        right_weight = 0.0
        if baseline_center_x > midpoint:
            right_weight = (baseline_center_x - midpoint) / max(1.0, right_center - midpoint)

        return (
            max(0.0, min(1.0, left_weight)),
            max(0.0, min(1.0, right_weight)),
        )

    def _scope_bounds_after_all_group(
        self,
        scope: str,
    ) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float]]:
        rects = [
            rect
            for key_id in self._baseline
            if self._key_matches_scope(key_id, scope)
            and (rect := self._all_group_transformed_rect(key_id)) is not None
        ]
        return self._calculate_bounds(rects), self._calculate_center_bounds(rects)

    def _key_matches_scope(self, key_id: int, scope: str) -> bool:
        if scope == "all":
            return True
        is_white = self._is_white_key_id(key_id)
        if scope == "white":
            return is_white
        if scope == "black":
            return not is_white
        return False

    def _is_white_key_id(self, key_id: int) -> bool:
        overlay = next(
            (candidate for candidate in self.app_state.overlays if candidate.key_id == key_id),
            None,
        )
        return overlay is not None and overlay.note_name_in_octave in WHITE_NOTE_NAMES

    def rotated_overlay_corners(self, overlay) -> Tuple[Tuple[float, float], ...]:
        return self._rotated_corners(
            OverlayGeometry(
                float(overlay.x),
                float(overlay.y),
                float(overlay.width),
                float(overlay.height),
                float(getattr(overlay, "rotation_degrees", 0.0) or 0.0),
            )
        )

    def _constrain_rect_to_keyboard_box(self, rect: OverlayGeometry) -> OverlayGeometry:
        box = self._keyboard_box
        if box is None:
            return rect
        box_width = max(1.0, box.right - box.left)
        box_height = max(1.0, box.bottom - box.top)
        width = max(1.0, rect.width)
        height = max(1.0, rect.height)
        rotation = rect.rotation_degrees
        x_margin, y_margin = self._rotated_margins(width, height, rotation)
        if x_margin * 2.0 > box_width or y_margin * 2.0 > box_height:
            scale = min(
                box_width / max(1.0, x_margin * 2.0),
                box_height / max(1.0, y_margin * 2.0),
            )
            width = max(1.0, width * scale)
            height = max(1.0, height * scale)
            x_margin, y_margin = self._rotated_margins(width, height, rotation)

        center_x = rect.x + (rect.width / 2.0)
        center_y = rect.y + (rect.height / 2.0)
        center_x = self._clamp(center_x, box.left + x_margin, box.right - x_margin)
        center_y = self._clamp(center_y, box.top + y_margin, box.bottom - y_margin)
        return OverlayGeometry(
            center_x - (width / 2.0),
            center_y - (height / 2.0),
            width,
            height,
            rotation,
        )

    @staticmethod
    def _rotated_margins(width: float, height: float, rotation_degrees: float) -> Tuple[float, float]:
        radians = math.radians(rotation_degrees)
        cos_value = abs(math.cos(radians))
        sin_value = abs(math.sin(radians))
        return (
            (width * cos_value + height * sin_value) / 2.0,
            (width * sin_value + height * cos_value) / 2.0,
        )

    @staticmethod
    def _rotated_corners(rect: OverlayGeometry) -> Tuple[Tuple[float, float], ...]:
        center_x = rect.x + (rect.width / 2.0)
        center_y = rect.y + (rect.height / 2.0)
        radians = math.radians(rect.rotation_degrees)
        cos_value = math.cos(radians)
        sin_value = math.sin(radians)
        corners = (
            (-rect.width / 2.0, -rect.height / 2.0),
            (rect.width / 2.0, -rect.height / 2.0),
            (rect.width / 2.0, rect.height / 2.0),
            (-rect.width / 2.0, rect.height / 2.0),
        )
        return tuple(
            (
                center_x + (x * cos_value) - (y * sin_value),
                center_y + (x * sin_value) + (y * cos_value),
            )
            for x, y in corners
        )

    def _persist_keyboard_box(self) -> None:
        self.app_state.calibration.manual_keyboard_box = self._keyboard_box_tuple()

    def _keyboard_box_tuple(self) -> Tuple[float, float, float, float] | None:
        if self._keyboard_box is None:
            return None
        return (
            self._keyboard_box.left,
            self._keyboard_box.top,
            self._keyboard_box.right,
            self._keyboard_box.bottom,
        )

    @staticmethod
    def _normalize_keyboard_box(left: float, top: float, right: float, bottom: float) -> KeyboardBox:
        box_left = min(float(left), float(right))
        box_right = max(float(left), float(right))
        box_top = min(float(top), float(bottom))
        box_bottom = max(float(top), float(bottom))
        if box_right <= box_left:
            box_right = box_left + 1.0
        if box_bottom <= box_top:
            box_bottom = box_top + 1.0
        return KeyboardBox(box_left, box_top, box_right, box_bottom)

    @staticmethod
    def _params_are_neutral(params: ManualFitParams) -> bool:
        return all(getattr(params, name) == 0 for name in CONTROL_PARAM_NAMES) and (
            params.group_dx == 0 and params.group_dy == 0
        )

    def _generate_baseline_from_setup_box(self, box: KeyboardBox) -> Dict[int, OverlayGeometry]:
        ordered_overlays = sorted(self.app_state.overlays, key=lambda overlay: overlay.key_id)
        white_overlays = [
            overlay for overlay in ordered_overlays if overlay.note_name_in_octave in WHITE_NOTE_NAMES
        ]
        white_count = max(1, len(white_overlays))
        white_width = max(1.0, (box.right - box.left) / white_count)
        black_width = max(1.0, white_width * 0.60)
        generated: Dict[int, OverlayGeometry] = {}
        white_index = 0

        for overlay in ordered_overlays:
            if overlay.note_name_in_octave in WHITE_NOTE_NAMES:
                x = box.left + (white_index * white_width)
                generated[overlay.key_id] = OverlayGeometry(
                    x,
                    box.top,
                    white_width,
                    box.bottom - box.top,
                    float(getattr(overlay, "rotation_degrees", 0.0) or 0.0),
                )
                white_index += 1
                continue

            boundary_x = box.left + (white_index * white_width)
            x = boundary_x - (black_width / 2.0)
            x = self._clamp(x, box.left, max(box.left, box.right - black_width))
            generated[overlay.key_id] = OverlayGeometry(
                x,
                box.top,
                black_width,
                max(1.0, self._setup_black_bottom_or_default() - box.top),
                float(getattr(overlay, "rotation_degrees", 0.0) or 0.0),
            )

        return generated

    def _require_setup_keyboard_box(self) -> KeyboardBox:
        if self._setup_keyboard_box is None:
            raise RuntimeError("Manual Fit keyboard box has not been set.")
        return self._setup_keyboard_box

    def _setup_black_bottom_or_default(self) -> float:
        if self._setup_black_bottom is not None:
            return self._setup_black_bottom
        return self.default_setup_black_bottom()

    def _setup_white_start_or_default(self) -> float:
        if self._setup_white_start is not None:
            return self._setup_white_start
        return self.default_setup_white_start()

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        if maximum < minimum:
            maximum = minimum
        return max(minimum, min(float(value), maximum))
