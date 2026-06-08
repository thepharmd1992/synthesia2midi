"""
Active monolithic auto-detect parameter specs and coercion helpers.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping


ACTIVE_AUTO_DETECT_PARAM_KEYS: List[str] = [
    "black_upper_ratio",
    "black_bottom_ratio",
    "black_threshold_method",
    "black_threshold",
    "black_adaptive_block_size",
    "black_adaptive_c",
    "black_column_ratio",
    "black_min_width",
    "black_max_width",
    "white_bottom_ratio",
    "white_initial_top_ratio",
    "white_strip_dark_threshold",
    "white_strip_dark_fraction",
    "white_strip_min_run",
    "white_strip_allow_failures",
    "white_sep_ratio",
    "white_sep_dyn_min",
    "white_sep_close_kernel",
    "white_sep_open_kernel",
    "white_sep_min_width",
    "type_aware_assignment",
    "black_recovery_enabled",
    "black_recovery_ratio",
    "black_recovery_column_ratio_scale",
    "black_split_max_factor",
    "padding_percent",
    "white_edge_left_shift_ticks",
    "white_edge_right_shift_ticks",
]

BASIC_AUTO_DETECT_PARAM_KEYS: List[str] = [
    "white_edge_left_shift_ticks",
    "white_edge_right_shift_ticks",
]

AUTO_DETECT_PARAM_CATEGORIES: List[str] = [
    "Edge Drift Correction",
    "Black Key Detection",
    "White Strip Selection",
    "White Separator Extraction",
    "Assignment and Recovery",
    "Geometry and Padding",
]

AUTO_DETECT_PARAM_SPECS: Dict[str, Dict[str, Any]] = {
    "black_upper_ratio": {"type": "float", "min": 0.20, "max": 0.90, "step": 0.01, "category": "Black Key Detection"},
    "black_bottom_ratio": {"type": "float", "min": 0.05, "max": 1.00, "step": 0.01, "category": "Black Key Detection"},
    "black_threshold_method": {"type": "enum", "options": ["fixed", "otsu", "adaptive"], "category": "Black Key Detection"},
    "black_threshold": {"type": "int", "min": 0, "max": 255, "step": 1, "category": "Black Key Detection"},
    "black_adaptive_block_size": {
        "type": "int",
        "min": 3,
        "max": 101,
        "step": 2,
        "odd_only": True,
        "category": "Black Key Detection",
    },
    "black_adaptive_c": {"type": "int", "min": -40, "max": 60, "step": 1, "category": "Black Key Detection"},
    "black_column_ratio": {"type": "float", "min": 0.01, "max": 0.50, "step": 0.005, "category": "Black Key Detection"},
    "black_min_width": {"type": "int", "min": 1, "max": 200, "step": 1, "category": "Black Key Detection"},
    "black_max_width": {"type": "int", "min": 2, "max": 300, "step": 1, "category": "Black Key Detection"},
    "white_bottom_ratio": {"type": "float", "min": 0.50, "max": 0.98, "step": 0.01, "category": "White Strip Selection"},
    "white_initial_top_ratio": {"type": "float", "min": 0.30, "max": 0.98, "step": 0.01, "category": "White Strip Selection"},
    "white_strip_dark_threshold": {"type": "int", "min": 0, "max": 255, "step": 1, "category": "White Strip Selection"},
    "white_strip_dark_fraction": {"type": "float", "min": 0.00, "max": 0.20, "step": 0.001, "category": "White Strip Selection"},
    "white_strip_min_run": {"type": "int", "min": 1, "max": 60, "step": 1, "category": "White Strip Selection"},
    "white_strip_allow_failures": {"type": "int", "min": 0, "max": 20, "step": 1, "category": "White Strip Selection"},
    "white_sep_ratio": {"type": "float", "min": 0.10, "max": 0.90, "step": 0.01, "category": "White Separator Extraction"},
    "white_sep_dyn_min": {"type": "int", "min": 1, "max": 100, "step": 1, "category": "White Separator Extraction"},
    "white_sep_close_kernel": {
        "type": "int",
        "min": 1,
        "max": 31,
        "step": 2,
        "odd_only": True,
        "category": "White Separator Extraction",
    },
    "white_sep_open_kernel": {
        "type": "int",
        "min": 1,
        "max": 31,
        "step": 2,
        "odd_only": True,
        "category": "White Separator Extraction",
    },
    "white_sep_min_width": {"type": "int", "min": 1, "max": 30, "step": 1, "category": "White Separator Extraction"},
    "type_aware_assignment": {"type": "bool", "category": "Assignment and Recovery"},
    "black_recovery_enabled": {"type": "bool", "category": "Assignment and Recovery"},
    "black_recovery_ratio": {"type": "float", "min": 0.10, "max": 1.00, "step": 0.01, "category": "Assignment and Recovery"},
    "black_recovery_column_ratio_scale": {
        "type": "float",
        "min": 0.10,
        "max": 1.00,
        "step": 0.01,
        "category": "Assignment and Recovery",
    },
    "black_split_max_factor": {"type": "float", "min": 1.00, "max": 3.00, "step": 0.05, "category": "Assignment and Recovery"},
    "padding_percent": {"type": "float", "min": 0.00, "max": 0.45, "step": 0.01, "category": "Geometry and Padding"},
    "white_edge_left_shift_ticks": {
        "type": "int",
        "min": 0,
        "max": 20,
        "step": 1,
        "category": "Edge Drift Correction",
    },
    "white_edge_right_shift_ticks": {
        "type": "int",
        "min": 0,
        "max": 20,
        "step": 1,
        "category": "Edge Drift Correction",
    },
}


def _base_active_defaults() -> Dict[str, Any]:
    from synthesia2midi.detection.monolithic_detector import DEFAULT_DETECTION_PARAMS

    defaults: Dict[str, Any] = {}
    for key in ACTIVE_AUTO_DETECT_PARAM_KEYS:
        defaults[key] = DEFAULT_DETECTION_PARAMS[key]
    return defaults


def humanize_auto_detect_param_name(key: str) -> str:
    return key.replace("_", " ").title()


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _coerce_bool(value: Any, default_value: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(default_value)


def _coerce_enum(value: Any, options: Iterable[str], default_value: str) -> str:
    if value is None:
        return default_value
    normalized = str(value).strip().lower()
    valid_options = [str(opt).strip().lower() for opt in options]
    if normalized in valid_options:
        return normalized
    return default_value


def _coerce_int(value: Any, *, minimum: int, maximum: int, default_value: int, odd_only: bool = False) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = int(default_value)

    parsed = int(_clamp(parsed, minimum, maximum))

    if odd_only and parsed % 2 == 0:
        if parsed < maximum:
            parsed += 1
        else:
            parsed -= 1
        parsed = int(_clamp(parsed, minimum, maximum))
        if parsed % 2 == 0:
            parsed = minimum if minimum % 2 == 1 else min(maximum, minimum + 1)

    return parsed


def _coerce_float(value: Any, *, minimum: float, maximum: float, default_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default_value)
    return float(_clamp(parsed, minimum, maximum))


def coerce_auto_detect_param_value(key: str, value: Any, default_value: Any = None) -> Any:
    spec = AUTO_DETECT_PARAM_SPECS[key]
    baseline_defaults = _base_active_defaults()
    fallback = baseline_defaults[key] if default_value is None else default_value
    value_type = spec["type"]

    if value_type == "bool":
        return _coerce_bool(value, bool(fallback))
    if value_type == "enum":
        return _coerce_enum(value, spec["options"], str(fallback))
    if value_type == "int":
        return _coerce_int(
            value,
            minimum=int(spec["min"]),
            maximum=int(spec["max"]),
            default_value=int(fallback),
            odd_only=bool(spec.get("odd_only", False)),
        )
    if value_type == "float":
        return _coerce_float(
            value,
            minimum=float(spec["min"]),
            maximum=float(spec["max"]),
            default_value=float(fallback),
        )
    return fallback


def coerce_auto_detect_params(values: Mapping[str, Any] | None) -> Dict[str, Any]:
    defaults = _base_active_defaults()
    source = dict(values or {})
    output: Dict[str, Any] = {}
    for key in ACTIVE_AUTO_DETECT_PARAM_KEYS:
        output[key] = coerce_auto_detect_param_value(
            key,
            source.get(key, defaults[key]),
            default_value=defaults[key],
        )
    return output


def get_active_auto_detect_defaults() -> Dict[str, Any]:
    return coerce_auto_detect_params(_base_active_defaults())


def get_category_param_keys(category: str) -> List[str]:
    return [
        key
        for key in ACTIVE_AUTO_DETECT_PARAM_KEYS
        if AUTO_DETECT_PARAM_SPECS[key]["category"] == category
    ]


def get_basic_auto_detect_param_keys() -> List[str]:
    return list(BASIC_AUTO_DETECT_PARAM_KEYS)


def get_advanced_auto_detect_param_keys() -> List[str]:
    basic_keys = set(BASIC_AUTO_DETECT_PARAM_KEYS)
    return [key for key in ACTIVE_AUTO_DETECT_PARAM_KEYS if key not in basic_keys]
