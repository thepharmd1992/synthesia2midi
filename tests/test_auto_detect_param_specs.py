from synthesia2midi.detection.auto_detect_param_specs import (
    ACTIVE_AUTO_DETECT_PARAM_KEYS,
    coerce_auto_detect_params,
)
from synthesia2midi.detection.detector_defaults import DEFAULT_DETECTION_PARAMS


def test_auto_detect_params_include_all_active_keys_and_drop_unknowns():
    params = coerce_auto_detect_params({"unknown_param": 123})

    assert set(params) == set(ACTIVE_AUTO_DETECT_PARAM_KEYS)
    assert "unknown_param" not in params


def test_auto_detect_params_clamp_numbers_and_normalize_bool_enum():
    params = coerce_auto_detect_params(
        {
            "black_upper_ratio": 999,
            "black_threshold_method": "ADAPTIVE",
            "type_aware_assignment": "false",
            "black_adaptive_block_size": 20,
        }
    )

    assert params["black_upper_ratio"] == 0.90
    assert params["black_threshold_method"] == "adaptive"
    assert params["type_aware_assignment"] is False
    assert params["black_adaptive_block_size"] % 2 == 1


def test_auto_detect_params_invalid_literals_fall_back_to_detector_defaults():
    params = coerce_auto_detect_params(
        {
            "black_threshold": "not-a-number",
            "black_threshold_method": "unsupported",
            "type_aware_assignment": "maybe",
            "black_adaptive_block_size": None,
        }
    )

    assert params["black_threshold"] == DEFAULT_DETECTION_PARAMS["black_threshold"]
    assert params["black_threshold_method"] == DEFAULT_DETECTION_PARAMS["black_threshold_method"]
    assert params["type_aware_assignment"] is DEFAULT_DETECTION_PARAMS["type_aware_assignment"]
    assert params["black_adaptive_block_size"] == DEFAULT_DETECTION_PARAMS["black_adaptive_block_size"]
