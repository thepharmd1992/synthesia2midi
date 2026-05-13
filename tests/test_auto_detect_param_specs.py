from synthesia2midi.detection.auto_detect_param_specs import (
    ACTIVE_AUTO_DETECT_PARAM_KEYS,
    coerce_auto_detect_params,
)


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
