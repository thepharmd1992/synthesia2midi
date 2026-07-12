from synthesia2midi.core.app_state import DetectionConfig


def test_detection_config_defaults_new_families_off_and_masks_them():
    config = DetectionConfig()

    assert config.exemplar_key_type_enabled["LW"] is True
    assert config.exemplar_key_type_enabled["COLOR_3_W"] is False
    config.exemplar_lit_colors["COLOR_3_W"] = (12, 34, 56)
    assert config.get_effective_exemplar_lit_colors()["COLOR_3_W"] is None

    config.exemplar_key_type_enabled["COLOR_3_W"] = True
    assert config.get_required_exemplar_types() == [
        "LW",
        "LB",
        "RW",
        "RB",
        "COLOR_3_W",
    ]
    assert config.get_required_base_exemplar_types() == config.get_required_exemplar_types()


def test_effective_exemplar_maps_preserve_unknown_existing_entries():
    config = DetectionConfig(
        exemplar_lit_colors={"LEGACY_DYNAMIC": (1, 2, 3)},
        exemplar_lit_histograms={"LEGACY_DYNAMIC": "histogram"},
        exemplar_key_type_enabled={},
    )

    assert config.get_effective_exemplar_lit_colors()["LEGACY_DYNAMIC"] == (1, 2, 3)
    assert config.get_effective_exemplar_lit_histograms()["LEGACY_DYNAMIC"] == "histogram"
    assert config.get_effective_exemplar_lit_colors()["COLOR_4_B"] is None
    assert config.get_effective_exemplar_lit_histograms()["COLOR_4_B"] is None
