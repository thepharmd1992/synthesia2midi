from synthesia2midi.core.color_families import (
    SUPPORTED_EXEMPLAR_SLOTS,
    active_family_numbers,
    exemplar_display_parts,
    family_for_slot,
    morphology_for_slot,
    slots_for_family,
)


def test_color_family_registry_preserves_legacy_slots_and_channels():
    assert SUPPORTED_EXEMPLAR_SLOTS == (
        "LW",
        "LB",
        "RW",
        "RB",
        "COLOR_3_W",
        "COLOR_3_B",
        "COLOR_4_W",
        "COLOR_4_B",
    )
    assert slots_for_family(1) == ("LW", "LB")
    assert slots_for_family(4) == ("COLOR_4_W", "COLOR_4_B")
    assert family_for_slot("COLOR_3_B").midi_channel == 2
    assert morphology_for_slot("RW") == "natural"
    assert morphology_for_slot("RB") == "accidental"
    assert exemplar_display_parts("COLOR_4_B") == (4, "Sharp / Flat")


def test_active_families_include_enabled_or_saved_slots_but_always_color_one():
    enabled = {slot: False for slot in SUPPORTED_EXEMPLAR_SLOTS}
    colors = {slot: None for slot in SUPPORTED_EXEMPLAR_SLOTS}
    colors["COLOR_4_W"] = (1, 2, 3)

    assert active_family_numbers(enabled, colors) == (1, 4)
