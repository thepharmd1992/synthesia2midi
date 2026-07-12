from types import SimpleNamespace

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.core.app_state import AppState
from synthesia2midi.workflows.conversion import ConversionWorkflow


def _conversion_state() -> AppState:
    state = AppState()
    overlay = OverlayConfig(
        key_id=1,
        note_octave=4,
        note_name_in_octave="C",
        x=0,
        y=0,
        width=10,
        height=40,
        key_type="white",
        unlit_reference_color=(12, 12, 12),
    )
    state.overlays = [overlay]
    state.detection.exemplar_lit_colors.update(
        {
            "LW": (255, 0, 0),
            "LB": (160, 0, 0),
            "RW": (0, 120, 255),
            "RB": (0, 70, 180),
        }
    )
    return state


def _workflow(state: AppState) -> ConversionWorkflow:
    return ConversionWorkflow(
        state,
        SimpleNamespace(),
        runtime_paths=SimpleNamespace(),
    )


def test_conversion_preflight_names_enabled_higher_family_missing_color():
    state = _conversion_state()
    state.detection.exemplar_key_type_enabled["COLOR_4_B"] = True
    state.detection.exemplar_lit_colors["COLOR_4_B"] = None

    errors = _workflow(state)._validate_prerequisites()

    assert any("Color 4 Sharp / Flat" in error for error in errors)


def test_conversion_preflight_ignores_unchecked_higher_family_slot():
    state = _conversion_state()
    state.detection.exemplar_key_type_enabled["COLOR_4_B"] = False
    state.detection.exemplar_lit_colors["COLOR_4_B"] = None

    errors = _workflow(state)._validate_prerequisites()

    assert not any("Missing exemplar colors" in error for error in errors)
