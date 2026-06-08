import pytest
from PySide6.QtWidgets import QApplication, QWidget

from synthesia2midi.core.app_state import AppState
from synthesia2midi.gui.wizard import CalibrationWizard


def _wizard_parent(width=1000, height=500):
    parent = QWidget()
    parent.video_session = type("VideoSessionStub", (), {"width": width, "height": height})()
    return parent


def test_manual_generated_black_keys_are_top_aligned_with_white_keys():
    QApplication.instance() or QApplication([])
    app_state = AppState()
    app_state.midi.leftmost_note_name = "C"
    app_state.midi.leftmost_note_octave = 4
    app_state.midi.total_keys = 5
    parent = _wizard_parent()
    wizard = CalibrationWizard(parent, app_state)

    try:
        wizard._generate_initial_overlays()
    finally:
        wizard.close()
        parent.close()

    white_keys = [overlay for overlay in app_state.overlays if "♯" not in overlay.note_name_in_octave]
    black_keys = [overlay for overlay in app_state.overlays if "♯" in overlay.note_name_in_octave]

    assert {black.note_name_in_octave: black.y for black in black_keys} == {
        "C♯": white_keys[0].y,
        "D♯": white_keys[0].y,
    }
    assert all(black.height < white_keys[0].height for black in black_keys)


def test_manual_generated_black_key_centers_split_adjacent_white_keys():
    QApplication.instance() or QApplication([])
    app_state = AppState()
    app_state.midi.leftmost_note_name = "G"
    app_state.midi.leftmost_note_octave = 4
    app_state.midi.total_keys = 3
    parent = _wizard_parent()
    wizard = CalibrationWizard(parent, app_state)

    try:
        wizard._generate_initial_overlays()
    finally:
        wizard.close()
        parent.close()

    g_key = next(overlay for overlay in app_state.overlays if overlay.note_name_in_octave == "G")
    g_sharp_key = next(overlay for overlay in app_state.overlays if overlay.note_name_in_octave == "G♯")
    a_key = next(overlay for overlay in app_state.overlays if overlay.note_name_in_octave == "A")

    g_sharp_center = g_sharp_key.x + (g_sharp_key.width / 2)
    expected_split = (g_key.x + (g_key.width / 2) + a_key.x + (a_key.width / 2)) / 2

    assert g_sharp_center == pytest.approx(expected_split)
