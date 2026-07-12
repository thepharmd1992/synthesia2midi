from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.core.app_state import AppState
from synthesia2midi.core.color_families import SUPPORTED_EXEMPLAR_SLOTS, slots_for_family
from synthesia2midi.detection.assisted_calibration import (
    AssistedCalibrationProposal,
    AssignedExemplar,
    ExemplarAssignmentResult,
    UnlitFrameAssessment,
)
from synthesia2midi.gui import calibration_wizard_controller as controller_module
from synthesia2midi.gui.assisted_calibration_dialog import AssistedCalibrationDecision
from synthesia2midi.gui.calibration_wizard_controller import CalibrationWizardController
from synthesia2midi.gui.controls_qt import ControlPanelQt


class _FakeTuningController:
    def set_apply_template_styles_callback(self, callback):
        self.apply_template_styles_callback = callback


class _FakeProgressDialog:
    def __init__(self, *_args, **_kwargs):
        self.closed = False

    def setWindowTitle(self, _title):
        pass

    def setMinimumDuration(self, _duration):
        pass

    def setMaximum(self, _maximum):
        pass

    def setValue(self, _value):
        pass

    def wasCanceled(self):
        return False

    def close(self):
        self.closed = True


def _overlay() -> OverlayConfig:
    overlay = OverlayConfig(
        key_id=1,
        note_octave=4,
        note_name_in_octave="C",
        x=0,
        y=0,
        width=4,
        height=4,
        key_type="white",
    )
    overlay.unlit_reference_color = (12, 13, 14)
    overlay.unlit_hist = np.array([0.25, 0.75], dtype=np.float32)
    return overlay


def _controller_with_seeded_state():
    QApplication.instance() or QApplication([])
    state = AppState()
    state.overlays = [_overlay()]
    for index, slot in enumerate(SUPPORTED_EXEMPLAR_SLOTS):
        state.detection.exemplar_key_type_enabled[slot] = index % 2 == 0
        state.detection.exemplar_lit_colors[slot] = (index + 1, index + 2, index + 3)
        state.detection.exemplar_lit_histograms[slot] = np.array(
            [index, index + 0.5], dtype=np.float32
        )
    state.detection.hand_assignment_enabled = True
    state.unsaved_changes = False
    app = SimpleNamespace(
        app_state=state,
        calibration_workflow=None,
        control_panel=SimpleNamespace(),
        keyboard_canvas=SimpleNamespace(),
        show_overlays_action=SimpleNamespace(),
        video_loading_workflow=None,
        video_session=SimpleNamespace(total_frames=5),
    )
    controller = CalibrationWizardController(
        app,
        auto_detect_tuning_controller=_FakeTuningController(),
    )
    return controller, state


def _proposal(*, family_number=3, canceled=False) -> AssistedCalibrationProposal:
    selected_slot = slots_for_family(family_number)[0]
    assignments = {
        slot: AssignedExemplar(
            slot=slot,
            rgb=(220, 40, 80) if slot == selected_slot else None,
            hist=np.array([1.0], dtype=np.float32) if slot == selected_slot else None,
            source=None,
            enabled=slot == selected_slot,
        )
        for slot in SUPPORTED_EXEMPLAR_SLOTS
    }
    return AssistedCalibrationProposal(
        baseline_frame_index=0,
        unlit_assessment=UnlitFrameAssessment(status="clean"),
        assignment_result=ExemplarAssignmentResult(
            assignments=assignments,
            missing_slots=(),
            disabled_slots=tuple(slot for slot in SUPPORTED_EXEMPLAR_SLOTS if slot != selected_slot),
            family_count=family_number,
            confidence=0.9,
        ),
        scanned_frame_count=4,
        candidate_count=1,
        canceled=canceled,
    )


def _partial_family_proposal(family_number=3) -> AssistedCalibrationProposal:
    natural_slot, accidental_slot = slots_for_family(family_number)
    assignments = {
        slot: AssignedExemplar(
            slot=slot,
            rgb=(220, 40, 80) if slot == natural_slot else None,
            hist=np.array([1.0], dtype=np.float32) if slot == natural_slot else None,
            source=None,
            enabled=slot in {natural_slot, accidental_slot},
        )
        for slot in SUPPORTED_EXEMPLAR_SLOTS
    }
    return AssistedCalibrationProposal(
        baseline_frame_index=0,
        unlit_assessment=UnlitFrameAssessment(status="clean"),
        assignment_result=ExemplarAssignmentResult(
            assignments=assignments,
            missing_slots=(accidental_slot,),
            disabled_slots=tuple(
                slot
                for slot in SUPPORTED_EXEMPLAR_SLOTS
                if slot not in {natural_slot, accidental_slot}
            ),
            family_count=1,
            confidence=0.9,
        ),
        scanned_frame_count=4,
        candidate_count=1,
    )


def _state_signature(state: AppState):
    return {
        "enabled": dict(state.detection.exemplar_key_type_enabled),
        "colors": dict(state.detection.exemplar_lit_colors),
        "histograms": {
            slot: None if histogram is None else np.asarray(histogram).tolist()
            for slot, histogram in state.detection.exemplar_lit_histograms.items()
        },
        "hand_assignment_enabled": state.detection.hand_assignment_enabled,
        "overlays": [
            (
                overlay.unlit_reference_color,
                None if overlay.unlit_hist is None else np.asarray(overlay.unlit_hist).tolist(),
            )
            for overlay in state.overlays
        ],
        "unsaved_changes": state.unsaved_changes,
    }


def _mutate_every_calibration_value(state: AppState) -> None:
    for index, slot in enumerate(SUPPORTED_EXEMPLAR_SLOTS):
        state.detection.exemplar_key_type_enabled[slot] = not state.detection.exemplar_key_type_enabled[slot]
        state.detection.exemplar_lit_colors[slot] = (200 + index, 100 + index, 50 + index)
        state.detection.exemplar_lit_histograms[slot] = np.array([99 + index], dtype=np.float32)
    state.detection.hand_assignment_enabled = False
    state.overlays[0].unlit_reference_color = (240, 240, 240)
    state.overlays[0].unlit_hist = np.array([9.0], dtype=np.float32)
    state.unsaved_changes = True


def _patch_assisted_dependencies(
    monkeypatch,
    state,
    proposal,
    *,
    decision=None,
    capture_error=None,
    scan_error=None,
):
    monkeypatch.setattr(controller_module, "QProgressDialog", _FakeProgressDialog)
    monkeypatch.setattr(
        controller_module,
        "assess_unlit_frame",
        lambda *_args, **_kwargs: UnlitFrameAssessment(status="clean"),
    )
    def capture_unlit(*_args, **_kwargs):
        _mutate_every_calibration_value(state)
        if capture_error is not None:
            raise capture_error

    monkeypatch.setattr(
        controller_module,
        "capture_unlit_references_from_frame",
        capture_unlit,
    )

    def build_proposal(*_args, **_kwargs):
        if scan_error is not None:
            raise scan_error
        return proposal

    monkeypatch.setattr(controller_module, "build_assisted_calibration_proposal", build_proposal)

    if decision is not None:
        class FakeDialog:
            def __init__(self, *_args, **_kwargs):
                self.decision = decision

            def exec(self):
                pass

        monkeypatch.setattr(controller_module, "AssistedCalibrationDialog", FakeDialog)


def test_progress_cancel_restores_all_eight_slots_and_assignment_flag(monkeypatch):
    controller, state = _controller_with_seeded_state()
    original = _state_signature(state)
    _patch_assisted_dependencies(monkeypatch, state, _proposal(canceled=True))

    assert controller._run_assisted_auto_calibration(np.zeros((4, 4, 3), dtype=np.uint8), 0) is False

    assert _state_signature(state) == original


def test_review_window_close_restores_all_eight_slots_and_assignment_flag(monkeypatch):
    controller, state = _controller_with_seeded_state()
    original = _state_signature(state)
    _patch_assisted_dependencies(
        monkeypatch,
        state,
        _proposal(),
        decision=AssistedCalibrationDecision.KEEP,
    )

    assert controller._run_assisted_auto_calibration(np.zeros((4, 4, 3), dtype=np.uint8), 0) is False

    assert _state_signature(state) == original


def test_scan_exception_restores_all_eight_slots_and_assignment_flag(monkeypatch):
    controller, state = _controller_with_seeded_state()
    original = _state_signature(state)
    _patch_assisted_dependencies(
        monkeypatch,
        state,
        _proposal(),
        scan_error=RuntimeError("scan failed"),
    )

    with pytest.raises(RuntimeError, match="scan failed"):
        controller._run_assisted_auto_calibration(np.zeros((4, 4, 3), dtype=np.uint8), 0)

    assert _state_signature(state) == original


def test_capture_exception_restores_all_eight_slots_and_assignment_flag(monkeypatch):
    controller, state = _controller_with_seeded_state()
    original = _state_signature(state)
    _patch_assisted_dependencies(
        monkeypatch,
        state,
        _proposal(),
        capture_error=RuntimeError("capture failed"),
    )

    with pytest.raises(RuntimeError, match="capture failed"):
        controller._run_assisted_auto_calibration(np.zeros((4, 4, 3), dtype=np.uint8), 0)

    assert _state_signature(state) == original


def test_apply_exception_restores_all_eight_slots_and_assignment_flag(monkeypatch):
    controller, state = _controller_with_seeded_state()
    original = _state_signature(state)
    _patch_assisted_dependencies(
        monkeypatch,
        state,
        _proposal(),
        decision=AssistedCalibrationDecision.USE,
    )

    def fail_during_apply(app_state, _proposal):
        _mutate_every_calibration_value(app_state)
        raise RuntimeError("apply failed")

    monkeypatch.setattr(
        controller_module,
        "apply_assisted_calibration_proposal",
        fail_during_apply,
    )

    with pytest.raises(RuntimeError, match="apply failed"):
        controller._run_assisted_auto_calibration(np.zeros((4, 4, 3), dtype=np.uint8), 0)

    assert _state_signature(state) == original


@pytest.mark.parametrize("family_number", [3, 4])
def test_accepting_higher_color_family_enables_separate_assignment(monkeypatch, family_number):
    controller, state = _controller_with_seeded_state()
    state.detection.hand_assignment_enabled = False
    proposal = _proposal(family_number=family_number)
    _patch_assisted_dependencies(
        monkeypatch,
        state,
        proposal,
        decision=AssistedCalibrationDecision.USE,
    )

    assert controller._run_assisted_auto_calibration(np.zeros((4, 4, 3), dtype=np.uint8), 0) is True

    selected_slot = slots_for_family(family_number)[0]
    assert state.detection.exemplar_lit_colors[selected_slot] == (220, 40, 80)
    assert state.detection.hand_assignment_enabled is True


def test_accepting_partial_new_family_keeps_missing_morphology_required(monkeypatch):
    controller, state = _controller_with_seeded_state()
    natural_slot, accidental_slot = slots_for_family(3)
    for slot in (natural_slot, accidental_slot):
        state.detection.exemplar_key_type_enabled[slot] = False
        state.detection.exemplar_lit_colors[slot] = None
        state.detection.exemplar_lit_histograms[slot] = None
    state.detection.hand_assignment_enabled = False
    proposal = _partial_family_proposal(3)
    _patch_assisted_dependencies(
        monkeypatch,
        state,
        proposal,
        decision=AssistedCalibrationDecision.USE,
    )

    assert controller._run_assisted_auto_calibration(
        np.zeros((4, 4, 3), dtype=np.uint8), 0
    ) is True

    assert state.detection.exemplar_lit_colors[natural_slot] == (220, 40, 80)
    assert state.detection.exemplar_key_type_enabled[accidental_slot] is True
    assert state.detection.exemplar_lit_colors[accidental_slot] is None
    assert state.detection.exemplar_lit_histograms[accidental_slot] is None
    assert accidental_slot in state.detection.get_required_exemplar_types()


def test_accepting_partial_new_family_refreshes_panel_and_shows_missing_row(monkeypatch):
    controller, state = _controller_with_seeded_state()
    natural_slot, accidental_slot = slots_for_family(3)
    for slot in (natural_slot, accidental_slot):
        state.detection.exemplar_key_type_enabled[slot] = False
        state.detection.exemplar_lit_colors[slot] = None
        state.detection.exemplar_lit_histograms[slot] = None

    panel = ControlPanelQt(app_state=state)
    controller.app.control_panel = panel
    refresh_calls = []
    original_refresh = panel.update_controls_from_state

    def record_refresh():
        refresh_calls.append(True)
        original_refresh()

    monkeypatch.setattr(panel, "update_controls_from_state", record_refresh)
    _patch_assisted_dependencies(
        monkeypatch,
        state,
        _partial_family_proposal(3),
        decision=AssistedCalibrationDecision.USE,
    )

    try:
        assert accidental_slot not in panel.color_family_grid.rows

        assert controller._run_assisted_auto_calibration(
            np.zeros((4, 4, 3), dtype=np.uint8), 0
        ) is True

        assert refresh_calls == [True]
        assert accidental_slot in panel.color_family_grid.rows
        assert panel.exemplar_buttons[accidental_slot].text() == "Set"
        assert panel.exemplar_presence_checkboxes[accidental_slot].text() == "Present"
    finally:
        panel.close()
        panel.deleteLater()


def test_proposal_summary_uses_dynamic_color_family_labels():
    controller, _state = _controller_with_seeded_state()
    proposal = _proposal(family_number=4)

    summary = controller._proposal_summary_text(proposal)

    assert "Color 4 Natural: found" in summary
    assert "Left/Right refer" not in summary
