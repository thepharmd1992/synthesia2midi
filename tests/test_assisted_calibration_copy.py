from types import SimpleNamespace

from PySide6.QtWidgets import QApplication

from synthesia2midi.core.app_state import AppState
from synthesia2midi.gui.calibration_wizard_controller import CalibrationWizardController
from synthesia2midi.workflows.calibration import CalibrationWorkflow


def test_assisted_calibration_summary_explains_color_families_without_rgb_first():
    QApplication.instance() or QApplication([])
    controller = CalibrationWizardController.__new__(CalibrationWizardController)
    proposal = SimpleNamespace(
        candidate_count=12,
        assignment_result=SimpleNamespace(
            family_count=2,
            assignments={
                "LW": SimpleNamespace(enabled=True, rgb=(255, 0, 0)),
                "LB": SimpleNamespace(enabled=True, rgb=None),
                "RW": SimpleNamespace(enabled=True, rgb=(0, 120, 255)),
                "RB": SimpleNamespace(enabled=False, rgb=None),
            },
        ),
    )

    text = controller._proposal_summary_text(proposal)

    assert "Assisted calibration found 12 possible pressed-key samples." in text
    assert "Found 2 Synthesia note color families." in text
    assert "Left/Right refer to Synthesia note colors" not in text
    assert "Color 1 Natural: found" in text
    assert "Color 1 Sharp / Flat: not found" in text
    assert "Color 2 Sharp / Flat: not present in this video" in text
    assert "{label}" not in text
    assert "(255, 0, 0)" not in text


def test_manual_exemplar_prompt_uses_canonical_color_family_label():
    workflow = CalibrationWorkflow(AppState(), SimpleNamespace())
    messages = []
    workflow._show_info = lambda title, message: messages.append((title, message))

    workflow.handle_calibrate_lit_exemplar_key_start("LB")

    assert messages == [
        (
            "Lit Exemplar Calibration",
            "Click a glowing key that matches Color 1 Sharp / Flat. The application "
            "will sample the color and histogram for Color 1 Sharp / Flat.",
        )
    ]
