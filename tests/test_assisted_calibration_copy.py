from types import SimpleNamespace

from PySide6.QtWidgets import QApplication

from synthesia2midi.gui.calibration_wizard_controller import CalibrationWizardController


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
