import numpy as np
from PySide6.QtWidgets import QApplication

from synthesia2midi.detection.assisted_calibration import (
    AssignedExemplar,
    AssistedCalibrationProposal,
    ExemplarAssignmentResult,
    UnlitFrameAssessment,
)
from synthesia2midi.gui.assisted_calibration_dialog import (
    AssistedCalibrationDecision,
    AssistedCalibrationDialog,
)


def _assignment(slot, rgb=None, *, enabled=True):
    return AssignedExemplar(
        slot=slot,
        rgb=rgb,
        hist=np.ones(4) if rgb is not None else None,
        source=None,
        enabled=enabled,
    )


def _proposal():
    assignments = {
        "LW": _assignment("LW", (220, 40, 30)),
        "LB": _assignment("LB"),
        "RW": _assignment("RW", (20, 100, 240)),
        "RB": _assignment("RB", enabled=False),
    }
    return AssistedCalibrationProposal(
        baseline_frame_index=10,
        unlit_assessment=UnlitFrameAssessment(status="clean"),
        assignment_result=ExemplarAssignmentResult(
            assignments=assignments,
            missing_slots=("LB",),
            disabled_slots=("RB",),
            family_count=2,
            confidence=0.9,
        ),
        scanned_frame_count=70,
        candidate_count=12,
    )


def test_assisted_dialog_shows_color_family_rows_and_real_swatches():
    QApplication.instance() or QApplication([])
    dialog = AssistedCalibrationDialog(_proposal())
    try:
        assert dialog.summary_label.text() == "12 samples found across 2 Synthesia note color families."
        assert "not the physical side" in dialog.color_family_note.text()
        assert list(dialog.rows) == ["LW", "LB", "RW", "RB"]
        assert dialog.rows["LW"].name_label.text() == "Left White"
        assert dialog.rows["LW"].status_label.text() == "Found"
        assert "rgb(220, 40, 30)" in dialog.rows["LW"].swatch.styleSheet()
        assert dialog.rows["LB"].status_label.text() == "Not found"
        assert dialog.rows["RB"].status_label.text() == "Not used"
        assert "(220, 40, 30)" not in dialog.summary_label.text()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_assisted_dialog_defaults_to_use_and_close_keeps_current_examples():
    QApplication.instance() or QApplication([])
    dialog = AssistedCalibrationDialog(_proposal())
    try:
        assert dialog.use_button.isDefault()
        assert dialog.decision is AssistedCalibrationDecision.KEEP
        dialog.try_another_button.click()
        assert dialog.decision is AssistedCalibrationDecision.RETRY
    finally:
        dialog.close()
        dialog.deleteLater()


def test_assisted_dialog_stacks_actions_for_long_translations():
    QApplication.instance() or QApplication([])
    dialog = AssistedCalibrationDialog(_proposal())
    try:
        button_layout = dialog.layout().itemAt(dialog.layout().count() - 1).layout()

        assert button_layout.rowCount() == 3
        assert button_layout.columnCount() == 1
    finally:
        dialog.close()
        dialog.deleteLater()
