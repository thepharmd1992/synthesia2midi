import numpy as np
from PySide6.QtWidgets import QApplication

from synthesia2midi.detection.assisted_calibration import (
    AssignedExemplar,
    AssistedCalibrationProposal,
    ExemplarAssignmentResult,
    UnlitFrameAssessment,
)
from synthesia2midi.detection.color_family_assignment import (
    ANCHOR_CONFLICT_WARNING,
    TOO_MANY_FAMILIES_WARNING,
)
from synthesia2midi.gui.assisted_calibration_dialog import (
    AssistedCalibrationDecision,
    AssistedCalibrationDialog,
)
from synthesia2midi.localization import install_translator


def _assignment(slot, rgb=None, *, enabled=True):
    return AssignedExemplar(
        slot=slot,
        rgb=rgb,
        hist=np.ones(4) if rgb is not None else None,
        source=None,
        enabled=enabled,
    )


def _proposal(*, warnings=(TOO_MANY_FAMILIES_WARNING,)):
    assignments = {
        "LW": _assignment("LW", (220, 40, 30)),
        "LB": _assignment("LB", (150, 25, 20)),
        "COLOR_3_W": _assignment("COLOR_3_W", (235, 185, 30)),
        "COLOR_3_B": _assignment("COLOR_3_B"),
    }
    return AssistedCalibrationProposal(
        baseline_frame_index=10,
        unlit_assessment=UnlitFrameAssessment(status="clean"),
        assignment_result=ExemplarAssignmentResult(
            assignments=assignments,
            missing_slots=("COLOR_3_B",),
            disabled_slots=(),
            family_count=2,
            confidence=0.9,
        ),
        scanned_frame_count=70,
        candidate_count=12,
        warnings=warnings,
    )


def test_assisted_dialog_shows_partial_color_family_and_scanner_warning():
    QApplication.instance() or QApplication([])
    dialog = AssistedCalibrationDialog(_proposal())
    try:
        assert dialog.summary_label.text() == "12 samples found across 2 Synthesia note color families."
        assert list(dialog.color_family_grid.rows) == [
            "LW",
            "LB",
            "COLOR_3_W",
            "COLOR_3_B",
        ]
        assert dialog.color_family_grid.family_heading(1).text() == "Color 1"
        assert dialog.color_family_grid.family_heading(3).text() == "Color 3"
        assert dialog.rows["LW"].label.text() == "Natural"
        assert dialog.rows["LW"].status.text() == "Found"
        assert "rgb(220, 40, 30)" in dialog.rows["LW"].swatch.styleSheet()
        assert dialog.rows["COLOR_3_W"].label.text() == "Natural"
        assert dialog.rows["COLOR_3_W"].status.text() == "Found"
        assert dialog.rows["COLOR_3_B"].label.text() == "Sharp / Flat"
        assert dialog.rows["COLOR_3_B"].status.text() == "Missing"
        assert dialog.warning_banner.text() == (
            "More than four stable color families were found."
        )
        assert not dialog.warning_banner.isHidden()
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


def test_assisted_dialog_translates_scanner_warning_codes():
    app = QApplication.instance() or QApplication([])
    assert install_translator(app, "es") == "es"
    dialog = AssistedCalibrationDialog(
        _proposal(warnings=(TOO_MANY_FAMILIES_WARNING, ANCHOR_CONFLICT_WARNING))
    )
    try:
        assert dialog.warning_banner.text() == (
            "Se encontraron más de cuatro familias de colores estables.\n"
            "La evidencia entra en conflicto con dos identidades de familias "
            "de colores guardadas."
        )
        assert TOO_MANY_FAMILIES_WARNING not in dialog.warning_banner.text()
        assert ANCHOR_CONFLICT_WARNING not in dialog.warning_banner.text()
    finally:
        dialog.close()
        dialog.deleteLater()
        install_translator(app, "en")


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
