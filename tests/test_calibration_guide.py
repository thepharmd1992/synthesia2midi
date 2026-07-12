import numpy as np
from types import SimpleNamespace
from PySide6.QtWidgets import QApplication

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.core.app_state import AppState
from synthesia2midi.gui.calibration_guide import (
    CalibrationGuideWidget,
    GuideStatus,
    derive_guide_snapshot,
)


def _overlay(key_id=1, *, unlit=False, histogram=False):
    return OverlayConfig(
        key_id=key_id,
        note_octave=4,
        note_name_in_octave="C",
        x=0,
        y=0,
        width=10,
        height=40,
        key_type="white",
        unlit_reference_color=(20, 20, 20) if unlit else None,
        unlit_hist=np.ones(4) if histogram else None,
    )


def test_guide_snapshot_advances_from_video_to_conversion():
    state = AppState()
    snapshot = derive_guide_snapshot(state, conversion_ready=False)
    assert [step.status for step in snapshot.steps] == [
        GuideStatus.NEXT,
        GuideStatus.NOT_READY,
        GuideStatus.NOT_READY,
        GuideStatus.NOT_READY,
        GuideStatus.NOT_READY,
    ]

    state.video.filepath = "/tmp/video.mp4"
    snapshot = derive_guide_snapshot(state, conversion_ready=False)
    assert snapshot.video.status is GuideStatus.DONE
    assert snapshot.overlays.status is GuideStatus.NEXT

    state.overlays = [_overlay()]
    snapshot = derive_guide_snapshot(state, conversion_ready=False)
    assert snapshot.overlays.status is GuideStatus.NEEDS_REVIEW
    assert snapshot.unlit.status is GuideStatus.NOT_READY

    state.overlays[0].unlit_reference_color = (20, 20, 20)
    snapshot = derive_guide_snapshot(state, conversion_ready=False)
    assert snapshot.overlays.status is GuideStatus.DONE
    assert snapshot.unlit.status is GuideStatus.DONE
    assert snapshot.exemplars.status is GuideStatus.NEXT

    for key_type in state.detection.get_required_base_exemplar_types():
        state.detection.exemplar_lit_colors[key_type] = (255, 0, 0)
    snapshot = derive_guide_snapshot(state, conversion_ready=True)
    assert snapshot.exemplars.status is GuideStatus.DONE
    assert snapshot.conversion.status is GuideStatus.NEXT


def test_no_key_step_requires_every_overlay_and_histograms_when_enabled():
    state = AppState()
    state.video.filepath = "/tmp/video.mp4"
    state.overlays = [_overlay(1, unlit=True, histogram=True), _overlay(2, unlit=False)]

    assert derive_guide_snapshot(state, False).unlit.status is GuideStatus.NEXT

    state.overlays[1].unlit_reference_color = (20, 20, 20)
    state.detection.use_histogram_detection = True
    assert derive_guide_snapshot(state, False).unlit.status is GuideStatus.NEXT

    state.overlays[1].unlit_hist = np.ones(4)
    assert derive_guide_snapshot(state, False).unlit.status is GuideStatus.DONE


def test_disabled_exemplar_families_are_not_required():
    state = AppState()
    state.video.filepath = "/tmp/video.mp4"
    state.overlays = [_overlay(unlit=True)]
    state.detection.exemplar_key_type_enabled["LB"] = False
    state.detection.exemplar_key_type_enabled["RB"] = False
    state.detection.exemplar_lit_colors["LW"] = (255, 0, 0)
    state.detection.exemplar_lit_colors["RW"] = (0, 0, 255)

    assert derive_guide_snapshot(state, True).exemplars.status is GuideStatus.DONE


def test_guide_widget_exposes_five_steps_and_routes_primary_actions():
    QApplication.instance() or QApplication([])
    widget = CalibrationGuideWidget()
    emitted = []
    widget.open_video_requested.connect(lambda: emitted.append("video"))
    widget.youtube_requested.connect(lambda: emitted.append("youtube"))
    widget.find_keyboard_requested.connect(lambda: emitted.append("keyboard"))
    widget.capture_unlit_requested.connect(lambda: emitted.append("unlit"))
    widget.assisted_scan_requested.connect(lambda: emitted.append("exemplars"))
    widget.convert_requested.connect(lambda: emitted.append("convert"))

    try:
        assert [row.title_label.text() for row in widget.step_rows] == [
            "1. Open or download a video",
            "2. Find and check the keyboard overlays",
            "3. Capture a no-key frame",
            "4. Find pressed-key colors",
            "5. Create MIDI",
        ]
        for row in widget.step_rows:
            row.primary_button.click()
        widget.youtube_button.click()
        assert emitted == ["video", "keyboard", "unlit", "exemplars", "convert", "youtube"]
    finally:
        widget.close()
        widget.deleteLater()


def test_guide_expands_only_current_step_and_compacts_completed_steps():
    QApplication.instance() or QApplication([])
    state = AppState()
    widget = CalibrationGuideWidget()
    try:
        widget.update_snapshot(derive_guide_snapshot(state, False))

        assert [not row.detail_widget.isHidden() for row in widget.step_rows] == [
            True,
            False,
            False,
            False,
            False,
        ]

        state.video.filepath = "/tmp/video.mp4"
        state.overlays = [_overlay(unlit=True)]
        for key_type in state.detection.get_required_base_exemplar_types():
            state.detection.exemplar_lit_colors[key_type] = (255, 0, 0)
        widget.update_snapshot(derive_guide_snapshot(state, True))

        assert [not row.detail_widget.isHidden() for row in widget.step_rows] == [
            False,
            False,
            False,
            False,
            True,
        ]
        for row in widget.step_rows[:4]:
            assert not row.completion_icon_label.pixmap().isNull()
            assert row.completion_icon_label.accessibleName() == "Done"
            assert not row.completion_icon_label.isHidden()
            assert row.status_label.text() == "Done"
            assert "#2e7d32" in row.status_label.styleSheet()
        assert widget.step_rows[4].completion_icon_label.isHidden()
    finally:
        widget.close()
        widget.deleteLater()


def test_overlay_step_routes_to_review_when_existing_overlays_need_review():
    QApplication.instance() or QApplication([])
    state = AppState()
    state.video.filepath = "/tmp/video.mp4"
    state.overlays = [_overlay()]
    widget = CalibrationGuideWidget()
    emitted = []
    widget.find_keyboard_requested.connect(lambda: emitted.append("find"))
    widget.review_alignment_requested.connect(lambda: emitted.append("review"))
    try:
        widget.update_snapshot(derive_guide_snapshot(state, False))
        assert widget.step_rows[1].primary_button.text() == "Review Alignment"
        widget.step_rows[1].primary_button.click()
        assert emitted == ["review"]
    finally:
        widget.close()
        widget.deleteLater()


def test_control_panel_places_guide_first():
    from synthesia2midi.gui.controls_qt import ControlPanelQt

    QApplication.instance() or QApplication([])
    panel = ControlPanelQt()
    try:
        assert panel.settings_section_rail.item(0).text() == "Guide"
        assert panel.settings_page_widgets[0] is panel.guide_page
        assert panel.tab_widget.widget(0) is panel.settings_page_scroll_areas[0]
    finally:
        panel.close()
        panel.deleteLater()


def test_review_current_alignment_uses_manual_fit_for_manual_overlays():
    from synthesia2midi.gui.calibration_wizard_controller import CalibrationWizardController

    calls = []
    controller = CalibrationWizardController.__new__(CalibrationWizardController)
    controller.app = SimpleNamespace(
        app_state=SimpleNamespace(
            calibration=SimpleNamespace(overlay_generation_source="manual"),
            overlays=[object()],
        ),
        manual_keyboard_fit_controller=SimpleNamespace(
            open=lambda **kwargs: calls.append(kwargs) or True
        ),
    )

    assert controller.review_current_alignment() is True
    assert calls == [{"start_setup": False}]


def test_review_current_auto_alignment_opens_tuning_without_showing_wizard():
    from synthesia2midi.gui.calibration_wizard_controller import CalibrationWizardController

    wizard = SimpleNamespace()
    open_calls = []
    tuning = SimpleNamespace(
        has_editable_context=lambda: True,
        open=lambda current_wizard, **kwargs: open_calls.append((current_wizard, kwargs)) or True,
    )
    controller = CalibrationWizardController.__new__(CalibrationWizardController)
    controller.app = SimpleNamespace(
        app_state=SimpleNamespace(
            calibration=SimpleNamespace(overlay_generation_source="auto"),
            overlays=[object()],
        ),
        calibration_workflow=SimpleNamespace(run_calibration_wizard=lambda: wizard),
    )
    controller.auto_detect_tuning_controller = tuning
    controller.calibration_wizard = None
    controller._pending_assisted_calibration_context = None

    assert controller.review_current_alignment() is True
    assert controller.calibration_wizard is wizard
    assert open_calls[0][0] is wizard
    assert open_calls[0][1]["use_wizard_context"] is False


def test_assisted_scan_from_current_frame_uses_visible_frame_as_baseline():
    from synthesia2midi.gui.calibration_wizard_controller import CalibrationWizardController

    frame = np.ones((4, 5, 3), dtype=np.uint8)
    calls = []
    controller = CalibrationWizardController.__new__(CalibrationWizardController)
    controller.app = SimpleNamespace(
        app_state=SimpleNamespace(video=SimpleNamespace(current_frame_index=23))
    )
    controller._frame_provider_rgb = lambda index: frame if index == 23 else None
    controller._run_assisted_auto_calibration = (
        lambda baseline, index: calls.append((baseline.copy(), index)) or True
    )

    assert controller.run_assisted_calibration_from_current_frame() is True
    assert calls[0][1] == 23
    assert np.array_equal(calls[0][0], frame)
