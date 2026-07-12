import json
from types import SimpleNamespace

import cv2
import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.gui.startup_dialog import StartupDialog
from synthesia2midi.gui.youtube_download_dialog import YouTubeDownloadDialog
from synthesia2midi.core.app_state import AppState
from synthesia2midi.detection.assisted_calibration import (
    AssignedExemplar,
    AssistedCalibrationProposal,
    ExemplarCandidate,
    ExemplarAssignmentResult,
    UnlitFrameAssessment,
)
from synthesia2midi.detection.factory import DetectionFactory
from synthesia2midi.detection.spark_integrated import SparkIntegratedDetection
from synthesia2midi.detection.standard import StandardDetection
from synthesia2midi.gui.auto_detect_tuning_controller import AutoDetectTuningController
from synthesia2midi.gui.calibration_interaction_controller import CalibrationInteractionController
from synthesia2midi.gui.calibration_wizard_controller import CalibrationWizardController
from synthesia2midi.gui.canvas.interaction import CanvasInteraction
from synthesia2midi.gui.main_action_controller import MainActionController
from synthesia2midi.gui.signal_manager import ControlSignalManager
from synthesia2midi.workflows.calibration import CalibrationWorkflow
from synthesia2midi.utils import ffmpeg_helper


def test_startup_youtube_choice_closes_startup_dialog_before_emitting_download_signal():
    QApplication.instance() or QApplication([])
    dialog = StartupDialog()
    emitted_results = []

    dialog.download_from_youtube.connect(lambda: emitted_results.append(dialog.result()))

    dialog._on_youtube_clicked()

    assert emitted_results == [QDialog.Accepted]
    assert dialog.result() == QDialog.Accepted


def test_youtube_download_dialog_closes_before_emitting_load_signal(monkeypatch, tmp_path):
    QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
    dialog = YouTubeDownloadDialog(default_output_dir=str(tmp_path))
    emitted = []

    dialog.video_downloaded.connect(lambda path: emitted.append((path, dialog.result())))

    dialog.on_download_finished("/tmp/downloaded.mp4")

    assert emitted == [("/tmp/downloaded.mp4", QDialog.Accepted)]
    assert dialog.result() == QDialog.Accepted


class RecordingSignal:
    def __init__(self):
        self.connected = []

    def connect(self, slot):
        self.connected.append(slot)

    def emit(self, *args):
        for slot in list(self.connected):
            slot(*args)

    def disconnect(self):
        self.connected.clear()


class DummyControlPanel:
    def __init__(self):
        signal_names = [
            "nav_interval_changed",
            "youtube_video_downloaded",
            "video_to_frames_requested",
            "start_frame_changed",
            "end_frame_changed",
            "detection_threshold_changed",
            "rise_delta_threshold_changed",
            "fall_delta_threshold_changed",
            "histogram_detection_toggled",
            "delta_detection_toggled",
            "winner_takes_black_changed",
            "hand_assignment_toggled",
            "histogram_threshold_changed",
            "similarity_ratio_changed",
            "calibrate_unlit_requested",
            "calibrate_lit_exemplar_requested",
            "exemplar_key_type_enabled_changed",
            "add_additional_color_requested",
            "remove_additional_color_requested",
            "calibration_wizard_requested",
            "refresh_overlay_display_requested",
            "align_white_keys_requested",
            "align_black_keys_requested",
            "manual_fit_requested",
            "overlay_size_adjustment_requested",
            "conversion_requested",
            "midi_touchup_requested",
            "trim_video_requested",
            "spark_roi_selection_requested",
            "spark_roi_visibility_toggled",
            "spark_calibration_requested",
            "auto_spark_calibration_requested",
            "spark_detection_toggled",
            "spark_detection_sensitivity_changed",
            "overlay_type_changed",
            "octave_transpose_changed",
            "fps_override_changed",
            "processing_start_frame_changed",
            "processing_end_frame_changed",
            "overlay_color_changed",
        ]
        for name in signal_names:
            setattr(self, name, RecordingSignal())


def test_ffprobe_frame_rate_is_parsed_without_executing_python(monkeypatch):
    payload = {
        "streams": [
            {
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "__import__('os').system('echo exploited')",
                "nb_frames": "10",
            }
        ]
    }

    monkeypatch.setattr(ffmpeg_helper.shutil, "which", lambda name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        ffmpeg_helper.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(payload)),
    )
    monkeypatch.setattr(
        "builtins.eval",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("eval must not be used")),
    )

    info = ffmpeg_helper.get_video_info("dummy.mp4")

    assert info == {"width": 1920, "height": 1080, "fps": 0.0, "frame_count": "10"}


def test_ffprobe_fractional_frame_rate_is_parsed_as_float(monkeypatch):
    payload = {
        "streams": [
            {"width": 1920, "height": 1080, "r_frame_rate": "30000/1001", "nb_frames": "10"}
        ]
    }

    monkeypatch.setattr(ffmpeg_helper.shutil, "which", lambda name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        ffmpeg_helper.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(payload)),
    )

    info = ffmpeg_helper.get_video_info("dummy.mp4")

    assert info is not None
    assert info["fps"] == 30000 / 1001


def test_calibration_tuning_fallback_unpacks_video_session_frame_tuple():
    frame_bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
    controller = AutoDetectTuningController(
        SimpleNamespace(
            keyboard_canvas=SimpleNamespace(current_frame_rgb=None),
            app_state=SimpleNamespace(video=SimpleNamespace(current_frame_index=7)),
            video_session=SimpleNamespace(get_frame=lambda frame_idx: (True, frame_bgr)),
        )
    )

    frame_rgb = controller.get_current_frame_rgb_for_tuning()

    assert np.array_equal(frame_rgb, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def test_calibration_tuning_fallback_returns_none_when_video_session_read_fails():
    controller = AutoDetectTuningController(
        SimpleNamespace(
            keyboard_canvas=SimpleNamespace(current_frame_rgb=None),
            app_state=SimpleNamespace(video=SimpleNamespace(current_frame_index=7)),
            video_session=SimpleNamespace(get_frame=lambda frame_idx: (False, None)),
        )
    )

    assert controller.get_current_frame_rgb_for_tuning() is None


class DummyWizardForController:
    def __init__(self, emit_signal_name: str, *, result=None):
        self.keyboard_region_selection_requested = RecordingSignal()
        self.edit_current_calibration_requested = RecordingSignal()
        self.emit_signal_name = emit_signal_name
        self.result = result
        self.edit_enabled = None

    def set_edit_current_calibration_enabled(self, enabled, tooltip=None):
        self.edit_enabled = (enabled, tooltip)

    def exec(self):
        getattr(self, self.emit_signal_name).emit()
        return QDialog.Accepted

    def get_auto_detect_tuning_context(self):
        return None


class DummyCalibrationWorkflowForController:
    def __init__(self, wizard):
        self.wizard = wizard
        self.completed = []

    def run_calibration_wizard(self):
        return self.wizard

    def handle_wizard_completed(self, success):
        self.completed.append(success)
        return False

    def apply_template_styles_to_overlays(self):
        raise AssertionError("template styles should not be applied in this test")


class DummyPassiveWizardForPlacement:
    def __init__(self):
        self.keyboard_region_selection_requested = RecordingSignal()
        self.edit_current_calibration_requested = RecordingSignal()
        self.result = False
        self.edit_enabled = None
        self.moved_to = None

    def set_edit_current_calibration_enabled(self, enabled, tooltip=None):
        self.edit_enabled = (enabled, tooltip)

    def frameGeometry(self):
        return QRect(0, 0, 600, 220)

    def move(self, x, y):
        self.moved_to = (x, y)

    def exec(self):
        return QDialog.Rejected

    def get_auto_detect_tuning_context(self):
        return None


class DummyAutoDetectTuningControllerForRestore:
    def __init__(self):
        self.open_kwargs = None
        self.open_calls = 0

    def set_apply_template_styles_callback(self, _callback):
        pass

    def has_editable_context(self):
        return True

    def open(self, wizard, **kwargs):
        self.open_calls += 1
        self.open_kwargs = {"wizard": wizard, **kwargs}
        return True

    def cache_context(self, context):
        self.cached_context = context


class DummySuccessfulWizardForController:
    def __init__(self, *, manual_overlays_generated=False):
        self.keyboard_region_selection_requested = RecordingSignal()
        self.edit_current_calibration_requested = RecordingSignal()
        self.result = True
        self.manual_overlays_generated = manual_overlays_generated

    def set_edit_current_calibration_enabled(self, enabled, tooltip=None):
        self.edit_enabled = (enabled, tooltip)

    def exec(self):
        return QDialog.Accepted


class DummySuccessfulCalibrationWorkflowForController:
    def __init__(self, wizard):
        self.wizard = wizard
        self.completed = []
        self.template_calls = 0

    def run_calibration_wizard(self):
        return self.wizard

    def handle_wizard_completed(self, success):
        self.completed.append(success)
        return success

    def apply_template_styles_to_overlays(self):
        self.template_calls += 1


class DummyShowOverlaysAction:
    def __init__(self):
        self.checked_values = []

    def setChecked(self, value):
        self.checked_values.append(value)


def _make_assigned_exemplar(slot, *, rgb=None, enabled=True):
    return AssignedExemplar(
        slot=slot,
        rgb=rgb,
        hist=np.array([1.0], dtype=np.float32) if rgb is not None and enabled else None,
        source=None,
        enabled=enabled,
    )


def _make_exemplar_candidate(slot_color, rgb, *, frame_index, key_id):
    hsv = cv2.cvtColor(np.array([[rgb]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
    return ExemplarCandidate(
        slot_color=slot_color,
        key_id=key_id,
        note_label="C4",
        frame_index=frame_index,
        rgb=rgb,
        hsv=(float(hsv[0]), float(hsv[1]), float(hsv[2])),
        delta_from_unlit=100.0,
        confidence=0.9,
        hist=np.array([1.0], dtype=np.float32),
    )


def _make_assisted_proposal(*, candidate_count, assignments, canceled=False, family_count=0):
    return AssistedCalibrationProposal(
        baseline_frame_index=3,
        unlit_assessment=UnlitFrameAssessment(status="clean"),
        assignment_result=ExemplarAssignmentResult(
            assignments=assignments,
            missing_slots=tuple(),
            disabled_slots=tuple(slot for slot, assignment in assignments.items() if not assignment.enabled),
            family_count=family_count,
            confidence=1.0 if candidate_count else 0.0,
        ),
        scanned_frame_count=4,
        candidate_count=candidate_count,
        canceled=canceled,
    )


def _make_assisted_calibration_controller(*, save_log):
    app_state = AppState()
    app_state.video.current_frame_index = 3
    app_state.overlays = [
        OverlayConfig(
            key_id=1,
            note_octave=4,
            note_name_in_octave="C",
            x=0,
            y=0,
            width=4,
            height=4,
            key_type="LW",
        )
    ]
    app = SimpleNamespace(
        app_state=app_state,
        video_loading_workflow=SimpleNamespace(save_current_config=lambda: save_log.append("save") or True),
        video_session=SimpleNamespace(total_frames=12, get_frame=lambda _index: (True, np.full((8, 8, 3), 240, dtype=np.uint8))),
    )
    return CalibrationWizardController(app, DummyAutoDetectTuningControllerForRestore())


def _patch_assisted_dialog(monkeypatch, decision):
    class FakeAssistedDialog:
        def __init__(self, proposal, parent=None):
            self.proposal = proposal
            self.parent = parent
            self.decision = decision

        def exec(self):
            return QDialog.Accepted if decision.value == "use" else QDialog.Rejected

    monkeypatch.setattr(
        "synthesia2midi.gui.calibration_wizard_controller.AssistedCalibrationDialog",
        FakeAssistedDialog,
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy assisted-calibration question dialog was used")
        ),
    )


def test_calibration_wizard_controller_keeps_wizard_for_keyboard_region_selection():
    wizard = DummyWizardForController("keyboard_region_selection_requested")
    workflow = DummyCalibrationWorkflowForController(wizard)
    selected_signal = RecordingSignal()
    cancelled_signal = RecordingSignal()
    interaction = SimpleNamespace(
        keyboard_region_selected=selected_signal,
        selection_cancelled=cancelled_signal,
        enter_keyboard_region_selection_mode=lambda: setattr(interaction, "entered", True),
        entered=False,
    )
    cursor_changes = []
    app = SimpleNamespace(
        app_state=SimpleNamespace(overlays=[], video=SimpleNamespace(current_frame_index=7)),
        calibration_workflow=workflow,
        control_panel=SimpleNamespace(),
        keyboard_canvas=SimpleNamespace(
            current_frame_rgb=None,
            interaction=interaction,
            setCursor=lambda cursor: cursor_changes.append(cursor),
        ),
        video_loading_workflow=None,
        video_session=None,
    )
    controller = CalibrationWizardController(app)

    controller.run_calibration_wizard()

    assert controller.calibration_wizard is wizard
    assert controller._keyboard_region_requested is True
    assert not hasattr(wizard, "_keyboard_region_requested")
    assert workflow.completed == []
    assert interaction.entered is True
    assert selected_signal.connected == [controller._handle_keyboard_region_selected]
    assert cursor_changes == [Qt.CrossCursor]
    assert cancelled_signal.connected == [controller._handle_canvas_selection_cancelled]

    cancelled_signal.emit("keyboard_region")

    assert controller.calibration_wizard is None
    assert controller._keyboard_region_requested is False
    assert selected_signal.connected == []
    assert cursor_changes == [Qt.CrossCursor, Qt.ArrowCursor]


def test_calibration_wizard_controller_places_wizard_in_upper_left_safe_zone():
    wizard = DummyPassiveWizardForPlacement()
    workflow = DummyCalibrationWorkflowForController(wizard)
    convert_button = SimpleNamespace(setEnabled=lambda enabled: setattr(convert_button, "enabled", enabled))
    display_calls = []
    app = SimpleNamespace(
        app_state=SimpleNamespace(overlays=[], video=SimpleNamespace(current_frame_index=7)),
        calibration_workflow=workflow,
        control_panel=SimpleNamespace(convert_button=convert_button),
        keyboard_canvas=SimpleNamespace(
            current_frame_rgb=None,
            display_frame=lambda frame_idx: display_calls.append(frame_idx),
        ),
        video_loading_workflow=None,
        video_session=None,
        screen=lambda: SimpleNamespace(availableGeometry=lambda: QRect(0, 24, 1440, 876)),
    )
    controller = CalibrationWizardController(app)

    controller.run_calibration_wizard()

    x, y = wizard.moved_to
    assert x <= 80
    assert 40 <= y <= 110


def test_calibration_wizard_controller_requests_settings_restore_after_tuning():
    app = SimpleNamespace()
    tuning_controller = DummyAutoDetectTuningControllerForRestore()
    controller = CalibrationWizardController(app, tuning_controller)
    wizard = SimpleNamespace()
    controller.calibration_wizard = wizard

    assert controller._open_auto_detect_tuning_dialog() is True

    assert tuning_controller.open_kwargs["wizard"] is wizard
    assert tuning_controller.open_kwargs["restore_settings_on_finish"] is True


def test_calibration_wizard_controller_shows_manual_overlays_after_success():
    wizard = DummySuccessfulWizardForController()
    workflow = DummySuccessfulCalibrationWorkflowForController(wizard)
    convert_button = SimpleNamespace(
        setEnabled=lambda enabled: setattr(convert_button, "enabled", enabled)
    )
    display_calls = []
    draw_calls = []
    app = SimpleNamespace(
        app_state=SimpleNamespace(
            overlays=[SimpleNamespace(key_id=0)],
            ui=SimpleNamespace(show_overlays=False),
            video=SimpleNamespace(current_frame_index=11),
        ),
        calibration_workflow=workflow,
        control_panel=SimpleNamespace(
            convert_button=convert_button,
            _can_convert=lambda: True,
        ),
        keyboard_canvas=SimpleNamespace(
            draw_overlays=lambda: draw_calls.append(True),
            display_frame=lambda frame_idx: display_calls.append(frame_idx),
        ),
        show_overlays_action=DummyShowOverlaysAction(),
        video_loading_workflow=None,
        video_session=None,
    )
    controller = CalibrationWizardController(app)

    controller.run_calibration_wizard()

    assert workflow.completed == [True]
    assert workflow.template_calls == 1
    assert app.app_state.ui.show_overlays is True
    assert app.show_overlays_action.checked_values == [True]
    assert convert_button.enabled is True
    assert draw_calls == [True]
    assert display_calls == [11]


def test_calibration_wizard_controller_opens_manual_fit_after_manual_overlay_generation():
    wizard = DummySuccessfulWizardForController(manual_overlays_generated=True)
    workflow = DummySuccessfulCalibrationWorkflowForController(wizard)
    manual_fit_calls = []
    convert_button = SimpleNamespace(setEnabled=lambda _enabled: None)
    app = SimpleNamespace(
        app_state=SimpleNamespace(
            overlays=[SimpleNamespace(key_id=0)],
            ui=SimpleNamespace(show_overlays=False),
            video=SimpleNamespace(current_frame_index=11),
        ),
        calibration_workflow=workflow,
        control_panel=SimpleNamespace(
            convert_button=convert_button,
            _can_convert=lambda: True,
        ),
        keyboard_canvas=SimpleNamespace(
            draw_overlays=lambda: None,
            display_frame=lambda _frame_idx: None,
        ),
        show_overlays_action=DummyShowOverlaysAction(),
        manual_keyboard_fit_controller=SimpleNamespace(open=lambda **kwargs: manual_fit_calls.append(kwargs)),
        video_loading_workflow=None,
        video_session=None,
    )
    controller = CalibrationWizardController(app)

    controller.run_calibration_wizard()

    assert manual_fit_calls == [{"start_setup": True}]
    assert controller.calibration_wizard is None


def test_calibration_wizard_controller_resets_edit_flag_when_tuning_context_missing(monkeypatch):
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)
    wizard = DummyWizardForController("edit_current_calibration_requested")
    workflow = DummyCalibrationWorkflowForController(wizard)
    convert_button = SimpleNamespace(setEnabled=lambda enabled: setattr(convert_button, "enabled", enabled))
    display_calls = []
    app = SimpleNamespace(
        app_state=SimpleNamespace(overlays=[], video=SimpleNamespace(current_frame_index=7)),
        calibration_workflow=workflow,
        control_panel=SimpleNamespace(convert_button=convert_button),
        keyboard_canvas=SimpleNamespace(
            current_frame_rgb=None,
            display_frame=lambda frame_idx: display_calls.append(frame_idx),
        ),
        video_loading_workflow=None,
        video_session=None,
    )
    controller = CalibrationWizardController(app)

    controller.run_calibration_wizard()

    assert controller.calibration_wizard is None
    assert controller._edit_current_calibration_requested is False
    assert not hasattr(wizard, "_edit_current_calibration_requested")
    assert workflow.completed == [False]
    assert convert_button.enabled is False
    assert display_calls == [7]


def test_edit_current_manual_calibration_opens_manual_fit_instead_of_auto_tuning(monkeypatch):
    wizard = DummyWizardForController("edit_current_calibration_requested")
    workflow = DummyCalibrationWorkflowForController(wizard)
    tuning_controller = DummyAutoDetectTuningControllerForRestore()
    manual_fit_calls = []
    app = SimpleNamespace(
        app_state=SimpleNamespace(
            overlays=[SimpleNamespace(key_id=1)],
            calibration=SimpleNamespace(overlay_generation_source="manual"),
            video=SimpleNamespace(current_frame_index=7),
        ),
        calibration_workflow=workflow,
        control_panel=SimpleNamespace(),
        keyboard_canvas=SimpleNamespace(current_frame_rgb=None),
        manual_keyboard_fit_controller=SimpleNamespace(open=lambda **kwargs: manual_fit_calls.append(kwargs)),
        video_loading_workflow=None,
        video_session=None,
    )
    controller = CalibrationWizardController(app, tuning_controller)

    controller.run_calibration_wizard()

    assert wizard.edit_enabled[0] is True
    assert "Manual Fit" in wizard.edit_enabled[1]
    assert manual_fit_calls == [{"start_setup": False}]
    assert tuning_controller.open_calls == 0
    assert controller.calibration_wizard is None


def test_auto_detect_keyboard_region_marks_overlay_generation_source_auto():
    class _Wizard:
        def handle_keyboard_region_selected(self, *_args):
            app.app_state.overlays = [SimpleNamespace(key_id=1)]
            return True

        def get_auto_detect_tuning_context(self):
            return {"frame_rgb": object(), "keyboard_roi": (1, 2, 3, 4)}

    app = SimpleNamespace(
        app_state=SimpleNamespace(
            overlays=[],
            calibration=SimpleNamespace(overlay_generation_source=None),
            ui=SimpleNamespace(show_overlays=False),
            video=SimpleNamespace(current_frame_index=7),
        ),
        calibration_workflow=SimpleNamespace(apply_template_styles_to_overlays=lambda: None),
        control_panel=SimpleNamespace(
            convert_button=SimpleNamespace(setEnabled=lambda _enabled: None),
            _can_convert=lambda: True,
            update_controls_from_state=lambda: None,
            update_trim_controls_from_state=lambda: None,
            update_selected_overlay_display=lambda: None,
        ),
        keyboard_canvas=SimpleNamespace(
            setCursor=lambda _cursor: None,
            display_frame=lambda _frame_idx: None,
        ),
        show_overlays_action=DummyShowOverlaysAction(),
    )
    tuning_controller = DummyAutoDetectTuningControllerForRestore()
    controller = CalibrationWizardController(app, tuning_controller)
    controller.calibration_wizard = _Wizard()

    controller._handle_keyboard_region_selected(1, 2, 3, 4)

    assert app.app_state.calibration.overlay_generation_source == "auto"


def test_keyboard_region_selection_defers_assisted_calibration_until_tuning_save(monkeypatch):
    QApplication.instance() or QApplication([])
    applied = []
    saved = []

    class _Wizard:
        auto_detect_source_frame_rgb = np.full((8, 8, 3), (245, 245, 235), dtype=np.uint8)

        def handle_keyboard_region_selected(self, *_args):
            app.app_state.overlays = [
                OverlayConfig(
                    key_id=1,
                    note_octave=4,
                    note_name_in_octave="C",
                    x=0,
                    y=0,
                    width=4,
                    height=4,
                    key_type="LW",
                )
            ]
            return True

        def get_auto_detect_tuning_context(self):
            return {"frame_rgb": self.auto_detect_source_frame_rgb, "keyboard_roi": (1, 2, 3, 4)}

    app = SimpleNamespace(
        app_state=AppState(),
        calibration_workflow=SimpleNamespace(apply_template_styles_to_overlays=lambda: None),
        control_panel=SimpleNamespace(
            convert_button=SimpleNamespace(setEnabled=lambda _enabled: None),
            _can_convert=lambda: True,
            update_controls_from_state=lambda: None,
            update_trim_controls_from_state=lambda: None,
            update_selected_overlay_display=lambda: None,
        ),
        keyboard_canvas=SimpleNamespace(
            setCursor=lambda _cursor: None,
            display_frame=lambda _frame_idx: None,
            update=lambda: None,
        ),
        show_overlays_action=DummyShowOverlaysAction(),
        video_loading_workflow=SimpleNamespace(save_current_config=lambda: saved.append("save") or True),
        video_session=SimpleNamespace(
            total_frames=12,
            get_frame=lambda _index: (True, np.full((8, 8, 3), (235, 245, 255), dtype=np.uint8)),
        ),
    )
    app.app_state.video.current_frame_index = 3
    tuning_controller = DummyAutoDetectTuningControllerForRestore()
    controller = CalibrationWizardController(app, tuning_controller)
    controller.calibration_wizard = _Wizard()

    monkeypatch.setattr(
        "synthesia2midi.gui.calibration_wizard_controller.apply_assisted_calibration_proposal",
        lambda app_state, proposal: applied.append(proposal),
    )
    monkeypatch.setattr(
        "synthesia2midi.gui.calibration_wizard_controller.build_assisted_calibration_proposal",
        lambda *_args, **_kwargs: _make_assisted_proposal(
            candidate_count=1,
            assignments={"LW": _make_assigned_exemplar("LW", rgb=(10, 20, 30))},
            family_count=1,
        ),
    )
    from synthesia2midi.gui.assisted_calibration_dialog import AssistedCalibrationDecision

    _patch_assisted_dialog(monkeypatch, AssistedCalibrationDecision.USE)

    controller._handle_keyboard_region_selected(1, 2, 3, 4)

    assert tuning_controller.open_calls == 1
    assert applied == []
    assert saved == []

    tuning_controller.open_kwargs["on_dialog_finished"](QDialog.Accepted)

    assert applied
    assert saved == ["save"]


def test_assisted_calibration_unlit_warning_cancel_skips_apply(monkeypatch):
    QApplication.instance() or QApplication([])
    applied = []
    app = SimpleNamespace(
        app_state=AppState(),
        video_session=SimpleNamespace(total_frames=4, get_frame=lambda _index: (True, np.zeros((8, 8, 3), dtype=np.uint8))),
        video_loading_workflow=SimpleNamespace(save_current_config=lambda: True),
        control_panel=SimpleNamespace(update_controls_from_state=lambda: None),
    )
    app.app_state.overlays = [
        OverlayConfig(key_id=i, note_octave=4, note_name_in_octave="E", x=i * 2, y=0, width=2, height=2, key_type="LW")
        for i in range(4)
    ]
    baseline = np.full((8, 12, 3), (245, 245, 235), dtype=np.uint8)
    baseline[0:2, 4:6] = (240, 140, 40)
    controller = CalibrationWizardController(app, DummyAutoDetectTuningControllerForRestore())

    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.StandardButton.Cancel)
    monkeypatch.setattr(
        "synthesia2midi.gui.calibration_wizard_controller.build_assisted_calibration_proposal",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("scan should not run after warning cancel")),
    )
    monkeypatch.setattr(
        "synthesia2midi.gui.calibration_wizard_controller.apply_assisted_calibration_proposal",
        lambda app_state, proposal: applied.append(proposal),
    )

    assert controller._run_assisted_auto_calibration(baseline, 0) is False
    assert applied == []


def test_assisted_calibration_scan_cancel_processes_events_and_stops(monkeypatch):
    QApplication.instance() or QApplication([])
    save_log = []
    controller = _make_assisted_calibration_controller(save_log=save_log)
    processed_events = []
    overlay = controller.app_state.overlays[0]
    overlay.unlit_reference_color = (1, 2, 3)
    overlay.unlit_hist = np.array([0.25, 0.75], dtype=np.float32)

    class FakeProgressDialog:
        latest = None

        def __init__(self, *_args, **_kwargs):
            self.maximum = None
            self.values = []
            self.closed = False
            self.canceled = False
            FakeProgressDialog.latest = self

        def setWindowTitle(self, _title):
            pass

        def setMinimumDuration(self, _duration):
            pass

        def setMaximum(self, value):
            self.maximum = value

        def setValue(self, value):
            self.values.append(value)

        def wasCanceled(self):
            return self.canceled

        def close(self):
            self.closed = True

    def fake_process_events():
        processed_events.append("processed")
        FakeProgressDialog.latest.canceled = True

    def fake_build_proposal(*_args, progress_callback=None, **_kwargs):
        keep_scanning = progress_callback(4, 11)
        return _make_assisted_proposal(
            candidate_count=1,
            assignments={"LW": _make_assigned_exemplar("LW", rgb=(10, 20, 30))},
            canceled=not keep_scanning,
            family_count=1,
        )

    def unexpected_question(*_args, **_kwargs):
        raise AssertionError("canceled scan should not reach confirmation")

    monkeypatch.setattr(
        "synthesia2midi.gui.calibration_wizard_controller.QProgressDialog",
        FakeProgressDialog,
    )
    monkeypatch.setattr(QApplication, "processEvents", staticmethod(fake_process_events))
    monkeypatch.setattr(
        "synthesia2midi.gui.calibration_wizard_controller.build_assisted_calibration_proposal",
        fake_build_proposal,
    )
    monkeypatch.setattr(QMessageBox, "question", unexpected_question)

    result = controller._run_assisted_auto_calibration(
        np.full((8, 8, 3), 245, dtype=np.uint8),
        3,
    )

    assert result is False
    assert processed_events == ["processed"]
    assert FakeProgressDialog.latest.values == [4]
    assert FakeProgressDialog.latest.closed is True
    assert save_log == []
    assert overlay.unlit_reference_color == (1, 2, 3)
    assert np.array_equal(overlay.unlit_hist, np.array([0.25, 0.75], dtype=np.float32))


def test_assisted_calibration_no_result_does_not_apply_or_save(monkeypatch):
    QApplication.instance() or QApplication([])
    save_log = []
    controller = _make_assisted_calibration_controller(save_log=save_log)
    info_calls = []
    applied = []
    detection = controller.app_state.detection
    overlay = controller.app_state.overlays[0]
    overlay.unlit_reference_color = (11, 22, 33)
    overlay.unlit_hist = np.array([0.1, 0.9], dtype=np.float32)
    original_enabled = dict(detection.exemplar_key_type_enabled)
    original_colors = dict(detection.exemplar_lit_colors)
    original_hists = dict(detection.exemplar_lit_histograms)

    proposal = _make_assisted_proposal(
        candidate_count=2,
        assignments={
            "LW": _make_assigned_exemplar("LW", rgb=None, enabled=False),
            "LB": _make_assigned_exemplar("LB", rgb=None, enabled=False),
            "RW": _make_assigned_exemplar("RW", rgb=None, enabled=False),
            "RB": _make_assigned_exemplar("RB", rgb=None, enabled=False),
        },
    )

    def unexpected_question(*_args, **_kwargs):
        raise AssertionError("empty proposals should not reach confirmation")

    monkeypatch.setattr(
        "synthesia2midi.gui.calibration_wizard_controller.build_assisted_calibration_proposal",
        lambda *_args, **_kwargs: proposal,
    )
    monkeypatch.setattr(
        "synthesia2midi.gui.calibration_wizard_controller.apply_assisted_calibration_proposal",
        lambda *_args, **_kwargs: applied.append("applied"),
    )
    monkeypatch.setattr(QMessageBox, "question", unexpected_question)
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: info_calls.append((args, kwargs)))

    result = controller._run_assisted_auto_calibration(
        np.full((8, 8, 3), 245, dtype=np.uint8),
        3,
    )

    assert result is False
    assert applied == []
    assert save_log == []
    assert len(info_calls) == 1
    assert "No lit examples were found" in info_calls[0][0][2]
    assert dict(detection.exemplar_key_type_enabled) == original_enabled
    assert dict(detection.exemplar_lit_colors) == original_colors
    assert dict(detection.exemplar_lit_histograms) == original_hists
    assert overlay.unlit_reference_color == (11, 22, 33)
    assert np.array_equal(overlay.unlit_hist, np.array([0.1, 0.9], dtype=np.float32))


def test_assisted_calibration_decline_does_not_apply_or_save(monkeypatch):
    QApplication.instance() or QApplication([])
    save_log = []
    controller = _make_assisted_calibration_controller(save_log=save_log)
    applied = []
    overlay = controller.app_state.overlays[0]
    overlay.unlit_reference_color = (9, 8, 7)
    overlay.unlit_hist = np.array([0.6, 0.4], dtype=np.float32)

    monkeypatch.setattr(
        "synthesia2midi.gui.calibration_wizard_controller.build_assisted_calibration_proposal",
        lambda *_args, **_kwargs: _make_assisted_proposal(
            candidate_count=1,
            assignments={"LW": _make_assigned_exemplar("LW", rgb=(10, 20, 30))},
            family_count=1,
        ),
    )
    monkeypatch.setattr(
        "synthesia2midi.gui.calibration_wizard_controller.apply_assisted_calibration_proposal",
        lambda *_args, **_kwargs: applied.append("applied"),
    )
    from synthesia2midi.gui.assisted_calibration_dialog import AssistedCalibrationDecision

    _patch_assisted_dialog(monkeypatch, AssistedCalibrationDecision.KEEP)

    result = controller._run_assisted_auto_calibration(
        np.full((8, 8, 3), 245, dtype=np.uint8),
        3,
    )

    assert result is False
    assert applied == []
    assert save_log == []
    assert overlay.unlit_reference_color == (9, 8, 7)
    assert np.array_equal(overlay.unlit_hist, np.array([0.6, 0.4], dtype=np.float32))


def test_assisted_calibration_retry_restores_complete_prior_calibration(monkeypatch):
    from synthesia2midi.gui.assisted_calibration_dialog import AssistedCalibrationDecision

    QApplication.instance() or QApplication([])
    save_log = []
    controller = _make_assisted_calibration_controller(save_log=save_log)
    state = controller.app_state
    overlay = state.overlays[0]
    overlay.unlit_reference_color = (9, 8, 7)
    overlay.unlit_hist = np.array([0.6, 0.4], dtype=np.float32)
    state.detection.exemplar_key_type_enabled.update({"LW": True, "LB": False})
    state.detection.exemplar_lit_colors.update({"LW": (1, 2, 3), "LB": None})
    state.detection.exemplar_lit_histograms.update(
        {"LW": np.array([0.2, 0.8], dtype=np.float32), "LB": None}
    )

    monkeypatch.setattr(
        "synthesia2midi.gui.calibration_wizard_controller.build_assisted_calibration_proposal",
        lambda *_args, **_kwargs: _make_assisted_proposal(
            candidate_count=1,
            assignments={"LW": _make_assigned_exemplar("LW", rgb=(100, 110, 120))},
            family_count=1,
        ),
    )
    _patch_assisted_dialog(monkeypatch, AssistedCalibrationDecision.RETRY)

    assert controller._run_assisted_auto_calibration(
        np.full((8, 8, 3), 245, dtype=np.uint8), 3
    ) is False
    assert overlay.unlit_reference_color == (9, 8, 7)
    assert np.array_equal(overlay.unlit_hist, np.array([0.6, 0.4], dtype=np.float32))
    assert state.detection.exemplar_key_type_enabled["LB"] is False
    assert state.detection.exemplar_lit_colors["LW"] == (1, 2, 3)
    assert np.array_equal(
        state.detection.exemplar_lit_histograms["LW"],
        np.array([0.2, 0.8], dtype=np.float32),
    )
    assert save_log == []


def test_assisted_calibration_accept_preserves_prior_slots_not_found_by_scan(monkeypatch):
    from synthesia2midi.gui.assisted_calibration_dialog import AssistedCalibrationDecision

    QApplication.instance() or QApplication([])
    save_log = []
    controller = _make_assisted_calibration_controller(save_log=save_log)
    refresh_calls = []
    controller.app.control_panel = SimpleNamespace(
        _update_conversion_readiness_display=lambda: refresh_calls.append("refresh")
    )
    state = controller.app_state
    state.detection.exemplar_key_type_enabled["LB"] = True
    state.detection.exemplar_lit_colors["LB"] = (4, 5, 6)
    state.detection.exemplar_lit_histograms["LB"] = np.array([0.3, 0.7], dtype=np.float32)
    proposal = _make_assisted_proposal(
        candidate_count=1,
        assignments={
            "LW": _make_assigned_exemplar("LW", rgb=(100, 110, 120)),
            "LB": _make_assigned_exemplar("LB", rgb=None, enabled=False),
        },
        family_count=1,
    )
    monkeypatch.setattr(
        "synthesia2midi.gui.calibration_wizard_controller.build_assisted_calibration_proposal",
        lambda *_args, **_kwargs: proposal,
    )
    _patch_assisted_dialog(monkeypatch, AssistedCalibrationDecision.USE)

    assert controller._run_assisted_auto_calibration(
        np.full((8, 8, 3), 245, dtype=np.uint8), 3
    ) is True
    assert state.detection.exemplar_lit_colors["LW"] == (100, 110, 120)
    assert state.detection.exemplar_key_type_enabled["LB"] is True
    assert state.detection.exemplar_lit_colors["LB"] == (4, 5, 6)
    assert np.array_equal(
        state.detection.exemplar_lit_histograms["LB"],
        np.array([0.3, 0.7], dtype=np.float32),
    )
    assert save_log == ["save"]
    assert refresh_calls == ["refresh"]


def test_assisted_calibration_controller_preserves_saved_family_identity_with_reversed_evidence(
    monkeypatch,
):
    from synthesia2midi.gui.assisted_calibration_dialog import AssistedCalibrationDecision

    QApplication.instance() or QApplication([])
    controller = _make_assisted_calibration_controller(save_log=[])
    detection = controller.app_state.detection
    saved_colors = {
        "LW": (70, 130, 230),
        "LB": (45, 95, 185),
        "RW": (235, 65, 65),
        "RB": (185, 35, 35),
    }
    detection.exemplar_lit_colors.update(saved_colors)

    candidates = list(
        reversed(
            [
                _make_exemplar_candidate("W", saved_colors["RW"], frame_index=10, key_id=1),
                _make_exemplar_candidate("W", saved_colors["RW"], frame_index=20, key_id=1),
                _make_exemplar_candidate("B", saved_colors["RB"], frame_index=11, key_id=2),
                _make_exemplar_candidate("B", saved_colors["RB"], frame_index=21, key_id=2),
                _make_exemplar_candidate("W", saved_colors["LW"], frame_index=40, key_id=3),
                _make_exemplar_candidate("W", saved_colors["LW"], frame_index=50, key_id=3),
                _make_exemplar_candidate("B", saved_colors["LB"], frame_index=41, key_id=4),
                _make_exemplar_candidate("B", saved_colors["LB"], frame_index=51, key_id=4),
            ]
        )
    )

    class FakeProgressDialog:
        def __init__(self, *_args, **_kwargs):
            pass

        def setWindowTitle(self, _title):
            pass

        def setMinimumDuration(self, _duration):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        "synthesia2midi.gui.calibration_wizard_controller.QProgressDialog",
        FakeProgressDialog,
    )
    monkeypatch.setattr(
        "synthesia2midi.detection.assisted_calibration.scan_lit_exemplar_candidates",
        lambda *_args, **_kwargs: (candidates, len(candidates), False),
    )
    _patch_assisted_dialog(monkeypatch, AssistedCalibrationDecision.USE)

    assert controller._run_assisted_auto_calibration(
        np.full((8, 8, 3), 245, dtype=np.uint8), 3
    ) is True
    assert detection.exemplar_lit_colors["LW"] == saved_colors["LW"]
    assert detection.exemplar_lit_colors["LB"] == saved_colors["LB"]
    assert detection.exemplar_lit_colors["RW"] == saved_colors["RW"]
    assert detection.exemplar_lit_colors["RB"] == saved_colors["RB"]


def test_main_action_controller_delegates_histogram_and_similarity_thresholds():
    calls = []
    detection_manager = SimpleNamespace(
        set_histogram_threshold=lambda value: calls.append(("histogram", value)),
        set_similarity_ratio=lambda value: calls.append(("similarity", value)),
    )
    controller = MainActionController(SimpleNamespace(detection_manager=detection_manager))

    controller.handle_histogram_threshold_change(0.42)
    controller.handle_similarity_ratio_change(0.73)

    assert calls == [("histogram", 0.42), ("similarity", 0.73)]


def test_signal_manager_wires_histogram_and_similarity_slider_signals():
    cp = DummyControlPanel()
    main_action_controller = SimpleNamespace(
        handle_detection_threshold_change=lambda value: None,
        handle_rise_delta_threshold_change=lambda value: None,
        handle_fall_delta_threshold_change=lambda value: None,
        toggle_hist_detection=lambda: None,
        toggle_delta_detection=lambda: None,
        toggle_winner_takes_black=lambda value: None,
        handle_hand_assignment_toggle=lambda value: None,
        handle_histogram_threshold_change=lambda value: None,
        handle_similarity_ratio_change=lambda value: None,
        handle_calibrate_unlit_all_keys=lambda: None,
        handle_calibrate_lit_exemplar_key_start=lambda value: None,
        handle_exemplar_key_type_enabled_change=lambda key_type, enabled: None,
        handle_add_additional_color=lambda: None,
        handle_remove_additional_color=lambda family_number: None,
        handle_refresh_selected_overlay_display=lambda: None,
        handle_align_white_keys_to_selected=lambda: None,
        handle_align_black_keys_to_selected=lambda: None,
        handle_manual_fit_request=lambda: None,
        handle_overlay_size_adjustment=lambda key_color, dimension, delta: None,
        handle_octave_transpose_change=lambda value: None,
        handle_fps_override_change=lambda value: None,
        handle_overlay_color_change=lambda value: None,
    )
    detection_manager = SimpleNamespace(
        set_detection_threshold=lambda value: None,
        set_rise_delta_threshold=lambda value: None,
        set_fall_delta_threshold=lambda value: None,
        set_histogram_detection_enabled=lambda value: None,
        set_delta_detection_enabled=lambda value: None,
        set_winner_takes_black_enabled=lambda value: None,
        set_hand_assignment_enabled=lambda value: None,
        set_histogram_threshold=lambda value: None,
        set_similarity_ratio=lambda value: None,
    )
    mw = SimpleNamespace(
        main_action_controller=main_action_controller,
        detection_manager=detection_manager,
        video_session_ui_controller=SimpleNamespace(
            update_nav_interval=lambda value: None,
            handle_youtube_video_downloaded=lambda value: None,
            handle_video_to_frames_request=lambda: None,
            handle_start_frame_change=lambda value: None,
            handle_end_frame_change=lambda value: None,
            handle_trim_video_request=lambda: None,
            handle_processing_start_frame_change=lambda value: None,
            handle_processing_end_frame_change=lambda value: None,
        ),
        frame_slider=SimpleNamespace(valueChanged=RecordingSignal()),
        video_controls=SimpleNamespace(on_frame_slider_changed=lambda value: None),
        calibration_wizard_controller=SimpleNamespace(run_calibration_wizard=lambda: None),
        midi_conversion_controller=SimpleNamespace(start_conversion_process=lambda: None),
        midi_touchup_controller=SimpleNamespace(open_from_picker=lambda: None),
        calibration_effects_controller=SimpleNamespace(
            spark=SimpleNamespace(
                select_spark_roi=lambda: None,
                set_spark_roi_visible=lambda value: None,
                request_spark_calibration=lambda value: None,
                start_auto_spark_calibration=lambda value: None,
                set_spark_detection_enabled=lambda value: None,
                set_spark_detection_sensitivity=lambda value: None,
            ),
            overlay=SimpleNamespace(handle_overlay_type_change=lambda value: None),
        ),
    )

    ControlSignalManager(cp, mw)

    assert getattr(cp, "calibration_wizard_requested").connected == [
        mw.calibration_wizard_controller.run_calibration_wizard
    ]
    assert cp.histogram_threshold_changed.connected == [
        detection_manager.set_histogram_threshold
    ]
    assert cp.similarity_ratio_changed.connected == [
        detection_manager.set_similarity_ratio
    ]
    assert cp.manual_fit_requested.connected == [
        main_action_controller.handle_manual_fit_request
    ]
    assert cp.spark_roi_selection_requested.connected == [
        mw.calibration_effects_controller.spark.select_spark_roi
    ]
    assert cp.spark_roi_visibility_toggled.connected == [
        mw.calibration_effects_controller.spark.set_spark_roi_visible
    ]
    assert cp.spark_calibration_requested.connected == [
        mw.calibration_effects_controller.spark.request_spark_calibration
    ]
    assert cp.auto_spark_calibration_requested.connected == [
        mw.calibration_effects_controller.spark.start_auto_spark_calibration
    ]
    assert cp.spark_detection_toggled.connected == [
        mw.calibration_effects_controller.spark.set_spark_detection_enabled
    ]
    assert cp.spark_detection_sensitivity_changed.connected == [
        mw.calibration_effects_controller.spark.set_spark_detection_sensitivity
    ]
    assert cp.overlay_type_changed.connected == [
        mw.calibration_effects_controller.overlay.handle_overlay_type_change
    ]


class RecordingEffectController:
    def __init__(self):
        self.calls = []

    def capture_spark_overlay_calibration(self, overlay, calibration_mode):
        self.calls.append((overlay, calibration_mode))

    def capture_shadow_overlay_calibration(self, overlay, calibration_mode):
        self.calls.append((overlay, calibration_mode))


def _calibration_interaction_app_for_effect_dispatch(calibration_mode):
    app_state = AppState()
    app_state.calibration.calibration_mode = calibration_mode
    app_state.overlays = [
        OverlayConfig(
            key_id=7,
            note_octave=4,
            note_name_in_octave="C",
            x=0,
            y=0,
            width=10,
            height=10,
            key_type="LW",
        )
    ]
    app_state.ui.selected_overlay_id = None
    spark = RecordingEffectController()
    shadow = RecordingEffectController()
    app = SimpleNamespace(
        app_state=app_state,
        calibration_effects_controller=SimpleNamespace(spark=spark, shadow=shadow),
        control_panel=SimpleNamespace(update_selected_overlay_display=lambda: None),
    )
    return app, spark, shadow


def test_calibration_interaction_dispatches_spark_overlay_to_focused_controller_without_app_wrapper():
    app, spark, shadow = _calibration_interaction_app_for_effect_dispatch("spark_bar_only")
    controller = CalibrationInteractionController(app)

    controller._handle_overlay_selection(7)

    assert spark.calls == [(app.app_state.overlays[0], "spark_bar_only")]
    assert shadow.calls == []
    assert app.app_state.ui.selected_overlay_id == 7
    assert not hasattr(app, "_capture_spark_overlay_calibration")


def test_calibration_interaction_dispatches_shadow_overlay_to_focused_controller_without_app_wrapper():
    app, spark, shadow = _calibration_interaction_app_for_effect_dispatch("shadow_lw_pressed")
    controller = CalibrationInteractionController(app)

    controller._handle_overlay_selection(7)

    assert shadow.calls == [(app.app_state.overlays[0], "shadow_lw_pressed")]
    assert spark.calls == []
    assert app.app_state.ui.selected_overlay_id == 7
    assert not hasattr(app, "_capture_shadow_overlay_calibration")


def test_lit_exemplar_rejects_sample_that_matches_unlit_reference(monkeypatch):
    warnings = []
    infos = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: warnings.append(args))
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: infos.append(args))
    app_state = AppState()
    app_state.calibration.calibration_mode = "lit_exemplar"
    app_state.calibration.current_calibration_key_type = "LW"
    app_state.detection.exemplar_lit_colors["LW"] = (10, 20, 30)
    app_state.detection.exemplar_lit_histograms["LW"] = np.array([1.0], dtype=np.float32)
    app_state.overlays = [
        OverlayConfig(
            key_id=7,
            note_octave=4,
            note_name_in_octave="C",
            x=0,
            y=0,
            width=4,
            height=4,
            key_type="LW",
            unlit_reference_color=(250, 250, 250),
        )
    ]
    control_panel = SimpleNamespace(
        advanced_updates=0,
        controls_updates=0,
        selected_updates=0,
        update_advanced_calibration_display=lambda: setattr(
            control_panel, "advanced_updates", control_panel.advanced_updates + 1
        ),
        update_controls_from_state=lambda: setattr(
            control_panel, "controls_updates", control_panel.controls_updates + 1
        ),
        update_selected_overlay_display=lambda: setattr(
            control_panel, "selected_updates", control_panel.selected_updates + 1
        ),
    )
    video_loading_workflow = SimpleNamespace(
        save_current_config=lambda: (_ for _ in ()).throw(
            AssertionError("invalid exemplar sample should not autosave")
        )
    )
    keyboard_canvas = SimpleNamespace(
        current_frame_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        get_average_color_for_overlay=lambda _frame, _overlay: (251, 249, 250),
        get_roi_bgr=lambda _overlay: np.full((4, 4, 3), (250, 250, 250), dtype=np.uint8),
    )
    app = SimpleNamespace(
        app_state=app_state,
        keyboard_canvas=keyboard_canvas,
        control_panel=control_panel,
        video_loading_workflow=video_loading_workflow,
    )

    CalibrationInteractionController(app)._handle_overlay_selection(7)

    assert app_state.detection.exemplar_lit_colors["LW"] == (10, 20, 30)
    assert app_state.detection.exemplar_lit_histograms["LW"].tolist() == [1.0]
    assert app_state.calibration.calibration_mode == "lit_exemplar"
    assert app_state.calibration.current_calibration_key_type == "LW"
    assert app_state.unsaved_changes is False
    assert infos == []
    assert warnings
    assert "frame where the key is lit" in warnings[0][2]
    assert control_panel.advanced_updates == 0
    assert control_panel.controls_updates == 0
    assert control_panel.selected_updates == 1


def test_lit_exemplar_accepts_sample_that_differs_from_unlit_reference(monkeypatch):
    warnings = []
    infos = []
    save_calls = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: warnings.append(args))
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: infos.append(args))
    app_state = AppState()
    app_state.calibration.calibration_mode = "lit_exemplar"
    app_state.calibration.current_calibration_key_type = "LW"
    app_state.overlays = [
        OverlayConfig(
            key_id=7,
            note_octave=4,
            note_name_in_octave="C",
            x=0,
            y=0,
            width=4,
            height=4,
            key_type="LW",
            unlit_reference_color=(250, 250, 250),
        )
    ]
    control_panel = SimpleNamespace(
        advanced_updates=0,
        controls_updates=0,
        selected_updates=0,
        update_advanced_calibration_display=lambda: setattr(
            control_panel, "advanced_updates", control_panel.advanced_updates + 1
        ),
        update_controls_from_state=lambda: setattr(
            control_panel, "controls_updates", control_panel.controls_updates + 1
        ),
        update_selected_overlay_display=lambda: setattr(
            control_panel, "selected_updates", control_panel.selected_updates + 1
        ),
    )
    video_loading_workflow = SimpleNamespace(
        save_current_config=lambda: save_calls.append("save") or True
    )
    keyboard_canvas = SimpleNamespace(
        current_frame_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        get_average_color_for_overlay=lambda _frame, _overlay: (220, 80, 40),
        get_roi_bgr=lambda _overlay: np.full((4, 4, 3), (40, 80, 220), dtype=np.uint8),
    )
    app = SimpleNamespace(
        app_state=app_state,
        keyboard_canvas=keyboard_canvas,
        control_panel=control_panel,
        video_loading_workflow=video_loading_workflow,
    )

    CalibrationInteractionController(app)._handle_overlay_selection(7)

    assert app_state.detection.exemplar_lit_colors["LW"] == (220, 80, 40)
    assert app_state.detection.exemplar_lit_histograms["LW"] is not None
    assert app_state.calibration.calibration_mode is None
    assert app_state.calibration.current_calibration_key_type is None
    assert app_state.unsaved_changes is False
    assert save_calls == ["save"]
    assert warnings == []
    assert infos
    assert control_panel.advanced_updates == 1
    assert control_panel.controls_updates == 1
    assert control_panel.selected_updates == 1


def _app_state_with_universal_spark_calibration():
    app_state = AppState()
    app_state.overlays = [
        OverlayConfig(
            key_id=1,
            note_octave=4,
            note_name_in_octave="C",
            x=0,
            y=0,
            width=10,
            height=10,
            key_type="LW",
        )
    ]
    detection = app_state.detection
    detection.spark_detection_enabled = True
    detection.spark_roi_top = 1
    detection.spark_roi_bottom = 20
    detection.spark_brightness_threshold = 0.5
    detection.spark_calibration_bar_only = {"mean_saturation": 0.8}
    detection.spark_calibration_dimmest_sparks = {"mean_saturation": 0.2}
    return app_state


def test_detection_factory_uses_spark_detector_with_universal_manual_calibration():
    app_state = _app_state_with_universal_spark_calibration()

    detector = DetectionFactory.create_from_app_state(app_state, app_state.overlays)

    assert isinstance(detector, SparkIntegratedDetection)


def test_spark_integrated_ready_with_universal_manual_calibration():
    app_state = _app_state_with_universal_spark_calibration()
    detector = SparkIntegratedDetection(app_state)

    assert detector._is_spark_detection_ready()


def test_spark_exposes_standard_winner_only_for_returned_keys_and_clears_on_reset():
    detector = SparkIntegratedDetection(AppState())
    detector.standard_detector.last_exemplar_matches = {
        1: "COLOR_4_W",
        2: "COLOR_3_B",
    }
    detector.previous_detected_keys = {1}

    assert detector.get_last_exemplar_match(1) == "COLOR_4_W"
    assert detector.get_last_exemplar_match(2) is None

    detector.reset_state()

    assert detector.get_last_exemplar_match(1) is None
    assert detector.standard_detector.get_last_exemplar_match(1) is None


def test_standard_detection_hand_assignment_restricts_color_exemplars_by_detected_hand():
    frame_bgr = np.full((4, 4, 3), (0, 255, 0), dtype=np.uint8)  # green, HSV hue near left hand
    overlay = OverlayConfig(
        key_id=1,
        note_octave=4,
        note_name_in_octave="C",
        x=0,
        y=0,
        width=4,
        height=4,
        key_type="LW",
        unlit_reference_color=(0, 0, 0),
    )
    detector = StandardDetection()

    pressed = detector.detect_frame(
        frame_bgr=frame_bgr,
        overlays=[overlay],
        exemplar_lit_colors={"LW": None, "RW": (255, 0, 0), "LB": None, "RB": None},
        exemplar_lit_histograms={"LW": None, "RW": None, "LB": None, "RB": None},
        detection_threshold=0.8,
        use_delta_detection=False,
        apply_black_filter=False,
        hand_assignment_enabled=True,
        hand_detection_calibrated=True,
        left_hand_hue_mean=60.0,
        right_hand_hue_mean=0.0,
    )

    assert pressed == set()


def test_canvas_interaction_exposes_shadow_roi_selection_modes():
    interaction = CanvasInteraction(
        canvas_widget=None,
        coord_manager=SimpleNamespace(image_height=100),
        app_state=AppState(),
    )

    interaction.enter_shadow_roi_selection_mode()
    assert interaction.is_in_roi_selection_mode()
    assert interaction._roi_selection_type == "shadow"

    interaction.enter_shadow_white_roi_selection_mode()
    assert interaction._roi_selection_type == "shadow_white"

    interaction.enter_shadow_black_roi_selection_mode()
    assert interaction._roi_selection_type == "shadow_black"


def test_unlit_calibration_warns_when_frame_has_likely_lit_key(monkeypatch):
    warnings = []
    infos = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: warnings.append(args) or QMessageBox.StandardButton.Cancel)
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: infos.append(args))

    app_state = AppState()
    app_state.overlays = [
        OverlayConfig(key_id=1, note_octave=4, note_name_in_octave="C", x=0, y=0, width=4, height=4, key_type="LW"),
        OverlayConfig(key_id=2, note_octave=4, note_name_in_octave="D", x=5, y=0, width=4, height=4, key_type="LW"),
        OverlayConfig(key_id=3, note_octave=4, note_name_in_octave="E", x=10, y=0, width=4, height=4, key_type="LW"),
        OverlayConfig(key_id=4, note_octave=4, note_name_in_octave="F", x=15, y=0, width=4, height=4, key_type="LW"),
    ]
    frame_rgb = np.full((8, 24, 3), (245, 245, 235), dtype=np.uint8)
    frame_rgb[0:4, 10:14] = (235, 150, 40)
    frame_bgr = frame_rgb[:, :, ::-1]
    canvas = SimpleNamespace(
        current_frame_rgb=frame_rgb,
        get_roi_bgr=lambda overlay: frame_bgr[
            int(overlay.y):int(overlay.y + overlay.height),
            int(overlay.x):int(overlay.x + overlay.width),
        ],
    )
    parent = SimpleNamespace(keyboard_canvas=canvas, control_panel=SimpleNamespace(update_controls_from_state=lambda: None))
    workflow = CalibrationWorkflow(app_state, SimpleNamespace(), parent)

    workflow.handle_calibrate_unlit_all_keys()

    assert warnings
    assert "E4" in warnings[0][2]
    assert infos == []
    assert all(overlay.unlit_reference_color is None for overlay in app_state.overlays)
