from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QRect
from PySide6.QtWidgets import QDialog

from synthesia2midi.gui.auto_detect_tuning_controller import AutoDetectTuningController
from synthesia2midi.gui.calibration_wizard_controller import CalibrationWizardController


class RecordingSignal:
    def __init__(self):
        self.connected = []

    def connect(self, callback):
        self.connected.append(callback)

    def disconnect(self, callback=None):
        if callback is None:
            self.connected.clear()
            return
        self.connected = [item for item in self.connected if item is not callback]

    def emit(self, *args):
        for callback in list(self.connected):
            callback(*args)


class FakeDialog:
    instances = []

    def __init__(
        self,
        parent,
        app_state,
        source_frame_rgb,
        keyboard_roi,
        *,
        initial_detection_results,
        fallback_used,
        apply_detection_callback,
    ):
        self.parent = parent
        self.app_state = app_state
        self.source_frame_rgb = source_frame_rgb
        self.keyboard_roi = keyboard_roi
        self.initial_detection_results = initial_detection_results
        self.fallback_used = fallback_used
        self.apply_detection_callback = apply_detection_callback
        self.finished = RecordingSignal()
        self.modal = None
        self.window_modality = None
        self.closed = False
        self.shown = False
        self.raised = False
        self.activated = False
        self.moved_to = None
        FakeDialog.instances.append(self)

    def setModal(self, value):
        self.modal = value

    def setWindowModality(self, value):
        self.window_modality = value

    def show(self):
        self.shown = True

    def raise_(self):
        self.raised = True

    def activateWindow(self):
        self.activated = True

    def frameGeometry(self):
        return QRect(0, 0, 640, 360)

    def move(self, x, y):
        self.moved_to = (x, y)

    def close(self):
        self.closed = True


class FakeWizard:
    def __init__(self, context=None):
        self.context = context
        self.applied_results = []
        self.auto_detect_latest_detection_result = (
            context.get("detection_results") if context is not None else None
        )
        self.detected_overlays = ["initial-overlay"]

    def get_auto_detect_tuning_context(self):
        if self.context is None:
            return None
        context = dict(self.context)
        context["detection_results"] = self.auto_detect_latest_detection_result
        return context

    def apply_auto_detect_results(self, detection_results):
        self.applied_results.append(detection_results)
        self.auto_detect_latest_detection_result = detection_results
        self.detected_overlays = ["preview-overlay"]
        if self.context is not None:
            self.context["detection_results"] = detection_results
        return True


class FakeAction:
    def __init__(self):
        self.checked_values = []

    def setChecked(self, value):
        self.checked_values.append(value)


class FakeButton:
    def __init__(self):
        self.enabled_values = []

    def setEnabled(self, value):
        self.enabled_values.append(value)


class FakeControlPanel:
    def __init__(self):
        self.convert_button = FakeButton()
        self.controls_updates = 0
        self.selected_overlay_updates = 0

    def _can_convert(self):
        return True

    def update_controls_from_state(self):
        self.controls_updates += 1

    def update_selected_overlay_display(self):
        self.selected_overlay_updates += 1


class FakeCanvas:
    def __init__(self):
        self.current_frame_rgb = None
        self.displayed_frames = []
        self.update_count = 0

    def display_frame(self, frame_index):
        self.displayed_frames.append(frame_index)

    def update(self):
        self.update_count += 1


class FakeVideoLoadingWorkflow:
    def __init__(self):
        self.save_count = 0

    def save_current_config(self):
        self.save_count += 1
        return True


class FakeSettingsToolWindow:
    def __init__(self, *, visible):
        self.visible = visible
        self.hide_count = 0
        self.show_count = 0
        self.restore_count = 0

    def isVisible(self):
        return self.visible

    def hide(self):
        self.hide_count += 1
        self.visible = False

    def show_near_parent(self):
        self.show_count += 1
        self.visible = True

    def show_preserving_geometry(self):
        self.restore_count += 1
        self.visible = True


def make_app(*, current_frame_index=4, unsaved_changes=True):
    return SimpleNamespace(
        app_state=SimpleNamespace(
            ui=SimpleNamespace(show_overlays=False),
            video=SimpleNamespace(current_frame_index=current_frame_index),
            calibration=SimpleNamespace(
                auto_detect_params={},
                overlay_generation_source="manual",
            ),
            midi=SimpleNamespace(total_keys=88, leftmost_note_name="A", leftmost_note_octave=0),
            overlays=[SimpleNamespace(x=0, y=0, width=3, height=2)],
            unsaved_changes=unsaved_changes,
        ),
        show_overlays_action=FakeAction(),
        control_panel=FakeControlPanel(),
        keyboard_canvas=FakeCanvas(),
        video_loading_workflow=FakeVideoLoadingWorkflow(),
        video_session=None,
        settings_tool_window=None,
        screen=lambda: None,
    )


def test_controller_retains_modeless_dialog_until_finished(monkeypatch):
    FakeDialog.instances.clear()
    monkeypatch.setattr(
        "synthesia2midi.gui.auto_detect_tuning_controller.AutoDetectTuningDialog",
        FakeDialog,
    )
    app = make_app()
    context = {
        "frame_rgb": np.zeros((2, 3, 3), dtype=np.uint8),
        "keyboard_roi": (0, 0, 3, 2),
        "fallback_used": True,
        "detection_results": {"total_keys": 88},
    }
    wizard = FakeWizard(context)
    finished = []
    controller = AutoDetectTuningController(app)

    opened = controller.open(
        wizard,
        use_wizard_context=True,
        on_dialog_finished=lambda result: finished.append(result),
    )

    assert opened is True
    dialog = FakeDialog.instances[-1]
    assert controller.active_dialog is dialog
    assert dialog.modal is False
    assert dialog.shown is True
    assert dialog.raised is True
    assert dialog.activated is True

    dialog.finished.emit(QDialog.Accepted)

    assert controller.active_dialog is None
    assert app.video_loading_workflow.save_count == 1
    assert finished == [QDialog.Accepted]


@pytest.mark.parametrize(
    ("initial_dirty", "preview_dirty"),
    [(False, True), (True, False)],
)
def test_rejected_tuning_restores_pre_dialog_state_without_saving(
    monkeypatch, initial_dirty, preview_dirty
):
    FakeDialog.instances.clear()
    monkeypatch.setattr(
        "synthesia2midi.gui.auto_detect_tuning_controller.AutoDetectTuningDialog",
        FakeDialog,
    )
    app = make_app(current_frame_index=12, unsaved_changes=initial_dirty)
    app.app_state.calibration.auto_detect_params = {"separator_threshold": 11}
    app.app_state.overlays = [SimpleNamespace(x=2, y=3, width=4, height=5)]
    initial_detection = {
        "total_keys": 88,
        "leftmost_note": "A",
        "leftmost_octave": 0,
        "detected_keys": [{"x": 2}],
    }
    wizard = FakeWizard(
        {
            "frame_rgb": np.zeros((2, 3, 3), dtype=np.uint8),
            "keyboard_roi": (0, 0, 3, 2),
            "fallback_used": False,
            "detection_results": initial_detection,
        }
    )
    controller = AutoDetectTuningController(app)

    assert controller.open(wizard, use_wizard_context=True) is True
    preview_detection = {
        "total_keys": 76,
        "leftmost_note": "C",
        "leftmost_octave": 2,
        "detected_keys": [{"x": 20}],
    }
    assert controller.apply_preview_result(preview_detection) is True
    app.app_state.calibration.auto_detect_params = {"separator_threshold": 37}
    app.app_state.calibration.overlay_generation_source = "auto"
    app.app_state.overlays = [SimpleNamespace(x=20, y=30, width=40, height=50)]
    app.app_state.midi.total_keys = 76
    app.app_state.midi.leftmost_note_name = "C"
    app.app_state.midi.leftmost_note_octave = 2
    app.app_state.unsaved_changes = preview_dirty

    FakeDialog.instances[-1].finished.emit(QDialog.Rejected)

    assert app.video_loading_workflow.save_count == 0
    assert app.app_state.calibration.auto_detect_params == {"separator_threshold": 11}
    assert app.app_state.calibration.overlay_generation_source == "manual"
    assert [(item.x, item.y, item.width, item.height) for item in app.app_state.overlays] == [
        (2, 3, 4, 5)
    ]
    assert (
        app.app_state.midi.total_keys,
        app.app_state.midi.leftmost_note_name,
        app.app_state.midi.leftmost_note_octave,
    ) == (88, "A", 0)
    assert app.app_state.ui.show_overlays is False
    assert app.app_state.unsaved_changes is initial_dirty
    assert app.show_overlays_action.checked_values[-1] is False
    assert wizard.auto_detect_latest_detection_result == initial_detection
    assert wizard.detected_overlays == ["initial-overlay"]
    assert controller.cached_context["detection_results"] == initial_detection
    assert app.keyboard_canvas.displayed_frames[-1] == 12
    assert app.control_panel.controls_updates == 2
    assert app.control_panel.selected_overlay_updates == 2


def test_controller_hides_visible_settings_tool_window_until_tuning_closes(monkeypatch):
    FakeDialog.instances.clear()
    monkeypatch.setattr(
        "synthesia2midi.gui.auto_detect_tuning_controller.AutoDetectTuningDialog",
        FakeDialog,
    )
    app = make_app()
    app.settings_tool_window = FakeSettingsToolWindow(visible=True)
    wizard = FakeWizard(
        {
            "frame_rgb": np.zeros((2, 3, 3), dtype=np.uint8),
            "keyboard_roi": (0, 0, 3, 2),
            "fallback_used": False,
            "detection_results": {"total_keys": 88},
        }
    )
    controller = AutoDetectTuningController(app)

    assert controller.open(wizard, use_wizard_context=True) is True

    dialog = FakeDialog.instances[-1]
    assert app.settings_tool_window.hide_count == 1
    assert app.settings_tool_window.visible is False

    dialog.finished.emit(0)

    assert app.settings_tool_window.show_count == 0
    assert app.settings_tool_window.restore_count == 1
    assert app.settings_tool_window.visible is True


def test_controller_restores_settings_when_calibration_flow_requests_it_even_if_hidden(monkeypatch):
    FakeDialog.instances.clear()
    monkeypatch.setattr(
        "synthesia2midi.gui.auto_detect_tuning_controller.AutoDetectTuningDialog",
        FakeDialog,
    )
    app = make_app()
    app.settings_tool_window = FakeSettingsToolWindow(visible=False)
    wizard = FakeWizard(
        {
            "frame_rgb": np.zeros((2, 3, 3), dtype=np.uint8),
            "keyboard_roi": (0, 0, 3, 2),
            "fallback_used": False,
            "detection_results": {"total_keys": 88},
        }
    )
    controller = AutoDetectTuningController(app)

    assert controller.open(
        wizard,
        use_wizard_context=True,
        restore_settings_on_finish=True,
    ) is True

    FakeDialog.instances[-1].finished.emit(0)

    assert app.settings_tool_window.restore_count == 1
    assert app.settings_tool_window.visible is True


def test_controller_places_tuning_dialog_top_center_safe_zone(monkeypatch):
    FakeDialog.instances.clear()
    monkeypatch.setattr(
        "synthesia2midi.gui.auto_detect_tuning_controller.AutoDetectTuningDialog",
        FakeDialog,
    )
    app = make_app()
    app.screen = lambda: SimpleNamespace(availableGeometry=lambda: QRect(0, 24, 1440, 876))
    wizard = FakeWizard(
        {
            "frame_rgb": np.zeros((2, 3, 3), dtype=np.uint8),
            "keyboard_roi": (0, 0, 3, 2),
            "fallback_used": False,
            "detection_results": {"total_keys": 88},
        }
    )
    controller = AutoDetectTuningController(app)

    assert controller.open(wizard, use_wizard_context=True) is True

    x, y = FakeDialog.instances[-1].moved_to
    assert 300 <= x <= 420
    assert 24 <= y <= 70


def test_preview_result_applies_to_wizard_and_refreshes_existing_ui_flow():
    app = make_app(current_frame_index=9)
    template_style_calls = []
    controller = AutoDetectTuningController(
        app,
        apply_template_styles_callback=lambda: template_style_calls.append("applied"),
    )
    wizard = FakeWizard(
        {
            "frame_rgb": np.zeros((2, 3, 3), dtype=np.uint8),
            "keyboard_roi": (0, 0, 3, 2),
            "fallback_used": False,
            "detection_results": {"total_keys": 88},
        }
    )
    assert controller.open(wizard, dialog_factory=lambda *args, **kwargs: FakeDialog(*args, **kwargs)) is True
    detection_results = {"total_keys": 76, "detected_keys": [{"x": 1}]}

    applied = controller.apply_preview_result(detection_results)

    assert applied is True
    assert wizard.applied_results == [detection_results]
    assert template_style_calls == ["applied"]
    assert app.app_state.ui.show_overlays is True
    assert app.show_overlays_action.checked_values == [True]
    assert app.control_panel.convert_button.enabled_values == [True]
    assert app.keyboard_canvas.displayed_frames == [9]
    assert app.control_panel.controls_updates == 1
    assert app.control_panel.selected_overlay_updates == 1
    assert controller.cached_context["detection_results"] == detection_results


def test_calibration_wizard_controller_delegates_tuning_state_to_dedicated_controller():
    app = make_app()
    tuning_controller = AutoDetectTuningController(app)

    calibration_controller = CalibrationWizardController(app, tuning_controller)

    assert calibration_controller.auto_detect_tuning_controller is tuning_controller
    assert "_auto_detect_tuning_dialog" not in vars(calibration_controller)
    assert "_last_auto_detect_tuning_context" not in vars(calibration_controller)
