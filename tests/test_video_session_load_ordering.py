from types import SimpleNamespace

import synthesia2midi.gui.video_session_ui_controller as ui_module
import synthesia2midi.workflows.video_session_coordinator as coordinator_module
from synthesia2midi.gui.video_session_ui_controller import VideoSessionUiController
from synthesia2midi.workflows.video_session_coordinator import VideoSessionCoordinator


class FakeSession:
    def __init__(self, calls, label, fps=24.0):
        self.calls = calls
        self.label = label
        self.fps = fps

    def close(self):
        self.calls.append(f"{self.label}.close")


class FakeStateManager:
    def __init__(self, calls):
        self.calls = calls

    def reset_to_defaults(self):
        self.calls.append("state.reset")


class FakeOverlays:
    def __init__(self, calls):
        self.calls = calls

    def clear(self):
        self.calls.append("state.overlays.clear")


class FakeVideoLoadingWorkflow:
    def __init__(self, calls, app_state, *, config_file):
        self.calls = calls
        self.app_state = app_state
        self.config_file = config_file

    def load_video_file(self, filepath):
        self.calls.append(f"loader.load:{filepath}")
        if self.config_file:
            self.app_state.video.processing_start_frame = 12
            self.app_state.video.current_frame_index = 4
        else:
            self.app_state.video.processing_start_frame = 0
            self.app_state.video.current_frame_index = 7
        return True, FakeSession(self.calls, "loaded", fps=24.0)

    def get_video_info(self):
        return {"config_file": self.config_file}


class FakeVideoControls:
    def __init__(self, calls):
        self.calls = calls

    def set_video_session(self, video_session):
        self.calls.append(f"controls.set_session:{video_session.label}")

    def update_frame_slider_for_video(self):
        self.calls.append("controls.update_slider")

    def display_frame_with_slider_update(self, frame):
        self.calls.append(f"controls.display_frame:{frame}")


class FakeKeyboardCanvas:
    def __init__(self, calls):
        self.calls = calls
        self.detect_pressed_func = None

    def set_video_session(self, video_session):
        self.calls.append(f"canvas.set_session:{video_session.label}")


class FakeButton:
    def __init__(self, calls, label):
        self.calls = calls
        self.label = label

    def setEnabled(self, enabled):
        self.calls.append(f"{self.label}.setEnabled:{enabled}")


class FakeControlPanel:
    def __init__(self, calls, app_state=None, dirty_on_update_controls=False):
        self.calls = calls
        self.app_state = app_state
        self.dirty_on_update_controls = dirty_on_update_controls
        self.convert_button = FakeButton(calls, "convert_button")
        self.wizard_button = FakeButton(calls, "wizard_button")

    def update_video_frame_limits(self):
        self.calls.append("panel.update_frame_limits")

    def update_video_info(self, fps):
        self.calls.append(f"panel.update_video_info:{fps}")

    def update_controls_from_state(self):
        self.calls.append("panel.update_controls_from_state")
        if self.dirty_on_update_controls and self.app_state is not None:
            self.app_state.unsaved_changes = True

    def update_trim_controls_from_state(self):
        self.calls.append("panel.update_trim_controls_from_state")

    def _can_convert(self):
        self.calls.append("panel.can_convert")
        return True


class FakeVideoSessionUiController:
    def __init__(self, calls):
        self.calls = calls

    def initialize_processing_range_defaults(self):
        self.calls.append("ui.initialize_processing_range_defaults")


class FakeMainActionController:
    def __init__(self, calls):
        self.calls = calls

    def create_detection_wrapper(self):
        self.calls.append("main_action.create_detection_wrapper")
        return lambda *_args, **_kwargs: None


class FakeWindowManager:
    def __init__(self, calls):
        self.calls = calls

    def resize_and_position_window(self):
        self.calls.append("window.resize_and_position")


class FakeAcceptedFileDialog:
    AnyFile = object()
    DontUseNativeDialog = object()
    selected_path = ""

    def __init__(self, _parent):
        pass

    def setWindowTitle(self, _title):
        pass

    def setFileMode(self, _file_mode):
        pass

    def setOption(self, _option, _enabled):
        pass

    def setNameFilter(self, _name_filter):
        pass

    def setDirectory(self, _directory):
        pass

    def findChild(self, _view_type):
        return None

    def selectedFiles(self):
        return [self.selected_path]


def _accepted_dialog_exec(_self):
    return ui_module.QDialog.Accepted


setattr(FakeAcceptedFileDialog, "exec", _accepted_dialog_exec)


def _fake_app(calls, *, config_file, dirty_on_update_controls=False):
    app_state = SimpleNamespace(
        video=SimpleNamespace(
            processing_start_frame=0,
            current_frame_index=0,
            video_is_trimmed=False,
            trim_start_frame=0,
        ),
        overlays=FakeOverlays(calls),
        unsaved_changes=True,
    )
    app = SimpleNamespace(
        app_state=app_state,
        video_session=FakeSession(calls, "old"),
        state_manager=FakeStateManager(calls),
        video_loading_workflow=FakeVideoLoadingWorkflow(calls, app_state, config_file=config_file),
        video_controls=FakeVideoControls(calls),
        keyboard_canvas=FakeKeyboardCanvas(calls),
        control_panel=FakeControlPanel(
            calls,
            app_state=app_state,
            dirty_on_update_controls=dirty_on_update_controls,
        ),
        video_session_ui_controller=FakeVideoSessionUiController(calls),
        main_action_controller=FakeMainActionController(calls),
        window_manager=FakeWindowManager(calls),
        _update_current_frame_display=lambda *_args, **_kwargs: None,
    )
    return app


def _patch_workflow_constructors(monkeypatch, calls):
    def constructor(label):
        def _construct(*_args, **_kwargs):
            calls.append(label)
            return SimpleNamespace(label=label)

        return _construct

    monkeypatch.setattr(coordinator_module, "CalibrationWorkflow", constructor("workflow.calibration"))
    monkeypatch.setattr(coordinator_module, "AutoCalibrationWorkflow", constructor("workflow.auto_calibration"))
    monkeypatch.setattr(coordinator_module, "DetectionManager", constructor("workflow.detection_manager"))
    monkeypatch.setattr(coordinator_module, "ConversionWorkflow", constructor("workflow.conversion"))


def test_local_file_open_path_preserves_configured_session_orchestration_order(monkeypatch):
    calls = []
    _patch_workflow_constructors(monkeypatch, calls)
    app = _fake_app(calls, config_file="local.ini")
    app.video_session_coordinator = VideoSessionCoordinator(app)
    selected_path = "/tmp/local-video.mp4"
    FakeAcceptedFileDialog.selected_path = selected_path
    monkeypatch.setattr(ui_module, "QFileDialog", FakeAcceptedFileDialog)

    VideoSessionUiController(app).open_video_file()

    assert calls == [
        "old.close",
        "state.reset",
        "loader.load:/tmp/local-video.mp4",
        "controls.set_session:loaded",
        "canvas.set_session:loaded",
        "controls.update_slider",
        "panel.update_frame_limits",
        "panel.update_video_info:24.0",
        "workflow.calibration",
        "workflow.auto_calibration",
        "workflow.detection_manager",
        "workflow.conversion",
        "main_action.create_detection_wrapper",
        "panel.update_controls_from_state",
        "panel.update_trim_controls_from_state",
        "ui.initialize_processing_range_defaults",
        "controls.display_frame:12",
        "panel.can_convert",
        "convert_button.setEnabled:True",
        "wizard_button.setEnabled:True",
        "window.resize_and_position",
    ]


def test_youtube_download_completion_preserves_no_config_session_orchestration_order(monkeypatch):
    calls = []
    _patch_workflow_constructors(monkeypatch, calls)
    app = _fake_app(calls, config_file=None)
    app.video_session_coordinator = VideoSessionCoordinator(app)

    VideoSessionUiController(app).handle_youtube_video_downloaded("/tmp/youtube-download.mp4")

    assert calls == [
        "old.close",
        "state.reset",
        "loader.load:/tmp/youtube-download.mp4",
        "controls.set_session:loaded",
        "canvas.set_session:loaded",
        "controls.update_slider",
        "panel.update_frame_limits",
        "workflow.calibration",
        "workflow.auto_calibration",
        "workflow.detection_manager",
        "workflow.conversion",
        "main_action.create_detection_wrapper",
        "state.overlays.clear",
        "panel.update_controls_from_state",
        "panel.update_trim_controls_from_state",
        "controls.display_frame:0",
        "convert_button.setEnabled:False",
        "wizard_button.setEnabled:True",
        "window.resize_and_position",
    ]
    assert "panel.update_video_info:24.0" not in calls


def test_load_path_finishes_clean_when_control_sync_emits_dirty_state(monkeypatch):
    calls = []
    _patch_workflow_constructors(monkeypatch, calls)
    app = _fake_app(calls, config_file=None, dirty_on_update_controls=True)
    app.app_state.unsaved_changes = False
    app.video_session_coordinator = VideoSessionCoordinator(app)

    VideoSessionUiController(app).handle_youtube_video_downloaded("/tmp/youtube-download.mp4")

    assert app.app_state.unsaved_changes is False
