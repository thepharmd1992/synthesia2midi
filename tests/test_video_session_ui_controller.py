from types import SimpleNamespace

from synthesia2midi.gui.video_session_ui_controller import VideoSessionUiController


class FakeAppState:
    def __init__(self, video):
        self.video = video
        self.unsaved_changes = False
        self.marked_unsaved = False

    def mark_unsaved(self):
        self.marked_unsaved = True
        self.unsaved_changes = True


class FakeControlPanel:
    def __init__(self):
        self.updated_controls = 0
        self.updated_limits = 0
        self.processing_start_frame_spin = SimpleNamespace(values=[], setValue=lambda value: self.processing_start_frame_spin.values.append(value))
        self.processing_end_frame_spin = SimpleNamespace(values=[], setValue=lambda value: self.processing_end_frame_spin.values.append(value))

    def update_controls_from_state(self):
        self.updated_controls += 1

    def update_video_frame_limits(self):
        self.updated_limits += 1


class FakeVideoControls:
    def __init__(self):
        self.slider_updates = 0
        self.displayed = []

    def update_frame_slider_for_video(self):
        self.slider_updates += 1

    def display_frame_with_slider_update(self, frame):
        self.displayed.append(frame)


class FakeVideoLoadingWorkflow:
    def __init__(self, success=True):
        self.success = success
        self.save_calls = 0

    def save_current_config(self):
        self.save_calls += 1
        return self.success


def _fake_video(**overrides):
    defaults = dict(
        start_frame=0,
        end_frame=0,
        processing_start_frame=0,
        processing_end_frame=0,
        video_is_trimmed=False,
        trim_start_frame=0,
        trim_end_frame=0,
        total_frames=100,
        current_nav_interval=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _fake_app(video):
    return SimpleNamespace(
        app_state=FakeAppState(video),
        control_panel=FakeControlPanel(),
        video_controls=FakeVideoControls(),
        video_loading_workflow=FakeVideoLoadingWorkflow(),
        parameter_manager=SimpleNamespace(update_nav_interval=lambda value: None),
        frame_nav_actions={},
    )


def test_start_and_end_frame_changes_update_legacy_and_processing_ranges():
    video = _fake_video(total_frames=100)
    app = _fake_app(video)
    controller = VideoSessionUiController(app)

    controller.handle_start_frame_change(8)
    controller.handle_end_frame_change(72)

    assert video.start_frame == 8
    assert video.processing_start_frame == 8
    assert video.end_frame == 72
    assert video.processing_end_frame == 72
    assert app.app_state.unsaved_changes is True


def test_frame_nav_interval_updates_state_actions_and_control_panel_signal():
    video = _fake_video(current_nav_interval=1)
    app = _fake_app(video)
    emitted = []
    app.control_panel.nav_interval_changed = SimpleNamespace(emit=lambda value: emitted.append(value))
    app.frame_nav_actions = {
        1: SimpleNamespace(checked=None, setChecked=lambda checked: setattr(app.frame_nav_actions[1], "checked", checked)),
        5: SimpleNamespace(checked=None, setChecked=lambda checked: setattr(app.frame_nav_actions[5], "checked", checked)),
    }

    VideoSessionUiController(app).handle_frame_nav_interval(5)

    assert video.current_nav_interval == 5
    assert app.app_state.unsaved_changes is True
    assert app.frame_nav_actions[1].checked is False
    assert app.frame_nav_actions[5].checked is True
    assert emitted == [5]


def test_update_nav_interval_syncs_parameter_manager_and_menu_actions():
    video = _fake_video(current_nav_interval=1)
    app = _fake_app(video)
    updated = []
    app.parameter_manager = SimpleNamespace(update_nav_interval=lambda value: updated.append(value))
    app.frame_nav_actions = {
        1: SimpleNamespace(checked=None, setChecked=lambda checked: setattr(app.frame_nav_actions[1], "checked", checked)),
        10: SimpleNamespace(checked=None, setChecked=lambda checked: setattr(app.frame_nav_actions[10], "checked", checked)),
    }

    VideoSessionUiController(app).update_nav_interval(10)

    assert updated == [10]
    assert app.frame_nav_actions[1].checked is False
    assert app.frame_nav_actions[10].checked is True


def test_processing_start_frame_clamps_and_rejects_crossing_end():
    video = _fake_video(processing_end_frame=50, total_frames=100)
    app = _fake_app(video)
    controller = VideoSessionUiController(app)

    controller.handle_processing_start_frame_change(120)

    assert video.processing_start_frame == 0
    assert app.app_state.marked_unsaved is False

    controller.handle_processing_start_frame_change(49)

    assert video.processing_start_frame == 49
    assert app.app_state.marked_unsaved is True


def test_processing_end_frame_clamps_and_rejects_crossing_start():
    video = _fake_video(processing_start_frame=20, total_frames=100)
    app = _fake_app(video)
    controller = VideoSessionUiController(app)

    controller.handle_processing_end_frame_change(10)

    assert video.processing_end_frame == 0
    assert app.app_state.marked_unsaved is False

    controller.handle_processing_end_frame_change(120)

    assert video.processing_end_frame == 99
    assert app.app_state.marked_unsaved is True


def test_trim_request_sets_trim_processing_range_updates_ui_and_saves():
    video = _fake_video(total_frames=100)
    app = _fake_app(video)

    VideoSessionUiController(app).handle_trim_video_request(10, -1)

    assert video.trim_start_frame == 10
    assert video.trim_end_frame == 99
    assert video.video_is_trimmed is True
    assert video.processing_start_frame == 10
    assert video.processing_end_frame == 99
    assert app.control_panel.updated_controls == 1
    assert app.control_panel.updated_limits == 1
    assert app.video_controls.slider_updates == 1
    assert app.video_controls.displayed == [10]
    assert app.video_loading_workflow.save_calls == 1
    assert app.app_state.marked_unsaved is True


def test_initialize_processing_range_defaults_uses_trim_or_full_video():
    trimmed_video = _fake_video(video_is_trimmed=True, trim_start_frame=12, trim_end_frame=80, total_frames=100)
    app = _fake_app(trimmed_video)

    VideoSessionUiController(app).initialize_processing_range_defaults()

    assert trimmed_video.processing_start_frame == 12
    assert trimmed_video.processing_end_frame == 80
    assert app.control_panel.processing_start_frame_spin.values == [12]
    assert app.control_panel.processing_end_frame_spin.values == [80]

    full_video = _fake_video(total_frames=42)
    app = _fake_app(full_video)

    VideoSessionUiController(app).initialize_processing_range_defaults()

    assert full_video.processing_start_frame == 0
    assert full_video.processing_end_frame == 41
