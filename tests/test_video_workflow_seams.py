from types import SimpleNamespace

from synthesia2midi.gui.signal_manager import ControlSignalManager
from synthesia2midi.gui.video_session_ui_controller import VideoSessionUiController
from synthesia2midi.workflows import video_session_coordinator as coordinator_module
from synthesia2midi.workflows.video_session_coordinator import VideoSessionCoordinator


class FakeSignal:
    def __init__(self, name, connections=None):
        self.name = name
        self.connections = connections if connections is not None else []

    def connect(self, slot):
        self.connections.append((self.name, slot))


class FakeAction:
    def __init__(self, name, events=None):
        self.name = name
        self.events = events if events is not None else []
        self.checked = None

    def setChecked(self, checked):
        self.checked = checked
        self.events.append((self.name, checked))


class FakeButton:
    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.enabled = None

    def setEnabled(self, enabled):
        self.enabled = enabled
        self.events.append((self.name, enabled))


class DynamicSignalPanel:
    def __init__(self, connections):
        self._connections = connections

    def __getattr__(self, name):
        signal = FakeSignal(name, self._connections)
        setattr(self, name, signal)
        return signal


def _sentinel(name):
    def slot(*args, **kwargs):
        return (name, args, kwargs)

    slot.__name__ = name
    return slot


def _namespace_with_slots(*names):
    return SimpleNamespace(**{name: _sentinel(name) for name in names})


def _patch_workflow_factories(monkeypatch, events):
    def calibration_factory(app_state, video_session, app):
        events.append("calibration_workflow")
        return "calibration-workflow"

    def auto_calibration_factory(app_state, video_session, app):
        events.append("auto_calibration_workflow")
        return "auto-calibration-workflow"

    def conversion_factory(app_state, video_session, app, detection_manager):
        events.append(("conversion_workflow", detection_manager))
        return "conversion-workflow"

    monkeypatch.setattr(coordinator_module, "CalibrationWorkflow", calibration_factory)
    monkeypatch.setattr(coordinator_module, "AutoCalibrationWorkflow", auto_calibration_factory)
    monkeypatch.setattr(coordinator_module, "ConversionWorkflow", conversion_factory)


def test_video_session_coordinator_load_path_closes_resets_loads_and_applies_session_in_order(monkeypatch):
    events = []
    new_session = SimpleNamespace(fps=30.0)
    old_session = SimpleNamespace(close=lambda: events.append("old_session_closed"))

    workflow = SimpleNamespace(
        load_video_file=lambda filepath: events.append(("load", filepath)) or (True, new_session),
    )
    app = SimpleNamespace(
        video_session=old_session,
        state_manager=SimpleNamespace(reset_to_defaults=lambda: events.append("reset")),
        video_loading_workflow=workflow,
    )
    coordinator = VideoSessionCoordinator(app)
    monkeypatch.setattr(
        coordinator,
        "apply_loaded_session",
        lambda session, *, log_prefix, update_fps_display: events.append(
            ("apply", session, log_prefix, update_fps_display)
        ),
    )

    assert coordinator.load_path("/tmp/video.mp4", log_prefix="file-open", update_fps_display=True) is True

    assert app.video_session is None
    assert events == [
        "old_session_closed",
        "reset",
        ("load", "/tmp/video.mp4"),
        ("apply", new_session, "file-open", True),
    ]


def test_video_session_coordinator_load_path_stops_after_failed_load(monkeypatch):
    events = []
    app = SimpleNamespace(
        video_session=None,
        state_manager=SimpleNamespace(reset_to_defaults=lambda: events.append("reset")),
        video_loading_workflow=SimpleNamespace(
            load_video_file=lambda filepath: events.append(("load", filepath)) or (False, None)
        ),
    )
    coordinator = VideoSessionCoordinator(app)
    monkeypatch.setattr(
        coordinator,
        "apply_loaded_session",
        lambda *args, **kwargs: events.append("unexpected_apply"),
    )

    assert coordinator.load_path("/tmp/missing.mp4", log_prefix="file-open", update_fps_display=True) is False
    assert events == ["reset", ("load", "/tmp/missing.mp4")]


def _fake_loaded_app(events, *, config_file, video_state):
    session = SimpleNamespace(fps=24.0, total_frames=120)
    convert_button = FakeButton("convert_enabled", events)
    wizard_button = FakeButton("wizard_enabled", events)
    control_panel = SimpleNamespace(
        update_video_frame_limits=lambda: events.append("limits"),
        update_video_info=lambda fps: events.append(("fps_info", fps)),
        update_controls_from_state=lambda: events.append("controls_from_state"),
        update_trim_controls_from_state=lambda: events.append("trim_from_state"),
        convert_button=convert_button,
        wizard_button=wizard_button,
    )
    control_panel._can_convert = lambda: events.append("can_convert") or True

    detection_manager = SimpleNamespace(
        reset_detector_cache=lambda: events.append("detection_cache_reset"),
        set_navigation_mode=lambda navigation_mode: events.append(("navigation_mode", navigation_mode)),
        create_detection_wrapper=lambda: events.append("detection_wrapper") or "detect-wrapper",
    )

    app = SimpleNamespace(
        app_state=SimpleNamespace(video=video_state, overlays={"old": object()}, unsaved_changes=True),
        video_session=None,
        video_controls=SimpleNamespace(
            set_video_session=lambda loaded: events.append(("controls_session", loaded)),
            update_frame_slider_for_video=lambda: events.append("slider"),
            display_frame_with_slider_update=lambda frame: events.append(("display", frame)),
        ),
        keyboard_canvas=SimpleNamespace(
            set_video_session=lambda loaded: events.append(("canvas_session", loaded)),
            detect_pressed_func=None,
        ),
        control_panel=control_panel,
        video_loading_workflow=SimpleNamespace(get_video_info=lambda: {"config_file": config_file}),
        video_session_ui_controller=SimpleNamespace(
            initialize_processing_range_defaults=lambda: events.append("initialize_processing_range_defaults")
        ),
        detection_manager=detection_manager,
        main_action_controller=SimpleNamespace(
            create_detection_wrapper=lambda: events.append("detection_wrapper") or "detect-wrapper"
        ),
        window_manager=SimpleNamespace(resize_and_position_window=lambda: events.append("resize")),
        _update_current_frame_display=_sentinel("update_current_frame_display"),
    )
    return app, session


def test_apply_loaded_session_with_config_runs_post_load_wiring_in_historical_order(monkeypatch):
    events = []
    _patch_workflow_factories(monkeypatch, events)
    video_state = SimpleNamespace(processing_start_frame=7, current_frame_index=3, video_is_trimmed=False)
    app, session = _fake_loaded_app(events, config_file="/tmp/video.ini", video_state=video_state)

    VideoSessionCoordinator(app).apply_loaded_session(session, log_prefix="file-open", update_fps_display=True)

    assert app.video_session is session
    assert app.keyboard_canvas.detect_pressed_func == "detect-wrapper"
    assert app.calibration_workflow == "calibration-workflow"
    assert app.auto_calibration_workflow == "auto-calibration-workflow"
    assert app.conversion_workflow == "conversion-workflow"
    assert events == [
        ("controls_session", session),
        ("canvas_session", session),
        "slider",
        "limits",
        ("fps_info", 24.0),
        "calibration_workflow",
        "auto_calibration_workflow",
        "detection_cache_reset",
        ("navigation_mode", True),
        ("conversion_workflow", app.detection_manager),
        "detection_wrapper",
        "controls_from_state",
        "trim_from_state",
        "initialize_processing_range_defaults",
        ("display", 7),
        "can_convert",
        ("convert_enabled", True),
        ("wizard_enabled", True),
        "resize",
    ]


def test_apply_loaded_session_without_config_clears_overlays_and_displays_trim_start(monkeypatch):
    events = []
    _patch_workflow_factories(monkeypatch, events)
    video_state = SimpleNamespace(
        processing_start_frame=0,
        current_frame_index=4,
        video_is_trimmed=True,
        trim_start_frame=12,
    )
    app, session = _fake_loaded_app(events, config_file=None, video_state=video_state)

    VideoSessionCoordinator(app).apply_loaded_session(session, log_prefix="file-open", update_fps_display=False)

    assert app.app_state.overlays == {}
    assert app.app_state.unsaved_changes is False
    assert ("display", 12) in events
    assert ("convert_enabled", False) in events
    assert "initialize_processing_range_defaults" not in events
    assert not any(event == ("fps_info", 24.0) for event in events)


def test_video_session_ui_controller_entrypoints_delegate_to_new_workflow_seams():
    coordinator_calls = []
    frame_controller_calls = []
    app = SimpleNamespace(
        video_session_coordinator=SimpleNamespace(
            load_path=lambda filepath, *, log_prefix, update_fps_display: coordinator_calls.append(
                (filepath, log_prefix, update_fps_display)
            )
        ),
        video_to_frames_controller=SimpleNamespace(
            handle_request=lambda: frame_controller_calls.append("request") or "frames-started"
        ),
    )
    controller = VideoSessionUiController(app)

    controller.handle_youtube_video_downloaded("/tmp/download.mp4")
    result = controller.handle_video_to_frames_request()

    assert coordinator_calls == [("/tmp/download.mp4", "_handle_youtube_video_downloaded", False)]
    assert frame_controller_calls == ["request"]
    assert result == "frames-started"


def test_control_signal_manager_wires_video_range_and_trim_surfaces_to_controllers():
    connections = []
    control_panel = DynamicSignalPanel(connections)
    video_session_ui_controller = _namespace_with_slots(
        "update_nav_interval",
        "handle_youtube_video_downloaded",
        "handle_video_to_frames_request",
        "handle_start_frame_change",
        "handle_end_frame_change",
        "handle_trim_video_request",
        "handle_processing_start_frame_change",
        "handle_processing_end_frame_change",
    )
    main_window = SimpleNamespace(
        frame_slider=SimpleNamespace(valueChanged=FakeSignal("frame_slider.valueChanged", connections)),
        video_session_ui_controller=video_session_ui_controller,
        video_controls=_namespace_with_slots("on_frame_slider_changed"),
        detection_manager=_namespace_with_slots(
            "set_detection_threshold",
            "set_rise_delta_threshold",
            "set_fall_delta_threshold",
            "set_histogram_detection_enabled",
            "set_delta_detection_enabled",
            "set_winner_takes_black_enabled",
            "set_hand_assignment_enabled",
            "set_histogram_threshold",
            "set_similarity_ratio",
        ),
        main_action_controller=_namespace_with_slots(
            "handle_detection_threshold_change",
            "handle_rise_delta_threshold_change",
            "handle_fall_delta_threshold_change",
            "toggle_hist_detection",
            "toggle_delta_detection",
            "toggle_winner_takes_black",
            "handle_hand_assignment_toggle",
            "handle_histogram_threshold_change",
            "handle_similarity_ratio_change",
            "handle_calibrate_unlit_all_keys",
            "handle_calibrate_lit_exemplar_key_start",
            "handle_exemplar_key_type_enabled_change",
            "handle_refresh_selected_overlay_display",
            "handle_align_white_keys_to_selected",
            "handle_align_black_keys_to_selected",
            "handle_manual_fit_request",
            "handle_overlay_size_adjustment",
            "handle_octave_transpose_change",
            "handle_fps_override_change",
            "handle_overlay_color_change",
        ),
        calibration_wizard_controller=_namespace_with_slots("run_calibration_wizard"),
        calibration_effects_controller=SimpleNamespace(
            spark=_namespace_with_slots(
                "select_spark_roi",
                "set_spark_roi_visible",
                "request_spark_calibration",
                "start_auto_spark_calibration",
                "set_spark_detection_enabled",
                "set_spark_detection_sensitivity",
            ),
            overlay=_namespace_with_slots("handle_overlay_type_change"),
        ),
        midi_conversion_controller=_namespace_with_slots("start_conversion_process"),
        midi_touchup_controller=_namespace_with_slots("open_from_picker"),
    )

    ControlSignalManager(control_panel, main_window)

    connected = dict(connections)
    assert connected["nav_interval_changed"] is video_session_ui_controller.update_nav_interval
    assert connected["youtube_video_downloaded"] is video_session_ui_controller.handle_youtube_video_downloaded
    assert connected["video_to_frames_requested"] is video_session_ui_controller.handle_video_to_frames_request
    assert connected["start_frame_changed"] is video_session_ui_controller.handle_start_frame_change
    assert connected["end_frame_changed"] is video_session_ui_controller.handle_end_frame_change
    assert connected["trim_video_requested"] is video_session_ui_controller.handle_trim_video_request
    assert connected["processing_start_frame_changed"] is video_session_ui_controller.handle_processing_start_frame_change
    assert connected["processing_end_frame_changed"] is video_session_ui_controller.handle_processing_end_frame_change
    assert connected["frame_slider.valueChanged"] is main_window.video_controls.on_frame_slider_changed
