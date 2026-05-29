from types import SimpleNamespace

from synthesia2midi.core.app_state import AppState
from synthesia2midi.gui.display_manager import DisplayManager
from synthesia2midi.gui.signal_manager import ControlSignalManager
from synthesia2midi.workflows.detection_manager import DetectionManager


class RecordingSignal:
    def __init__(self, name, connections):
        self.name = name
        self.connections = connections

    def connect(self, slot):
        self.connections[self.name] = slot


class DynamicSignalPanel:
    def __init__(self, connections):
        self._connections = connections

    def __getattr__(self, name):
        signal = RecordingSignal(name, self._connections)
        setattr(self, name, signal)
        return signal


def _slots(*names):
    return SimpleNamespace(**{name: (lambda *args, _name=name: None) for name in names})


def _assert_same_bound_method(actual, expected):
    assert actual.__self__ is expected.__self__
    assert actual.__func__ is expected.__func__


def _signal_manager_window(connections):
    return SimpleNamespace(
        frame_slider=SimpleNamespace(valueChanged=RecordingSignal("frame_slider.valueChanged", connections)),
        video_session_ui_controller=_slots(
            "update_nav_interval",
            "handle_youtube_video_downloaded",
            "handle_video_to_frames_request",
            "handle_start_frame_change",
            "handle_end_frame_change",
            "handle_trim_video_request",
            "handle_processing_start_frame_change",
            "handle_processing_end_frame_change",
        ),
        video_controls=_slots("on_frame_slider_changed"),
        detection_manager=_slots(
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
        main_action_controller=_slots(
            "handle_calibrate_unlit_all_keys",
            "handle_calibrate_lit_exemplar_key_start",
            "handle_exemplar_key_type_enabled_change",
            "handle_refresh_selected_overlay_display",
            "handle_align_white_keys_to_selected",
            "handle_align_black_keys_to_selected",
            "handle_overlay_size_adjustment",
            "handle_octave_transpose_change",
            "handle_fps_override_change",
            "handle_overlay_color_change",
        ),
        calibration_wizard_controller=_slots("run_calibration_wizard"),
        calibration_effects_controller=SimpleNamespace(
            spark=_slots(
                "select_spark_roi",
                "set_spark_roi_visible",
                "request_spark_calibration",
                "start_auto_spark_calibration",
                "set_spark_detection_enabled",
                "set_spark_detection_sensitivity",
            ),
            overlay=_slots("handle_overlay_type_change"),
        ),
        midi_conversion_controller=_slots("start_conversion_process"),
        midi_touchup_controller=_slots("open_from_picker"),
    )


def test_signal_manager_wires_detection_signals_directly_to_detection_manager():
    connections = {}
    control_panel = DynamicSignalPanel(connections)
    main_window = _signal_manager_window(connections)

    ControlSignalManager(control_panel, main_window)

    assert connections["detection_threshold_changed"] is main_window.detection_manager.set_detection_threshold
    assert connections["rise_delta_threshold_changed"] is main_window.detection_manager.set_rise_delta_threshold
    assert connections["fall_delta_threshold_changed"] is main_window.detection_manager.set_fall_delta_threshold
    assert connections["histogram_detection_toggled"] is main_window.detection_manager.set_histogram_detection_enabled
    assert connections["delta_detection_toggled"] is main_window.detection_manager.set_delta_detection_enabled
    assert connections["winner_takes_black_changed"] is main_window.detection_manager.set_winner_takes_black_enabled
    assert connections["hand_assignment_toggled"] is main_window.detection_manager.set_hand_assignment_enabled
    assert connections["histogram_threshold_changed"] is main_window.detection_manager.set_histogram_threshold
    assert connections["similarity_ratio_changed"] is main_window.detection_manager.set_similarity_ratio


def test_detection_manager_toggle_setters_use_emitted_boolean_values():
    app_state = AppState()
    app_state.detection.use_histogram_detection = False
    app_state.detection.use_delta_detection = True
    refreshes = []
    manager = DetectionManager(app_state, lambda: refreshes.append("refresh"))

    manager.set_histogram_detection_enabled(True)
    manager.set_delta_detection_enabled(False)

    assert app_state.detection.use_histogram_detection is True
    assert app_state.detection.use_delta_detection is False
    assert app_state.unsaved_changes is True


def test_detection_manager_numeric_setters_update_state_and_refresh_loaded_video():
    class LoadedUi:
        def has_video_loaded(self):
            return True

    app_state = AppState()
    refreshes = []
    manager = DetectionManager(app_state, lambda: refreshes.append("refresh"), LoadedUi())

    manager.set_rise_delta_threshold(0.31)
    manager.set_fall_delta_threshold(0.12)
    manager.set_histogram_threshold(0.44)
    manager.set_similarity_ratio(0.67)

    assert app_state.detection.rise_delta_threshold == 0.31
    assert app_state.detection.fall_delta_threshold == 0.12
    assert app_state.detection.hist_ratio_threshold == 0.44
    assert app_state.detection.similarity_ratio == 0.67
    assert app_state.unsaved_changes is True
    assert refreshes == ["refresh", "refresh", "refresh", "refresh"]


def test_display_manager_uses_checked_state_for_live_feedback_and_visual_monitor():
    class LoadedUi:
        def __init__(self):
            self.live_action_updates = []
            self.visual_action_updates = []
            self.refreshes = 0

        def has_video_loaded(self):
            return True

        def update_live_detection_action(self, checked):
            self.live_action_updates.append(checked)

        def update_visual_threshold_monitor_action(self, checked):
            self.visual_action_updates.append(checked)

        def refresh_canvas(self):
            self.refreshes += 1

    app_state = AppState()
    ui = LoadedUi()
    manager = DisplayManager(app_state, ui)

    manager.set_live_detection_feedback_enabled(False)
    manager.set_visual_threshold_monitor_enabled(True)

    assert app_state.ui.live_detection_feedback is False
    assert app_state.ui.visual_threshold_monitor_enabled is True
    assert app_state.unsaved_changes is True
    assert ui.live_action_updates == [False]
    assert ui.visual_action_updates == [True]
    assert ui.refreshes == 2
