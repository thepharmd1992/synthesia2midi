from types import SimpleNamespace

from PySide6.QtWidgets import QMessageBox

from synthesia2midi.core.color_families import SUPPORTED_EXEMPLAR_SLOTS
from synthesia2midi.gui.main_action_controller import MainActionController
from synthesia2midi.gui.signal_manager import ControlSignalManager


class FakeAppState:
    def __init__(self):
        self.unsaved_changes = False
        self.marked_unsaved = False
        self.detection = SimpleNamespace(
            rise_delta_threshold=0.1,
            fall_delta_threshold=0.1,
            winner_takes_black_enabled=False,
            hand_assignment_enabled=False,
            exemplar_key_type_enabled={
                slot: slot in {"LW", "LB"} for slot in SUPPORTED_EXEMPLAR_SLOTS
            },
            exemplar_lit_colors={slot: None for slot in SUPPORTED_EXEMPLAR_SLOTS},
            exemplar_lit_histograms={slot: None for slot in SUPPORTED_EXEMPLAR_SLOTS},
        )
        self.calibration = SimpleNamespace(calibration_mode=None, current_calibration_key_type=None)
        self.video = SimpleNamespace(fps_override=None)
        self.ui = SimpleNamespace(overlay_color="red", visual_threshold_monitor_enabled=False)
        self.midi = SimpleNamespace(octave_transpose=0)

    def mark_unsaved(self):
        self.marked_unsaved = True
        self.unsaved_changes = True


def test_exemplar_key_type_toggle_updates_state_and_cancels_active_calibration():
    app_state = FakeAppState()
    app_state.calibration.calibration_mode = "lit_exemplar"
    app_state.calibration.current_calibration_key_type = "LW"
    control_panel = SimpleNamespace(updated=0, update_controls_from_state=lambda: setattr(control_panel, "updated", control_panel.updated + 1))
    app = SimpleNamespace(app_state=app_state, control_panel=control_panel)

    MainActionController(app).handle_exemplar_key_type_enabled_change("LW", False)

    assert app_state.detection.exemplar_key_type_enabled["LW"] is False
    assert app_state.calibration.calibration_mode is None
    assert app_state.calibration.current_calibration_key_type is None
    assert app_state.unsaved_changes is True
    assert control_panel.updated == 1


def test_add_color_family_enables_first_unused_family_and_stops_at_four():
    app_state = FakeAppState()
    app_state.detection.exemplar_key_type_enabled.update(
        {"COLOR_3_W": True, "COLOR_3_B": True}
    )
    control_panel = SimpleNamespace(
        updated=0,
        update_controls_from_state=lambda: setattr(
            control_panel, "updated", control_panel.updated + 1
        ),
    )
    app = SimpleNamespace(app_state=app_state, control_panel=control_panel)
    controller = MainActionController(app)

    controller.handle_add_additional_color()

    assert app_state.detection.exemplar_key_type_enabled["RW"] is True
    assert app_state.detection.exemplar_key_type_enabled["RB"] is True
    assert app_state.unsaved_changes is True
    assert control_panel.updated == 1

    app_state.detection.exemplar_key_type_enabled.update(
        {slot: True for slot in SUPPORTED_EXEMPLAR_SLOTS}
    )
    controller.handle_add_additional_color()
    assert control_panel.updated == 1


def test_remove_calibrated_color_family_no_response_preserves_all_data(monkeypatch):
    app_state = FakeAppState()
    slots = ("COLOR_3_W", "COLOR_3_B")
    app_state.detection.exemplar_key_type_enabled.update({slot: True for slot in slots})
    app_state.detection.exemplar_lit_colors.update(
        {"COLOR_3_W": (230, 180, 30), "COLOR_3_B": (170, 120, 20)}
    )
    histograms = {"COLOR_3_W": object(), "COLOR_3_B": object()}
    app_state.detection.exemplar_lit_histograms.update(histograms)
    control_panel = SimpleNamespace(
        updated=0,
        update_controls_from_state=lambda: setattr(
            control_panel, "updated", control_panel.updated + 1
        ),
    )
    app = SimpleNamespace(app_state=app_state, control_panel=control_panel)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.No)

    MainActionController(app).handle_remove_additional_color(3)

    assert all(app_state.detection.exemplar_key_type_enabled[slot] for slot in slots)
    assert app_state.detection.exemplar_lit_colors["COLOR_3_W"] == (230, 180, 30)
    assert app_state.detection.exemplar_lit_colors["COLOR_3_B"] == (170, 120, 20)
    assert app_state.detection.exemplar_lit_histograms["COLOR_3_W"] is histograms["COLOR_3_W"]
    assert app_state.detection.exemplar_lit_histograms["COLOR_3_B"] is histograms["COLOR_3_B"]
    assert app_state.unsaved_changes is False
    assert control_panel.updated == 0


def test_remove_approved_color_family_clears_colors_histograms_and_flags(monkeypatch):
    app_state = FakeAppState()
    slots = ("COLOR_3_W", "COLOR_3_B")
    app_state.detection.exemplar_key_type_enabled.update({slot: True for slot in slots})
    app_state.detection.exemplar_lit_colors.update(
        {"COLOR_3_W": (230, 180, 30), "COLOR_3_B": (170, 120, 20)}
    )
    app_state.detection.exemplar_lit_histograms.update(
        {"COLOR_3_W": object(), "COLOR_3_B": object()}
    )
    control_panel = SimpleNamespace(updated=0)
    control_panel.update_controls_from_state = lambda: setattr(
        control_panel, "updated", control_panel.updated + 1
    )
    app = SimpleNamespace(app_state=app_state, control_panel=control_panel)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    MainActionController(app).handle_remove_additional_color(3)

    assert all(not app_state.detection.exemplar_key_type_enabled[slot] for slot in slots)
    assert all(app_state.detection.exemplar_lit_colors[slot] is None for slot in slots)
    assert all(app_state.detection.exemplar_lit_histograms[slot] is None for slot in slots)
    assert app_state.unsaved_changes is True
    assert control_panel.updated == 1


class _FakeSignal:
    def __init__(self):
        self.connected = []

    def connect(self, callback):
        self.connected.append(callback)


def test_signal_manager_connects_color_family_add_and_remove_actions():
    calibration_signal_names = (
        "calibrate_unlit_requested",
        "calibrate_lit_exemplar_requested",
        "exemplar_key_type_enabled_changed",
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
        "add_additional_color_requested",
        "remove_additional_color_requested",
    )
    control_panel = SimpleNamespace(
        **{name: _FakeSignal() for name in calibration_signal_names}
    )
    main_actions = SimpleNamespace(
        handle_calibrate_unlit_all_keys=lambda: None,
        handle_calibrate_lit_exemplar_key_start=lambda _slot: None,
        handle_exemplar_key_type_enabled_change=lambda _slot, _enabled: None,
        handle_refresh_selected_overlay_display=lambda: None,
        handle_align_white_keys_to_selected=lambda: None,
        handle_align_black_keys_to_selected=lambda: None,
        handle_manual_fit_request=lambda: None,
        handle_overlay_size_adjustment=lambda *_args: None,
        handle_add_additional_color=lambda: None,
        handle_remove_additional_color=lambda _family: None,
    )
    main_window = SimpleNamespace(
        main_action_controller=main_actions,
        calibration_wizard_controller=SimpleNamespace(run_calibration_wizard=lambda: None),
        midi_conversion_controller=SimpleNamespace(start_conversion_process=lambda: None),
        midi_touchup_controller=SimpleNamespace(open_from_picker=lambda: None),
        video_session_ui_controller=SimpleNamespace(handle_trim_video_request=lambda *_args: None),
        calibration_effects_controller=SimpleNamespace(
            spark=SimpleNamespace(
                select_spark_roi=lambda: None,
                set_spark_roi_visible=lambda _visible: None,
                request_spark_calibration=lambda _step: None,
                start_auto_spark_calibration=lambda _slot: None,
                set_spark_detection_enabled=lambda _enabled: None,
                set_spark_detection_sensitivity=lambda _value: None,
            ),
            overlay=SimpleNamespace(handle_overlay_type_change=lambda _value: None),
        ),
    )
    manager = ControlSignalManager.__new__(ControlSignalManager)
    manager.control_panel = control_panel
    manager.main_window = main_window
    manager.logger = SimpleNamespace(debug=lambda *_args: None)

    manager._connect_calibration_signals()

    assert control_panel.add_additional_color_requested.connected == [
        main_actions.handle_add_additional_color
    ]
    assert control_panel.remove_additional_color_requested.connected == [
        main_actions.handle_remove_additional_color
    ]


def test_overlay_color_and_octave_handlers_refresh_through_display_manager_and_mark_unsaved():
    app_state = FakeAppState()
    display_manager = SimpleNamespace(refreshed=0)
    display_manager.refresh_canvas_overlays = lambda: setattr(
        display_manager,
        "refreshed",
        display_manager.refreshed + 1,
    )
    canvas = SimpleNamespace(
        update=lambda: (_ for _ in ()).throw(AssertionError("direct canvas update not expected")),
        draw_overlays=lambda: (_ for _ in ()).throw(AssertionError("direct overlay draw not expected")),
    )
    app = SimpleNamespace(app_state=app_state, display_manager=display_manager, keyboard_canvas=canvas)
    controller = MainActionController(app)

    controller.handle_overlay_color_change("BLUE")
    controller.handle_octave_transpose_change(2)

    assert app_state.ui.overlay_color == "blue"
    assert app_state.midi.octave_transpose == 2
    assert display_manager.refreshed == 2
    assert app_state.marked_unsaved is True


def test_fps_override_updates_state_and_refreshes_detected_fps_display_when_loaded():
    app_state = FakeAppState()
    control_panel = SimpleNamespace(values=[], update_video_info=lambda value: control_panel.values.append(value))
    app = SimpleNamespace(app_state=app_state, video_session=SimpleNamespace(fps=29.97), control_panel=control_panel)

    MainActionController(app).handle_fps_override_change(60.0)

    assert app_state.video.fps_override == 60.0
    assert app_state.unsaved_changes is True
    assert control_panel.values == [29.97]
