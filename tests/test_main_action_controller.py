from types import SimpleNamespace

from synthesia2midi.gui.main_action_controller import MainActionController


class FakeAppState:
    def __init__(self):
        self.unsaved_changes = False
        self.marked_unsaved = False
        self.detection = SimpleNamespace(
            rise_delta_threshold=0.1,
            fall_delta_threshold=0.1,
            winner_takes_black_enabled=False,
            hand_assignment_enabled=False,
            exemplar_key_type_enabled={"LW": True, "LB": True, "RW": True, "RB": True},
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
