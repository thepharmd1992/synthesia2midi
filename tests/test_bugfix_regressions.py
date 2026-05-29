import json
from types import SimpleNamespace

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.gui.startup_dialog import StartupDialog
from synthesia2midi.gui.youtube_download_dialog import YouTubeDownloadDialog
from synthesia2midi.core.app_state import AppState
from synthesia2midi.detection.factory import DetectionFactory
from synthesia2midi.detection.spark_integrated import SparkIntegratedDetection
from synthesia2midi.detection.standard import StandardDetection
from synthesia2midi.gui.auto_detect_tuning_controller import AutoDetectTuningController
from synthesia2midi.gui.calibration_interaction_controller import CalibrationInteractionController
from synthesia2midi.gui.calibration_wizard_controller import CalibrationWizardController
from synthesia2midi.gui.canvas.interaction import CanvasInteraction
from synthesia2midi.gui.main_action_controller import MainActionController
from synthesia2midi.gui.signal_manager import ControlSignalManager
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
            "calibration_wizard_requested",
            "refresh_overlay_display_requested",
            "align_white_keys_requested",
            "align_black_keys_requested",
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


def test_calibration_wizard_controller_keeps_wizard_for_keyboard_region_selection():
    wizard = DummyWizardForController("keyboard_region_selection_requested")
    workflow = DummyCalibrationWorkflowForController(wizard)
    selected_signal = RecordingSignal()
    interaction = SimpleNamespace(
        keyboard_region_selected=selected_signal,
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
        handle_refresh_selected_overlay_display=lambda: None,
        handle_align_white_keys_to_selected=lambda: None,
        handle_align_black_keys_to_selected=lambda: None,
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
