import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "synthesia2midi"


IMPORT_SMOKE_MODULES = [
    # Script entry points: importing them must not launch setup, FFmpeg, or GUI work.
    "run",
    "setup_env",
    # The package launcher/main modules configure logging at import time. The test
    # subprocess redirects logs to tmp_path and relies on __main__ guards to avoid
    # QApplication/window creation.
    "synthesia2midi.main",
    "synthesia2midi.run",
    # Core configuration and state.
    "synthesia2midi.app_config",
    "synthesia2midi.config_manager",
    "synthesia2midi.core.app_state",
    "synthesia2midi.core.logging_config",
    "synthesia2midi.core.state_manager",
    # Detection/auto-detection public surfaces and refactor-prone stage modules.
    "synthesia2midi.detection.auto_detect_adapter",
    "synthesia2midi.detection.auto_detect_param_specs",
    "synthesia2midi.detection.base",
    "synthesia2midi.detection.black_key_detector",
    "synthesia2midi.detection.black_note_assignment",
    "synthesia2midi.detection.black_note_center_map",
    "synthesia2midi.detection.black_residual_warp",
    "synthesia2midi.detection.detection_utils",
    "synthesia2midi.detection.detector_defaults",
    "synthesia2midi.detection.detector_geometry",
    "synthesia2midi.detection.detector_visualization",
    "synthesia2midi.detection.factory",
    "synthesia2midi.detection.hand_detection",
    "synthesia2midi.detection.hsv_cache_strategy",
    "synthesia2midi.detection.monolithic_detector",
    "synthesia2midi.detection.note_assignment",
    "synthesia2midi.detection.note_parsing",
    "synthesia2midi.detection.roi_cache",
    "synthesia2midi.detection.roi_utils",
    "synthesia2midi.detection.spark_calibration",
    "synthesia2midi.detection.spark_integrated",
    "synthesia2midi.detection.spark_mapper",
    "synthesia2midi.detection.standard",
    "synthesia2midi.detection.white_key_boundary_solver",
    "synthesia2midi.detection.white_key_geometry",
    "synthesia2midi.detection.white_key_lattice_model",
    "synthesia2midi.detection.white_key_lattice_solver",
    "synthesia2midi.detection.white_note_assignment",
    # File/video/MIDI helpers: import only, no local media, ffmpeg, or network calls.
    "synthesia2midi.frame_cache",
    "synthesia2midi.image_sequence_loader",
    "synthesia2midi.midi_generator",
    "synthesia2midi.midi_reader",
    "synthesia2midi.utils.ffmpeg_helper",
    "synthesia2midi.utils.font_helper",
    "synthesia2midi.video_loader",
    "synthesia2midi.youtube_downloader",
    # GUI modules and controllers: import-only smoke, no QApplication/window init.
    "synthesia2midi.gui.auto_detect_tuning_dialog",
    "synthesia2midi.gui.calibration_effects_controller",
    "synthesia2midi.gui.calibration_interaction_controller",
    "synthesia2midi.gui.calibration_wizard_controller",
    "synthesia2midi.gui.canvas.coordinates",
    "synthesia2midi.gui.canvas.interaction",
    "synthesia2midi.gui.controls_qt",
    "synthesia2midi.gui.display_manager",
    "synthesia2midi.gui.keyboard_canvas",
    "synthesia2midi.gui.main_action_controller",
    "synthesia2midi.gui.midi_conversion_controller",
    "synthesia2midi.gui.midi_touchup_controller",
    "synthesia2midi.gui.overlay_interaction_controller",
    "synthesia2midi.gui.shadow_calibration_controller",
    "synthesia2midi.gui.signal_manager",
    "synthesia2midi.gui.spark_calibration_controller",
    "synthesia2midi.gui.spinbox_utils",
    "synthesia2midi.gui.startup_dialog",
    "synthesia2midi.gui.ui_update_interface",
    "synthesia2midi.gui.video_controls",
    "synthesia2midi.gui.video_session_ui_controller",
    "synthesia2midi.gui.window_manager",
    "synthesia2midi.gui.wizard",
    "synthesia2midi.gui.youtube_download_dialog",
    # Workflow orchestration modules.
    "synthesia2midi.workflows.auto_calibration",
    "synthesia2midi.workflows.calibration",
    "synthesia2midi.workflows.conversion",
    "synthesia2midi.workflows.detection_manager",
    "synthesia2midi.workflows.overlay_manager",
    "synthesia2midi.workflows.parameter_manager",
    "synthesia2midi.workflows.midi_export",
    "synthesia2midi.workflows.video_loading",
    "synthesia2midi.workflows.video_session_coordinator",
    "synthesia2midi.workflows.video_to_frames",
]


def test_key_modules_import_in_clean_process_without_external_resources(tmp_path):
    script = textwrap.dedent(
        """
        import importlib
        import json
        import os
        import sys

        modules = json.loads(os.environ["S2M_IMPORT_SMOKE_MODULES"])
        for module in modules:
            importlib.import_module(module)
        """
    )

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PACKAGE_ROOT) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["SYNTHESIA2MIDI_LOG_DIR"] = str(tmp_path / "logs")
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    env["XDG_CACHE_HOME"] = str(tmp_path / "xdg-cache")
    env["S2M_IMPORT_SMOKE_MODULES"] = json.dumps(IMPORT_SMOKE_MODULES)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
