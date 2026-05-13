import importlib


def test_core_import_smoke():
    modules = [
        "synthesia2midi.app_config",
        "synthesia2midi.config_manager",
        "synthesia2midi.core.app_state",
        "synthesia2midi.detection.factory",
        "synthesia2midi.detection.monolithic_detector",
        "synthesia2midi.detection.auto_detect_adapter",
        "synthesia2midi.detection.black_key_detector",
        "synthesia2midi.detection.black_note_assignment",
        "synthesia2midi.detection.black_note_center_map",
        "synthesia2midi.detection.black_residual_warp",
        "synthesia2midi.detection.detector_defaults",
        "synthesia2midi.detection.detector_geometry",
        "synthesia2midi.detection.detector_visualization",
        "synthesia2midi.detection.note_assignment",
        "synthesia2midi.detection.note_parsing",
        "synthesia2midi.detection.white_key_boundary_solver",
        "synthesia2midi.detection.white_key_geometry",
        "synthesia2midi.detection.white_key_lattice_model",
        "synthesia2midi.detection.white_key_lattice_solver",
        "synthesia2midi.detection.white_note_assignment",
        "synthesia2midi.midi_generator",
        "synthesia2midi.video_loader",
    ]

    for module in modules:
        importlib.import_module(module)
