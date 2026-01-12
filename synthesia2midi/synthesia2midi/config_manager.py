"""
Configuration management for synthesia2midi application state.

Handles loading and saving of application configuration using INI files for
settings and JSON files for overlay data. Manages the complex process of
persisting detection parameters, calibration data, and overlay configurations
across application sessions.

Key Responsibilities:
- INI file management for detection settings and calibration data
- JSON file management for overlay configurations and exemplar histograms
- State validation and error handling during load/save operations
- Backward compatibility with previous configuration formats
- Automatic file path generation based on video file names
"""
import configparser
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from synthesia2midi.app_config import NOTE_NAMES_SHARP, OverlayConfig
from synthesia2midi.core.app_state import AppState


class ConfigManager:
    """Handles INI file operations for application state."""

    def __init__(self, app_state: AppState):
        self.app_state = app_state

    def _get_ini_path(self, video_filepath: str) -> str:
        """Generates the INI filepath based on the video filepath."""
        if not video_filepath:
            # If video_filepath is empty, it might be a direct call with a template path
            # or an error. Handle gracefully or let load_config manage it.
            return "" # Return empty, load_config will handle non-existence or use direct path
        # Normalize path separators to fix cross-platform compatibility
        normalized_path = os.path.normpath(video_filepath)
        base, _ = os.path.splitext(normalized_path)
        return f"{base}.ini"
    
    def _get_overlay_json_path(self, video_filepath: str) -> str:
        """Generates the overlay JSON filepath based on the video filepath."""
        if not video_filepath:
            return ""
        # Normalize path separators to fix cross-platform compatibility
        normalized_path = os.path.normpath(video_filepath)
        base, _ = os.path.splitext(normalized_path)
        return f"{base}_overlays.json"
    
    def _save_overlay_data(self, video_filepath: str) -> bool:
        """Save overlay data and exemplar histograms to JSON file."""
        overlay_json_path = self._get_overlay_json_path(video_filepath)
        if not overlay_json_path:
            logging.error("Failed to determine overlay JSON path for saving.")
            return False
        
        overlay_data = {
            "overlays": [],
            "exemplar_lit_histograms": {}
        }
        
        # Save overlay configurations
        for overlay in self.app_state.overlays:
            overlay_dict = {
                "key_id": overlay.key_id,
                "note_octave": overlay.note_octave,
                "note_name_in_octave": overlay.note_name_in_octave,
                "x": overlay.x,
                "y": overlay.y,
                "width": overlay.width,
                "height": overlay.height,
                "unlit_reference_color": list(overlay.unlit_reference_color) if overlay.unlit_reference_color else None,
                "key_type": overlay.key_type,
                "overlay_type": overlay.overlay_type if hasattr(overlay, 'overlay_type') else 'key',
                "unlit_hist": overlay.unlit_hist.tolist() if overlay.unlit_hist is not None else None
            }
            # Debug logging for unlit calibration
            if overlay.unlit_hist is not None:
                logging.debug(f"[CONFIG-SAVE] Key {overlay.key_id}: Saving unlit_hist with shape {overlay.unlit_hist.shape}")
            if overlay.unlit_reference_color:
                logging.debug(f"[CONFIG-SAVE] Key {overlay.key_id}: Saving unlit_reference_color {overlay.unlit_reference_color}")
            overlay_data["overlays"].append(overlay_dict)
        
        # Save exemplar lit histograms
        for key_type_abbr, hist_array in self.app_state.detection.exemplar_lit_histograms.items():
            if hist_array is not None:
                overlay_data["exemplar_lit_histograms"][key_type_abbr] = hist_array.tolist()
            else:
                overlay_data["exemplar_lit_histograms"][key_type_abbr] = None
        
        try:
            with open(overlay_json_path, 'w', encoding='utf-8') as f:
                json.dump(overlay_data, f, indent=2)
            logging.info(f"Overlay data saved to: {overlay_json_path}")
            logging.info(f"[CONFIG-SAVE] [OK] Overlay data saved to: {overlay_json_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving overlay data to {overlay_json_path}: {e}")
            return False
    
    def _load_overlay_data(self, video_filepath: str) -> bool:
        """Load overlay data and exemplar histograms from JSON file."""
        overlay_json_path = self._get_overlay_json_path(video_filepath)
        if not overlay_json_path or not os.path.exists(overlay_json_path):
            logging.info(f"Overlay JSON file not found: {overlay_json_path}")
            return False
        
        try:
            with open(overlay_json_path, 'r', encoding='utf-8') as f:
                overlay_data = json.load(f)
            
            # Clear existing overlays
            self.app_state.overlays.clear()
            
            # Load overlays
            for overlay_dict in overlay_data.get("overlays", []):
                overlay = OverlayConfig(
                    key_id=overlay_dict["key_id"],
                    note_octave=overlay_dict["note_octave"],
                    note_name_in_octave=overlay_dict["note_name_in_octave"],
                    x=overlay_dict["x"],
                    y=overlay_dict["y"],
                    width=overlay_dict["width"],
                    height=overlay_dict["height"],
                    unlit_reference_color=tuple(overlay_dict["unlit_reference_color"]) if overlay_dict["unlit_reference_color"] else None,
                    key_type=overlay_dict["key_type"],
                    overlay_type=overlay_dict.get("overlay_type", "key")  # Default to 'key' for backward compatibility
                )
                
                
                # Load histogram if present
                if overlay_dict.get("unlit_hist") is not None:
                    overlay.unlit_hist = np.array(overlay_dict["unlit_hist"])
                    logging.debug(f"[CONFIG-LOAD] Key {overlay.key_id}: Loaded unlit_hist with shape {overlay.unlit_hist.shape}")
                else:
                    logging.debug(f"[CONFIG-LOAD] Key {overlay.key_id}: No unlit_hist found")
                
                # Log unlit reference color
                if overlay.unlit_reference_color:
                    logging.debug(f"[CONFIG-LOAD] Key {overlay.key_id}: Loaded unlit_reference_color {overlay.unlit_reference_color}")
                else:
                    logging.debug(f"[CONFIG-LOAD] Key {overlay.key_id}: No unlit_reference_color found")
                
                self.app_state.overlays.append(overlay)
            
            # Load exemplar lit histograms
            for key_type_abbr, hist_list in overlay_data.get("exemplar_lit_histograms", {}).items():
                if hist_list is not None:
                    self.app_state.detection.exemplar_lit_histograms[key_type_abbr] = np.array(hist_list)
                else:
                    self.app_state.detection.exemplar_lit_histograms[key_type_abbr] = None
            
            # Debug: Log unlit calibration status after loading
            unlit_count = 0
            for overlay in self.app_state.overlays:
                if overlay.unlit_hist is not None or overlay.unlit_reference_color is not None:
                    unlit_count += 1
                    logging.info(f"[CONFIG-LOAD-DEBUG] Overlay {overlay.key_id} loaded with unlit calibration: "
                               f"hist={overlay.unlit_hist is not None}, "
                               f"color={overlay.unlit_reference_color}")
            logging.info(f"[CONFIG-LOAD-DEBUG] Total overlays with unlit calibration after load: {unlit_count}/{len(self.app_state.overlays)}")
            
            logging.info(f"Loaded {len(self.app_state.overlays)} overlays from JSON")
            logging.info(f"Loaded {len(self.app_state.detection.exemplar_lit_histograms)} exemplar histograms")
            logging.info(f"[CONFIG-LOAD] [OK] Overlay data loaded from: {overlay_json_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading overlay data from {overlay_json_path}: {e}")
            return False

    def load_config(self, config_filepath: str, is_template: bool = False) -> bool:
        """Load configuration from file. config_filepath can be a video-specific path or a template path."""
        logging.info(f"Attempting to load config from: {config_filepath}, is_template: {is_template}")
        if not config_filepath: # Check if the provided path is empty
            logging.warning("load_config called with an empty config_filepath.")
            return False
            
        # The config_filepath is now the direct path to the .ini file (either video-specific or template)
        ini_path = config_filepath 

        if not os.path.exists(ini_path):
            # This log might be redundant if main.py already checks existence, but good for direct calls
            logging.info(f"INI file not found at: {ini_path}")
            return False
            
        try:
            config = configparser.ConfigParser()
            config.read(ini_path, encoding='utf-8')
            
            logging.debug(f"Overlays before loading {len(self.app_state.overlays)}: {self.app_state.overlays}")
            # Clear existing overlays
            self.app_state.overlays.clear()
            
            # Load general settings if a [Settings] section exists
            if config.has_section('Settings'):
                settings_data = config['Settings']
                # Legacy: ignore old sensitivity setting
                settings_data.getint('sensitivity', 0)  # Read but don't use
                self.app_state.midi.tempo = settings_data.getint('tempo', self.app_state.midi.tempo)
                self.app_state.video.current_nav_interval = settings_data.getint('current_frame_nav_interval', self.app_state.video.current_nav_interval)
                self.app_state.video.start_frame = settings_data.getint('start_frame', self.app_state.video.start_frame)
                self.app_state.video.end_frame = settings_data.getint('end_frame', self.app_state.video.end_frame)
                self.app_state.video.processing_start_frame = settings_data.getint('processing_start_frame', self.app_state.video.processing_start_frame)
                self.app_state.video.processing_end_frame = settings_data.getint('processing_end_frame', self.app_state.video.processing_end_frame)
                self.app_state.video.trim_start_frame = settings_data.getint('trim_start_frame', self.app_state.video.trim_start_frame)
                self.app_state.video.trim_end_frame = settings_data.getint('trim_end_frame', self.app_state.video.trim_end_frame)
                self.app_state.video.video_is_trimmed = settings_data.getboolean('video_is_trimmed', self.app_state.video.video_is_trimmed)
                self.app_state.ui.show_overlays = settings_data.getboolean('show_overlays', self.app_state.ui.show_overlays)
                self.app_state.ui.live_detection_feedback = settings_data.getboolean('live_detection_feedback', self.app_state.ui.live_detection_feedback)
                self.app_state.detection.detection_threshold = settings_data.getfloat('detection_threshold', self.app_state.detection.detection_threshold)
                self.app_state.detection.hist_ratio_threshold = settings_data.getfloat('hist_ratio_threshold', self.app_state.detection.hist_ratio_threshold)
                self.app_state.detection.use_histogram_detection = settings_data.getboolean('use_histogram_detection', self.app_state.detection.use_histogram_detection)
                self.app_state.detection.use_delta_detection = settings_data.getboolean('use_delta_detection', self.app_state.detection.use_delta_detection)
                self.app_state.detection.spark_detection_enabled = settings_data.getboolean('spark_detection_enabled', self.app_state.detection.spark_detection_enabled)
                self.app_state.detection.hand_assignment_enabled = settings_data.getboolean('hand_assignment_enabled', self.app_state.detection.hand_assignment_enabled)
                self.app_state.detection.rise_delta_threshold = settings_data.getfloat('rise_delta_threshold', self.app_state.detection.rise_delta_threshold)
                self.app_state.detection.fall_delta_threshold = settings_data.getfloat('fall_delta_threshold', self.app_state.detection.fall_delta_threshold)
                self.app_state.detection.winner_takes_black_enabled = settings_data.getboolean('winner_takes_black_enabled', self.app_state.detection.winner_takes_black_enabled)
                self.app_state.detection.similarity_ratio = settings_data.getfloat('similarity_ratio', self.app_state.detection.similarity_ratio)
                self.app_state.video.current_frame_index = settings_data.getint('current_video_frame_index', self.app_state.video.current_frame_index)
                self.app_state.ui.visual_threshold_monitor_enabled = settings_data.getboolean('visual_threshold_monitor_enabled', self.app_state.ui.visual_threshold_monitor_enabled)
                self.app_state.ui.overlay_color = settings_data.get('overlay_color', self.app_state.ui.overlay_color)
                # FPS settings - always use 30.0 FPS regardless of config file value
                config_fps = settings_data.getfloat('fps', 30.0)
                if config_fps != 30.0:
                    logging.info(f"Config file has fps={config_fps}, but forcing to 30.0 FPS for consistent timing")
                self.app_state.video.fps = 30.0
                fps_override_str = settings_data.get('fps_override', '')
                self.app_state.video.fps_override = float(fps_override_str) if fps_override_str else None
                # Hand detection hue calibration
                self.app_state.detection.left_hand_hue_mean = settings_data.getfloat('left_hand_hue_mean', self.app_state.detection.left_hand_hue_mean)
                self.app_state.detection.right_hand_hue_mean = settings_data.getfloat('right_hand_hue_mean', self.app_state.detection.right_hand_hue_mean)
                self.app_state.detection.hand_detection_calibrated = settings_data.getboolean('hand_detection_calibrated', self.app_state.detection.hand_detection_calibrated)
                # Octave transpose setting
                self.app_state.midi.octave_transpose = settings_data.getint('octave_transpose', self.app_state.midi.octave_transpose)
                
                # Log loaded hand detection values
                logging.debug(f"[CONFIG-LOAD] Loaded hand detection calibration:")
                logging.debug(f"[CONFIG-LOAD]   left_hand_hue_mean: {self.app_state.detection.left_hand_hue_mean}")
                logging.debug(f"[CONFIG-LOAD]   right_hand_hue_mean: {self.app_state.detection.right_hand_hue_mean}")
                logging.debug(f"[CONFIG-LOAD]   hand_detection_calibrated: {self.app_state.detection.hand_detection_calibrated}")
                # Add other AppState fields here if they need to be loaded

            # Try to load overlay data from JSON file first (new format)
            # Extract the base path from the INI file path (remove .ini extension)
            base_path = config_filepath[:-4] if config_filepath.endswith('.ini') else config_filepath
            
            # Try multiple strategies to find overlay data
            overlay_data_loaded = False
            
            # Strategy 1: Check if we have a video filepath stored in the config
            if config.has_section('Video') and config.has_option('Video', 'filepath'):
                stored_video_path = config['Video']['filepath']
                if os.path.exists(stored_video_path):
                    logging.info(f"[CONFIG-LOAD] Using video path from config: {stored_video_path}")
                    overlay_data_loaded = self._load_overlay_data(stored_video_path)
            
            # Strategy 2: Try to find video file with same base name as INI
            if not overlay_data_loaded:
                video_path = None
                for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                    potential_path = base_path + ext
                    if os.path.exists(potential_path):
                        video_path = potential_path
                        break
                
                if video_path:
                    logging.info(f"[CONFIG-LOAD] Found video file: {video_path}")
                    overlay_data_loaded = self._load_overlay_data(video_path)
            
            # Strategy 3: Try to load JSON directly with expected name
            if not overlay_data_loaded:
                overlay_json_path = base_path + "_overlays.json"
                if os.path.exists(overlay_json_path):
                    logging.info(f"[CONFIG-LOAD] Loading overlay JSON directly: {overlay_json_path}")
                    # Create a dummy video path to generate the correct JSON filename
                    dummy_video_path = base_path + ".mp4"
                    overlay_data_loaded = self._load_overlay_data(dummy_video_path)
                else:
                    logging.info(f"[CONFIG-LOAD] No overlay JSON file found at: {overlay_json_path}")
            
            # Fallback: Load overlays from individual INI sections (legacy format)
            if not overlay_data_loaded:
                logging.info("JSON overlay data not found, loading from INI sections (legacy format)")
                for section in config.sections():
                    if section.startswith('Overlay_'):
                        try:
                            overlay_data = config[section]
                            
                            unlit_ref_color_str = overlay_data.get('unlit_reference_color', '')
                            unlit_reference_color = None
                            if unlit_ref_color_str:
                                try:
                                    unlit_reference_color = tuple(map(int, unlit_ref_color_str.split(',')))
                                except ValueError:
                                    logging.warning(f"Could not parse unlit_reference_color '{unlit_ref_color_str}' for {section}")
                            
                            key_type_str = overlay_data.get('key_type', '')
                            key_type = key_type_str if key_type_str else None

                            # If key_type is still None after loading (e.g., from an older INI), assign a default
                            if key_type is None:
                                try:
                                    temp_key_id = int(overlay_data['key_id'])
                                    temp_note_name = overlay_data['note_name_in_octave']
                                    
                                    white_key_names = {name for name in NOTE_NAMES_SHARP if "♯" not in name and "♭" not in name}
                                    is_white_key = temp_note_name in white_key_names
                                    
                                    # Default L/R split (consistent with wizard)
                                    hand_prefix = "L" if temp_key_id < 39 else "R" 
                                    color_suffix = "W" if is_white_key else "B"
                                    key_type = f"{hand_prefix}{color_suffix}"
                                    logging.info(f"Assigned default key_type '{key_type}' to Overlay {temp_key_id} during load.")
                                except Exception as e_kt:
                                    logging.warning(f"Could not assign default key_type for Overlay section {section}: {e_kt}")
                                    # key_type remains None if default assignment fails

                            overlay = OverlayConfig(
                                key_id=int(overlay_data['key_id']),
                                note_octave=int(overlay_data['note_octave']),
                                note_name_in_octave=overlay_data['note_name_in_octave'],
                                x=float(overlay_data['x']),
                                y=float(overlay_data['y']),
                                width=float(overlay_data['width']),
                                height=float(overlay_data['height']),
                                unlit_reference_color=unlit_reference_color,
                                key_type=key_type # Use the potentially defaulted key_type
                            )
                            
                            # Load histograms if present
                            unlit_hist_str = overlay_data.get('unlit_hist', '')
                            if unlit_hist_str:
                                try:
                                    hist_values = list(map(float, unlit_hist_str.split(',')))
                                    overlay.unlit_hist = np.array(hist_values)
                                except ValueError as e:
                                    logging.warning(f"Could not parse unlit_hist for overlay {overlay.key_id}: {e}")
                            
                            # Note: per-overlay lit_hist is not loaded; exemplar histograms are used for lit references.
                            
                            self.app_state.overlays.append(overlay)
                        except (ValueError, KeyError) as e:
                            logging.error(f"Error loading overlay from section {section}: {e}")
                            continue
                
                # Load Exemplar Lit Histograms from INI (legacy format)
                if config.has_section('ExemplarLitHistograms'):
                    for key_type_abbr, hist_str in config.items('ExemplarLitHistograms'):
                        try:
                            if hist_str: # Ensure not empty string
                                hist_values = list(map(float, hist_str.split(',')))
                                self.app_state.detection.exemplar_lit_histograms[key_type_abbr.upper()] = np.array(hist_values)
                            else:
                                self.app_state.detection.exemplar_lit_histograms[key_type_abbr.upper()] = None
                        except ValueError as e:
                            logging.error(f"Error loading exemplar lit histogram '{key_type_abbr}': {hist_str} - {e}")
                            continue
            
            logging.info(f"Overlays loaded ({len(self.app_state.overlays)}):")
            for o_idx, ov in enumerate(self.app_state.overlays):
                logging.info(f"  Overlay {o_idx}: id={ov.key_id}, x={ov.x}, y={ov.y}, w={ov.width}, h={ov.height}, type={ov.key_type}")

            # ColorMap INI section is ignored (color-to-channel mapping is not loaded here).

            # Load Exemplar Lit Colors
            if config.has_section('ExemplarLitColors'):
                for key_type_abbr, color_str in config.items('ExemplarLitColors'):
                    try:
                        if color_str: # Ensure not empty string
                             r, g, b = map(int, color_str.split(','))
                             self.app_state.detection.exemplar_lit_colors[key_type_abbr.upper()] = (r, g, b) # Ensure key is upper for consistency
                        else:
                            self.app_state.detection.exemplar_lit_colors[key_type_abbr.upper()] = None
                    except ValueError as e:
                        logging.error(f"Error loading exemplar lit color '{key_type_abbr}': {color_str} - {e}")
                        continue

            # Note: ExemplarLitHistograms now loaded from JSON or INI fallback above
            
            # Load Spark Detection settings
            if config.has_section('SparkDetection'):
                spark_data = config['SparkDetection']
                self.app_state.detection.spark_roi_top = spark_data.getint('spark_roi_top', self.app_state.detection.spark_roi_top)
                self.app_state.detection.spark_roi_bottom = spark_data.getint('spark_roi_bottom', self.app_state.detection.spark_roi_bottom)
                self.app_state.detection.spark_roi_visible = spark_data.getboolean('spark_roi_visible', self.app_state.detection.spark_roi_visible)
                self.app_state.detection.spark_brightness_threshold = spark_data.getfloat('spark_brightness_threshold', self.app_state.detection.spark_brightness_threshold)
                # Handle legacy parameter name
                if 'spark_off_threshold' in spark_data:
                    # Convert legacy spark_off_threshold to spark_detection_sensitivity (approximate mapping)
                    legacy_threshold = spark_data.getfloat('spark_off_threshold', 0.12)
                    self.app_state.detection.spark_detection_sensitivity = min(1.0, legacy_threshold * 4.0)  # Scale 0.12->0.48
                else:
                    self.app_state.detection.spark_detection_sensitivity = spark_data.getfloat('spark_detection_sensitivity', self.app_state.detection.spark_detection_sensitivity)
                self.app_state.detection.spark_detection_confidence = spark_data.getfloat('spark_detection_confidence', self.app_state.detection.spark_detection_confidence)
            
            # Load Spark Calibration data
            if config.has_section('SparkCalibration'):
                import json
                
                def load_calibration_dict(json_str):
                    """Convert JSON string back to calibration dict."""
                    if not json_str:
                        return None
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        return None
                
                spark_calib = config['SparkCalibration']
                self.app_state.detection.spark_calibration_background = load_calibration_dict(spark_calib.get('background', ''))
                self.app_state.detection.spark_calibration_lh_bar_only = load_calibration_dict(spark_calib.get('lh_bar_only', ''))
                self.app_state.detection.spark_calibration_lh_dimmest_sparks = load_calibration_dict(spark_calib.get('lh_dimmest_sparks', ''))
                self.app_state.detection.spark_calibration_lh_brightest_sparks = load_calibration_dict(spark_calib.get('lh_brightest_sparks', ''))
                self.app_state.detection.spark_calibration_rh_bar_only = load_calibration_dict(spark_calib.get('rh_bar_only', ''))
                self.app_state.detection.spark_calibration_rh_dimmest_sparks = load_calibration_dict(spark_calib.get('rh_dimmest_sparks', ''))
                self.app_state.detection.spark_calibration_rh_brightest_sparks = load_calibration_dict(spark_calib.get('rh_brightest_sparks', ''))
                self.app_state.detection.spark_calibration_bar_only = load_calibration_dict(spark_calib.get('bar_only', ''))
                self.app_state.detection.spark_calibration_dimmest_sparks = load_calibration_dict(spark_calib.get('dimmest_sparks', ''))
                
                # Load key-type-specific calibration data
                self.app_state.detection.spark_calibration_lw_bar_only = load_calibration_dict(spark_calib.get('lw_bar_only', ''))
                self.app_state.detection.spark_calibration_lw_dimmest_sparks = load_calibration_dict(spark_calib.get('lw_dimmest_sparks', ''))
                self.app_state.detection.spark_calibration_lw_brightest_sparks = load_calibration_dict(spark_calib.get('lw_brightest_sparks', ''))
                self.app_state.detection.spark_calibration_lb_bar_only = load_calibration_dict(spark_calib.get('lb_bar_only', ''))
                self.app_state.detection.spark_calibration_lb_dimmest_sparks = load_calibration_dict(spark_calib.get('lb_dimmest_sparks', ''))
                self.app_state.detection.spark_calibration_lb_brightest_sparks = load_calibration_dict(spark_calib.get('lb_brightest_sparks', ''))
                self.app_state.detection.spark_calibration_rw_bar_only = load_calibration_dict(spark_calib.get('rw_bar_only', ''))
                self.app_state.detection.spark_calibration_rw_dimmest_sparks = load_calibration_dict(spark_calib.get('rw_dimmest_sparks', ''))
                self.app_state.detection.spark_calibration_rw_brightest_sparks = load_calibration_dict(spark_calib.get('rw_brightest_sparks', ''))
                self.app_state.detection.spark_calibration_rb_bar_only = load_calibration_dict(spark_calib.get('rb_bar_only', ''))
                self.app_state.detection.spark_calibration_rb_dimmest_sparks = load_calibration_dict(spark_calib.get('rb_dimmest_sparks', ''))
                self.app_state.detection.spark_calibration_rb_brightest_sparks = load_calibration_dict(spark_calib.get('rb_brightest_sparks', ''))
            
            
            # After successful loading, manage the video_filepath_ini_used state
            if not is_template:
                # If it's not a template, it means this INI is specific to the self.app_state.video.filepath
                # So, record that this ini_path was used for the current video.
                self.app_state.video.filepath_ini_used = ini_path
            else:
                # If it IS a template, we clear video_filepath_ini_used.
                # This ensures that if the user saves, it saves to a NEW file based on app_state.video.filepath,
                # not overwriting the template file.
                self.app_state.video.filepath_ini_used = None
            logging.info(f"Config loaded. video_filepath_ini_used set to: {self.app_state.video.filepath_ini_used}")

            self.app_state.unsaved_changes = False # Loading a config means it's "saved" in that state
            return True
        except Exception as e:
            logging.error(f"Error loading config file: {e}")
            return False

    def save_config(self, video_filepath: str) -> bool:
        """Save current configuration to file."""
        if not video_filepath:
            logging.warning("save_config called with an empty video_filepath.")
            return False
        logging.info(f"Attempting to save config for video: {video_filepath}")
            
        # Determine the correct .ini path. This will be based on the video_filepath,
        # regardless of whether a template was loaded initially.
        ini_path = self._get_ini_path(video_filepath)
        if not ini_path: # Should not happen if video_filepath is valid
            logging.error("Failed to determine INI path for saving.")
            return False
        logging.info(f"Saving config to INI path: {ini_path}")

        config = configparser.ConfigParser()
        
        # Save video file path
        config['Video'] = {
            'filepath': video_filepath
        }
        
        # Save general AppState settings
        config['Settings'] = {
            'tempo': str(self.app_state.midi.tempo),
            'current_frame_nav_interval': str(self.app_state.video.current_nav_interval),
            'start_frame': str(self.app_state.video.start_frame),
            'end_frame': str(self.app_state.video.end_frame),
            'processing_start_frame': str(self.app_state.video.processing_start_frame),
            'processing_end_frame': str(self.app_state.video.processing_end_frame),
            'trim_start_frame': str(self.app_state.video.trim_start_frame),
            'trim_end_frame': str(self.app_state.video.trim_end_frame),
            'video_is_trimmed': str(self.app_state.video.video_is_trimmed),
            'show_overlays': str(self.app_state.ui.show_overlays),
            'live_detection_feedback': str(self.app_state.ui.live_detection_feedback),
            'detection_threshold': str(self.app_state.detection.detection_threshold),
            'hist_ratio_threshold': str(self.app_state.detection.hist_ratio_threshold),
            'use_histogram_detection': str(self.app_state.detection.use_histogram_detection),
            'use_delta_detection': str(self.app_state.detection.use_delta_detection),
            'spark_detection_enabled': str(self.app_state.detection.spark_detection_enabled),
            'hand_assignment_enabled': str(self.app_state.detection.hand_assignment_enabled),
            'rise_delta_threshold': str(self.app_state.detection.rise_delta_threshold),
            'fall_delta_threshold': str(self.app_state.detection.fall_delta_threshold),
            'winner_takes_black_enabled': str(self.app_state.detection.winner_takes_black_enabled),
            'similarity_ratio': str(self.app_state.detection.similarity_ratio),
            'current_video_frame_index': str(self.app_state.video.current_frame_index),
            'visual_threshold_monitor_enabled': str(self.app_state.ui.visual_threshold_monitor_enabled),
            'overlay_color': str(self.app_state.ui.overlay_color),
            # FPS settings - always save 30.0 for consistent timing
            'fps': '30.0',
            'fps_override': str(self.app_state.video.fps_override) if self.app_state.video.fps_override else '',
            # Hand detection hue calibration
            'left_hand_hue_mean': str(self.app_state.detection.left_hand_hue_mean),
            'right_hand_hue_mean': str(self.app_state.detection.right_hand_hue_mean),
            'hand_detection_calibrated': str(self.app_state.detection.hand_detection_calibrated),
            # Octave transpose
            'octave_transpose': str(self.app_state.midi.octave_transpose)
        }
        
        # Log hand detection values being saved
        logging.debug(f"[CONFIG-SAVE] Saving hand detection calibration:")
        logging.debug(f"[CONFIG-SAVE]   left_hand_hue_mean: {self.app_state.detection.left_hand_hue_mean}")
        logging.debug(f"[CONFIG-SAVE]   right_hand_hue_mean: {self.app_state.detection.right_hand_hue_mean}")
        logging.debug(f"[CONFIG-SAVE]   hand_detection_calibrated: {self.app_state.detection.hand_detection_calibrated}")
        
        # Save overlay data to separate JSON file
        # Debug: Log unlit calibration status before saving
        unlit_count = 0
        for overlay in self.app_state.overlays:
            if overlay.unlit_hist is not None or overlay.unlit_reference_color is not None:
                unlit_count += 1
                logging.info(f"[CONFIG-SAVE-DEBUG] Overlay {overlay.key_id} has unlit calibration: "
                           f"hist={overlay.unlit_hist is not None}, "
                           f"color={overlay.unlit_reference_color}")
        logging.info(f"[CONFIG-SAVE-DEBUG] Total overlays with unlit calibration: {unlit_count}/{len(self.app_state.overlays)}")
        
        overlay_save_success = self._save_overlay_data(video_filepath)
        if not overlay_save_success:
            logging.warning("Failed to save overlay data to JSON, but continuing with INI save")
        
        logging.info(f"Saving overlays ({len(self.app_state.overlays)}) to JSON file:")
        for o_idx, ov in enumerate(self.app_state.overlays):
            logging.info(f"  Overlay {o_idx}: id={ov.key_id}, x={ov.x}, y={ov.y}, w={ov.width}, h={ov.height}, type={ov.key_type}")
        
        # ColorMap INI section is not written (color-to-channel mapping is not persisted here).

        # Save Exemplar Lit Colors
        config['ExemplarLitColors'] = {}
        for key_type_abbr, color_tuple in self.app_state.detection.exemplar_lit_colors.items():
            if color_tuple:
                config['ExemplarLitColors'][key_type_abbr.lower()] = ",".join(map(str, color_tuple))
            else:
                 config['ExemplarLitColors'][key_type_abbr.lower()] = ""

        # Note: Exemplar Lit Histograms now saved to JSON file along with overlay data
        
        # Save Spark Detection Settings
        config['SparkDetection'] = {
            'spark_roi_top': str(self.app_state.detection.spark_roi_top),
            'spark_roi_bottom': str(self.app_state.detection.spark_roi_bottom),
            'spark_roi_visible': str(self.app_state.detection.spark_roi_visible),
            'spark_brightness_threshold': str(self.app_state.detection.spark_brightness_threshold),
            'spark_detection_sensitivity': str(self.app_state.detection.spark_detection_sensitivity),
            'spark_detection_confidence': str(self.app_state.detection.spark_detection_confidence)
        }
        
        # Save spark calibration data
        def save_calibration_dict(data_dict):
            """Convert calibration dict to string format for saving."""
            if data_dict is None:
                return ""
            # Save as JSON string for complex dictionary
            import json
            return json.dumps(data_dict)
        
        config['SparkCalibration'] = {
            'background': save_calibration_dict(self.app_state.detection.spark_calibration_background),
            'lh_bar_only': save_calibration_dict(self.app_state.detection.spark_calibration_lh_bar_only),
            'lh_dimmest_sparks': save_calibration_dict(self.app_state.detection.spark_calibration_lh_dimmest_sparks),
            'lh_brightest_sparks': save_calibration_dict(self.app_state.detection.spark_calibration_lh_brightest_sparks),
            'rh_bar_only': save_calibration_dict(self.app_state.detection.spark_calibration_rh_bar_only),
            'rh_dimmest_sparks': save_calibration_dict(self.app_state.detection.spark_calibration_rh_dimmest_sparks),
            'rh_brightest_sparks': save_calibration_dict(self.app_state.detection.spark_calibration_rh_brightest_sparks),
            'bar_only': save_calibration_dict(self.app_state.detection.spark_calibration_bar_only),
            'dimmest_sparks': save_calibration_dict(self.app_state.detection.spark_calibration_dimmest_sparks),
            # Key-type-specific calibration data
            'lw_bar_only': save_calibration_dict(self.app_state.detection.spark_calibration_lw_bar_only),
            'lw_dimmest_sparks': save_calibration_dict(self.app_state.detection.spark_calibration_lw_dimmest_sparks),
            'lw_brightest_sparks': save_calibration_dict(self.app_state.detection.spark_calibration_lw_brightest_sparks),
            'lb_bar_only': save_calibration_dict(self.app_state.detection.spark_calibration_lb_bar_only),
            'lb_dimmest_sparks': save_calibration_dict(self.app_state.detection.spark_calibration_lb_dimmest_sparks),
            'lb_brightest_sparks': save_calibration_dict(self.app_state.detection.spark_calibration_lb_brightest_sparks),
            'rw_bar_only': save_calibration_dict(self.app_state.detection.spark_calibration_rw_bar_only),
            'rw_dimmest_sparks': save_calibration_dict(self.app_state.detection.spark_calibration_rw_dimmest_sparks),
            'rw_brightest_sparks': save_calibration_dict(self.app_state.detection.spark_calibration_rw_brightest_sparks),
            'rb_bar_only': save_calibration_dict(self.app_state.detection.spark_calibration_rb_bar_only),
            'rb_dimmest_sparks': save_calibration_dict(self.app_state.detection.spark_calibration_rb_dimmest_sparks),
            'rb_brightest_sparks': save_calibration_dict(self.app_state.detection.spark_calibration_rb_brightest_sparks)
        }
        

        # Add timestamp comment at the top
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            with open(ini_path, 'w', encoding='utf-8') as configfile:
                configfile.write(f"# Configuration saved on {timestamp}\n")
                configfile.write(f"# synthesia2midi Configuration\n\n")
                config.write(configfile)
            # After a successful save, update video_filepath_ini_used to the path we just saved to.
            self.app_state.video.filepath_ini_used = ini_path
            logging.info(f"Config saved successfully at {timestamp}. video_filepath_ini_used set to: {self.app_state.video.filepath_ini_used}")
            logging.info(f"[CONFIG-SAVE] [OK] INI file saved successfully to: {ini_path}")
            logging.debug(f"[CONFIG-SAVE] Hand detection values in saved file:")
            logging.debug(f"[CONFIG-SAVE]   left_hand_hue_mean = {self.app_state.detection.left_hand_hue_mean}")
            logging.debug(f"[CONFIG-SAVE]   right_hand_hue_mean = {self.app_state.detection.right_hand_hue_mean}")
            logging.debug(f"[CONFIG-SAVE]   hand_detection_calibrated = {self.app_state.detection.hand_detection_calibrated}")
            self.app_state.unsaved_changes = False
            return True  # Return success even if overlay save failed - INI is the primary config
        except Exception as e:
            logging.error(f"Error writing config file: {e}")
            return False

    def parse_overlays_from_file(self, ini_filepath: str) -> Optional[List[OverlayConfig]]:
        """Parses only OverlayConfig objects from an INI file without modifying app_state."""
        if not ini_filepath or not os.path.exists(ini_filepath):
            logging.warning(f"parse_overlays_from_file: INI file not found or path empty: {ini_filepath}")
            return None

        parsed_overlays: List[OverlayConfig] = []
        try:
            config = configparser.ConfigParser()
            config.read(ini_filepath, encoding='utf-8')

            for section in config.sections():
                if section.startswith('Overlay_'):
                    try:
                        overlay_data = config[section]
                        
                        unlit_ref_color_str = overlay_data.get('unlit_reference_color', '')
                        unlit_reference_color = None
                        if unlit_ref_color_str:
                            try:
                                unlit_reference_color = tuple(map(int, unlit_ref_color_str.split(',')))
                            except ValueError:
                                logging.warning(f"Could not parse unlit_reference_color '{unlit_ref_color_str}' for {section} in {ini_filepath}")
                        
                        key_type_str = overlay_data.get('key_type', '')
                        key_type = key_type_str if key_type_str else None

                        # Basic check for essential fields to avoid crashing on malformed template overlays
                        if not all(k in overlay_data for k in ['key_id', 'note_octave', 'note_name_in_octave', 'x', 'y', 'width', 'height']):
                            logging.warning(f"Skipping overlay section {section} in {ini_filepath} due to missing essential fields.")
                            continue

                        overlay = OverlayConfig(
                            key_id=int(overlay_data['key_id']),
                            note_octave=int(overlay_data['note_octave']),
                            note_name_in_octave=overlay_data['note_name_in_octave'],
                            x=float(overlay_data['x']),
                            y=float(overlay_data['y']),
                            width=float(overlay_data['width']),
                            height=float(overlay_data['height']),
                            unlit_reference_color=unlit_reference_color,
                            key_type=key_type
                        )
                        parsed_overlays.append(overlay)
                    except (ValueError, KeyError) as e:
                        logging.error(f"Error parsing overlay from section {section} in {ini_filepath}: {e}")
                        continue # Skip this overlay but try others
            
            if not parsed_overlays:
                logging.info(f"No overlays found or parsed from {ini_filepath}")
                # Return None if no overlays were successfully parsed, to distinguish from an empty list from a valid file
                # However, an INI file might legitimately have settings but no overlays. Let's return empty list in that case.
                # The check for `if template_overlay_list:` in main.py will handle empty list.

            return parsed_overlays

        except configparser.Error as e:
            logging.error(f"ConfigParser error reading {ini_filepath}: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error parsing overlays from {ini_filepath}: {e}")
            return None
