"""
Calibration wizard for setting up initial key overlays.
"""
# Standard library imports
import copy
import logging
import os
from typing import List, Optional, Tuple

# Third-party imports
import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QSlider, QSpinBox, QVBoxLayout,
    QWidget
)

# Local imports
from synthesia2midi.app_config import (
    IDEALIZED_AVG_BLACK_KEY_WIDTH, IDEALIZED_AVG_WHITE_KEY_WIDTH,
    IDEALIZED_BLACK_KEY_HEIGHT, IDEALIZED_BLACK_KEY_X_START_FACTOR,
    IDEALIZED_BLACK_KEY_Y, IDEALIZED_WHITE_KEY_HEIGHT,
    IDEALIZED_WHITE_KEY_Y, NOTE_NAMES_SHARP, OverlayConfig
)
from synthesia2midi.core.app_state import AppState
from synthesia2midi.detection.auto_detect_adapter import AutoDetectAdapter      
from synthesia2midi.gui.spinbox_utils import install_spinbox_wheel_filter       


class AutoDetectTuningDialog(QDialog):
    """Popup dialog for tuning auto-detect parameters."""

    DEFAULT_PARAMS = {
        "type_aware_assignment": True,
        "black_threshold_method": "otsu",
        "black_threshold": 60,
        "black_column_ratio": 0.06,
        "black_min_width": 6,
        "black_max_width": 140,
        "black_adaptive_block_size": 31,
        "black_adaptive_c": 5,
        "black_recovery_enabled": True,
        "black_recovery_ratio": 0.5,
        "black_recovery_column_ratio_scale": 0.45,
        "black_split_max_factor": 1.6,
        "white_strip_dark_threshold": 75,
        "white_strip_dark_fraction": 0.03,
        "white_strip_min_run": 8,
        "white_strip_allow_failures": 1,
        "white_sep_ratio": 0.55,
        "white_sep_dyn_min": 8,
        "white_sep_close_kernel": 5,
        "white_sep_open_kernel": 3,
        "white_sep_min_width": 2,
        "white_initial_top_ratio": 0.65,
    }

    def __init__(
        self,
        parent,
        adapter: AutoDetectAdapter,
        frame: np.ndarray,
        keyboard_region: Tuple[int, int, int, int],
        apply_callback,
        initial_params: Optional[dict] = None,
        initial_results: Optional[dict] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Auto-Detect Tuning")
        self.setModal(True)
        self.adapter = adapter
        self.frame = frame
        self.keyboard_region = keyboard_region
        self.apply_callback = apply_callback
        self._suppress_updates = True
        self._last_detection_results = initial_results

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._run_detection)

        self._controls = {}
        self._base_params = {}

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Adjust sliders to refine lenient auto-detection. Changes rerun detection automatically."))

        params = self.DEFAULT_PARAMS.copy()
        if initial_params:
            params.update(initial_params)
        self._base_params = params.copy()

        layout.addWidget(self._build_black_group(params))
        layout.addWidget(self._build_white_group(params))

        if initial_results:
            status_text = (
                f"Detected {initial_results['total_keys']} keys "
                f"(leftmost: {initial_results['leftmost_note']}{initial_results['leftmost_octave']})"
            )
        else:
            status_text = "Ready"
        self.status_label = QLabel(status_text)
        layout.addWidget(self.status_label)

        button_layout = QHBoxLayout()
        reset_button = QPushButton("Reset Sliders")
        reset_button.clicked.connect(self._reset_sliders)
        self.save_button = QPushButton("Save Settings")
        self.save_button.clicked.connect(self._on_save)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(reset_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        self._suppress_updates = False

    def _build_black_group(self, params: dict):
        group = QGroupBox("Black Key Detection")
        form = QFormLayout()

        self._add_choice_slider(
            form,
            "black_threshold_method",
            "Threshold method",
            ["fixed", "adaptive", "otsu"],
            params,
        )
        self._add_slider(form, "black_threshold", "Threshold", 20, 120, 1, 0, params)
        self._add_slider(form, "black_column_ratio", "Column ratio", 0.02, 0.15, 0.001, 3, params)
        self._add_slider(form, "black_min_width", "Min width", 4, 20, 1, 0, params)
        self._add_slider(form, "black_max_width", "Max width", 40, 220, 1, 0, params)
        self._add_slider(form, "black_adaptive_block_size", "Adaptive block size", 3, 101, 2, 0, params)
        self._add_slider(form, "black_adaptive_c", "Adaptive C", -20, 20, 1, 0, params)
        self._add_slider(form, "black_recovery_ratio", "Recovery ratio", 0.2, 0.9, 0.05, 2, params)
        self._add_slider(form, "black_recovery_column_ratio_scale", "Recovery column scale", 0.2, 1.0, 0.05, 2, params)
        self._add_slider(form, "black_split_max_factor", "Split max factor", 1.2, 3.0, 0.1, 2, params)

        group.setLayout(form)
        return group

    def _build_white_group(self, params: dict):
        group = QGroupBox("White Key Detection")
        form = QFormLayout()

        self._add_slider(form, "white_strip_dark_threshold", "Strip dark threshold", 40, 120, 1, 0, params)
        self._add_slider(form, "white_strip_dark_fraction", "Strip dark fraction", 0.01, 0.08, 0.001, 3, params)
        self._add_slider(form, "white_strip_min_run", "Strip min run", 2, 20, 1, 0, params)
        self._add_slider(form, "white_strip_allow_failures", "Strip allow failures", 0, 4, 1, 0, params)
        self._add_slider(form, "white_sep_ratio", "Separator ratio", 0.35, 0.75, 0.01, 2, params)
        self._add_slider(form, "white_sep_dyn_min", "Separator dyn min", 0, 20, 1, 0, params)
        self._add_slider(form, "white_sep_close_kernel", "Separator close kernel", 1, 15, 1, 0, params)
        self._add_slider(form, "white_sep_open_kernel", "Separator open kernel", 1, 15, 1, 0, params)
        self._add_slider(form, "white_sep_min_width", "Separator min width", 1, 8, 1, 0, params)
        self._add_slider(form, "white_initial_top_ratio", "Top ratio", 0.4, 0.9, 0.01, 2, params)

        group.setLayout(form)
        return group

    def _add_slider(self, form, key, label, min_val, max_val, step, decimals, params):
        scale = 10 ** decimals
        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(round(min_val * scale)), int(round(max_val * scale)))
        slider.setSingleStep(int(round(step * scale)))
        value = float(params.get(key, self.DEFAULT_PARAMS.get(key, min_val)))
        slider.setValue(int(round(value * scale)))

        value_label = QLabel()

        def update_label(val):
            display = val / scale
            if decimals > 0:
                value_label.setText(f"{display:.{decimals}f}")
            else:
                value_label.setText(str(int(display)))

        update_label(slider.value())

        slider.valueChanged.connect(update_label)
        slider.valueChanged.connect(self._schedule_detection)

        row = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(slider)
        row_layout.addWidget(value_label)
        row.setLayout(row_layout)

        form.addRow(QLabel(label), row)
        self._controls[key] = {
            "slider": slider,
            "scale": scale,
            "decimals": decimals,
        }

    def _add_choice_slider(self, form, key, label, choices, params):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, len(choices) - 1)
        value = params.get(key, self.DEFAULT_PARAMS.get(key, choices[0]))
        try:
            idx = choices.index(value)
        except ValueError:
            idx = 0
        slider.setValue(idx)

        value_label = QLabel(choices[idx])

        def update_label(val):
            value_label.setText(choices[val])

        slider.valueChanged.connect(update_label)
        slider.valueChanged.connect(self._schedule_detection)

        row = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(slider)
        row_layout.addWidget(value_label)
        row.setLayout(row_layout)

        form.addRow(QLabel(label), row)
        self._controls[key] = {
            "slider": slider,
            "choices": choices,
        }

    def _schedule_detection(self):
        if self._suppress_updates:
            return
        self.status_label.setText("Re-running detection...")
        self._timer.start(200)

    def _gather_params(self):
        params = self._base_params.copy()
        for key, meta in self._controls.items():
            if "choices" in meta:
                params[key] = meta["choices"][meta["slider"].value()]
                continue
            raw = meta["slider"].value()
            value = raw / meta["scale"]
            if meta["decimals"] == 0:
                value = int(value)
            params[key] = value
        params["black_recovery_enabled"] = True
        params["type_aware_assignment"] = True
        block_size = params.get("black_adaptive_block_size")
        if block_size is not None and block_size % 2 == 0:
            params["black_adaptive_block_size"] = block_size + 1
        return params

    def _reset_sliders(self):
        self._suppress_updates = True
        for key, meta in self._controls.items():
            if "choices" in meta:
                try:
                    idx = meta["choices"].index(self._base_params.get(key, meta["choices"][0]))
                except ValueError:
                    idx = 0
                meta["slider"].setValue(idx)
                continue
            scale = meta["scale"]
            value = float(self._base_params.get(key, 0))
            meta["slider"].setValue(int(round(value * scale)))
        self._suppress_updates = False
        self._schedule_detection()

    def _run_detection(self):
        params = self._gather_params()
        results = self.adapter.detect_from_frame_with_params(
            self.frame,
            keyboard_region=self.keyboard_region,
            detection_params=params,
        )

        if results is None:
            self.status_label.setText("Detection failed. Adjust parameters and try again.")
            return

        self._last_detection_results = results
        self.apply_callback(results, self.adapter)
        self.status_label.setText(
            f"Detected {results['total_keys']} keys (leftmost: {results['leftmost_note']}{results['leftmost_octave']})"
        )

    def _on_save(self):
        if self._last_detection_results is None:
            QMessageBox.warning(self, "Auto-Detect", "No valid detection results yet.")
            return
        self.accept()

class CalibrationWizard(QDialog):
    """Modal dialog for initial keyboard calibration."""
    
    # Signal to request keyboard region selection
    keyboard_region_selection_requested = Signal()

    def __init__(self, parent, app_state: AppState):
        super().__init__(parent)
        self.setWindowTitle("Calibration Wizard")
        self.setModal(True)
        self.app_state = app_state
        self.parent_app = parent  # Store reference to access video frame
        self.result: Optional[bool] = None # True if submitted, False/None if cancelled
        self.detected_overlays: Optional[List[OverlayConfig]] = None  # Store auto-detected overlays
        self._pending_tuning_state: Optional[dict] = None

        # Set wider window size
        self.setMinimumWidth(600)  # Make window twice as wide

        # Create layout
        layout = QGridLayout()

        # Auto-detection button
        auto_selection_button = QPushButton("Select Keyboard Region With Autodetector")
        auto_selection_button.setMinimumWidth(550)  # Wide button
        auto_selection_button.setToolTip("Automatically detect piano keys in a selected region")
        auto_selection_button.clicked.connect(self._handle_manual_keyboard_selection)
        layout.addWidget(auto_selection_button, 0, 0, 1, 3)

        # Manual calibration section
        manual_label = QLabel("Or use manual calibration:")
        manual_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(manual_label, 1, 0, 1, 3)

        # Leftmost key selection
        leftmost_label = QLabel("Leftmost Key:")
        layout.addWidget(leftmost_label, 2, 0)
        
        self.leftmost_note_combo = QComboBox()
        self.leftmost_note_combo.addItems(NOTE_NAMES_SHARP)
        self.leftmost_note_combo.setCurrentText(self.app_state.midi.leftmost_note_name)
        layout.addWidget(self.leftmost_note_combo, 2, 1)
        
        self.leftmost_octave_spin = QSpinBox()
        self.leftmost_octave_spin.setRange(-2, 8)
        self.leftmost_octave_spin.setValue(self.app_state.midi.leftmost_note_octave)
        install_spinbox_wheel_filter(self.leftmost_octave_spin)
        layout.addWidget(self.leftmost_octave_spin, 2, 2)

        # Total keys selection
        total_keys_label = QLabel("Total Keys:")
        layout.addWidget(total_keys_label, 3, 0)
        
        self.total_keys_spin = QSpinBox()
        self.total_keys_spin.setRange(1, 128)
        self.total_keys_spin.setValue(self.app_state.midi.total_keys)
        self.total_keys_spin.setToolTip("Number of keys on the keyboard")
        install_spinbox_wheel_filter(self.total_keys_spin)
        layout.addWidget(self.total_keys_spin, 3, 1, 1, 2)

        # Manual submit button
        manual_submit_button = QPushButton("Generate Manual Overlays")
        manual_submit_button.clicked.connect(self._submit_manual)
        layout.addWidget(manual_submit_button, 4, 0, 1, 3)

        # Buttons
        button_layout = QHBoxLayout()
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self._cancel)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout, 5, 0, 1, 3)
        
        self.setLayout(layout)

    def _submit(self):
        logging.info("=== WIZARD _SUBMIT CALLED ===")
        
        # Check if we have autodetected values
        if not self.detected_overlays:
            logging.error("No keyboard detection performed yet")
            QMessageBox.critical(self, "Error", "Please select keyboard region first.")
            return
        
        logging.info("Using autodetected values")
        
        logging.info("Calling _generate_initial_overlays")
        self._generate_initial_overlays()
        
        self.app_state.unsaved_changes = True
        self.result = True
        
        logging.info("Calling accept() to close wizard")
        self.accept()

    def _submit_manual(self):
        """Handle manual calibration submission."""
        logging.info("=== MANUAL CALIBRATION SUBMIT ===")
        
        # Update app state with manual values
        self.app_state.midi.leftmost_note_name = self.leftmost_note_combo.currentText()
        self.app_state.midi.leftmost_note_octave = self.leftmost_octave_spin.value()
        self.app_state.midi.total_keys = self.total_keys_spin.value()
        
        logging.info(f"Manual settings: {self.app_state.midi.total_keys} keys, "
                    f"leftmost: {self.app_state.midi.leftmost_note_name}{self.app_state.midi.leftmost_note_octave}")
        
        # Clear detected_overlays to force manual generation
        self.detected_overlays = None
        
        # Generate overlays using existing logic
        self._generate_initial_overlays()
        
        logging.info(f"Generated {len(self.app_state.overlays)} overlays after manual calibration")
        
        self.app_state.unsaved_changes = True
        self.result = True
        
        logging.info("Manual calibration complete, closing wizard")
        self.accept()

    def _cancel(self):
        self.result = False
        self.reject()

    def _handle_manual_keyboard_selection(self):
        """Handle manual keyboard region selection."""
        logging.info("=== MANUAL KEYBOARD SELECTION STARTED ===")
        
        # Show instructions to the user
        QMessageBox.information(
            self, 
            "Select Keyboard Region", 
            "Please navigate to a frame where the keyboard is fully visible.\n\n"
            "After clicking OK:\n"
            "1. Click and drag to draw a rectangle around the entire keyboard\n"
            "2. The system will detect keys within the selected region\n"
            "3. Right-click to cancel the selection"
        )
        
        logging.info("Emitting keyboard_region_selection_requested signal")
        # Close the wizard and emit signal to start selection mode
        self.keyboard_region_selection_requested.emit()
        logging.info("Accepting dialog (closing wizard)")
        self.accept()  # Close dialog with success
    
    def handle_keyboard_region_selected(self, x: int, y: int, width: int, height: int):
        """Handle the keyboard region selection from the canvas."""
        logging.info("=== WIZARD HANDLING KEYBOARD REGION ===")
        logging.info(f"Received region: x={x}, y={y}, width={width}, height={height}")
        
        try:
            # Get current video frame
            logging.debug("Getting current video frame")
            current_frame = self._get_current_frame()
            if current_frame is None:
                logging.error("No video frame available")
                QMessageBox.warning(self, "Detection Error", 
                                  "No video frame available. Please ensure a video is loaded.")
                return
            logging.info(f"Got video frame with shape: {current_frame.shape}")
            
            # Crop the frame to the selected region
            logging.info(f"Cropping frame from y:{y} to y:{y+height}, x:{x} to x:{x+width}")
            cropped_frame = current_frame[y:y+height, x:x+width]
            logging.info(f"Cropped frame shape: {cropped_frame.shape}")
            
            # Create adapter and run detection on the cropped region
            logging.info("Creating AutoDetectAdapter")
            adapter = AutoDetectAdapter()
            logging.info("Running detection on cropped frame")
            # FIXED: Pass original ROI coordinates, not (0,0), so coordinate conversion works correctly
            detection_results = adapter.detect_from_frame(cropped_frame, keyboard_region=(x, y, width, height))

            if detection_results is None:
                logging.error("Detection returned None")
                reason = getattr(adapter, "last_failure_reason", None)
                if reason == "low_quality":
                    message = "Video quality is too blurry for autodetector. Please assign overlays manually."
                else:
                    message = "Failed to detect keys in the selected region. Please try again."
                QMessageBox.warning(self, "Detection Error", message)
                return
            
            logging.info(f"Detection successful: {detection_results['total_keys']} keys detected")

            if adapter.last_successful_profile_name == "default":
                self._apply_detection_results(detection_results, adapter, mark_unsaved=True)
                self.result = True
                return

            original_state = self._snapshot_state()
            self._apply_detection_results(detection_results, adapter, mark_unsaved=False)

            initial_params = adapter.last_successful_profile_params or {}
            self._pending_tuning_state = {
                "adapter": adapter,
                "cropped_frame": cropped_frame,
                "keyboard_region": (x, y, width, height),
                "initial_params": initial_params,
                "initial_results": detection_results,
                "original_state": original_state,
            }
            QTimer.singleShot(50, self._open_tuning_dialog)
            return
            
        except Exception as e:
            logging.error(f"=== KEYBOARD DETECTION FAILED ===")
            logging.error(f"Error: {e}", exc_info=True)
            QMessageBox.critical(self, "Detection Error", 
                               f"Key detection failed: {str(e)}")

    def _get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current video frame from the parent application."""
        try:
            # Access the keyboard canvas through parent
            if hasattr(self.parent_app, 'keyboard_canvas'):
                canvas = self.parent_app.keyboard_canvas
                if hasattr(canvas, 'current_frame_rgb') and canvas.current_frame_rgb is not None:
                    return canvas.current_frame_rgb
            
            # Try to get frame from video session
            if hasattr(self.parent_app, 'video_session') and self.parent_app.video_session:
                video_session = self.parent_app.video_session
                if self.app_state.video.current_frame_index is not None:
                    frame = video_session.get_frame(self.app_state.video.current_frame_index)
                    if frame is not None:
                        # Convert BGR to RGB
                        import cv2
                        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            return None
        except Exception as e:
            logging.error(f"Failed to get current frame: {e}")
            return None

    def _apply_detection_results(self, detection_results: dict, adapter: AutoDetectAdapter, mark_unsaved: bool):
        logging.info("Updating app state with detection results")
        self.app_state.midi.total_keys = detection_results['total_keys']
        self.app_state.midi.leftmost_note_name = detection_results['leftmost_note']
        self.app_state.midi.leftmost_note_octave = detection_results['leftmost_octave']
        logging.info(
            "App state updated: %s keys, leftmost: %s%s",
            detection_results['total_keys'],
            detection_results['leftmost_note'],
            detection_results['leftmost_octave'],
        )

        logging.info("Creating overlays from detection results")
        self.detected_overlays = adapter.create_overlays_from_detection(
            detection_results, self.app_state.overlays)
        logging.info("Created %s overlays", len(self.detected_overlays) if self.detected_overlays else 0)

        logging.info("Generating overlays from detected results")
        self._generate_initial_overlays()

        if mark_unsaved:
            self.app_state.unsaved_changes = True

        self._refresh_canvas()
        self._refresh_ui()
        QApplication.processEvents()

    def _snapshot_state(self):
        return {
            "overlays": copy.deepcopy(self.app_state.overlays),
            "total_keys": self.app_state.midi.total_keys,
            "leftmost_note": self.app_state.midi.leftmost_note_name,
            "leftmost_octave": self.app_state.midi.leftmost_note_octave,
            "unsaved_changes": self.app_state.unsaved_changes,
        }

    def _restore_state(self, snapshot: dict):
        self.app_state.overlays = snapshot["overlays"]
        self.app_state.midi.total_keys = snapshot["total_keys"]
        self.app_state.midi.leftmost_note_name = snapshot["leftmost_note"]
        self.app_state.midi.leftmost_note_octave = snapshot["leftmost_octave"]
        self.app_state.unsaved_changes = snapshot["unsaved_changes"]
        self.detected_overlays = None
        self._refresh_canvas()
        self._refresh_ui()

    def _refresh_canvas(self):
        if not self.parent_app or not hasattr(self.parent_app, 'keyboard_canvas'):
            return
        canvas = self.parent_app.keyboard_canvas
        if self.app_state.video.current_frame_index is not None:
            canvas.display_frame(self.app_state.video.current_frame_index)
        else:
            canvas.update()

    def _refresh_ui(self):
        if not self.parent_app:
            return
        if hasattr(self.parent_app, "control_panel"):
            self.parent_app.control_panel.update_controls_from_state()
            self.parent_app.control_panel.update_trim_controls_from_state()
            self.parent_app.control_panel.update_selected_overlay_display()
        if hasattr(self.parent_app, "keyboard_canvas"):
            self.parent_app.keyboard_canvas.draw_overlays()

    def _open_tuning_dialog(self):
        if not self._pending_tuning_state:
            return

        state = self._pending_tuning_state
        self._pending_tuning_state = None

        QApplication.processEvents()
        tuning_dialog = AutoDetectTuningDialog(
            self,
            state["adapter"],
            state["cropped_frame"],
            state["keyboard_region"],
            lambda results, adapt=state["adapter"]: self._apply_detection_results(results, adapt, mark_unsaved=False),
            initial_params=state["initial_params"],
            initial_results=state["initial_results"],
        )
        dialog_result = tuning_dialog.exec()
        if dialog_result == QDialog.Accepted:
            self.app_state.unsaved_changes = True
            self.result = True
        else:
            self._restore_state(state["original_state"])

    def _generate_initial_overlays(self):
        """Generates an idealized piano keyboard layout based on hardcoded stylistic constants."""
        logging.info("=== _GENERATE_INITIAL_OVERLAYS CALLED ===")
        logging.info(f"Current overlays count: {len(self.app_state.overlays)}")
        logging.info(f"Detected overlays available: {self.detected_overlays is not None}")
        
        # Preserve unlit calibration data from existing overlays
        # Store by both key_id and position for better matching
        unlit_calibration_by_id = {}
        unlit_calibration_by_position = {}
        for overlay in self.app_state.overlays:
            if overlay.unlit_hist is not None or overlay.unlit_reference_color is not None:
                calib_data = {
                    'unlit_hist': overlay.unlit_hist.copy() if overlay.unlit_hist is not None else None,
                    'unlit_reference_color': overlay.unlit_reference_color
                }
                unlit_calibration_by_id[overlay.key_id] = calib_data
                # Create a position key for matching by location
                pos_key = (round(overlay.x), round(overlay.y), round(overlay.width), round(overlay.height))
                unlit_calibration_by_position[pos_key] = calib_data
                logging.info(f"[WIZARD] Preserving unlit calibration for key {overlay.key_id} at position {pos_key}")
        
        self.app_state.overlays.clear()
        
        # Use detected overlays if available
        if self.detected_overlays:
            logging.info(f"Using {len(self.detected_overlays)} auto-detected overlays")
            
            logging.info("=== FINAL WHITE KEY OVERLAY POSITIONS DEBUG ===")
            white_key_count = 0
            for i, overlay in enumerate(self.detected_overlays):
                if 'W' in overlay.key_type:  # White key
                    white_key_count += 1
                    logging.info(f"WHITE KEY OVERLAY {white_key_count} (key_id={overlay.key_id}): x={overlay.x}, y={overlay.y}, w={overlay.width}, h={overlay.height}")
                    logging.info(f"  Note: {overlay.note_name_in_octave}{overlay.note_octave}, Key Type: {overlay.key_type}")
                elif i < 5:  # Also log first few keys regardless of type
                    logging.info(f"Overlay {i} (key_id={overlay.key_id}): x={overlay.x}, y={overlay.y}, w={overlay.width}, h={overlay.height}, type={overlay.key_type}")
            
            logging.info(f"Total white key overlays: {white_key_count}")
            logging.info("=== END FINAL WHITE KEY OVERLAY DEBUG ===")
            
            self.app_state.overlays.extend(self.detected_overlays)
            logging.info(f"App state now has {len(self.app_state.overlays)} overlays")
            
            # Restore unlit calibration data
            restored_count = 0
            for overlay in self.app_state.overlays:
                calib_data = None
                
                # First try to match by key_id
                if overlay.key_id in unlit_calibration_by_id:
                    calib_data = unlit_calibration_by_id[overlay.key_id]
                    logging.info(f"[WIZARD] Found calibration by key_id for key {overlay.key_id}")
                else:
                    # Try to match by position (with small tolerance)
                    for dx in [-2, -1, 0, 1, 2]:  # Allow small position variations
                        for dy in [-2, -1, 0, 1, 2]:
                            pos_key = (round(overlay.x + dx), round(overlay.y + dy), round(overlay.width), round(overlay.height))
                            if pos_key in unlit_calibration_by_position:
                                calib_data = unlit_calibration_by_position[pos_key]
                                logging.info(f"[WIZARD] Found calibration by position for key {overlay.key_id} at {pos_key}")
                                break
                        if calib_data:
                            break
                
                # Restore calibration data if found
                if calib_data:
                    overlay.unlit_hist = calib_data['unlit_hist']
                    overlay.unlit_reference_color = calib_data['unlit_reference_color']
                    restored_count += 1
                    logging.info(f"[WIZARD] Restored unlit calibration for key {overlay.key_id}")
            
            if restored_count > 0:
                logging.info(f"[WIZARD] Successfully restored unlit calibration for {restored_count} out of {len(self.app_state.overlays)} overlays")
            
            return
        
        logging.info("Generating idealized piano layout for wizard.")

        num_keys_to_generate = self.app_state.midi.total_keys
        start_note_index = NOTE_NAMES_SHARP.index(self.app_state.midi.leftmost_note_name)
        start_octave = self.app_state.midi.leftmost_note_octave
        
        # For manual calibration, position overlays in center of video
        # Get video dimensions if available
        video_height = 1080  # Default
        video_width = 1920   # Default
        if hasattr(self.parent_app, 'video_session') and self.parent_app.video_session:
            video_height = self.parent_app.video_session.height or 1080
            video_width = self.parent_app.video_session.width or 1920
        
        # Position keyboard in lower third of video
        keyboard_y_position = int(video_height * 0.6)  # 60% down from top
        white_key_y = keyboard_y_position
        white_key_height = int(video_height * 0.15)  # 15% of video height
        black_key_height = int(white_key_height * 0.6)  # 60% of white key height
        black_key_y = white_key_y - (black_key_height - white_key_height)
        
        # Calculate key widths based on video width and number of keys
        # Count white keys
        white_key_count = 0
        temp_idx = start_note_index
        for i in range(num_keys_to_generate):
            note_name = NOTE_NAMES_SHARP[temp_idx % 12]
            if '♯' not in note_name and 'b' not in note_name:
                white_key_count += 1
            temp_idx += 1
        
        # Calculate white key width to fit in video with some margin
        margin = int(video_width * 0.1)  # 10% margin on each side
        available_width = video_width - (2 * margin)
        white_key_width = available_width / white_key_count if white_key_count > 0 else 30
        black_key_width = white_key_width * 0.6
        
        # Starting X position
        start_x = margin

        white_key_x_positions = []
        current_x = float(start_x)

        # Pass 1: Calculate X positions for all white keys to establish the base layout
        temp_note_idx = start_note_index
        temp_octave = start_octave
        white_key_count_generated = 0
        for i in range(num_keys_to_generate): 
            note_name_full = NOTE_NAMES_SHARP[temp_note_idx % 12]
            is_black_key = '♯' in note_name_full or 'b' in note_name_full

            if not is_black_key:
                white_key_x_positions.append(current_x)
                current_x += white_key_width
                white_key_count_generated += 1
            
            temp_note_idx += 1
            # Octave increment logic
            if note_name_full == 'B' and temp_note_idx % 12 == NOTE_NAMES_SHARP.index('C'):
                temp_octave += 1
        
        # If no white keys were requested (e.g. user asks for only 1 black key, though unlikely via UI)
        # provide a default starting x for black keys to prevent errors.
        if not white_key_x_positions and num_keys_to_generate > 0:
            white_key_x_positions.append(float(start_x)) # Default anchor if only black keys are somehow generated first


        # Pass 2: Generate all keys, positioning black keys relative to white keys
        current_note_idx = start_note_index
        current_octave = start_octave
        white_key_abs_idx = 0 # To iterate through white_key_x_positions

        for i in range(num_keys_to_generate):
            note_name = NOTE_NAMES_SHARP[current_note_idx % 12]
            is_black_key = '♯' in note_name or 'b' in note_name

            x_pos: float
            y_pos: float
            width: float
            height: float
            key_type_suffix: str

            if is_black_key:
                y_pos = float(black_key_y)
                height = float(black_key_height)
                width = float(black_key_width)
                key_type_suffix = "B"

                # Find the X position of the preceding white key for reference
                ref_white_key_x = float(start_x)
                if white_key_abs_idx > 0:
                    ref_white_key_x = white_key_x_positions[white_key_abs_idx - 1]
                elif white_key_x_positions:
                    ref_white_key_x = white_key_x_positions[0]
                
                x_pos = ref_white_key_x + (white_key_width * IDEALIZED_BLACK_KEY_X_START_FACTOR) - (width / 2)

            else: # White key
                y_pos = float(white_key_y)
                height = float(white_key_height)
                width = float(white_key_width)
                key_type_suffix = "W"
                
                if white_key_abs_idx < len(white_key_x_positions):
                    x_pos = white_key_x_positions[white_key_abs_idx]
                    white_key_abs_idx += 1
                else:
                    # Fallback
                    logging.error("Ran out of pre-calculated white key X positions. Defaulting X.")
                    x_pos = i * white_key_width + start_x
            
            # Determine L/R hand (simple split for now)
            # For an 88 key piano, A0-D#4 (key_id 0-39) is often considered left.
            # Middle C (C4) is key_id 39 if A0 is key_id 0.
            # Let's make it so that if the note is C4 or higher, it is "R"
            # This needs to map `i` (generated key index) to a more global midi-like concept if possible.
            # For simplicity, let's use i < num_keys_to_generate / 2 as a rough split.
            hand_prefix = "L" if i < (num_keys_to_generate / 2.0) else "R"
            assigned_key_type = f"{hand_prefix}{key_type_suffix}"

            self.app_state.overlays.append(OverlayConfig(
                key_id=i,
                note_octave=current_octave,
                note_name_in_octave=note_name,
                x=x_pos, # Store as float, will be int when drawn
                y=y_pos,
                width=width,
                height=height,
                key_type=assigned_key_type
            ))

            current_note_idx += 1
            if note_name == 'B' and current_note_idx % 12 == NOTE_NAMES_SHARP.index('C'):
                current_octave += 1

        logging.info(f"Generated {len(self.app_state.overlays)} overlays for the wizard.")
        
        # Restore unlit calibration data for manually generated overlays
        for overlay in self.app_state.overlays:
            if overlay.key_id in unlit_calibration_by_id:
                calib_data = unlit_calibration_by_id[overlay.key_id]
                overlay.unlit_hist = calib_data['unlit_hist']
                overlay.unlit_reference_color = calib_data['unlit_reference_color']
                logging.info(f"[WIZARD] Restored unlit calibration for key {overlay.key_id}")


def show_calibration_wizard(parent, app_state: AppState) -> bool:
    """Displays the calibration wizard and returns True if submitted, False otherwise."""
    wizard = CalibrationWizard(parent, app_state)
    result = wizard.exec()
    return wizard.result is True
