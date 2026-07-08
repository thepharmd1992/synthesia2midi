"""
Calibration wizard for setting up initial key overlays.
"""
# Standard library imports
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
from PySide6.QtCore import QCoreApplication, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QGridLayout, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QSpinBox, QVBoxLayout
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

class CalibrationWizard(QDialog):
    """Modal dialog for initial keyboard calibration."""
    
    # Signal to request keyboard region selection
    keyboard_region_selection_requested = Signal()
    edit_current_calibration_requested = Signal()

    def __init__(self, parent, app_state: AppState):
        super().__init__(parent)
        self.setWindowTitle(QCoreApplication.translate("CalibrationWizard", "Calibration Wizard"))
        self.setModal(True)
        self.app_state = app_state
        self.parent_app = parent  # Store reference to access video frame
        self.result: Optional[bool] = None # True if submitted, False/None if cancelled
        self.detected_overlays: Optional[List[OverlayConfig]] = None  # Store auto-detected overlays
        self.manual_overlays_generated = False
        self.auto_detect_source_frame_rgb: Optional[np.ndarray] = None
        self.auto_detect_keyboard_roi: Optional[Tuple[int, int, int, int]] = None
        self.auto_detect_saved_params_fallback_used: bool = False
        self.auto_detect_latest_detection_result: Optional[Dict[str, Any]] = None

        # Set wider window size
        self.setMinimumWidth(600)  # Make window twice as wide

        # Create layout
        layout = QGridLayout()

        auto_instruction_label = QLabel(
            QCoreApplication.translate(
                "CalibrationWizard",
                "Pause on a clear frame where the full keyboard is visible.",
            )
        )
        auto_instruction_label.setWordWrap(True)
        layout.addWidget(auto_instruction_label, 0, 0, 1, 3)

        # Auto-detection button
        auto_selection_button = QPushButton(
            QCoreApplication.translate("CalibrationWizard", "Draw Keyboard Box and Find Keys")
        )
        auto_selection_button.setMinimumWidth(550)  # Wide button
        auto_selection_button.setStyleSheet(
            "QPushButton {"
            "background-color: #2e7d32;"
            "color: #ffffff;"
            "border: 2px solid #1b5e20;"
            "padding: 8px 12px;"
            "font-weight: 600;"
            "font-size: 14px;"
            "}"
            "QPushButton:hover { background-color: #388e3c; }"
            "QPushButton:pressed { background-color: #2c6e30; }"
            "QPushButton:disabled {"
            "background-color: #c8e6c9;"
            "color: #1f1f1f;"
            "border: 2px solid #a5d6a7;"
            "}"
        )
        auto_selection_button.setToolTip(QCoreApplication.translate("CalibrationWizard", "Automatically detect piano keys in a selected region"))
        auto_selection_button.clicked.connect(self._handle_manual_keyboard_selection)
        layout.addWidget(auto_selection_button, 1, 0, 1, 3)

        self.edit_current_calibration_button = QPushButton(QCoreApplication.translate("CalibrationWizard", "Edit Current Calibration"))
        self.edit_current_calibration_button.setMinimumWidth(270)
        self.edit_current_calibration_button.setMaximumWidth(270)
        self.edit_current_calibration_button.setToolTip(
            QCoreApplication.translate("CalibrationWizard", "Open the auto-detect tuning panel using your current calibration.")
        )
        self.edit_current_calibration_button.setEnabled(False)
        self.edit_current_calibration_button.clicked.connect(self._handle_edit_current_calibration)
        layout.addWidget(self.edit_current_calibration_button, 2, 0)

        self.edit_current_reason_label = QLabel(
            QCoreApplication.translate(
                "CalibrationWizard",
                "Edit becomes available after you create key overlays.",
            )
        )
        self.edit_current_reason_label.setWordWrap(True)
        self.edit_current_reason_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.edit_current_reason_label, 3, 0, 1, 3)

        # Manual calibration section
        manual_label = QLabel(QCoreApplication.translate("CalibrationWizard", "Or use manual calibration:"))
        manual_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(manual_label, 4, 0, 1, 3)

        # Leftmost key selection
        leftmost_label = QLabel(QCoreApplication.translate("CalibrationWizard", "Leftmost Key:"))
        layout.addWidget(leftmost_label, 5, 0)
        
        self.leftmost_note_combo = QComboBox()
        self.leftmost_note_combo.addItems(NOTE_NAMES_SHARP)
        self.leftmost_note_combo.setCurrentText(self.app_state.midi.leftmost_note_name)
        layout.addWidget(self.leftmost_note_combo, 5, 1)
        
        self.leftmost_octave_spin = QSpinBox()
        self.leftmost_octave_spin.setRange(-2, 8)
        self.leftmost_octave_spin.setValue(self.app_state.midi.leftmost_note_octave)
        install_spinbox_wheel_filter(self.leftmost_octave_spin)
        layout.addWidget(self.leftmost_octave_spin, 5, 2)

        # Total keys selection
        total_keys_label = QLabel(QCoreApplication.translate("CalibrationWizard", "Total Keys:"))
        layout.addWidget(total_keys_label, 6, 0)
        
        self.total_keys_spin = QSpinBox()
        self.total_keys_spin.setRange(1, 128)
        self.total_keys_spin.setValue(self.app_state.midi.total_keys)
        self.total_keys_spin.setToolTip(QCoreApplication.translate("CalibrationWizard", "Number of keys on the keyboard"))
        install_spinbox_wheel_filter(self.total_keys_spin)
        layout.addWidget(self.total_keys_spin, 6, 1, 1, 2)

        # Manual submit button
        manual_submit_button = QPushButton(QCoreApplication.translate("CalibrationWizard", "Generate Manual Overlays"))
        manual_submit_button.clicked.connect(self._submit_manual)
        layout.addWidget(manual_submit_button, 7, 0, 1, 3)

        # Buttons
        button_layout = QHBoxLayout()
        
        cancel_button = QPushButton(QCoreApplication.translate("CalibrationWizard", "Cancel"))
        cancel_button.clicked.connect(self._cancel)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout, 8, 0, 1, 3)
        
        self.setLayout(layout)

    def set_edit_current_calibration_enabled(self, enabled: bool, tooltip: Optional[str] = None) -> None:
        self.edit_current_calibration_button.setEnabled(enabled)
        self.edit_current_reason_label.setVisible(not enabled)
        if tooltip:
            self.edit_current_calibration_button.setToolTip(tooltip)
            if not enabled:
                self.edit_current_reason_label.setText(tooltip)

    def _submit(self):
        logging.info("=== WIZARD _SUBMIT CALLED ===")
        
        # Check if we have autodetected values
        if not self.detected_overlays:
            logging.error("No keyboard detection performed yet")
            QMessageBox.critical(
                self,
                QCoreApplication.translate("CalibrationWizard", "Error"),
                QCoreApplication.translate("CalibrationWizard", "Please select keyboard region first."),
            )
            return
        
        logging.info("Using autodetected values")
        self.manual_overlays_generated = False
        
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
        self.manual_overlays_generated = True
        self.app_state.calibration.overlay_generation_source = "manual"
        
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
            QCoreApplication.translate("CalibrationWizard", "Select Keyboard Region"),
            QCoreApplication.translate(
                "CalibrationWizard",
                "Please navigate to a frame where the keyboard is fully visible.\n\nAfter clicking OK:\n1. Click and drag to draw a rectangle around the entire keyboard\n2. The system will detect keys within the selected region\n3. Right-click to cancel the selection",
            ),
        )
        
        logging.info("Emitting keyboard_region_selection_requested signal")
        # Close the wizard and emit signal to start selection mode
        self.keyboard_region_selection_requested.emit()
        logging.info("Accepting dialog (closing wizard)")
        self.accept()  # Close dialog with success

    def _handle_edit_current_calibration(self):
        logging.info("=== EDIT CURRENT CALIBRATION REQUESTED ===")
        self.edit_current_calibration_requested.emit()
        self.accept()
    
    def handle_keyboard_region_selected(self, x: int, y: int, width: int, height: int) -> bool:
        """Handle the keyboard region selection from the canvas."""
        logging.info("=== WIZARD HANDLING KEYBOARD REGION ===")
        logging.info(f"Received region: x={x}, y={y}, width={width}, height={height}")

        try:
            current_frame = self._get_current_frame()
            if current_frame is None:
                logging.error("No video frame available")
                QMessageBox.warning(
                    self,
                    QCoreApplication.translate("CalibrationWizard", "Detection Error"),
                    QCoreApplication.translate(
                        "CalibrationWizard", "No video frame available. Please ensure a video is loaded."
                    ),
                )
                return False
            logging.info(f"Got video frame with shape: {current_frame.shape}")

            # Keep a stable source frame/ROI for live tuning reruns.
            self.auto_detect_source_frame_rgb = np.copy(current_frame)
            self.auto_detect_keyboard_roi = (x, y, width, height)

            cropped_frame = current_frame[y:y + height, x:x + width]
            if cropped_frame.size == 0:
                QMessageBox.warning(
                    self,
                    QCoreApplication.translate("CalibrationWizard", "Detection Error"),
                    QCoreApplication.translate(
                        "CalibrationWizard", "Selected region is empty. Please draw a valid keyboard region."
                    ),
                )
                return False

            adapter = AutoDetectAdapter()

            # Always run a clean built-in profile chain for each new overlay calibration pass.
            detection_results = adapter.detect_from_frame(
                cropped_frame,
                keyboard_region=(x, y, width, height),
                tuning_params=None,
                use_profile_fallback=True,
            )

            if detection_results is None:
                logging.error("Detection returned None")
                reason = getattr(adapter, "last_failure_reason", None)
                if reason == "low_quality":
                    message = QCoreApplication.translate(
                        "CalibrationWizard",
                        "Video quality is too blurry for autodetector. Please assign overlays manually.",
                    )
                else:
                    message = QCoreApplication.translate(
                        "CalibrationWizard",
                        "Failed to detect keys in the selected region. Please try again.",
                    )
                QMessageBox.warning(
                    self,
                    QCoreApplication.translate("CalibrationWizard", "Detection Error"),
                    message,
                )
                self.auto_detect_saved_params_fallback_used = False
                return False

            fallback_used = bool(detection_results.get("fallback_used", False))
            self.auto_detect_saved_params_fallback_used = fallback_used
            self.auto_detect_latest_detection_result = detection_results
            logging.info(
                "Detection successful with clean profile '%s' (fallback_used=%s): %s keys detected",
                detection_results.get("profile_name", "unknown"),
                fallback_used,
                detection_results["total_keys"],
            )

            applied = self.apply_auto_detect_results(detection_results, adapter=adapter)
            if not applied:
                QMessageBox.warning(
                    self,
                    QCoreApplication.translate("CalibrationWizard", "Detection Error"),
                    QCoreApplication.translate(
                        "CalibrationWizard", "Autodetection produced no overlays. Please try another region."
                    ),
                )
                return False

            self.result = True
            return True

        except Exception as e:
            logging.error("=== KEYBOARD DETECTION FAILED ===")
            logging.error(f"Error: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                QCoreApplication.translate("CalibrationWizard", "Detection Error"),
                QCoreApplication.translate("CalibrationWizard", "Key detection failed: {error}").format(
                    error=str(e)
                ),
            )
            return False

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

    def get_auto_detect_tuning_context(self) -> Optional[Dict[str, Any]]:
        if self.auto_detect_source_frame_rgb is None or self.auto_detect_keyboard_roi is None:
            return None

        return {
            "frame_rgb": np.copy(self.auto_detect_source_frame_rgb),
            "keyboard_roi": self.auto_detect_keyboard_roi,
            "fallback_used": bool(self.auto_detect_saved_params_fallback_used),
            "detection_results": self.auto_detect_latest_detection_result,
        }

    def apply_auto_detect_results(
        self,
        detection_results: Dict[str, Any],
        *,
        adapter: Optional[AutoDetectAdapter] = None,
    ) -> bool:
        detector_adapter = adapter or AutoDetectAdapter()
        detected_overlays = detector_adapter.create_overlays_from_detection(
            detection_results,
            self.app_state.overlays,
        )
        if not detected_overlays:
            return False

        self.app_state.midi.total_keys = detection_results["total_keys"]
        self.app_state.midi.leftmost_note_name = detection_results["leftmost_note"]
        self.app_state.midi.leftmost_note_octave = detection_results["leftmost_octave"]
        self.auto_detect_latest_detection_result = detection_results
        self.detected_overlays = detected_overlays

        self._apply_detected_overlays(detected_overlays)

        self.app_state.calibration.overlay_generation_source = "auto"
        self.app_state.unsaved_changes = True
        logging.info(
            "Applied autodetect overlays: %s overlays, leftmost=%s%s",
            len(detected_overlays),
            detection_results["leftmost_note"],
            detection_results["leftmost_octave"],
        )
        return True

    def _capture_unlit_calibration_data(self) -> Tuple[Dict[int, Dict[str, Any]], Dict[Tuple[int, int, int, int], Dict[str, Any]]]:
        unlit_calibration_by_id: Dict[int, Dict[str, Any]] = {}
        unlit_calibration_by_position: Dict[Tuple[int, int, int, int], Dict[str, Any]] = {}
        for overlay in self.app_state.overlays:
            if overlay.unlit_hist is None and overlay.unlit_reference_color is None:
                continue
            calib_data = {
                "unlit_hist": overlay.unlit_hist.copy() if overlay.unlit_hist is not None else None,
                "unlit_reference_color": overlay.unlit_reference_color,
            }
            unlit_calibration_by_id[overlay.key_id] = calib_data
            pos_key = (
                round(overlay.x),
                round(overlay.y),
                round(overlay.width),
                round(overlay.height),
            )
            unlit_calibration_by_position[pos_key] = calib_data
        return unlit_calibration_by_id, unlit_calibration_by_position

    def _restore_unlit_calibration_data(
        self,
        overlays: List[OverlayConfig],
        unlit_calibration_by_id: Dict[int, Dict[str, Any]],
        unlit_calibration_by_position: Dict[Tuple[int, int, int, int], Dict[str, Any]],
    ) -> int:
        restored_count = 0
        for overlay in overlays:
            calib_data = unlit_calibration_by_id.get(overlay.key_id)
            if calib_data is None:
                for dx in [-2, -1, 0, 1, 2]:
                    for dy in [-2, -1, 0, 1, 2]:
                        pos_key = (
                            round(overlay.x + dx),
                            round(overlay.y + dy),
                            round(overlay.width),
                            round(overlay.height),
                        )
                        if pos_key in unlit_calibration_by_position:
                            calib_data = unlit_calibration_by_position[pos_key]
                            break
                    if calib_data is not None:
                        break

            if calib_data is None:
                continue

            overlay.unlit_hist = calib_data["unlit_hist"]
            overlay.unlit_reference_color = calib_data["unlit_reference_color"]
            restored_count += 1

        return restored_count

    def _apply_detected_overlays(self, detected_overlays: List[OverlayConfig]) -> None:
        unlit_calibration_by_id, unlit_calibration_by_position = self._capture_unlit_calibration_data()

        self.app_state.overlays.clear()
        self.app_state.overlays.extend(detected_overlays)

        restored_count = self._restore_unlit_calibration_data(
            self.app_state.overlays,
            unlit_calibration_by_id,
            unlit_calibration_by_position,
        )
        if restored_count > 0:
            logging.info(
                "[WIZARD] Restored unlit calibration for %s overlays after autodetect apply",
                restored_count,
            )

    def _generate_initial_overlays(self):
        """Generates an idealized piano keyboard layout based on hardcoded stylistic constants."""
        logging.info("=== _GENERATE_INITIAL_OVERLAYS CALLED ===")
        logging.info(f"Current overlays count: {len(self.app_state.overlays)}")
        logging.info(f"Detected overlays available: {self.detected_overlays is not None}")

        if self.detected_overlays:
            logging.info(f"Using {len(self.detected_overlays)} auto-detected overlays")
            self._apply_detected_overlays(self.detected_overlays)
            return

        unlit_calibration_by_id, unlit_calibration_by_position = self._capture_unlit_calibration_data()
        self.app_state.overlays.clear()

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
        black_key_y = white_key_y
        
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

        restored = self._restore_unlit_calibration_data(
            self.app_state.overlays,
            unlit_calibration_by_id,
            unlit_calibration_by_position,
        )
        if restored > 0:
            logging.info(f"[WIZARD] Restored unlit calibration for {restored} manually generated overlays")


def show_calibration_wizard(parent, app_state: AppState) -> bool:
    """Displays the calibration wizard and returns True if submitted, False otherwise."""
    wizard = CalibrationWizard(parent, app_state)
    result = wizard.exec()
    return wizard.result is True
