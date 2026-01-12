"""
Tab-based control panel UI for Synthesia2MIDI.

This module defines the right-hand settings pane and emits signals that the main
window connects to workflows/state updates.
"""

import logging
from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSizePolicy,
    QSlider, QSpinBox, QTabWidget, QToolButton, QVBoxLayout, QWidget
)

from synthesia2midi.core.app_state import AppState

# Key type constants
KEY_TYPES = ["LW", "LB", "RW", "RB"]
KEY_TYPE_LABELS = {
    "LW": "Left Hand White",
    "LB": "Left Hand Black", 
    "RW": "Right Hand White",
    "RB": "Right Hand Black"
}


class CollapsibleSection(QWidget):
    def __init__(self, title: str, *, expanded: bool = False, parent: QWidget | None = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._toggle.toggled.connect(self._handle_toggled)
        layout.addWidget(self._toggle)

        self._content = QWidget()
        self._content.setVisible(expanded)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(20, 0, 0, 0)
        self._content_layout.setSpacing(6)
        layout.addWidget(self._content)

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    def _handle_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._content.setVisible(checked)


class ControlPanelQt(QWidget):
    """Qt version of the Control Panel containing various controls for the application.
    
    Uses clean tabs and standard Qt widgets.
    """
    
    # Signals exposed to the main window
    
    # ==================== Calibration Signals ====================
    calibrate_unlit_requested = Signal()
    calibrate_lit_exemplar_requested = Signal(str)  # key_type
    calibration_wizard_requested = Signal()
    align_white_keys_requested = Signal()
    align_black_keys_requested = Signal()
    add_additional_color_requested = Signal()
    remove_additional_color_requested = Signal(str)  # key_type
    
    # ==================== Spark Calibration Signals ====================
    spark_calibration_requested = Signal(str)  # calibration step
    auto_spark_calibration_requested = Signal(str)  # key_type
    
    # ==================== Detection Settings Signals ====================
    detection_threshold_changed = Signal(float)
    histogram_detection_toggled = Signal(bool)
    histogram_threshold_changed = Signal(float)
    delta_detection_toggled = Signal(bool)
    use_delta_detection_toggled = Signal(bool)
    visual_threshold_monitor_toggled = Signal(bool)
    similarity_ratio_changed = Signal(float)
    add_histogram_detection_changed = Signal(bool)
    use_delta_detection_changed = Signal(bool)
    rise_delta_threshold_changed = Signal(float)
    fall_delta_threshold_changed = Signal(float)
    winner_takes_black_changed = Signal(bool)
    filter_similarity_ratio_changed = Signal(float)
    hist_thresh_changed = Signal(float)
    hand_assignment_toggled = Signal(bool)
    
    # ==================== Spark Detection Signals ====================
    spark_roi_selection_requested = Signal()
    spark_roi_changed = Signal(int, int)  # top, bottom
    spark_roi_visibility_toggled = Signal(bool)
    spark_detection_toggled = Signal(bool)
    spark_detection_sensitivity_changed = Signal(float)
    
    # ==================== Video/Frame Navigation Signals ====================
    start_frame_changed = Signal(int)
    end_frame_changed = Signal(int)
    timeline_seek_requested = Signal(int)
    nav_interval_changed = Signal(int)
    youtube_video_downloaded = Signal(str)
    video_to_frames_requested = Signal()
    
    # ==================== Overlay Management Signals ====================
    refresh_overlay_display_requested = Signal()
    overlay_color_changed = Signal(str)
    overlay_type_changed = Signal(str)
    overlay_size_adjustment_requested = Signal(str, str, int)
    
    # ==================== MIDI Processing Signals ====================
    octave_transpose_changed = Signal(int)
    processing_start_frame_changed = Signal(int)
    processing_end_frame_changed = Signal(int)
    fps_override_changed = Signal(object)  # float or None
    
    # ==================== Main Action Signals ====================
    conversion_requested = Signal()
    trim_video_requested = Signal(int, int)  # start_frame, end_frame
    
    def __init__(self, parent=None, app_state: AppState = None, state_manager=None):
        super().__init__(parent)
        self.app_state = app_state or AppState()
        self.state_manager = state_manager
        
        # Widget references for state updates
        self.widgets = {}
        
        self._setup_ui()
        self.update_controls_from_state()
    
    def _setup_ui(self):
        """Create the main UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 2, 10, 10)  # Reduced top margin from 10 to 2
        main_layout.setSpacing(5)  # Reduce spacing between elements
        main_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)  # Align content to left side
        
        # Always-visible elements at top
        self._create_always_visible_section(main_layout)
        
        # Main tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("main_tabs")
        self.tab_widget.setMaximumHeight(600)  # Constrain height to force scrolling in tabs
        self.tab_widget.setMaximumWidth(850)  # Increased for 14pt font
        
        # Create all tabs
        self._create_mandatory_calibration_tab()
        self._create_overlay_settings_tab()
        self._create_basic_detection_tab()
        self._create_spark_detection_tab()
        self._create_midi_settings_tab()
        self._create_video_trim_tab()
        self._create_optional_settings_tab()
        
        main_layout.addWidget(self.tab_widget)
        
        # Add stretch at bottom to prevent spacing between elements above
        main_layout.addStretch(1)
    
    def _create_always_visible_section(self, parent_layout):
        """Create elements that are always visible regardless of tab."""
        always_visible_group = QGroupBox("Main Actions")
        always_visible_group.setMaximumWidth(720)  # Increased for 14pt font
        always_visible_layout = QVBoxLayout(always_visible_group)
        always_visible_layout.setContentsMargins(10, 5, 10, 10)  # Reduce top margin inside group box
        
        # Convert to MIDI button
        convert_layout = QHBoxLayout()
        self.convert_button = QPushButton("Convert to MIDI")
        self.convert_button.setObjectName("convert_button")
        self.convert_button.clicked.connect(self._handle_conversion_request)
        self.convert_button.setMinimumHeight(40)
        convert_layout.addWidget(self.convert_button)
        
        # Status labels
        status_layout = QVBoxLayout()
        self.conversion_status = QLabel("Ready to convert")
        status_layout.addWidget(self.conversion_status)
        
        
        convert_layout.addLayout(status_layout)
        
        always_visible_layout.addLayout(convert_layout)
        
        # Keyboard selection (if needed)
        keyboard_layout = QHBoxLayout()
        keyboard_layout.addWidget(QLabel("Selected Overlay:"))
        self.selected_overlay_label = QLabel("None")
        keyboard_layout.addWidget(self.selected_overlay_label)
        keyboard_layout.addStretch()
        
        always_visible_layout.addLayout(keyboard_layout)
        
        parent_layout.addWidget(always_visible_group)
    
    def _create_mandatory_calibration_tab(self):
        """Tab 1: Mandatory Calibration"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)  # Add some padding
        layout.setSpacing(15)  # Space between sections

        help_section = CollapsibleSection("Help", expanded=False)
        help_layout = help_section.content_layout()
        help_lines = [
            "Initial calibration directions (recommended order):",
            "1) Calibrate Key Overlays: create overlays that line up with the keyboard in your video.",
            "2) Unlit Key Calibration: pause on a frame where no notes are highlighted, then click Calibrate.",
            ("3) Lit Key Exemplars: for each button you need (Left/Right x White/Black), pause on a frame "
             "where that kind of overlay is highlighted, click the button, then click that highlighted "
             "overlay in the video."),
            "Octave Transpose: shifts the generated MIDI up/down by octaves."
        ]
        for line in help_lines:
            label = QLabel(line)
            label.setWordWrap(True)
            label.setStyleSheet("font-size: 9pt;")  # slightly smaller help text
            help_layout.addWidget(label)
        layout.addWidget(help_section)
        
        # Key Overlay Calibration - inline layout
        overlay_layout = QHBoxLayout()
        overlay_label = QLabel("Key Overlay Calibration")
        overlay_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        overlay_label.setFixedWidth(216)  # Scaled for 14pt font
        overlay_layout.addWidget(overlay_label)
        
        self.calibration_wizard_button = QPushButton("Calibrate Key Overlays")
        self.calibration_wizard_button.setFixedWidth(264)
        # Center-aligned text (default for QPushButton)
        self.calibration_wizard_button.clicked.connect(self.calibration_wizard_requested.emit)
        self.calibration_wizard_button.setToolTip(
            "Creates overlays for the keyboard in your video. Re-run if overlays don't line up."
        )
        overlay_layout.addWidget(self.calibration_wizard_button)
        
        # Octave transpose control
        overlay_layout.addSpacing(20)
        overlay_layout.addWidget(QLabel("Octave Transpose:"))
        self.octave_transpose_spin = QSpinBox()
        self.octave_transpose_spin.setRange(-5, 5)
        self.octave_transpose_spin.setValue(0)
        self.octave_transpose_spin.valueChanged.connect(self.octave_transpose_changed.emit)
        self.octave_transpose_spin.setToolTip("Shifts the MIDI output up/down by octaves.")
        overlay_layout.addWidget(self.octave_transpose_spin)
        
        overlay_layout.addStretch()
        
        layout.addLayout(overlay_layout)
        layout.addSpacing(10)  # Extra space before next section
        
        # Unlit key calibration - inline layout with status
        unlit_layout = QHBoxLayout()
        unlit_label = QLabel("Unlit Key Calibration")
        unlit_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        unlit_label.setFixedWidth(216)  # Scaled for 14pt font - same as above
        unlit_layout.addWidget(unlit_label)
        
        self.calibrate_unlit_button = QPushButton("Calibrate")
        self.calibrate_unlit_button.setFixedWidth(180)
        # Center-aligned text (default for QPushButton)
        self.calibrate_unlit_button.clicked.connect(self.calibrate_unlit_requested.emit)
        self.calibrate_unlit_button.setToolTip(
            "Captures what unpressed overlays look like from the current frame. "
            "Pause on a frame with no highlighted notes first."
        )
        unlit_layout.addWidget(self.calibrate_unlit_button)
        
        # Status indicator for unlit calibration
        self.unlit_status_label = QLabel("Not Set")
        self.unlit_status_label.setStyleSheet("font-style: italic; color: #888;")
        unlit_layout.addWidget(self.unlit_status_label)
        unlit_layout.addStretch()
        
        layout.addLayout(unlit_layout)
        layout.addSpacing(10)  # Extra space before next section
        
        # Lit exemplar calibration - plain text label
        exemplar_label = QLabel("Lit Key Exemplars")
        exemplar_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(exemplar_label)
        
        # Create horizontal layout for exemplar buttons
        exemplar_container = QHBoxLayout()
        
        self.exemplar_buttons = {}
        self.exemplar_swatches = {}
        
        # Left column container
        left_column = QVBoxLayout()
        left_column.setSpacing(10)
        
        # Left column - LW and LB
        for key_type, label in [("LW", "Left White"), ("LB", "Left Black")]:
            button = QPushButton(f"Calibrate {label}")
            button.setMinimumWidth(180)
            button.setMaximumWidth(240)
            button.clicked.connect(lambda checked, kt=key_type: self.calibrate_lit_exemplar_requested.emit(kt))
            button.setToolTip(
                "Captures a pressed-overlay example for this type. "
                "Pause on a frame where that type is highlighted, click the button, "
                "then click that highlighted overlay."
            )
            self.exemplar_buttons[key_type] = button
            
            # Color swatch next to button
            color_swatch = QLabel("")
            color_swatch.setFixedSize(20, 20)
            color_swatch.setStyleSheet("border: 1px solid black; background-color: gray;")
            self.exemplar_swatches[key_type] = color_swatch
            
            # Add to left column
            button_layout = QHBoxLayout()
            button_layout.setContentsMargins(0, 0, 0, 0)
            button_layout.addWidget(button)
            button_layout.addSpacing(10)  # Add spacing between button and color swatch
            button_layout.addWidget(color_swatch)
            left_column.addLayout(button_layout)
        
        # Right column container
        right_column = QVBoxLayout()
        right_column.setSpacing(10)
        
        # Right column - RW and RB
        for key_type, label in [("RW", "Right White"), ("RB", "Right Black")]:
            button = QPushButton(f"Calibrate {label}")
            button.setMinimumWidth(180)
            button.setMaximumWidth(240)
            button.clicked.connect(lambda checked, kt=key_type: self.calibrate_lit_exemplar_requested.emit(kt))
            button.setToolTip(
                "Captures a pressed-overlay example for this type. "
                "Pause on a frame where that type is highlighted, click the button, "
                "then click that highlighted overlay."
            )
            self.exemplar_buttons[key_type] = button
            
            # Color swatch next to button
            color_swatch = QLabel("")
            color_swatch.setFixedSize(20, 20)
            color_swatch.setStyleSheet("border: 1px solid black; background-color: gray;")
            self.exemplar_swatches[key_type] = color_swatch
            
            # Add to right column
            button_layout = QHBoxLayout()
            button_layout.setContentsMargins(0, 0, 0, 0)
            button_layout.addWidget(button)
            button_layout.addSpacing(10)  # Add spacing between button and color swatch
            button_layout.addWidget(color_swatch)
            right_column.addLayout(button_layout)
        
        # Add columns to container with spacing
        exemplar_container.addLayout(left_column)
        exemplar_container.addSpacing(50)  # 50 pixels between left and right columns
        exemplar_container.addLayout(right_column)
        exemplar_container.addStretch()  # Push everything to the left
        
        layout.addLayout(exemplar_container)
        
        layout.addStretch()  # Push everything to the top
        
        self.tab_widget.addTab(tab, "Calibration")
    
    def _create_overlay_settings_tab(self):
        """Tab 2: Overlay Settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)  # Make grey containers flush with tab header
        layout.setSpacing(5)  # Minimal spacing between grey containers
        
        # Key alignment
        alignment_group = QGroupBox("Key Alignment")
        alignment_group.setObjectName("first_in_tab")  # For CSS styling
        alignment_layout = QVBoxLayout(alignment_group)
        
        align_layout = QHBoxLayout()
        self.align_white_button = QPushButton("Align White Keys")
        self.align_white_button.setMaximumWidth(432)  # Increased by 20% from 360
        self.align_white_button.clicked.connect(self.align_white_keys_requested.emit)
        self.align_black_button = QPushButton("Align Black Keys")
        self.align_black_button.setMaximumWidth(432)  # Increased by 20% from 360
        self.align_black_button.clicked.connect(self.align_black_keys_requested.emit)
        
        # Place buttons side by side
        button_row = QHBoxLayout()
        button_row.addWidget(self.align_white_button)
        button_row.addSpacing(15)  # Move align black keys button to the right
        button_row.addWidget(self.align_black_button)
        button_row.addStretch()
        align_layout.addLayout(button_row)
        alignment_layout.addLayout(align_layout)
        
        layout.addWidget(alignment_group)
        
        # Overlay size adjustment
        size_group = QGroupBox("Overlay Size Adjustment")
        size_layout = QVBoxLayout(size_group)
        
        # White key dimensions - horizontal layout
        white_row = QHBoxLayout()
        # Set fixed width for height label to ensure alignment
        height_label_white = QLabel("White Key Height:")
        height_label_white.setFixedWidth(180)  # Prevent text cutoff at narrower panel widths
        white_row.addWidget(height_label_white)
        white_row.addSpacing(10)  # Add spacing between label and buttons
        
        # Height adjustment buttons
        self.white_height_dec_button = QPushButton("-")
        self.white_height_dec_button.setFixedSize(30, 30)
        self.white_height_dec_button.clicked.connect(lambda: self.overlay_size_adjustment_requested.emit("white", "height", -2))
        white_row.addWidget(self.white_height_dec_button)
        
        self.white_height_inc_button = QPushButton("+")
        self.white_height_inc_button.setFixedSize(30, 30)
        self.white_height_inc_button.clicked.connect(lambda: self.overlay_size_adjustment_requested.emit("white", "height", 2))
        white_row.addWidget(self.white_height_inc_button)
        
        white_row.addSpacing(60)  # Further increased space between height and width to prevent cutoff
        # Set fixed width for width label to ensure alignment
        width_label_white = QLabel("White Key Width:")
        width_label_white.setFixedWidth(180)  # Prevent text cutoff at narrower panel widths
        white_row.addWidget(width_label_white)
        white_row.addSpacing(10)  # Add spacing between label and buttons
        
        # Width adjustment buttons
        self.white_width_dec_button = QPushButton("-")
        self.white_width_dec_button.setFixedSize(30, 30)
        self.white_width_dec_button.clicked.connect(lambda: self.overlay_size_adjustment_requested.emit("white", "width", -2))
        white_row.addWidget(self.white_width_dec_button)
        
        self.white_width_inc_button = QPushButton("+")
        self.white_width_inc_button.setFixedSize(30, 30)
        self.white_width_inc_button.clicked.connect(lambda: self.overlay_size_adjustment_requested.emit("white", "width", 2))
        white_row.addWidget(self.white_width_inc_button)
        
        white_row.addStretch()
        size_layout.addLayout(white_row)
        
        # Black key dimensions - horizontal layout
        black_row = QHBoxLayout()
        # Set fixed width for height label to ensure alignment
        height_label_black = QLabel("Black Key Height:")
        height_label_black.setFixedWidth(180)  # Prevent text cutoff at narrower panel widths
        black_row.addWidget(height_label_black)
        black_row.addSpacing(10)  # Add spacing between label and buttons
        
        # Height adjustment buttons
        self.black_height_dec_button = QPushButton("-")
        self.black_height_dec_button.setFixedSize(30, 30)
        self.black_height_dec_button.clicked.connect(lambda: self.overlay_size_adjustment_requested.emit("black", "height", -2))
        black_row.addWidget(self.black_height_dec_button)
        
        self.black_height_inc_button = QPushButton("+")
        self.black_height_inc_button.setFixedSize(30, 30)
        self.black_height_inc_button.clicked.connect(lambda: self.overlay_size_adjustment_requested.emit("black", "height", 2))
        black_row.addWidget(self.black_height_inc_button)
        
        black_row.addSpacing(60)  # Further increased space between height and width to prevent cutoff
        # Set fixed width for width label to ensure alignment
        width_label_black = QLabel("Black Key Width:")
        width_label_black.setFixedWidth(180)  # Prevent text cutoff at narrower panel widths
        black_row.addWidget(width_label_black)
        black_row.addSpacing(10)  # Add spacing between label and buttons
        
        # Width adjustment buttons
        self.black_width_dec_button = QPushButton("-")
        self.black_width_dec_button.setFixedSize(30, 30)
        self.black_width_dec_button.clicked.connect(lambda: self.overlay_size_adjustment_requested.emit("black", "width", -2))
        black_row.addWidget(self.black_width_dec_button)
        
        self.black_width_inc_button = QPushButton("+")
        self.black_width_inc_button.setFixedSize(30, 30)
        self.black_width_inc_button.clicked.connect(lambda: self.overlay_size_adjustment_requested.emit("black", "width", 2))
        black_row.addWidget(self.black_width_inc_button)
        
        black_row.addStretch()
        size_layout.addLayout(black_row)
        
        layout.addWidget(size_group)
        
        # Overlay color
        color_group = QGroupBox("Overlay Appearance")
        color_layout = QVBoxLayout(color_group)
        
        # Horizontal layout for color dropdown and square
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Overlay Color:"))
        
        self.overlay_color_combo = QComboBox()
        self.overlay_color_combo.setMaximumWidth(160)  # Doubled width
        self.overlay_color_combo.addItems(["Red", "Green", "Blue", "Yellow", "Cyan", "Magenta", "White"])
        self.overlay_color_combo.currentTextChanged.connect(self.overlay_color_changed.emit)
        self.overlay_color_combo.currentTextChanged.connect(self._update_color_square)
        color_row.addWidget(self.overlay_color_combo)
        
        # Color square indicator
        self.color_square = QLabel("")
        self.color_square.setFixedSize(20, 20)
        self._update_color_square("Red")  # Initialize with default color
        color_row.addWidget(self.color_square)
        color_row.addStretch()
        
        color_layout.addLayout(color_row)
        
        layout.addWidget(color_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Overlays")
    
    def _create_basic_detection_tab(self):
        """Tab 3: Basic Detection Settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)  # Make grey containers flush with tab header
        layout.setSpacing(5)  # Minimal spacing between grey containers

        help_section = CollapsibleSection("Help", expanded=False)
        help_section.setStyleSheet("QLabel { font-size: 9pt; }")  # shrink detection help text
        help_layout = help_section.content_layout()
        help_lines = [
            "Before tuning detection: run Unlit Key Calibration + at least one Lit Key Exemplar.",
            "Detection Threshold: main sensitivity setting for pressed vs unpressed.",
            "If notes are missed: lower the threshold. If you get false notes: raise the threshold.",
            ("Histogram Detection: uses a color-pattern match inside each overlay. "
             "Use when pressed overlays have strong gradients or uneven lighting."),
            ("Delta Detection: uses frame-to-frame change to confirm press/release. "
             "Use when the pressed color fades in/out gradually instead of switching cleanly."),
            "Black Key Filter: reduces false black-key presses caused by nearby overlays."
        ]
        for line in help_lines:
            label = QLabel(line)
            label.setWordWrap(True)
            help_layout.addWidget(label)
        layout.addWidget(help_section)
        
        # Detection threshold
        threshold_group = QGroupBox("Detection Threshold")
        threshold_group.setObjectName("first_in_tab")  # For CSS styling
        threshold_layout = QVBoxLayout(threshold_group)
        threshold_layout.setContentsMargins(15, 10, 15, 10)
        
        self.detection_threshold_slider = QSlider(Qt.Horizontal)
        self.detection_threshold_slider.setMaximumWidth(150)  # Half of default width
        self.detection_threshold_slider.setRange(0, 100)
        self.detection_threshold_slider.setValue(50)
        self.detection_threshold_slider.valueChanged.connect(self._handle_detection_threshold_change)
        self.detection_threshold_slider.setToolTip(
            "Main sensitivity. Lower = detects more; higher = fewer false notes."
        )
        
        self.detection_threshold_label = QLabel("50%")
        self.detection_threshold_label.setToolTip(
            "Main sensitivity. Lower = detects more; higher = fewer false notes."
        )
        
        threshold_layout.addWidget(QLabel("Detection Threshold:"))
        threshold_layout.addWidget(self.detection_threshold_slider)
        threshold_layout.addWidget(self.detection_threshold_label)
        
        layout.addWidget(threshold_group)
        
        # Detection modes
        modes_group = QGroupBox("Detection Modes")
        modes_layout = QVBoxLayout(modes_group)
        modes_layout.setContentsMargins(15, 10, 15, 10)
        
        # Histogram detection with sensitivity slider
        histogram_layout = QHBoxLayout()
        self.histogram_detection_cb = QCheckBox("Enable Histogram Detection")
        self.histogram_detection_cb.toggled.connect(self.histogram_detection_toggled.emit)
        self.histogram_detection_cb.toggled.connect(self._update_histogram_slider_state)
        self.histogram_detection_cb.setToolTip(
            "Uses a color-pattern match inside the overlay. Helpful with gradients/uneven lighting."
        )
        histogram_layout.addWidget(self.histogram_detection_cb)
        histogram_layout.addStretch(2)  # Give checkbox text room; keep slider nearer center-right
        
        # Add histogram threshold slider
        self.histogram_threshold_slider = QSlider(Qt.Horizontal)
        self.histogram_threshold_slider.setMaximumWidth(150)
        self.histogram_threshold_slider.setRange(10, 100)  # 0.1 to 1.0
        self.histogram_threshold_slider.setValue(80)  # Default 0.8
        self.histogram_threshold_slider.valueChanged.connect(self._handle_histogram_threshold_change)
        self.histogram_threshold_slider.setEnabled(False)  # Initially disabled
        self.histogram_threshold_slider.setToolTip(
            "How strong the histogram match must be (only used when Histogram Detection is enabled)."
        )
        histogram_layout.addWidget(self.histogram_threshold_slider)
        
        self.histogram_threshold_label = QLabel("0.80")
        self.histogram_threshold_label.setMinimumWidth(40)
        self.histogram_threshold_label.setToolTip(
            "How strong the histogram match must be (only used when Histogram Detection is enabled)."
        )
        histogram_layout.addWidget(self.histogram_threshold_label)
        histogram_layout.addStretch(1)
        
        modes_layout.addLayout(histogram_layout)
        
        # Delta detection with rise/fall sliders
        self.delta_detection_cb = QCheckBox("Enable Delta Detection")
        self.delta_detection_cb.toggled.connect(self.delta_detection_toggled.emit)
        self.delta_detection_cb.toggled.connect(self._update_delta_sliders_state)
        self.delta_detection_cb.setToolTip(
            "Uses frame-to-frame change to confirm press/release (helps when color fades)."
        )
        modes_layout.addWidget(self.delta_detection_cb)
        
        # Rise delta threshold
        rise_layout = QHBoxLayout()
        rise_layout.setContentsMargins(20, 0, 0, 0)  # Indent to show it's under Delta Detection
        rise_label = QLabel("Rise Threshold:")
        rise_label.setFixedWidth(260)  # Fixed width for alignment (280-20 indent)
        rise_layout.addWidget(rise_label)
        rise_layout.addStretch(2)
        
        self.rise_delta_slider = QSlider(Qt.Horizontal)
        self.rise_delta_slider.setMaximumWidth(150)
        self.rise_delta_slider.setRange(1, 50)  # 0.01 to 0.50
        self.rise_delta_slider.setValue(15)  # Default 0.15
        self.rise_delta_slider.valueChanged.connect(self._handle_rise_delta_change)
        self.rise_delta_slider.setEnabled(False)  # Initially disabled
        self.rise_delta_slider.setToolTip(
            "How big the change must be to count as a press (only used when Delta Detection is enabled)."
        )
        rise_layout.addWidget(self.rise_delta_slider)
        
        self.rise_delta_label = QLabel("0.15")
        self.rise_delta_label.setMinimumWidth(40)
        self.rise_delta_label.setToolTip(
            "How big the change must be to count as a press (only used when Delta Detection is enabled)."
        )
        rise_layout.addWidget(self.rise_delta_label)
        rise_layout.addStretch(1)
        
        modes_layout.addLayout(rise_layout)
        
        # Fall delta threshold
        fall_layout = QHBoxLayout()
        fall_layout.setContentsMargins(20, 0, 0, 0)  # Indent to show it's under Delta Detection
        fall_label = QLabel("Fall Threshold:")
        fall_label.setFixedWidth(260)  # Fixed width for alignment (280-20 indent)
        fall_layout.addWidget(fall_label)
        fall_layout.addStretch(2)
        
        self.fall_delta_slider = QSlider(Qt.Horizontal)
        self.fall_delta_slider.setMaximumWidth(150)
        self.fall_delta_slider.setRange(1, 50)  # 0.01 to 0.50
        self.fall_delta_slider.setValue(15)  # Default 0.15
        self.fall_delta_slider.valueChanged.connect(self._handle_fall_delta_change)
        self.fall_delta_slider.setEnabled(False)  # Initially disabled
        self.fall_delta_slider.setToolTip(
            "How big the change must be to count as a release (only used when Delta Detection is enabled)."
        )
        fall_layout.addWidget(self.fall_delta_slider)
        
        self.fall_delta_label = QLabel("0.15")
        self.fall_delta_label.setMinimumWidth(40)
        self.fall_delta_label.setToolTip(
            "How big the change must be to count as a release (only used when Delta Detection is enabled)."
        )
        fall_layout.addWidget(self.fall_delta_label)
        fall_layout.addStretch(1)
        
        modes_layout.addLayout(fall_layout)
        
        # Black key filter with similarity ratio slider
        filter_layout = QHBoxLayout()
        self.black_key_filter_cb = QCheckBox("Enable Black Key Filter")
        self.black_key_filter_cb.toggled.connect(self.winner_takes_black_changed.emit)
        self.black_key_filter_cb.toggled.connect(self._update_similarity_slider_state)
        self.black_key_filter_cb.setToolTip(
            "Reduces false black-key presses from nearby overlays."
        )
        filter_layout.addWidget(self.black_key_filter_cb)
        filter_layout.addStretch(2)  # Give checkbox text room; keep slider nearer center-right
        
        # Add similarity ratio slider
        self.similarity_ratio_slider = QSlider(Qt.Horizontal)
        self.similarity_ratio_slider.setMaximumWidth(150)
        self.similarity_ratio_slider.setRange(10, 100)  # 0.1 to 1.0
        self.similarity_ratio_slider.setValue(60)  # Default 0.6
        self.similarity_ratio_slider.valueChanged.connect(self._handle_similarity_ratio_change)
        self.similarity_ratio_slider.setEnabled(False)  # Initially disabled
        self.similarity_ratio_slider.setToolTip(
            "Controls how strict black-key filtering is (only used when Black Key Filter is enabled)."
        )
        filter_layout.addWidget(self.similarity_ratio_slider)
        
        self.similarity_ratio_label = QLabel("0.60")
        self.similarity_ratio_label.setMinimumWidth(40)
        self.similarity_ratio_label.setToolTip(
            "Controls how strict black-key filtering is (only used when Black Key Filter is enabled)."
        )
        filter_layout.addWidget(self.similarity_ratio_label)
        filter_layout.addStretch(1)
        
        modes_layout.addLayout(filter_layout)
        
        layout.addWidget(modes_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Detection")
    
    def _create_spark_detection_tab(self):
        """Tab 4: Spark Detection (scrollable to avoid clipping)."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        help_section = CollapsibleSection("Help", expanded=False)
        help_layout = help_section.content_layout()
        help_lines = [
            "Use Spark Detection only when:",
            "1) Key overlays stay ON across repeated notes (false continuous press).",
            "2) Key overlays are solid color (no fading or gradients)."
        ]
        for line in help_lines:
            label = QLabel(line)
            label.setWordWrap(True)
            label.setStyleSheet("font-size: 9pt;")  # slightly smaller help text
            help_layout.addWidget(label)
        layout.addWidget(help_section)

        # Spark detection toggle
        main_group = QGroupBox("Spark Detection")
        main_group.setObjectName("first_in_tab")  # For CSS styling
        main_layout = QVBoxLayout(main_group)

        self.spark_detection_cb = QCheckBox("Enable Spark Detection")
        self.spark_detection_cb.toggled.connect(self.spark_detection_toggled.emit)
        self.spark_detection_cb.toggled.connect(self._update_spark_controls_state)
        self.spark_detection_cb.setToolTip(
            "Use only when key overlays stay ON across repeated notes (false continuous press), "
            "and the overlays are solid color (no fading or gradients)."
        )
        main_layout.addWidget(self.spark_detection_cb)

        # Sensitivity
        main_layout.addWidget(QLabel("Sensitivity:"))
        self.spark_sensitivity_slider = QSlider(Qt.Horizontal)
        self.spark_sensitivity_slider.setMaximumWidth(150)  # Half of default width
        self.spark_sensitivity_slider.setRange(0, 100)
        self.spark_sensitivity_slider.setValue(50)
        self.spark_sensitivity_slider.valueChanged.connect(self._handle_spark_sensitivity_change)
        self.spark_sensitivity_slider.setToolTip(
            "Controls how aggressively Spark Detection splits false continuous notes."
        )
        main_layout.addWidget(self.spark_sensitivity_slider)

        self.spark_sensitivity_label = QLabel("50%")
        self.spark_sensitivity_label.setToolTip(
            "Controls how aggressively Spark Detection splits false continuous notes."
        )
        main_layout.addWidget(self.spark_sensitivity_label)

        layout.addWidget(main_group)

        # Spark calibration
        calibration_group = QGroupBox("Spark Calibration")
        calibration_layout = QVBoxLayout(calibration_group)

        # ROI selection
        roi_layout = QHBoxLayout()
        roi_button = QPushButton("Select Spark ROI")
        roi_button.setMaximumWidth(264)  # Increased by 10% from 240
        roi_button.clicked.connect(self.spark_roi_selection_requested.emit)
        roi_button.setToolTip(
            "Select the region above the keys where spark bars and sparks appear."
        )
        roi_layout.addWidget(roi_button)

        # Add spacing between buttons
        roi_layout.addSpacing(35)  # Increased spacing to move Hide button more to the right

        # Add toggle button for showing/hiding spark overlays
        self.spark_roi_toggle_button = QPushButton("Hide Spark Overlays")
        self.spark_roi_toggle_button.setMaximumWidth(396)  # Increased by 10% from 360
        self.spark_roi_toggle_button.setCheckable(True)
        self.spark_roi_toggle_button.clicked.connect(self._toggle_spark_roi_visibility)
        self.spark_roi_toggle_button.setToolTip(
            "Show or hide the spark ROI overlay on the video."
        )
        roi_layout.addWidget(self.spark_roi_toggle_button)
        roi_layout.addStretch()

        calibration_layout.addLayout(roi_layout)

        manual_section = CollapsibleSection("Manual Calibration", expanded=False)
        calib_buttons_layout = manual_section.content_layout()

        # Step 1: Calibrate Background
        step1_layout = QHBoxLayout()
        step1_label = QLabel("Step 1)")
        step1_label.setFixedWidth(60)  # Fixed width for alignment
        step1_layout.addWidget(step1_label)
        self.spark_bg_button = QPushButton("Calibrate Background")
        self.spark_bg_button.setFixedWidth(300)  # Fixed width for exact alignment
        self.spark_bg_button.clicked.connect(lambda: self.spark_calibration_requested.emit("background"))
        self.spark_bg_button.setToolTip(
            "Manual calibration: capture baseline brightness when there are no bars or sparks."
        )
        step1_layout.addWidget(self.spark_bg_button)
        step1_layout.addStretch()
        calib_buttons_layout.addLayout(step1_layout)

        # Step 2: Calibrate Bar Only
        step2_layout = QHBoxLayout()
        step2_label = QLabel("Step 2)")
        step2_label.setFixedWidth(60)  # Fixed width for alignment
        step2_layout.addWidget(step2_label)
        self.spark_bar_button = QPushButton("Calibrate Bar Only")
        self.spark_bar_button.setFixedWidth(300)  # Fixed width for exact alignment
        self.spark_bar_button.clicked.connect(lambda: self.spark_calibration_requested.emit("bar_only"))
        self.spark_bar_button.setToolTip(
            "Manual calibration: click an overlay showing colored bars with no sparks."
        )
        step2_layout.addWidget(self.spark_bar_button)
        step2_layout.addStretch()
        calib_buttons_layout.addLayout(step2_layout)

        # Step 3: Calibrate Dimmest Sparks
        step3_layout = QHBoxLayout()
        step3_label = QLabel("Step 3)")
        step3_label.setFixedWidth(60)  # Fixed width for alignment
        step3_layout.addWidget(step3_label)
        self.spark_brightest_button = QPushButton("Calibrate Dimmest Sparks")
        self.spark_brightest_button.setFixedWidth(300)  # Fixed width for exact alignment
        self.spark_brightest_button.clicked.connect(lambda: self.spark_calibration_requested.emit("dimmest_sparks"))
        self.spark_brightest_button.setToolTip(
            "Manual calibration: click an overlay where sparks are just barely visible."
        )
        step3_layout.addWidget(self.spark_brightest_button)
        step3_layout.addStretch()
        calib_buttons_layout.addLayout(step3_layout)

        calibration_layout.addWidget(manual_section)

        # Auto calibration
        calibration_layout.addWidget(QLabel("Auto Calibration:"))
        auto_layout = QVBoxLayout()
        auto_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for left alignment

        # Store button and status references for updates
        self.auto_calib_buttons = {}
        self.auto_calib_status_labels = {}

        # Create white keys row (LW and RW aligned horizontally)
        white_keys_layout = QHBoxLayout()
        white_keys_layout.setContentsMargins(0, 0, 0, 0)

        # Left Hand White
        lw_row = QHBoxLayout()
        lw_row.setSpacing(10)  # Space between button and status

        lw_button = QPushButton(f"Auto {KEY_TYPE_LABELS['LW']}")
        lw_button.setFixedWidth(220)  # Fixed width for alignment
        lw_button.clicked.connect(lambda: self.auto_spark_calibration_requested.emit("LW"))
        lw_button.setToolTip(
            "Recommended: auto-calibrate spark detection for this key type. "
            "Navigate to the frame where a key first turns ON, then click that overlay."
        )
        self.auto_calib_buttons["LW"] = lw_button
        lw_row.addWidget(lw_button)

        lw_status = QLabel("Not Set")
        lw_status.setStyleSheet("color: grey; font-style: italic;")
        lw_status.setFixedWidth(80)
        self.auto_calib_status_labels["LW"] = lw_status
        lw_row.addWidget(lw_status)

        white_keys_layout.addLayout(lw_row)
        white_keys_layout.addSpacing(50)  # Space between left and right hand buttons

        # Right Hand White
        rw_row = QHBoxLayout()
        rw_row.setSpacing(10)  # Space between button and status

        rw_button = QPushButton(f"Auto {KEY_TYPE_LABELS['RW']}")
        rw_button.setFixedWidth(220)  # Fixed width for alignment
        rw_button.clicked.connect(lambda: self.auto_spark_calibration_requested.emit("RW"))
        rw_button.setToolTip(
            "Recommended: auto-calibrate spark detection for this key type. "
            "Navigate to the frame where a key first turns ON, then click that overlay."
        )
        self.auto_calib_buttons["RW"] = rw_button
        rw_row.addWidget(rw_button)

        rw_status = QLabel("Not Set")
        rw_status.setStyleSheet("color: grey; font-style: italic;")
        rw_status.setFixedWidth(80)
        self.auto_calib_status_labels["RW"] = rw_status
        rw_row.addWidget(rw_status)

        white_keys_layout.addLayout(rw_row)
        white_keys_layout.addStretch()  # Push everything left

        auto_layout.addLayout(white_keys_layout)

        # Create black keys row (LB and RB aligned horizontally)
        black_keys_layout = QHBoxLayout()
        black_keys_layout.setContentsMargins(0, 0, 0, 0)

        # Left Hand Black
        lb_row = QHBoxLayout()
        lb_row.setSpacing(10)  # Space between button and status

        lb_button = QPushButton(f"Auto {KEY_TYPE_LABELS['LB']}")
        lb_button.setFixedWidth(220)  # Fixed width for alignment
        lb_button.clicked.connect(lambda: self.auto_spark_calibration_requested.emit("LB"))
        lb_button.setToolTip(
            "Recommended: auto-calibrate spark detection for this key type. "
            "Navigate to the frame where a key first turns ON, then click that overlay."
        )
        self.auto_calib_buttons["LB"] = lb_button
        lb_row.addWidget(lb_button)

        lb_status = QLabel("Not Set")
        lb_status.setStyleSheet("color: grey; font-style: italic;")
        lb_status.setFixedWidth(80)
        self.auto_calib_status_labels["LB"] = lb_status
        lb_row.addWidget(lb_status)

        black_keys_layout.addLayout(lb_row)
        black_keys_layout.addSpacing(50)  # Space between left and right hand buttons

        # Right Hand Black
        rb_row = QHBoxLayout()
        rb_row.setSpacing(10)  # Space between button and status

        rb_button = QPushButton(f"Auto {KEY_TYPE_LABELS['RB']}")
        rb_button.setFixedWidth(220)  # Fixed width for alignment
        rb_button.clicked.connect(lambda: self.auto_spark_calibration_requested.emit("RB"))
        rb_button.setToolTip(
            "Recommended: auto-calibrate spark detection for this key type. "
            "Navigate to the frame where a key first turns ON, then click that overlay."
        )
        self.auto_calib_buttons["RB"] = rb_button
        rb_row.addWidget(rb_button)

        rb_status = QLabel("Not Set")
        rb_status.setStyleSheet("color: grey; font-style: italic;")
        rb_status.setFixedWidth(80)
        self.auto_calib_status_labels["RB"] = rb_status
        rb_row.addWidget(rb_status)

        black_keys_layout.addLayout(rb_row)
        black_keys_layout.addStretch()  # Push everything left

        auto_layout.addLayout(black_keys_layout)

        calibration_layout.addLayout(auto_layout)

        layout.addWidget(calibration_group)

        # Preview / status
        preview_group = QGroupBox("Spark Preview / Status")
        preview_layout = QVBoxLayout(preview_group)
        self.spark_preview_label = QLabel("Preview will show spark calibration status here.")
        self.spark_preview_label.setWordWrap(True)
        preview_layout.addWidget(self.spark_preview_label)

        self.spark_status_label = QLabel("Preview not available yet.")
        self.spark_status_label.setWordWrap(True)
        self.spark_status_label.setStyleSheet("color: grey; font-style: italic;")
        preview_layout.addWidget(self.spark_status_label)

        layout.addWidget(preview_group)

        layout.addStretch()

        scroll_area.setWidget(content)
        tab_layout.addWidget(scroll_area)
        self.tab_widget.addTab(tab, "Spark Detection")
    
    def _create_midi_settings_tab(self):
        """Tab 5: MIDI Settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)  # Make grey containers flush with tab header
        layout.setSpacing(5)  # Minimal spacing between grey containers
        
        # FPS Override
        fps_group = QGroupBox("Frame Rate Override")
        fps_group.setObjectName("first_in_tab")  # For CSS styling
        fps_layout = QVBoxLayout(fps_group)
        
        fps_button_layout = QHBoxLayout()
        fps_button_layout.setSpacing(5)  # Minimal spacing between buttons
        
        self.fps_30_button = QPushButton("30 FPS")
        self.fps_30_button.setMaximumWidth(132)
        self.fps_30_button.setCheckable(True)
        self.fps_30_button.clicked.connect(lambda: self._set_fps_override(30))
        
        self.fps_60_button = QPushButton("60 FPS")
        self.fps_60_button.setMaximumWidth(132)
        self.fps_60_button.setCheckable(True)
        self.fps_60_button.clicked.connect(lambda: self._set_fps_override(60))
        
        self.fps_auto_button = QPushButton("Auto")
        self.fps_auto_button.setMaximumWidth(132)
        self.fps_auto_button.setCheckable(True)
        self.fps_auto_button.setChecked(True)
        self.fps_auto_button.clicked.connect(lambda: self._set_fps_override(None))
        
        fps_button_layout.addWidget(self.fps_30_button)
        fps_button_layout.addWidget(self.fps_60_button)
        fps_button_layout.addWidget(self.fps_auto_button)
        fps_button_layout.addStretch()  # Push buttons to the left
        
        fps_layout.addLayout(fps_button_layout)
        
        # Current FPS display
        self.fps_display_label = QLabel("Current FPS: Auto-detected")
        fps_layout.addWidget(self.fps_display_label)
        
        layout.addWidget(fps_group)
        
        # Custom MIDI Processing Range
        processing_range_group = QGroupBox("Custom MIDI Processing Range")
        processing_range_layout = QVBoxLayout(processing_range_group)
        
        # Frame controls in grid for alignment
        processing_grid = QGridLayout()
        processing_grid.setHorizontalSpacing(25)  # Increased spacing to move buttons to the right
        processing_grid.setColumnStretch(3, 1)  # Push everything to the left
        
        # Processing start frame
        processing_start_label = QLabel("Start Frame:")
        processing_start_label.setFixedWidth(144)
        processing_grid.addWidget(processing_start_label, 0, 0)
        
        self.processing_start_frame_spin = QSpinBox()
        self.processing_start_frame_spin.setMaximumWidth(180)  # widened for readability
        self.processing_start_frame_spin.setRange(0, 999999)
        self.processing_start_frame_spin.setValue(0)
        self.processing_start_frame_spin.valueChanged.connect(self.processing_start_frame_changed.emit)
        processing_grid.addWidget(self.processing_start_frame_spin, 0, 1)
        
        self.processing_start_set_button = QPushButton("Set to Current")
        self.processing_start_set_button.setMaximumWidth(240)  # Increased by 20% from 200
        self.processing_start_set_button.clicked.connect(self._set_processing_start_to_current)
        processing_grid.addWidget(self.processing_start_set_button, 0, 2)
        
        # Processing end frame
        processing_end_label = QLabel("End Frame:")
        processing_end_label.setFixedWidth(144)
        processing_grid.addWidget(processing_end_label, 1, 0)
        
        self.processing_end_frame_spin = QSpinBox()
        self.processing_end_frame_spin.setMaximumWidth(180)  # widened for readability
        self.processing_end_frame_spin.setRange(0, 999999)
        self.processing_end_frame_spin.setValue(0)
        self.processing_end_frame_spin.valueChanged.connect(self.processing_end_frame_changed.emit)
        processing_grid.addWidget(self.processing_end_frame_spin, 1, 1)
        
        self.processing_end_set_button = QPushButton("Set to Current")
        self.processing_end_set_button.setMaximumWidth(240)  # Increased by 20% from 200
        self.processing_end_set_button.clicked.connect(self._set_processing_end_to_current)
        processing_grid.addWidget(self.processing_end_set_button, 1, 2)
        
        processing_range_layout.addLayout(processing_grid)
        layout.addWidget(processing_range_group)
        
        # Octave transpose is configured in the Calibration tab.
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "MIDI")
    
    def _create_video_trim_tab(self):
        """Tab 6: Video Trim Settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)  # Make grey containers flush with tab header
        layout.setSpacing(5)  # Minimal spacing between grey containers
        
        trim_group = QGroupBox("Video Processing Range")
        trim_group.setObjectName("first_in_tab")  # For CSS styling
        trim_layout = QVBoxLayout(trim_group)
        
        # Frame controls in grid for alignment
        frame_grid = QGridLayout()
        frame_grid.setHorizontalSpacing(10)
        frame_grid.setColumnStretch(3, 1)  # Push everything to the left
        
        # Start frame
        start_label = QLabel("Start Frame:")
        start_label.setFixedWidth(144)  # Scaled for 14pt font
        frame_grid.addWidget(start_label, 0, 0)
        
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setMaximumWidth(60)  # 1/10 of default width
        self.start_frame_spin.setRange(0, 999999)
        self.start_frame_spin.setValue(0)
        self.start_frame_spin.valueChanged.connect(self.start_frame_changed.emit)
        frame_grid.addWidget(self.start_frame_spin, 0, 1)
        
        self.trim_start_set_button = QPushButton("Set to Current")
        self.trim_start_set_button.setMaximumWidth(200)
        self.trim_start_set_button.clicked.connect(self._set_trim_start_to_current)
        frame_grid.addWidget(self.trim_start_set_button, 0, 2)
        
        # End frame
        end_label = QLabel("End Frame:")
        end_label.setFixedWidth(144)  # Scaled for 14pt font
        frame_grid.addWidget(end_label, 1, 0)
        
        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setMaximumWidth(60)  # 1/10 of default width
        self.end_frame_spin.setRange(-1, 999999)
        self.end_frame_spin.setValue(-1)
        self.end_frame_spin.valueChanged.connect(self.end_frame_changed.emit)
        frame_grid.addWidget(self.end_frame_spin, 1, 1)
        
        self.trim_end_set_button = QPushButton("Set to Current")
        self.trim_end_set_button.setMaximumWidth(200)
        self.trim_end_set_button.clicked.connect(self._set_trim_end_to_current)
        frame_grid.addWidget(self.trim_end_set_button, 1, 2)
        
        trim_layout.addLayout(frame_grid)
        
        # Trim Video button
        self.trim_video_button = QPushButton("Trim Video")
        self.trim_video_button.setMaximumWidth(200)
        self.trim_video_button.clicked.connect(self._handle_trim_video_request)
        trim_layout.addWidget(self.trim_video_button)
        
        layout.addWidget(trim_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Video Trim")
    
    def _create_optional_settings_tab(self):
        """Tab 7: Optional Settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)  # Make grey containers flush with tab header
        layout.setSpacing(5)  # Minimal spacing between grey containers
        
        optional_group = QGroupBox("Optional Features")
        optional_group.setObjectName("first_in_tab")  # For CSS styling
        optional_layout = QVBoxLayout(optional_group)
        
        # Hand assignment
        self.hand_assignment_cb = QCheckBox("Enable Hand Assignment (MIDI Channels)")
        self.hand_assignment_cb.toggled.connect(self.hand_assignment_toggled.emit)
        optional_layout.addWidget(self.hand_assignment_cb)
        
        
        # Add more optional settings here as needed
        
        layout.addWidget(optional_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Optional")
    
    def _handle_conversion_request(self):
        """Handle conversion button click."""
        self.convert_button.setText("Converting...")
        self.convert_button.setEnabled(False)
        self.conversion_status.setText("Converting video to MIDI...")
        self.conversion_requested.emit()
    
    def _handle_detection_threshold_change(self, value):
        """Handle detection threshold slider change."""
        threshold = value / 100.0
        self.detection_threshold_label.setText(f"{value}%")
        self.detection_threshold_changed.emit(threshold)
    
    def _handle_spark_sensitivity_change(self, value):
        """Handle spark sensitivity slider change."""
        sensitivity = value / 100.0
        self.spark_sensitivity_label.setText(f"{value}%")
        self.spark_detection_sensitivity_changed.emit(sensitivity)
    
    def _handle_histogram_threshold_change(self, value):
        """Handle histogram threshold slider change."""
        threshold = value / 100.0
        self.histogram_threshold_label.setText(f"{threshold:.2f}")
        self.histogram_threshold_changed.emit(threshold)
    
    def _handle_rise_delta_change(self, value):
        """Handle rise delta threshold slider change."""
        threshold = value / 100.0
        self.rise_delta_label.setText(f"{threshold:.2f}")
        self.rise_delta_threshold_changed.emit(threshold)
    
    def _handle_fall_delta_change(self, value):
        """Handle fall delta threshold slider change."""
        threshold = value / 100.0
        self.fall_delta_label.setText(f"{threshold:.2f}")
        self.fall_delta_threshold_changed.emit(threshold)
    
    def _handle_similarity_ratio_change(self, value):
        """Handle similarity ratio slider change."""
        ratio = value / 100.0
        self.similarity_ratio_label.setText(f"{ratio:.2f}")
        self.similarity_ratio_changed.emit(ratio)
    
    def _update_histogram_slider_state(self, checked):
        """Enable/disable histogram threshold slider based on checkbox state."""
        self.histogram_threshold_slider.setEnabled(checked)
    
    def _update_delta_sliders_state(self, checked):
        """Enable/disable delta threshold sliders based on checkbox state."""
        self.rise_delta_slider.setEnabled(checked)
        self.fall_delta_slider.setEnabled(checked)
    
    def _update_similarity_slider_state(self, checked):
        """Enable/disable similarity ratio slider based on checkbox state."""
        self.similarity_ratio_slider.setEnabled(checked)
    
    def _update_spark_controls_state(self, spark_enabled):
        """Enable/disable all spark detection controls based on main checkbox state."""
        # Sensitivity slider and label
        self.spark_sensitivity_slider.setEnabled(spark_enabled)
        
        # ROI selection and toggle buttons
        for widget in self.findChildren(QPushButton):
            if widget.text() == "Select Spark ROI":
                widget.setEnabled(spark_enabled)
                break
        self.spark_roi_toggle_button.setEnabled(spark_enabled)
        
        # Manual calibration buttons
        self.spark_bg_button.setEnabled(spark_enabled)
        self.spark_bar_button.setEnabled(spark_enabled)
        self.spark_brightest_button.setEnabled(spark_enabled)
        
        # Auto calibration buttons
        for key_type in ["LW", "LB", "RW", "RB"]:
            if key_type in self.auto_calib_buttons:
                self.auto_calib_buttons[key_type].setEnabled(spark_enabled)
    
    def _set_processing_start_to_current(self):
        """Set processing start frame to current video frame."""
        if hasattr(self, 'app_state') and self.app_state and hasattr(self.app_state.video, 'current_frame_index'):
            current_frame = self.app_state.video.current_frame_index
            self.processing_start_frame_spin.setValue(current_frame)
    
    def _set_processing_end_to_current(self):
        """Set processing end frame to current video frame."""
        if hasattr(self, 'app_state') and self.app_state and hasattr(self.app_state.video, 'current_frame_index'):
            current_frame = self.app_state.video.current_frame_index
            self.processing_end_frame_spin.setValue(current_frame)
    
    def _set_trim_start_to_current(self):
        """Set trim start frame to current video frame."""
        if hasattr(self, 'app_state') and self.app_state and hasattr(self.app_state.video, 'current_frame_index'):
            current_frame = self.app_state.video.current_frame_index
            self.start_frame_spin.setValue(current_frame)
    
    def _set_trim_end_to_current(self):
        """Set trim end frame to current video frame."""
        if hasattr(self, 'app_state') and self.app_state and hasattr(self.app_state.video, 'current_frame_index'):
            current_frame = self.app_state.video.current_frame_index
            self.end_frame_spin.setValue(current_frame)
    
    def _handle_trim_video_request(self):
        """Handle trim video button click with confirmation dialog."""
        from PySide6.QtWidgets import QMessageBox
        
        start_frame = self.start_frame_spin.value()
        end_frame = self.end_frame_spin.value()
        
        # Validate trim range
        if end_frame != -1 and start_frame >= end_frame:
            QMessageBox.warning(self, "Invalid Trim Range", 
                              "Start frame must be less than end frame.")
            return
        
        # Create red warning dialog
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("⚠️ Trim Video - Irreversible Action")
        msg_box.setIcon(QMessageBox.Warning)
        
        # Red styling for the dialog
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #2b2b2b;
                color: white;
            }
            QMessageBox QLabel {
                color: #ff6b6b;
                font-weight: bold;
                font-size: 14px;
            }
            QMessageBox QPushButton {
                background-color: #ff4757;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QMessageBox QPushButton:hover {
                background-color: #ff3742;
            }
            QMessageBox QPushButton:pressed {
                background-color: #ff2731;
            }
        """)
        
        end_text = f"frame {end_frame}" if end_frame != -1 else "end of video"
        msg_box.setText(f"""
<b>⚠️ WARNING: This action is IRREVERSIBLE</b><br><br>
This will permanently trim the video session to frames {start_frame} to {end_text}.<br><br>
<b>After trimming:</b><br>
• Frames outside this range will become inaccessible<br>
• Video navigation will be restricted to this range<br>
• MIDI processing will be limited to this range<br><br>
<b>Are you sure you want to proceed?</b>
        """)
        
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        msg_box.setDefaultButton(QMessageBox.Cancel)
        
        # Make the Yes button red too
        yes_button = msg_box.button(QMessageBox.Yes)
        yes_button.setText("⚠️ YES, TRIM VIDEO")
        
        result = msg_box.exec()
        
        if result == QMessageBox.Yes:
            # Emit signal to main window to handle the actual trimming
            self.trim_video_requested.emit(start_frame, end_frame)
    
    def _update_color_square(self, color_name: str):
        """Update the color square to match the selected overlay color."""
        color_map = {
            "Red": "#FF0000",
            "Green": "#00FF00",
            "Blue": "#0000FF",
            "Yellow": "#FFFF00",
            "Cyan": "#00FFFF",
            "Magenta": "#FF00FF",
            "White": "#FFFFFF"
        }
        color_hex = color_map.get(color_name, "#FF0000")
        self.color_square.setStyleSheet(f"background-color: {color_hex}; border: 1px solid black;")
    
    def _toggle_spark_roi_visibility(self):
        """Toggle spark ROI overlay visibility."""
        is_visible = not self.spark_roi_toggle_button.isChecked()
        self.spark_roi_visibility_toggled.emit(is_visible)
        # Update button text based on state
        if is_visible:
            self.spark_roi_toggle_button.setText("Hide Spark Overlays")
        else:
            self.spark_roi_toggle_button.setText("Show Spark Overlays")
    
    def _set_fps_override(self, fps, emit_signal=True):
        """Set FPS override and update button states."""
        # Update button states
        self.fps_30_button.setChecked(fps == 30)
        self.fps_60_button.setChecked(fps == 60)
        self.fps_auto_button.setChecked(fps is None)
        
        # Update display
        if fps is None:
            self.fps_display_label.setText("Current FPS: Auto-detected")
        else:
            self.fps_display_label.setText(f"Current FPS: {fps} (override)")
        
        # Emit signal to update app state (only when user clicks, not when updating from state)
        if emit_signal:
            self.fps_override_changed.emit(fps)
    
    
    def update_video_info(self, detected_fps: float):
        """Update video-related information displays.
        
        Args:
            detected_fps: The detected FPS from the video file
        """
        # Update FPS display to show detected FPS
        if hasattr(self, 'fps_display_label'):
            fps_override = self.app_state.video.fps_override if hasattr(self.app_state, 'video') else None
            if fps_override:
                self.fps_display_label.setText(f"Current FPS: {fps_override} (override, detected: {detected_fps:.2f})")
            else:
                self.fps_display_label.setText(f"Current FPS: {detected_fps:.2f} (auto-detected)")
    
    def update_controls_from_state(self):
        """Update all controls to match the current app state."""
        if not self.app_state:
            return
        
        try:
            # Update detection settings
            if hasattr(self.app_state, 'detection'):
                threshold_percent = int(self.app_state.detection.detection_threshold * 100)
                self.detection_threshold_slider.setValue(threshold_percent)
                self.detection_threshold_label.setText(f"{threshold_percent}%")
                
                self.histogram_detection_cb.setChecked(self.app_state.detection.use_histogram_detection)
                self.delta_detection_cb.setChecked(self.app_state.detection.use_delta_detection)
                
                # Update black key filter checkbox
                if hasattr(self.app_state.detection, 'winner_takes_black_enabled'):
                    self.black_key_filter_cb.setChecked(self.app_state.detection.winner_takes_black_enabled)
                
                # Update spark detection checkbox
                if hasattr(self.app_state.detection, 'spark_detection_enabled'):
                    self.spark_detection_cb.setChecked(self.app_state.detection.spark_detection_enabled)
                    # Update spark controls enabled state
                    self._update_spark_controls_state(self.app_state.detection.spark_detection_enabled)
                
                # Update spark sensitivity slider
                if hasattr(self.app_state.detection, 'spark_detection_sensitivity'):
                    sensitivity_percent = int(self.app_state.detection.spark_detection_sensitivity * 100)
                    self.spark_sensitivity_slider.setValue(sensitivity_percent)
                    self.spark_sensitivity_label.setText(f"{sensitivity_percent}%")
                
                # Update histogram threshold slider
                if hasattr(self.app_state.detection, 'hist_ratio_threshold'):
                    hist_thresh_percent = int(self.app_state.detection.hist_ratio_threshold * 100)
                    self.histogram_threshold_slider.setValue(hist_thresh_percent)
                    self.histogram_threshold_label.setText(f"{self.app_state.detection.hist_ratio_threshold:.2f}")
                    self.histogram_threshold_slider.setEnabled(self.app_state.detection.use_histogram_detection)
                
                # Update delta threshold sliders
                if hasattr(self.app_state.detection, 'rise_delta_threshold'):
                    rise_percent = int(self.app_state.detection.rise_delta_threshold * 100)
                    self.rise_delta_slider.setValue(rise_percent)
                    self.rise_delta_label.setText(f"{self.app_state.detection.rise_delta_threshold:.2f}")
                    
                if hasattr(self.app_state.detection, 'fall_delta_threshold'):
                    fall_percent = int(self.app_state.detection.fall_delta_threshold * 100)
                    self.fall_delta_slider.setValue(fall_percent)
                    self.fall_delta_label.setText(f"{self.app_state.detection.fall_delta_threshold:.2f}")
                
                # Enable/disable delta sliders based on delta detection state
                delta_enabled = self.app_state.detection.use_delta_detection
                self.rise_delta_slider.setEnabled(delta_enabled)
                self.fall_delta_slider.setEnabled(delta_enabled)
                
                # Update similarity ratio slider
                if hasattr(self.app_state.detection, 'similarity_ratio'):
                    sim_percent = int(self.app_state.detection.similarity_ratio * 100)
                    self.similarity_ratio_slider.setValue(sim_percent)
                    self.similarity_ratio_label.setText(f"{self.app_state.detection.similarity_ratio:.2f}")
                    self.similarity_ratio_slider.setEnabled(self.app_state.detection.winner_takes_black_enabled)
            
            # Update overlay settings
            if hasattr(self.app_state, 'ui'):
                # Find overlay color index
                colors = ["Red", "Green", "Blue", "Yellow", "Cyan", "Magenta", "White"]
                if hasattr(self.app_state.ui, 'overlay_color'):
                    color_name = self.app_state.ui.overlay_color.title()
                    if color_name in colors:
                        self.overlay_color_combo.setCurrentText(color_name)
            
            # Update video trim settings
            if hasattr(self.app_state, 'video'):
                if hasattr(self.app_state.video, 'start_frame'):
                    self.start_frame_spin.setValue(self.app_state.video.start_frame)
                if hasattr(self.app_state.video, 'end_frame'):
                    self.end_frame_spin.setValue(self.app_state.video.end_frame)
                
                # Update processing frame settings
                if hasattr(self.app_state.video, 'processing_start_frame'):
                    self.processing_start_frame_spin.setValue(self.app_state.video.processing_start_frame)
                if hasattr(self.app_state.video, 'processing_end_frame'):
                    self.processing_end_frame_spin.setValue(self.app_state.video.processing_end_frame)
                
                # Update FPS override buttons and display
                if hasattr(self.app_state.video, 'fps_override'):
                    fps_override = self.app_state.video.fps_override
                    self._set_fps_override(fps_override, emit_signal=False)
            
            # Update MIDI settings
            if hasattr(self.app_state, 'midi'):
                if hasattr(self.app_state.midi, 'octave_transpose'):
                    self.octave_transpose_spin.setValue(self.app_state.midi.octave_transpose)
            
            # Update optional settings
            if hasattr(self.app_state, 'detection'):
                if hasattr(self.app_state.detection, 'hand_assignment_enabled'):
                    self.hand_assignment_cb.setChecked(self.app_state.detection.hand_assignment_enabled)
                
            
            # Update unlit calibration status
            if hasattr(self, 'unlit_status_label') and hasattr(self.app_state, 'overlays'):
                # Check if any overlay has unlit calibration
                has_unlit_calibration = any(
                    hasattr(overlay, 'unlit_reference_color') and overlay.unlit_reference_color is not None 
                    for overlay in self.app_state.overlays
                )
                
                if has_unlit_calibration:
                    self.unlit_status_label.setText("Unlit State Calibrated")
                    self.unlit_status_label.setStyleSheet("color: #4CAF50; font-style: italic;")
                else:
                    self.unlit_status_label.setText("Not Set")
                    self.unlit_status_label.setStyleSheet("color: #888; font-style: italic;")
            
            # Update exemplar color swatches
            if hasattr(self.app_state, 'detection') and hasattr(self.app_state.detection, 'exemplar_lit_colors'):
                for key_type, color_tuple in self.app_state.detection.exemplar_lit_colors.items():
                    if key_type in self.exemplar_swatches and color_tuple is not None:
                        # Convert RGB tuple to hex color
                        r, g, b = color_tuple
                        hex_color = f"#{r:02x}{g:02x}{b:02x}"
                        self.exemplar_swatches[key_type].setStyleSheet(f"border: 1px solid black; background-color: {hex_color};")
            
            # Update spark ROI visibility button state
            if hasattr(self.app_state, 'detection') and hasattr(self, 'spark_roi_toggle_button'):
                is_visible = self.app_state.detection.spark_roi_visible
                self.spark_roi_toggle_button.setChecked(not is_visible)
                if is_visible:
                    self.spark_roi_toggle_button.setText("Hide Spark Overlays")
                else:
                    self.spark_roi_toggle_button.setText("Show Spark Overlays")
            
            # Update auto calibration status indicators
            self._update_auto_calibration_status()
            
        except Exception as e:
            logging.warning(f"Error updating controls from state: {e}")
    
    def set_conversion_result(self, success: bool, message: str):
        """Update the conversion status."""
        self.convert_button.setText("Convert to MIDI")
        self.convert_button.setEnabled(True)
        
        if success:
            self.conversion_status.setText(f"Success: {message}")
        else:
            self.conversion_status.setText(f"Error: {message}")
    
    def update_selected_overlay(self, overlay_id: Optional[int]):
        """Update the selected overlay display."""
        if overlay_id is None:
            self.selected_overlay_label.setText("None")
        else:
            self.selected_overlay_label.setText(f"Overlay {overlay_id}")
    
    # ==================== Compatibility Methods ====================
    # These methods/properties are referenced by the main window code.
    
    def update_video_frame_limits(self):
        """Update the frame limit controls based on video total frames."""
        if not self.app_state or not hasattr(self.app_state, 'video'):
            return
        
        total_frames = getattr(self.app_state.video, 'total_frames', 0)
        if total_frames > 0:
            # Update start frame range (0 to total_frames - 1)
            self.start_frame_spin.setRange(0, total_frames - 1)
            
            # Update end frame range (-1 for "end of video", or 0 to total_frames - 1)
            self.end_frame_spin.setRange(-1, total_frames - 1)
            
            # Update processing frame ranges - constrain to trim range if trimmed
            if self.app_state.video.video_is_trimmed:
                min_frame = self.app_state.video.trim_start_frame
                max_frame = self.app_state.video.trim_end_frame
                self.processing_start_frame_spin.setRange(min_frame, max_frame)
                self.processing_end_frame_spin.setRange(min_frame, max_frame)
            else:
                self.processing_start_frame_spin.setRange(0, total_frames - 1)
                self.processing_end_frame_spin.setRange(0, total_frames - 1)
            
    
    def update_trim_controls_from_state(self):
        """Update video trim controls from app state."""
        if hasattr(self.app_state, 'video'):
            if hasattr(self.app_state.video, 'start_frame'):
                self.start_frame_spin.setValue(self.app_state.video.start_frame)
            if hasattr(self.app_state.video, 'end_frame'):
                self.end_frame_spin.setValue(self.app_state.video.end_frame)
    
    def _is_key_type_calibrated(self, key_type: str) -> bool:
        """Check if a specific key type is fully calibrated.
        
        Args:
            key_type: One of "LW", "LB", "RW", "RB"
            
        Returns:
            True if both bar_only and brightest_sparks calibrations exist
        """
        if not hasattr(self.app_state, 'detection'):
            return False
            
        detection_state = self.app_state.detection
        key_type_lower = key_type.lower()
        
        # Check both required calibrations exist (not None)
        bar_only_attr = f"spark_calibration_{key_type_lower}_bar_only"
        brightest_attr = f"spark_calibration_{key_type_lower}_brightest_sparks"
        
        bar_only_cal = getattr(detection_state, bar_only_attr, None)
        brightest_cal = getattr(detection_state, brightest_attr, None)
        
        return bar_only_cal is not None and brightest_cal is not None
    
    def _update_auto_calibration_status(self):
        """Update the status labels for all auto calibration buttons."""
        if not hasattr(self, 'auto_calib_status_labels'):
            return
            
        for key_type in ["LW", "LB", "RW", "RB"]:
            if key_type in self.auto_calib_status_labels:
                is_calibrated = self._is_key_type_calibrated(key_type)
                label = self.auto_calib_status_labels[key_type]
                
                if is_calibrated:
                    label.setText("Calibrated")
                    label.setStyleSheet("color: green; font-style: italic; font-size: 12px;")
                else:
                    label.setText("Not Set")
                    label.setStyleSheet("color: grey; font-style: italic;")

    def update_selected_overlay_display(self):
        """Update selected overlay display (compatibility wrapper)."""
        # Get the selected overlay ID from app state
        selected_id = None
        if hasattr(self.app_state, 'ui') and hasattr(self.app_state.ui, 'selected_overlay_id'):
            selected_id = self.app_state.ui.selected_overlay_id
        self.update_selected_overlay(selected_id)
    
    def update_advanced_calibration_display(self):
        """Update advanced calibration display (compatibility no-op)."""
        # This control panel does not require additional handling for this update.
        pass
    
    def update_spark_calibration_display(self):
        """Update spark calibration display (compatibility wrapper)."""
        # Update auto calibration status indicators
        self._update_auto_calibration_status()
    
    def update_shadow_calibration_display(self):
        """Update shadow calibration display (compatibility no-op)."""
        # This control panel does not require additional handling for this update.
        pass
    
    # Compatibility properties that the main window expects to exist
    @property
    def wizard_button(self):
        """Compatibility property: return the calibration wizard button."""
        return self.calibration_wizard_button
    
    @property
    def video_to_frames_button(self):
        """Compatibility property: placeholder for a video-to-frames button."""
        # This control panel does not provide a dedicated button; return a small mock.
        class MockButton:
            def setEnabled(self, enabled): pass
            def setText(self, text): pass
        return MockButton()
    
    @property
    def detection_threshold_spin(self):
        """Compatibility property: adapt the detection threshold slider to a spinbox-like API."""
        class SliderAsSpinBox:
            def __init__(self, slider):
                self._slider = slider
            def setValue(self, value):
                # Convert 0.0-1.0 range to 0-100 for slider
                self._slider.setValue(int(value * 100))
        return SliderAsSpinBox(self.detection_threshold_slider)
    
    # Properties for canvas and video controls (set by main.py)
    
    @property
    def canvas_refresh_callback(self):
        """Canvas refresh callback property."""
        return getattr(self, '_canvas_refresh_callback', None)
    
    @canvas_refresh_callback.setter
    def canvas_refresh_callback(self, callback):
        """Set canvas refresh callback."""
        self._canvas_refresh_callback = callback
    
    @property
    def video_controls(self):
        """Video controls property."""
        return getattr(self, '_video_controls', None)
    
    @video_controls.setter
    def video_controls(self, controls):
        """Set video controls reference."""
        self._video_controls = controls
    
    @property
    def keyboard_canvas(self):
        """Keyboard canvas property."""
        return getattr(self, '_keyboard_canvas', None)
    
    @keyboard_canvas.setter
    def keyboard_canvas(self, canvas):
        """Set keyboard canvas reference."""
        self._keyboard_canvas = canvas
