"""
Tab-based control panel UI for Synthesia2MIDI.

This module defines the right-hand settings pane and emits signals that the main
window connects to workflows/state updates.
"""

import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QScrollArea, QSizePolicy,
    QSlider, QSpinBox, QStackedWidget, QToolButton, QVBoxLayout, QWidget
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
    exemplar_key_type_enabled_changed = Signal(str, bool)  # key_type, enabled
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
    manual_fit_requested = Signal()
    
    # ==================== MIDI Processing Signals ====================
    octave_transpose_changed = Signal(int)
    processing_start_frame_changed = Signal(int)
    processing_end_frame_changed = Signal(int)
    fps_override_changed = Signal(object)  # float or None
    
    # ==================== Main Action Signals ====================
    conversion_requested = Signal()
    midi_touchup_requested = Signal()
    trim_video_requested = Signal(int, int)  # start_frame, end_frame

    DEFAULT_DETECTION_THRESHOLD = 50
    DEFAULT_HISTOGRAM_THRESHOLD = 80
    DEFAULT_RISE_DELTA_THRESHOLD = 15
    DEFAULT_FALL_DELTA_THRESHOLD = 5
    DEFAULT_SIMILARITY_RATIO = 60
    
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

        self._create_global_action_widgets()
        
        settings_layout = QHBoxLayout()
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(8)

        self.settings_section_rail_container = QWidget()
        self.settings_section_rail_container.setObjectName("settings_section_rail_container")
        self.settings_section_rail_container.setFixedWidth(98)
        self.settings_section_rail_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        settings_rail_layout = QVBoxLayout(self.settings_section_rail_container)
        settings_rail_layout.setContentsMargins(0, 0, 0, 0)
        settings_rail_layout.setSpacing(8)

        self.settings_section_rail = QListWidget()
        self.settings_section_rail.setObjectName("settings_section_rail")
        self.settings_section_rail.setFixedWidth(98)
        self.settings_section_rail.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.settings_section_rail.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.settings_section_rail.currentRowChanged.connect(self._set_settings_section)

        self.tab_widget = QStackedWidget()
        self.tab_widget.setObjectName("settings_section_stack")
        self.tab_widget.setMaximumWidth(760)  # Wide enough for readable settings
        self.tab_widget.currentChanged.connect(self.settings_section_rail.setCurrentRow)
        
        # Create all tabs
        self._create_mandatory_calibration_tab()
        self._create_overlay_settings_tab()
        self._create_basic_detection_tab()
        self._create_spark_detection_tab()
        self._create_midi_settings_tab()
        self._create_video_trim_tab()
        self._create_optional_settings_tab()

        self._fit_settings_section_rail_to_items()
        settings_rail_layout.addWidget(self.settings_section_rail)
        settings_rail_layout.addStretch(1)
        self._create_settings_rail_actions(settings_rail_layout)
        
        settings_layout.addWidget(self.settings_section_rail_container)
        settings_layout.addWidget(self.tab_widget, 1)
        main_layout.addLayout(settings_layout, 1)

        self.settings_section_rail.setCurrentRow(0)

    def _add_settings_section(self, widget: QWidget, label: str) -> None:
        item = QListWidgetItem(label)
        item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.settings_section_rail.addItem(item)
        self.tab_widget.addWidget(widget)

    def _set_settings_section(self, index: int) -> None:
        if index >= 0:
            self.tab_widget.setCurrentIndex(index)

    def _fit_settings_section_rail_to_items(self) -> None:
        if self.settings_section_rail.count() == 0:
            return

        row_height = self.settings_section_rail.sizeHintForRow(0)
        if row_height <= 0:
            row_height = 30
        frame = self.settings_section_rail.frameWidth() * 2
        self.settings_section_rail.setFixedHeight(
            (row_height * self.settings_section_rail.count()) + frame + 4
        )
    
    def _create_global_action_widgets(self):
        """Create section-independent actions shown in the lower rail."""
        self.convert_button = QPushButton("Convert")
        self.convert_button.setObjectName("convert_button")
        self.convert_button.clicked.connect(self._handle_conversion_request)
        self.convert_button.setMinimumHeight(34)

        self.conversion_status = QLabel("Ready to convert")
        self.conversion_status.setWordWrap(True)

        self.midi_touchup_button = QPushButton("Edit MIDI")
        self.midi_touchup_button.setObjectName("midi_touchup_button")
        self.midi_touchup_button.setMinimumHeight(34)
        self.midi_touchup_button.clicked.connect(self.midi_touchup_requested.emit)

        self.selected_overlay_caption = QLabel("Overlay")
        self.selected_overlay_label = QLabel("None")
        self.selected_overlay_label.setWordWrap(True)

    def _create_settings_rail_actions(self, parent_layout):
        self.settings_rail_actions = QWidget()
        self.settings_rail_actions.setObjectName("settings_rail_actions")
        self.settings_rail_actions.setFixedWidth(self.settings_section_rail.width())
        self.settings_rail_actions.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        actions_layout = QVBoxLayout(self.settings_rail_actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(6)
        actions_layout.addWidget(self.convert_button)
        actions_layout.addWidget(self.conversion_status)
        actions_layout.addSpacing(4)
        actions_layout.addWidget(self.midi_touchup_button)
        actions_layout.addSpacing(4)
        overlay_row = QHBoxLayout()
        overlay_row.setContentsMargins(0, 0, 0, 0)
        overlay_row.setSpacing(4)
        overlay_row.addWidget(self.selected_overlay_caption)
        overlay_row.addWidget(self.selected_overlay_label)
        overlay_row.addStretch()
        actions_layout.addLayout(overlay_row)

        parent_layout.addWidget(self.settings_rail_actions, alignment=Qt.AlignBottom)
    
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
            "If a key type is not present in this video, uncheck its 'Present in Video' box.",
            "Octave Transpose: shifts the generated MIDI up/down by octaves."
        ]
        for line in help_lines:
            label = QLabel(line)
            label.setWordWrap(True)
            label.setStyleSheet("font-size: 9pt;")  # slightly smaller help text
            help_layout.addWidget(label)
        layout.addWidget(help_section)
        
        # Key Overlay Calibration - compact grid. Avoid fixed-width rows that
        # overlap when the settings pane is at its 300px minimum.
        calibration_grid = QGridLayout()
        calibration_grid.setHorizontalSpacing(8)
        calibration_grid.setVerticalSpacing(8)
        calibration_grid.setColumnStretch(1, 1)

        overlay_label = QLabel("Overlays")
        overlay_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        calibration_grid.addWidget(overlay_label, 0, 0)
        
        self.calibration_wizard_button = QPushButton("Calibrate")
        self.calibration_wizard_button.setMinimumWidth(88)
        # Center-aligned text (default for QPushButton)
        self.calibration_wizard_button.clicked.connect(self.calibration_wizard_requested.emit)
        self.calibration_wizard_button.setToolTip(
            "Creates overlays for the keyboard in your video. Re-run if overlays don't line up."
        )
        calibration_grid.addWidget(self.calibration_wizard_button, 0, 1)
        
        # Octave transpose control
        octave_label = QLabel("Octave")
        octave_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        calibration_grid.addWidget(octave_label, 1, 0)
        self.octave_transpose_spin = QSpinBox()
        self.octave_transpose_spin.setRange(-5, 5)
        self.octave_transpose_spin.setValue(0)
        self.octave_transpose_spin.setFixedWidth(64)
        self.octave_transpose_spin.valueChanged.connect(self.octave_transpose_changed.emit)
        self.octave_transpose_spin.setToolTip("Shifts the MIDI output up/down by octaves.")
        calibration_grid.addWidget(self.octave_transpose_spin, 1, 1)
        
        # Unlit key calibration with status
        unlit_label = QLabel("Unlit")
        unlit_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        calibration_grid.addWidget(unlit_label, 2, 0)
        
        self.calibrate_unlit_button = QPushButton("Calibrate")
        self.calibrate_unlit_button.setMinimumWidth(88)
        # Center-aligned text (default for QPushButton)
        self.calibrate_unlit_button.clicked.connect(self.calibrate_unlit_requested.emit)
        self.calibrate_unlit_button.setToolTip(
            "Captures what unpressed overlays look like from the current frame. "
            "Pause on a frame with no highlighted notes first."
        )
        unlit_value_layout = QVBoxLayout()
        unlit_value_layout.setContentsMargins(0, 0, 0, 0)
        unlit_value_layout.setSpacing(3)
        unlit_value_layout.addWidget(self.calibrate_unlit_button)

        self.unlit_status_label = QLabel("Not Set")
        self.unlit_status_label.setStyleSheet("font-style: italic; color: #888;")
        unlit_value_layout.addWidget(self.unlit_status_label)
        calibration_grid.addLayout(unlit_value_layout, 2, 1)
        
        layout.addLayout(calibration_grid)
        layout.addSpacing(10)  # Extra space before next section
        
        # Lit exemplar calibration - plain text label
        exemplar_label = QLabel("Lit Key Exemplars")
        exemplar_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(exemplar_label)
        
        # Create vertical layout for exemplar buttons
        exemplar_container = QVBoxLayout()
        exemplar_container.setSpacing(10)
        
        self.exemplar_buttons = {}
        self.exemplar_swatches = {}
        self.exemplar_presence_checkboxes = {}
        
        # Single column order: LW, LB, RW, RB
        for key_type, label in [("LW", "Left White"), ("LB", "Left Black"), ("RW", "Right White"), ("RB", "Right Black")]:
            button = QPushButton(f"Set {label}")
            button.setMinimumWidth(110)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
            presence_cb = QCheckBox("Present")
            presence_cb.setChecked(True)
            presence_cb.setToolTip("Uncheck if this key type never appears in this video.")
            presence_cb.toggled.connect(
                lambda checked, kt=key_type: self._handle_exemplar_key_type_presence_toggled(kt, checked)
            )
            self.exemplar_presence_checkboxes[key_type] = presence_cb
            
            # Add to right column
            button_layout = QHBoxLayout()
            button_layout.setContentsMargins(0, 0, 0, 0)
            button_layout.addWidget(button, 1)
            button_layout.addSpacing(4)
            button_layout.addWidget(color_swatch)
            button_layout.addSpacing(4)
            button_layout.addWidget(presence_cb)
            button_layout.addStretch()
            exemplar_container.addLayout(button_layout)
        
        layout.addLayout(exemplar_container)
        
        layout.addStretch()  # Push everything to the top
        
        self._add_settings_section(tab, "Calibration")
    
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
        self.manual_fit_button = QPushButton("Manual Fit")
        self.manual_fit_button.setMaximumWidth(432)
        self.manual_fit_button.clicked.connect(self.manual_fit_requested.emit)
        
        # Place buttons side by side
        button_row = QHBoxLayout()
        button_row.addWidget(self.align_white_button)
        button_row.addSpacing(15)  # Move align black keys button to the right
        button_row.addWidget(self.align_black_button)
        button_row.addStretch()
        align_layout.addLayout(button_row)
        fit_row = QHBoxLayout()
        fit_row.addWidget(self.manual_fit_button)
        fit_row.addStretch()
        align_layout.addLayout(fit_row)
        alignment_layout.addLayout(align_layout)
        
        layout.addWidget(alignment_group)
        
        # Overlay size adjustment
        size_group = QGroupBox("Overlay Size Adjustment")
        size_layout = QVBoxLayout(size_group)

        size_grid = QGridLayout()
        size_grid.setHorizontalSpacing(14)
        size_grid.setVerticalSpacing(10)

        def add_size_control(row, column, label, dec_button, inc_button):
            cell = QVBoxLayout()
            cell.setContentsMargins(0, 0, 0, 0)
            cell.setSpacing(4)
            cell.addWidget(label)
            button_row = QHBoxLayout()
            button_row.setContentsMargins(0, 0, 0, 0)
            button_row.setSpacing(4)
            button_row.addWidget(dec_button)
            button_row.addWidget(inc_button)
            button_row.addStretch()
            cell.addLayout(button_row)
            size_grid.addLayout(cell, row, column)

        self.white_height_label = QLabel("White Key Height")
        self.white_height_dec_button = QPushButton("-")
        self.white_height_dec_button.setFixedSize(30, 30)
        self.white_height_dec_button.clicked.connect(lambda: self.overlay_size_adjustment_requested.emit("white", "height", -2))
        self.white_height_inc_button = QPushButton("+")
        self.white_height_inc_button.setFixedSize(30, 30)
        self.white_height_inc_button.clicked.connect(lambda: self.overlay_size_adjustment_requested.emit("white", "height", 2))
        add_size_control(0, 0, self.white_height_label, self.white_height_dec_button, self.white_height_inc_button)

        self.white_width_label = QLabel("White Key Width")
        self.white_width_dec_button = QPushButton("-")
        self.white_width_dec_button.setFixedSize(30, 30)
        self.white_width_dec_button.clicked.connect(lambda: self.overlay_size_adjustment_requested.emit("white", "width", -2))
        self.white_width_inc_button = QPushButton("+")
        self.white_width_inc_button.setFixedSize(30, 30)
        self.white_width_inc_button.clicked.connect(lambda: self.overlay_size_adjustment_requested.emit("white", "width", 2))
        add_size_control(0, 1, self.white_width_label, self.white_width_dec_button, self.white_width_inc_button)

        self.black_height_label = QLabel("Black Key Height")
        self.black_height_dec_button = QPushButton("-")
        self.black_height_dec_button.setFixedSize(30, 30)
        self.black_height_dec_button.clicked.connect(lambda: self.overlay_size_adjustment_requested.emit("black", "height", -2))
        self.black_height_inc_button = QPushButton("+")
        self.black_height_inc_button.setFixedSize(30, 30)
        self.black_height_inc_button.clicked.connect(lambda: self.overlay_size_adjustment_requested.emit("black", "height", 2))
        add_size_control(1, 0, self.black_height_label, self.black_height_dec_button, self.black_height_inc_button)

        self.black_width_label = QLabel("Black Key Width")
        self.black_width_dec_button = QPushButton("-")
        self.black_width_dec_button.setFixedSize(30, 30)
        self.black_width_dec_button.clicked.connect(lambda: self.overlay_size_adjustment_requested.emit("black", "width", -2))
        self.black_width_inc_button = QPushButton("+")
        self.black_width_inc_button.setFixedSize(30, 30)
        self.black_width_inc_button.clicked.connect(lambda: self.overlay_size_adjustment_requested.emit("black", "width", 2))
        add_size_control(1, 1, self.black_width_label, self.black_width_dec_button, self.black_width_inc_button)

        self.left_slant_label = QLabel("Left Slant")
        self.left_slant_dec_button = QPushButton("-")
        self.left_slant_dec_button.setFixedSize(30, 30)
        self.left_slant_dec_button.clicked.connect(lambda: self.overlay_size_adjustment_requested.emit("all", "left_slant", -1))
        self.left_slant_inc_button = QPushButton("+")
        self.left_slant_inc_button.setFixedSize(30, 30)
        self.left_slant_inc_button.clicked.connect(lambda: self.overlay_size_adjustment_requested.emit("all", "left_slant", 1))
        add_size_control(2, 0, self.left_slant_label, self.left_slant_dec_button, self.left_slant_inc_button)

        self.right_slant_label = QLabel("Right Slant")
        self.right_slant_dec_button = QPushButton("-")
        self.right_slant_dec_button.setFixedSize(30, 30)
        self.right_slant_dec_button.clicked.connect(lambda: self.overlay_size_adjustment_requested.emit("all", "right_slant", -1))
        self.right_slant_inc_button = QPushButton("+")
        self.right_slant_inc_button.setFixedSize(30, 30)
        self.right_slant_inc_button.clicked.connect(lambda: self.overlay_size_adjustment_requested.emit("all", "right_slant", 1))
        add_size_control(2, 1, self.right_slant_label, self.right_slant_dec_button, self.right_slant_inc_button)

        size_layout.addLayout(size_grid)

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
        self._add_settings_section(tab, "Overlays")
    
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
        self.detection_threshold_slider.setValue(self.DEFAULT_DETECTION_THRESHOLD)
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
        
        slider_label_width = 62

        def add_slider_row(label_text, slider, value_label, *, indent=0):
            row = QHBoxLayout()
            row.setContentsMargins(indent, 0, 0, 0)
            row.setSpacing(6)
            label = QLabel(label_text)
            label.setFixedWidth(max(1, slider_label_width - indent))
            row.addWidget(label)
            row.addWidget(slider)
            row.addWidget(value_label)
            row.addStretch()
            modes_layout.addLayout(row)

        # Histogram detection with sensitivity slider
        self.histogram_detection_cb = QCheckBox("Enable Histogram Detection")
        self.histogram_detection_cb.toggled.connect(self.histogram_detection_toggled.emit)
        self.histogram_detection_cb.toggled.connect(self._update_histogram_slider_state)
        self.histogram_detection_cb.setToolTip(
            "Uses a color-pattern match inside the overlay. Helpful with gradients/uneven lighting."
        )
        modes_layout.addWidget(self.histogram_detection_cb)
        
        # Add histogram threshold slider
        self.histogram_threshold_slider = QSlider(Qt.Horizontal)
        self.histogram_threshold_slider.setFixedWidth(110)
        self.histogram_threshold_slider.setRange(10, 100)  # 0.1 to 1.0
        self.histogram_threshold_slider.setValue(self.DEFAULT_HISTOGRAM_THRESHOLD)  # Default 0.8
        self.histogram_threshold_slider.valueChanged.connect(self._handle_histogram_threshold_change)
        self.histogram_threshold_slider.setEnabled(False)  # Initially disabled
        self.histogram_threshold_slider.setToolTip(
            "How strong the histogram match must be (only used when Histogram Detection is enabled)."
        )
        self.histogram_threshold_label = QLabel("0.80")
        self.histogram_threshold_label.setMinimumWidth(40)
        self.histogram_threshold_label.setToolTip(
            "How strong the histogram match must be (only used when Histogram Detection is enabled)."
        )
        add_slider_row("Strength:", self.histogram_threshold_slider, self.histogram_threshold_label)
        
        # Delta detection with rise/fall sliders
        self.delta_detection_cb = QCheckBox("Enable Delta Detection")
        self.delta_detection_cb.toggled.connect(self.delta_detection_toggled.emit)
        self.delta_detection_cb.toggled.connect(self._update_delta_sliders_state)
        self.delta_detection_cb.setToolTip(
            "Uses frame-to-frame change to confirm press/release (helps when color fades)."
        )
        modes_layout.addWidget(self.delta_detection_cb)
        
        self.rise_delta_slider = QSlider(Qt.Horizontal)
        self.rise_delta_slider.setFixedWidth(110)
        self.rise_delta_slider.setRange(1, 50)  # 0.01 to 0.50
        self.rise_delta_slider.setValue(self.DEFAULT_RISE_DELTA_THRESHOLD)  # Default 0.15
        self.rise_delta_slider.valueChanged.connect(self._handle_rise_delta_change)
        self.rise_delta_slider.setEnabled(False)  # Initially disabled
        self.rise_delta_slider.setToolTip(
            "How big the change must be to count as a press (only used when Delta Detection is enabled)."
        )
        self.rise_delta_label = QLabel("0.15")
        self.rise_delta_label.setMinimumWidth(40)
        self.rise_delta_label.setToolTip(
            "How big the change must be to count as a press (only used when Delta Detection is enabled)."
        )
        add_slider_row("Rise:", self.rise_delta_slider, self.rise_delta_label, indent=16)
        
        self.fall_delta_slider = QSlider(Qt.Horizontal)
        self.fall_delta_slider.setFixedWidth(110)
        self.fall_delta_slider.setRange(1, 50)  # 0.01 to 0.50
        self.fall_delta_slider.setValue(self.DEFAULT_FALL_DELTA_THRESHOLD)  # Default 0.05
        self.fall_delta_slider.valueChanged.connect(self._handle_fall_delta_change)
        self.fall_delta_slider.setEnabled(False)  # Initially disabled
        self.fall_delta_slider.setToolTip(
            "How big the change must be to count as a release (only used when Delta Detection is enabled)."
        )
        self.fall_delta_label = QLabel("0.05")
        self.fall_delta_label.setMinimumWidth(40)
        self.fall_delta_label.setToolTip(
            "How big the change must be to count as a release (only used when Delta Detection is enabled)."
        )
        add_slider_row("Fall:", self.fall_delta_slider, self.fall_delta_label, indent=16)
        
        # Black key filter with similarity ratio slider
        self.black_key_filter_cb = QCheckBox("Enable Black Key Filter")
        self.black_key_filter_cb.toggled.connect(self.winner_takes_black_changed.emit)
        self.black_key_filter_cb.toggled.connect(self._update_similarity_slider_state)
        self.black_key_filter_cb.setToolTip(
            "Reduces false black-key presses from nearby overlays."
        )
        modes_layout.addWidget(self.black_key_filter_cb)
        
        # Add similarity ratio slider
        self.similarity_ratio_slider = QSlider(Qt.Horizontal)
        self.similarity_ratio_slider.setFixedWidth(110)
        self.similarity_ratio_slider.setRange(10, 100)  # 0.1 to 1.0
        self.similarity_ratio_slider.setValue(self.DEFAULT_SIMILARITY_RATIO)  # Default 0.6
        self.similarity_ratio_slider.valueChanged.connect(self._handle_similarity_ratio_change)
        self.similarity_ratio_slider.setEnabled(False)  # Initially disabled
        self.similarity_ratio_slider.setToolTip(
            "Controls how strict black-key filtering is (only used when Black Key Filter is enabled)."
        )
        self.similarity_ratio_label = QLabel("0.60")
        self.similarity_ratio_label.setMinimumWidth(40)
        self.similarity_ratio_label.setToolTip(
            "Controls how strict black-key filtering is (only used when Black Key Filter is enabled)."
        )
        add_slider_row("Similarity:", self.similarity_ratio_slider, self.similarity_ratio_label)
        
        layout.addWidget(modes_group)

        self.restore_detection_defaults_button = QPushButton("Restore Defaults")
        self.restore_detection_defaults_button.setToolTip(
            "Reset detection threshold and detection mode parameter sliders to their defaults. "
            "Detection mode checkboxes stay unchanged."
        )
        self.restore_detection_defaults_button.clicked.connect(self._restore_detection_defaults)
        layout.addWidget(self.restore_detection_defaults_button, alignment=Qt.AlignLeft)
        
        layout.addStretch()
        self._add_settings_section(tab, "Detection")
    
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
        roi_layout = QVBoxLayout()
        roi_layout.setContentsMargins(0, 0, 0, 0)
        roi_layout.setSpacing(6)
        self.spark_roi_select_button = QPushButton("Select Spark ROI")
        self.spark_roi_select_button.setMaximumWidth(264)
        self.spark_roi_select_button.clicked.connect(self.spark_roi_selection_requested.emit)
        self.spark_roi_select_button.setToolTip(
            "Select the region above the keys where spark bars and sparks appear."
        )
        roi_layout.addWidget(self.spark_roi_select_button)

        # Add toggle button for showing/hiding spark overlays
        self.spark_roi_toggle_button = QPushButton("Hide Spark Overlays")
        self.spark_roi_toggle_button.setMaximumWidth(264)
        self.spark_roi_toggle_button.setCheckable(True)
        self.spark_roi_toggle_button.clicked.connect(self._toggle_spark_roi_visibility)
        self.spark_roi_toggle_button.setToolTip(
            "Show or hide the spark ROI overlay on the video."
        )
        roi_layout.addWidget(self.spark_roi_toggle_button)

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

        # Create one vertical auto-calibration row per key type so the Spark tab
        # fits in the settings pane without horizontal scrolling.
        for key_type in ["LW", "LB", "RW", "RB"]:
            row = QVBoxLayout()
            row.setSpacing(5)
            row.setContentsMargins(0, 0, 0, 0)

            button = QPushButton(f"Auto {KEY_TYPE_LABELS[key_type]}")
            button.setMaximumWidth(210)
            button.clicked.connect(lambda checked=False, kt=key_type: self.auto_spark_calibration_requested.emit(kt))
            button.setToolTip(
                "Recommended: auto-calibrate spark detection for this key type. "
                "Navigate to the frame where a key first turns ON, then click that overlay."
            )
            self.auto_calib_buttons[key_type] = button
            row.addWidget(button)

            status = QLabel("Not Set")
            status.setStyleSheet("color: grey; font-style: italic;")
            self.auto_calib_status_labels[key_type] = status
            row.addWidget(status)

            auto_layout.addLayout(row)

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
        self._add_settings_section(tab, "Spark")
    
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
        self._add_settings_section(tab, "MIDI")
    
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
        self._add_settings_section(tab, "Trim")
    
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
        self._add_settings_section(tab, "Optional")
    
    def _handle_conversion_request(self):
        """Handle conversion button click."""
        self.convert_button.setText("Converting...")
        self.convert_button.setEnabled(False)
        self.conversion_status.setText("Converting video to MIDI...")
        self.conversion_requested.emit()

    def _handle_exemplar_key_type_presence_toggled(self, key_type: str, checked: bool):
        """Handle per-key-type exemplar availability toggle."""
        self._update_exemplar_key_type_ui_state(key_type)
        self.exemplar_key_type_enabled_changed.emit(key_type, checked)

    def _update_exemplar_key_type_ui_state(self, key_type: str):
        """Update button and swatch styling for one exemplar key type."""
        button = self.exemplar_buttons.get(key_type)
        swatch = self.exemplar_swatches.get(key_type)
        checkbox = self.exemplar_presence_checkboxes.get(key_type)
        if button is None or swatch is None or checkbox is None:
            return

        is_enabled = checkbox.isChecked()
        button.setEnabled(is_enabled)

        color_tuple = None
        if hasattr(self.app_state, "detection") and hasattr(self.app_state.detection, "exemplar_lit_colors"):
            color_tuple = self.app_state.detection.exemplar_lit_colors.get(key_type)

        if not is_enabled:
            swatch.setStyleSheet("border: 1px dashed #888; background-color: #d0d0d0;")
            return

        if color_tuple is None:
            swatch.setStyleSheet("border: 1px solid #666; background-color: #9e9e9e;")
            return

        r, g, b = color_tuple
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        swatch.setStyleSheet(f"border: 1px solid black; background-color: {hex_color};")
    
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

    def _restore_detection_defaults(self):
        """Reset detection parameter sliders without changing mode toggles."""
        self.detection_threshold_slider.setValue(self.DEFAULT_DETECTION_THRESHOLD)
        self.histogram_threshold_slider.setValue(self.DEFAULT_HISTOGRAM_THRESHOLD)
        self.rise_delta_slider.setValue(self.DEFAULT_RISE_DELTA_THRESHOLD)
        self.fall_delta_slider.setValue(self.DEFAULT_FALL_DELTA_THRESHOLD)
        self.similarity_ratio_slider.setValue(self.DEFAULT_SIMILARITY_RATIO)
    
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
            
            # Update exemplar availability controls and swatches
            if hasattr(self.app_state, 'detection') and hasattr(self.app_state.detection, 'exemplar_lit_colors'):
                enabled_map = getattr(self.app_state.detection, 'exemplar_key_type_enabled', {})
                for key_type in KEY_TYPES:
                    checkbox = self.exemplar_presence_checkboxes.get(key_type)
                    if checkbox:
                        old_block_state = checkbox.blockSignals(True)
                        checkbox.setChecked(enabled_map.get(key_type, True))
                        checkbox.blockSignals(old_block_state)
                    self._update_exemplar_key_type_ui_state(key_type)
            
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

            # Update convert button availability based on prerequisites
            if hasattr(self, "convert_button"):
                self.convert_button.setEnabled(self._can_convert())

        except Exception as e:
            logging.warning(f"Error updating controls from state: {e}")
    
    def set_conversion_result(self, success: bool, message: str):
        """Update the conversion status."""
        self.convert_button.setText("Convert")
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
            self.selected_overlay_label.setText(str(overlay_id))
    
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

    def _can_convert(self) -> bool:
        """Return True if MIDI conversion prerequisites are satisfied."""
        if not self.app_state or not hasattr(self.app_state, 'video'):
            return False

        if not getattr(self.app_state.video, 'filepath', None):
            return False

        overlays = getattr(self.app_state, 'overlays', None) or []
        if not overlays:
            return False

        missing_unlit = [
            overlay.key_id
            for overlay in overlays
            if getattr(overlay, 'unlit_reference_color', None) is None
        ]
        if missing_unlit:
            return False

        if getattr(self.app_state.detection, 'use_histogram_detection', False):
            missing_hist = [
                overlay.key_id
                for overlay in overlays
                if getattr(overlay, 'unlit_hist', None) is None
            ]
            if missing_hist:
                return False

        required_exemplars = self.app_state.detection.get_required_base_exemplar_types()
        if not required_exemplars:
            return False

        exemplar_colors = self.app_state.detection.get_effective_exemplar_lit_colors()
        for exemplar in required_exemplars:
            if exemplar_colors.get(exemplar) is None:
                return False

        detection_threshold = getattr(self.app_state.detection, 'detection_threshold', 0.0)
        if not 0.1 <= detection_threshold <= 0.99:
            return False

        if getattr(self.app_state.midi, 'tempo', 0) <= 0:
            return False

        return True
    
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
