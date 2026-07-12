"""
Tab-based control panel UI for Synthesia2MIDI.

This module defines the right-hand settings pane and emits signals that the main
window connects to workflows/state updates.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QCoreApplication, QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractScrollArea, QCheckBox, QComboBox, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QScrollArea, QSizePolicy,
    QSlider, QSpinBox, QStackedWidget, QToolButton, QVBoxLayout, QWidget
)

from synthesia2midi.core.app_state import AppState
from synthesia2midi.core.color_families import active_family_numbers, exemplar_display_parts
from synthesia2midi.gui.calibration_guide import CalibrationGuideWidget, derive_guide_snapshot
from synthesia2midi.gui.color_family_grid import ColorFamilyGrid
from synthesia2midi.gui.repeated_notes_tool_window import RepeatedNotesToolWindow
from synthesia2midi.gui.ui_glossary import UiGlossary
from synthesia2midi.localization import (
    load_preferred_locale,
    locale_display_name,
    save_preferred_locale,
    supported_user_locales,
)

# Spark auto-calibration currently exposes the two legacy persisted families.
KEY_TYPES = ["LW", "LB", "RW", "RB"]

translate = QCoreApplication.translate


def _exemplar_display_label(slot: str) -> str:
    family_number, morphology = exemplar_display_parts(slot)
    family_label = translate("ControlPanelQt", "Color {number}").format(
        number=family_number
    )
    morphology_label = (
        translate("ControlPanelQt", "Natural")
        if morphology == "Natural"
        else translate("ControlPanelQt", "Sharp / Flat")
    )
    return f"{family_label} {morphology_label}"


@dataclass(frozen=True)
class ConversionReadiness:
    can_convert: bool
    status_text: str


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
    remove_additional_color_requested = Signal(int)  # family number
    
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
    OVERLAY_ADJUSTMENT_INDETERMINATE = "--"
    
    def __init__(self, parent=None, app_state: AppState = None, state_manager=None, settings=None):
        super().__init__(parent)
        self.app_state = app_state or AppState()
        self.state_manager = state_manager
        self.settings = settings or QSettings("Synthesia2MIDI", "Synthesia2MIDI")
        
        # Widget references for state updates
        self.widgets = {}
        self._overlay_adjustment_values: dict[tuple[str, str], int | None] = {}
        self._overlay_adjustment_value_labels: dict[tuple[str, str], QLabel] = {}
        self._overlay_adjustment_reset_buttons: dict[tuple[str, str], QPushButton] = {}
        self._overlay_adjustment_pending_previous_values: dict[tuple[str, str], int | None] = {}
        self._overlay_adjustment_basis: tuple[tuple[int, float, float, float, float, float], ...] | None = None
        
        self._setup_ui()
        self.update_controls_from_state()
    
    def _setup_ui(self):
        """Create the main UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 2, 10, 10)  # Reduced top margin from 10 to 2
        main_layout.setSpacing(5)  # Reduce spacing between elements

        self._create_global_action_widgets()
        self.settings_page_widgets: list[QWidget] = []
        self.settings_page_scroll_areas: list[QScrollArea] = []
        
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
        self.settings_section_rail.setWordWrap(True)
        self.settings_section_rail.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.settings_section_rail.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.settings_section_rail.currentRowChanged.connect(self._set_settings_section)

        self.tab_widget = QStackedWidget()
        self.tab_widget.setObjectName("settings_section_stack")
        self.tab_widget.setMaximumWidth(760)  # Wide enough for readable settings
        self.tab_widget.currentChanged.connect(self.settings_section_rail.setCurrentRow)
        
        # Create all tabs
        self._create_guide_tab()
        self._create_mandatory_calibration_tab()
        self._create_overlay_settings_tab()
        self._create_basic_detection_tab()
        self._create_spark_detection_tab()
        self._create_midi_settings_tab()
        self._create_video_trim_tab()
        self._create_advanced_settings_tab()
        self._create_optional_settings_tab()
        self._create_language_settings_tab()

        self._fit_settings_section_rail_to_items()
        settings_rail_layout.addWidget(self.settings_section_rail)
        settings_rail_layout.addStretch(1)

        self.settings_content_container = QWidget()
        settings_content_layout = QVBoxLayout(self.settings_content_container)
        settings_content_layout.setContentsMargins(0, 0, 0, 0)
        settings_content_layout.setSpacing(8)
        settings_content_layout.addWidget(self.tab_widget, 1)
        self._create_settings_footer(settings_content_layout)

        settings_layout.addWidget(self.settings_section_rail_container)
        settings_layout.addWidget(self.settings_content_container, 1)
        main_layout.addLayout(settings_layout, 1)

        self.settings_section_rail.setCurrentRow(0)
        QWidget.setTabOrder(
            self.settings_section_rail,
            self.guide_page.step_rows[0].primary_button,
        )
        guide_buttons = [row.primary_button for row in self.guide_page.step_rows]
        QWidget.setTabOrder(guide_buttons[0], self.guide_page.youtube_button)
        QWidget.setTabOrder(self.guide_page.youtube_button, guide_buttons[1])
        for current_button, next_button in zip(guide_buttons[1:], guide_buttons[2:]):
            QWidget.setTabOrder(current_button, next_button)
        QWidget.setTabOrder(guide_buttons[-1], self.convert_button)
        QWidget.setTabOrder(self.convert_button, self.midi_touchup_button)

    def _create_guide_tab(self) -> None:
        self.guide_page = CalibrationGuideWidget()
        self._add_settings_section(self.guide_page, self.tr("Guide"))

    def _add_settings_section(self, widget: QWidget, label: str) -> None:
        item = QListWidgetItem(label)
        item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.settings_section_rail.addItem(item)
        scroll_area = QScrollArea()
        scroll_area.setObjectName(f"settings_page_scroll_{len(self.settings_page_scroll_areas)}")
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        scroll_area.setWidget(widget)
        self.settings_page_widgets.append(widget)
        self.settings_page_scroll_areas.append(scroll_area)
        self.tab_widget.addWidget(scroll_area)

    def _set_settings_section(self, index: int) -> None:
        if index >= 0:
            self.tab_widget.setCurrentIndex(index)

    def _fit_settings_section_rail_to_items(self) -> None:
        if self.settings_section_rail.count() == 0:
            return

        metrics = self.settings_section_rail.fontMetrics()
        widest_text = max(
            metrics.horizontalAdvance(self.settings_section_rail.item(index).text())
            for index in range(self.settings_section_rail.count())
        )
        rail_width = min(144, max(98, widest_text + 28))
        self.settings_section_rail.setFixedWidth(rail_width)
        self.settings_section_rail_container.setFixedWidth(rail_width)
        frame = self.settings_section_rail.frameWidth() * 2
        self.settings_section_rail.setFixedHeight(
            sum(
                max(30, self.settings_section_rail.sizeHintForRow(index))
                for index in range(self.settings_section_rail.count())
            )
            + frame
            + 4
        )

    def fit_settings_section_rail(self) -> None:
        """Recalculate rail geometry after a font or translation change."""
        self._fit_settings_section_rail_to_items()
    
    def _create_global_action_widgets(self):
        """Create section-independent actions shown in the lower rail."""
        self.convert_button = QPushButton(QCoreApplication.translate("ControlPanelQt", "Convert"))
        self.convert_button.setObjectName("convert_button")
        self.convert_button.clicked.connect(self._handle_conversion_request)
        self.convert_button.setMinimumHeight(36)

        self.conversion_status = QLabel(
            QCoreApplication.translate("ControlPanelQt", "Load a video to convert.")
        )
        self.conversion_status.setWordWrap(True)

        self.midi_touchup_button = QPushButton(QCoreApplication.translate("ControlPanelQt", "Edit MIDI"))
        self.midi_touchup_button.setObjectName("midi_touchup_button")
        self.midi_touchup_button.setMinimumHeight(36)
        self.midi_touchup_button.clicked.connect(self.midi_touchup_requested.emit)

        self.selected_overlay_caption = QLabel(QCoreApplication.translate("ControlPanelQt", "Overlay"))
        self.selected_overlay_label = QLabel(QCoreApplication.translate("ControlPanelQt", "None"))
        self.selected_overlay_label.setWordWrap(True)

    def _create_settings_footer(self, parent_layout: QVBoxLayout) -> None:
        self.settings_footer = QWidget()
        self.settings_footer.setObjectName("settings_footer")
        self.settings_footer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.settings_footer.setStyleSheet(
            "QWidget#settings_footer { border-top: 1px solid #b8b8b8; }"
        )

        footer_layout = QGridLayout(self.settings_footer)
        footer_layout.setContentsMargins(0, 8, 0, 0)
        footer_layout.setHorizontalSpacing(8)
        footer_layout.setVerticalSpacing(6)
        footer_layout.addWidget(self.conversion_status, 0, 0)
        overlay_row = QHBoxLayout()
        overlay_row.setContentsMargins(0, 0, 0, 0)
        overlay_row.setSpacing(4)
        overlay_row.addWidget(self.selected_overlay_caption)
        overlay_row.addWidget(self.selected_overlay_label)
        overlay_row.addStretch()
        footer_layout.addLayout(overlay_row, 1, 0)
        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(8)
        action_row.addStretch()
        action_row.addWidget(self.midi_touchup_button)
        action_row.addWidget(self.convert_button)
        footer_layout.addLayout(action_row, 2, 0)
        footer_layout.setColumnStretch(0, 1)
        parent_layout.addWidget(self.settings_footer)
        self.settings_footer.setMinimumHeight(self.settings_footer.sizeHint().height())

    def _create_language_settings_tab(self):
        """Language settings shown as a first-class settings section."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        language_group = QGroupBox(translate("ControlPanelQt", "Language"))
        language_group.setObjectName("first_in_tab")
        language_layout = QGridLayout(language_group)
        language_layout.addWidget(QLabel(translate("ControlPanelQt", "Language:")), 0, 0)

        self.language_combo = QComboBox()
        self.language_combo.setObjectName("language_combo")
        current_locale = load_preferred_locale(self.settings)
        self.language_combo.blockSignals(True)
        for locale_name in supported_user_locales():
            self.language_combo.addItem(locale_display_name(locale_name), locale_name)
        selected_index = self.language_combo.findData(current_locale)
        if selected_index >= 0:
            self.language_combo.setCurrentIndex(selected_index)
        self.language_combo.blockSignals(False)
        self.language_combo.currentIndexChanged.connect(self._handle_language_changed)
        language_layout.addWidget(self.language_combo, 0, 1)

        layout.addWidget(language_group)
        layout.addStretch()
        self._add_settings_section(tab, translate("ControlPanelQt", "Language"))
    
    def _create_mandatory_calibration_tab(self):
        """Tab 1: Mandatory Calibration"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)  # Add some padding
        layout.setSpacing(15)  # Space between sections

        help_section = CollapsibleSection(translate("ControlPanelQt", "Help"), expanded=False)
        help_layout = help_section.content_layout()
        help_lines = [
            translate("ControlPanelQt", "Initial calibration directions (recommended order):"),
            translate(
                "ControlPanelQt",
                "1) Find Keyboard Box: create overlays that line up with the keyboard in your video.",
            ),
            translate(
                "ControlPanelQt",
                "2) Capture No-Key Frame: pause where no keys are glowing, then click Capture No-Key Frame.",
            ),
            translate(
                "ControlPanelQt",
                "3) Capture Pressed-Key Examples: for each Color family, capture a Natural and Sharp / Flat example that appears in the video.",
            ),
            translate("ControlPanelQt", "If a key type is not present in this video, uncheck its 'Present in Video' box."),
            translate("ControlPanelQt", "Octave Transpose: shifts the generated MIDI up/down by octaves."),
        ]
        for line in help_lines:
            label = QLabel(line)
            label.setWordWrap(True)
            label.setStyleSheet("font-size: 9pt;")  # slightly smaller help text
            help_layout.addWidget(label)
        layout.addWidget(help_section)
        
        self.calibration_instruction_labels = {}

        def add_instruction_row(row_key: str, title: str, instruction: str, action_widget: QWidget) -> None:
            row_widget = QWidget()
            row_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row = QVBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)

            title_label = QLabel(title)
            title_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
            row.addWidget(title_label)

            instruction_label = QLabel(instruction)
            instruction_label.setWordWrap(True)
            instruction_label.setStyleSheet("color: #555;")
            self.calibration_instruction_labels[row_key] = instruction_label
            row.addWidget(instruction_label)
            row.addWidget(action_widget)

            layout.addWidget(row_widget)

        self.calibration_wizard_button = QPushButton(
            translate("ControlPanelQt", "Draw Keyboard Box and Find Keys")
        )
        self.calibration_wizard_button.setMinimumWidth(
            max(180, self.calibration_wizard_button.sizeHint().width())
        )
        self.calibration_wizard_button.setMinimumHeight(
            max(36, self.calibration_wizard_button.sizeHint().height())
        )
        self.calibration_wizard_button.clicked.connect(self.calibration_wizard_requested.emit)
        self.calibration_wizard_button.setToolTip(
            translate(
                "ControlPanelQt",
                "Creates overlays for the keyboard in your video. Re-run if overlays don't line up.",
            )
        )
        add_instruction_row(
            "keyboard",
            translate("ControlPanelQt", "Find the keyboard"),
            translate("ControlPanelQt", "Pause on a clear frame where the full keyboard is visible."),
            self.calibration_wizard_button,
        )

        octave_grid = QGridLayout()
        octave_grid.setHorizontalSpacing(8)
        octave_label = QLabel(translate("ControlPanelQt", "Octave"))
        octave_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        octave_grid.addWidget(octave_label, 0, 0)
        self.octave_transpose_spin = QSpinBox()
        self.octave_transpose_spin.setRange(-5, 5)
        self.octave_transpose_spin.setValue(0)
        self.octave_transpose_spin.setFixedWidth(64)
        self.octave_transpose_spin.valueChanged.connect(self.octave_transpose_changed.emit)
        self.octave_transpose_spin.setToolTip(
            translate("ControlPanelQt", "Shifts the MIDI output up/down by octaves.")
        )
        octave_grid.addWidget(self.octave_transpose_spin, 0, 1)
        octave_grid.setColumnStretch(2, 1)
        layout.addLayout(octave_grid)

        self.calibrate_unlit_button = QPushButton(
            translate("ControlPanelQt", "Capture No-Key Frame")
        )
        self.calibrate_unlit_button.setMinimumWidth(180)
        self.calibrate_unlit_button.setMinimumHeight(36)
        self.calibrate_unlit_button.clicked.connect(self.calibrate_unlit_requested.emit)
        self.calibrate_unlit_button.setToolTip(
            translate(
                "ControlPanelQt",
                "Captures what unpressed overlays look like from the current frame. Pause on a frame with no highlighted notes first.",
            )
        )

        unlit_title = QLabel(translate("ControlPanelQt", "Capture no-key frame"))
        unlit_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(unlit_title)

        unlit_instruction = QLabel(translate("ControlPanelQt", "Pause where no keys are glowing."))
        unlit_instruction.setWordWrap(True)
        unlit_instruction.setStyleSheet("color: #555;")
        self.calibration_instruction_labels["unlit"] = unlit_instruction
        layout.addWidget(unlit_instruction)

        unlit_stack = QVBoxLayout()
        unlit_stack.setContentsMargins(0, 0, 0, 0)
        unlit_stack.setSpacing(0)
        unlit_stack.addWidget(self.calibrate_unlit_button)
        unlit_stack.addSpacing(8)

        self.unlit_status_label = QLabel(translate("ControlPanelQt", "Not Set"))
        self.unlit_status_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.unlit_status_label.setFixedHeight(20)
        self.unlit_status_label.setStyleSheet("font-style: italic; color: #595959;")
        unlit_stack.addWidget(self.unlit_status_label)
        layout.addLayout(unlit_stack)

        pressed_title = QLabel(translate("ControlPanelQt", "Capture pressed-key examples"))
        pressed_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(pressed_title)

        pressed_instruction = QLabel(
            translate("ControlPanelQt", "Pause where a key is glowing, then click that key.")
        )
        pressed_instruction.setWordWrap(True)
        pressed_instruction.setStyleSheet("color: #555;")
        self.calibration_instruction_labels["pressed"] = pressed_instruction
        layout.addWidget(pressed_instruction)

        layout.addSpacing(10)  # Extra space before next section
        
        # Lit exemplar calibration - plain text label
        exemplar_label = QLabel(translate("ControlPanelQt", "Lit Key Exemplars"))
        exemplar_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(exemplar_label)
        
        self.color_family_grid = ColorFamilyGrid(mode="calibration")
        self.color_family_grid.exemplar_requested.connect(
            self.calibrate_lit_exemplar_requested.emit
        )
        self.color_family_grid.exemplar_enabled_changed.connect(
            self._handle_exemplar_key_type_presence_toggled
        )
        self.color_family_grid.family_add_requested.connect(
            self.add_additional_color_requested.emit
        )
        self.color_family_grid.family_remove_requested.connect(
            self.remove_additional_color_requested.emit
        )
        self.exemplar_buttons = {}
        self.exemplar_swatches = {}
        self.exemplar_presence_checkboxes = {}
        self._refresh_color_family_grid()
        layout.addWidget(self.color_family_grid)
        
        layout.addStretch()  # Push everything to the top
        
        self._add_settings_section(tab, translate("ControlPanelQt", "Calibration"))
    
    def _create_overlay_settings_tab(self):
        """Tab 2: Overlay Settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)  # Make grey containers flush with tab header
        layout.setSpacing(5)  # Minimal spacing between grey containers
        
        # Key alignment
        alignment_group = QGroupBox(translate("ControlPanelQt", "Key Alignment"))
        alignment_group.setObjectName("first_in_tab")  # For CSS styling
        alignment_layout = QVBoxLayout(alignment_group)
        
        align_layout = QVBoxLayout()
        align_layout.setSpacing(6)
        self.align_white_button = QPushButton(translate("ControlPanelQt", "Align White Keys"))
        self.align_white_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.align_white_button.clicked.connect(self.align_white_keys_requested.emit)
        self.align_black_button = QPushButton(translate("ControlPanelQt", "Align Black Keys"))
        self.align_black_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.align_black_button.clicked.connect(self.align_black_keys_requested.emit)
        self.manual_fit_button = QPushButton(translate("ControlPanelQt", "Manual Fit"))
        self.manual_fit_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.manual_fit_button.clicked.connect(self.manual_fit_requested.emit)

        align_layout.addWidget(self.align_white_button)
        align_layout.addWidget(self.align_black_button)
        align_layout.addWidget(self.manual_fit_button)
        alignment_layout.addLayout(align_layout)
        
        layout.addWidget(alignment_group)
        
        # Overlay size adjustment
        size_group = QGroupBox(translate("ControlPanelQt", "Overlay Size Adjustment"))
        size_layout = QVBoxLayout(size_group)

        size_grid = QGridLayout()
        size_grid.setHorizontalSpacing(14)
        size_grid.setVerticalSpacing(10)

        def add_size_control(row, column, label, dec_button, inc_button, key_color, dimension):
            key = self._overlay_adjustment_key(key_color, dimension)
            cell = QVBoxLayout()
            cell.setContentsMargins(0, 0, 0, 0)
            cell.setSpacing(4)
            cell.addWidget(label)

            value_row = QHBoxLayout()
            value_row.setContentsMargins(0, 0, 0, 0)
            value_row.setSpacing(6)
            value_caption = QLabel(translate("ControlPanelQt", "Current:"))
            value_label = QLabel("0")
            value_label.setMinimumWidth(24)
            reset_button = QPushButton(translate("ControlPanelQt", "Reset"))
            reset_button.clicked.connect(
                lambda checked=False, kc=key_color, dim=dimension: self._reset_overlay_adjustment(kc, dim)
            )
            value_row.addWidget(value_caption)
            value_row.addWidget(value_label)
            value_row.addWidget(reset_button)
            value_row.addStretch()
            cell.addLayout(value_row)

            button_row = QHBoxLayout()
            button_row.setContentsMargins(0, 0, 0, 0)
            button_row.setSpacing(4)
            button_row.addWidget(dec_button)
            button_row.addWidget(inc_button)
            decrease_text = translate("ControlPanelQt", "Decrease {setting}").format(
                setting=label.text()
            )
            increase_text = translate("ControlPanelQt", "Increase {setting}").format(
                setting=label.text()
            )
            dec_button.setAccessibleName(decrease_text)
            dec_button.setAccessibleDescription(decrease_text)
            inc_button.setAccessibleName(increase_text)
            inc_button.setAccessibleDescription(increase_text)
            button_row.addStretch()
            cell.addLayout(button_row)
            self._overlay_adjustment_value_labels[key] = value_label
            self._overlay_adjustment_reset_buttons[key] = reset_button
            self._set_overlay_adjustment_value(key_color, dimension, 0)
            size_grid.addLayout(cell, row, column)
            return value_label, reset_button

        self.white_height_label = QLabel(translate("ControlPanelQt", "White Key Height"))
        self.white_height_dec_button = QPushButton("-")
        self.white_height_dec_button.setFixedSize(36, 36)
        self.white_height_dec_button.clicked.connect(lambda: self._apply_overlay_adjustment("white", "height", -2))
        self.white_height_inc_button = QPushButton("+")
        self.white_height_inc_button.setFixedSize(36, 36)
        self.white_height_inc_button.clicked.connect(lambda: self._apply_overlay_adjustment("white", "height", 2))
        self.white_height_value_label, self.white_height_reset_button = add_size_control(
            0,
            0,
            self.white_height_label,
            self.white_height_dec_button,
            self.white_height_inc_button,
            "white",
            "height",
        )

        self.white_width_label = QLabel(translate("ControlPanelQt", "White Key Width"))
        self.white_width_dec_button = QPushButton("-")
        self.white_width_dec_button.setFixedSize(36, 36)
        self.white_width_dec_button.clicked.connect(lambda: self._apply_overlay_adjustment("white", "width", -2))
        self.white_width_inc_button = QPushButton("+")
        self.white_width_inc_button.setFixedSize(36, 36)
        self.white_width_inc_button.clicked.connect(lambda: self._apply_overlay_adjustment("white", "width", 2))
        self.white_width_value_label, self.white_width_reset_button = add_size_control(
            0,
            1,
            self.white_width_label,
            self.white_width_dec_button,
            self.white_width_inc_button,
            "white",
            "width",
        )

        self.black_height_label = QLabel(translate("ControlPanelQt", "Black Key Height"))
        self.black_height_dec_button = QPushButton("-")
        self.black_height_dec_button.setFixedSize(36, 36)
        self.black_height_dec_button.clicked.connect(lambda: self._apply_overlay_adjustment("black", "height", -2))
        self.black_height_inc_button = QPushButton("+")
        self.black_height_inc_button.setFixedSize(36, 36)
        self.black_height_inc_button.clicked.connect(lambda: self._apply_overlay_adjustment("black", "height", 2))
        self.black_height_value_label, self.black_height_reset_button = add_size_control(
            1,
            0,
            self.black_height_label,
            self.black_height_dec_button,
            self.black_height_inc_button,
            "black",
            "height",
        )

        self.black_width_label = QLabel(translate("ControlPanelQt", "Black Key Width"))
        self.black_width_dec_button = QPushButton("-")
        self.black_width_dec_button.setFixedSize(36, 36)
        self.black_width_dec_button.clicked.connect(lambda: self._apply_overlay_adjustment("black", "width", -2))
        self.black_width_inc_button = QPushButton("+")
        self.black_width_inc_button.setFixedSize(36, 36)
        self.black_width_inc_button.clicked.connect(lambda: self._apply_overlay_adjustment("black", "width", 2))
        self.black_width_value_label, self.black_width_reset_button = add_size_control(
            1,
            1,
            self.black_width_label,
            self.black_width_dec_button,
            self.black_width_inc_button,
            "black",
            "width",
        )

        self.left_slant_label = QLabel(translate("ControlPanelQt", "Left Slant"))
        self.left_slant_dec_button = QPushButton("-")
        self.left_slant_dec_button.setFixedSize(36, 36)
        self.left_slant_dec_button.clicked.connect(lambda: self._apply_overlay_adjustment("all", "left_slant", -1))
        self.left_slant_inc_button = QPushButton("+")
        self.left_slant_inc_button.setFixedSize(36, 36)
        self.left_slant_inc_button.clicked.connect(lambda: self._apply_overlay_adjustment("all", "left_slant", 1))
        self.left_slant_value_label, self.left_slant_reset_button = add_size_control(
            2,
            0,
            self.left_slant_label,
            self.left_slant_dec_button,
            self.left_slant_inc_button,
            "all",
            "left_slant",
        )

        self.right_slant_label = QLabel(translate("ControlPanelQt", "Right Slant"))
        self.right_slant_dec_button = QPushButton("-")
        self.right_slant_dec_button.setFixedSize(36, 36)
        self.right_slant_dec_button.clicked.connect(lambda: self._apply_overlay_adjustment("all", "right_slant", -1))
        self.right_slant_inc_button = QPushButton("+")
        self.right_slant_inc_button.setFixedSize(36, 36)
        self.right_slant_inc_button.clicked.connect(lambda: self._apply_overlay_adjustment("all", "right_slant", 1))
        self.right_slant_value_label, self.right_slant_reset_button = add_size_control(
            2,
            1,
            self.right_slant_label,
            self.right_slant_dec_button,
            self.right_slant_inc_button,
            "all",
            "right_slant",
        )

        size_layout.addLayout(size_grid)

        layout.addWidget(size_group)
        
        # Overlay color
        color_group = QGroupBox(translate("ControlPanelQt", "Overlay Appearance"))
        color_layout = QVBoxLayout(color_group)
        
        # Horizontal layout for color dropdown and square
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel(translate("ControlPanelQt", "Overlay Color:")))
        
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
        self._add_settings_section(tab, translate("ControlPanelQt", "Overlays"))
    
    def _create_basic_detection_tab(self):
        """Tab 3: Basic Detection Settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)  # Make grey containers flush with tab header
        layout.setSpacing(5)  # Minimal spacing between grey containers

        help_section = CollapsibleSection(translate("ControlPanelQt", "Help"), expanded=False)
        help_section.setStyleSheet("QLabel { font-size: 9pt; }")  # shrink detection help text
        help_layout = help_section.content_layout()
        help_lines = [
            translate("ControlPanelQt", "Before tuning detection: capture a no-key frame and at least one pressed-key example."),
            translate("ControlPanelQt", "Detection Sensitivity: main setting for pressed vs unpressed keys."),
            translate("ControlPanelQt", "Missing notes? Lower it. Extra notes? Raise it."),
            translate(
                "ControlPanelQt",
                "Histogram Detection helps when pressed colors have gradients or uneven lighting.",
            ),
            translate(
                "ControlPanelQt",
                "Delta Detection helps when pressed colors fade in or out instead of switching cleanly.",
            ),
            translate("ControlPanelQt", "Black Key Filter reduces false black-key notes caused by nearby overlays."),
        ]
        for line in help_lines:
            label = QLabel(line)
            label.setWordWrap(True)
            help_layout.addWidget(label)
        layout.addWidget(help_section)
        
        # Detection threshold
        threshold_group = QGroupBox(translate("ControlPanelQt", "Detection Sensitivity"))
        threshold_group.setObjectName("first_in_tab")  # For CSS styling
        threshold_layout = QVBoxLayout(threshold_group)
        threshold_layout.setContentsMargins(15, 10, 15, 10)
        
        self.detection_threshold_slider = QSlider(Qt.Horizontal)
        self.detection_threshold_slider.setMaximumWidth(150)  # Half of default width
        self.detection_threshold_slider.setRange(0, 100)
        self.detection_threshold_slider.setValue(self.DEFAULT_DETECTION_THRESHOLD)
        self.detection_threshold_slider.valueChanged.connect(self._handle_detection_threshold_change)
        self.detection_threshold_slider.setToolTip(
            translate("ControlPanelQt", "Main sensitivity. Lower = detects more; higher = fewer false notes.")
        )
        
        self.detection_threshold_label = QLabel("50%")
        self.detection_threshold_label.setToolTip(
            translate("ControlPanelQt", "Main sensitivity. Lower = detects more; higher = fewer false notes.")
        )
        
        threshold_layout.addWidget(QLabel(translate("ControlPanelQt", "Detection Sensitivity:")))
        threshold_layout.addWidget(self.detection_threshold_slider)
        threshold_layout.addWidget(self.detection_threshold_label)
        threshold_layout.addWidget(QLabel(translate("ControlPanelQt", "Missing notes? Lower it. Extra notes? Raise it.")))
        
        layout.addWidget(threshold_group)
        
        # Specialist modes are composed into the collapsed Advanced page.
        self.histogram_advanced_widget = QWidget()
        histogram_layout = QVBoxLayout(self.histogram_advanced_widget)
        histogram_layout.setContentsMargins(0, 0, 0, 0)
        self.delta_advanced_widget = QWidget()
        delta_layout = QVBoxLayout(self.delta_advanced_widget)
        delta_layout.setContentsMargins(0, 0, 0, 0)
        self.black_key_advanced_widget = QWidget()
        black_key_layout = QVBoxLayout(self.black_key_advanced_widget)
        black_key_layout.setContentsMargins(0, 0, 0, 0)

        self.advanced_slider_labels = []
        advanced_slider_label_specs = []

        def add_slider_row(target_layout, label_text, slider, value_label, *, indent=0):
            row = QHBoxLayout()
            row.setContentsMargins(indent, 0, 0, 0)
            row.setSpacing(6)
            label = QLabel(label_text)
            self.advanced_slider_labels.append(label)
            advanced_slider_label_specs.append((label, indent))
            row.addWidget(label)
            row.addWidget(slider)
            row.addWidget(value_label)
            row.addStretch()
            target_layout.addLayout(row)

        # Histogram detection with sensitivity slider
        self.histogram_detection_cb = QCheckBox(translate("ControlPanelQt", "Enable Histogram Detection"))
        self.histogram_detection_cb.toggled.connect(self.histogram_detection_toggled.emit)
        self.histogram_detection_cb.toggled.connect(self._update_histogram_slider_state)
        self.histogram_detection_cb.setToolTip(
            translate("ControlPanelQt", "Uses a color-pattern match inside the overlay. Helpful with gradients/uneven lighting.")
        )
        histogram_layout.addWidget(self.histogram_detection_cb)
        
        # Add histogram threshold slider
        self.histogram_threshold_slider = QSlider(Qt.Horizontal)
        self.histogram_threshold_slider.setFixedWidth(110)
        self.histogram_threshold_slider.setRange(10, 100)  # 0.1 to 1.0
        self.histogram_threshold_slider.setValue(self.DEFAULT_HISTOGRAM_THRESHOLD)  # Default 0.8
        self.histogram_threshold_slider.valueChanged.connect(self._handle_histogram_threshold_change)
        self.histogram_threshold_slider.setEnabled(False)  # Initially disabled
        self.histogram_threshold_slider.setToolTip(
            translate("ControlPanelQt", "How strong the histogram match must be (only used when Histogram Detection is enabled).")
        )
        self.histogram_threshold_label = QLabel("0.80")
        self.histogram_threshold_label.setMinimumWidth(40)
        self.histogram_threshold_label.setToolTip(
            translate("ControlPanelQt", "How strong the histogram match must be (only used when Histogram Detection is enabled).")
        )
        add_slider_row(histogram_layout, translate("ControlPanelQt", "Strength:"), self.histogram_threshold_slider, self.histogram_threshold_label)
        
        # Delta detection with rise/fall sliders
        self.delta_detection_cb = QCheckBox(translate("ControlPanelQt", "Enable Delta Detection"))
        self.delta_detection_cb.toggled.connect(self.delta_detection_toggled.emit)
        self.delta_detection_cb.toggled.connect(self._update_delta_sliders_state)
        self.delta_detection_cb.setToolTip(
            translate("ControlPanelQt", "Uses frame-to-frame change to confirm press/release (helps when color fades).")
        )
        delta_layout.addWidget(self.delta_detection_cb)
        
        self.rise_delta_slider = QSlider(Qt.Horizontal)
        self.rise_delta_slider.setFixedWidth(110)
        self.rise_delta_slider.setRange(1, 50)  # 0.01 to 0.50
        self.rise_delta_slider.setValue(self.DEFAULT_RISE_DELTA_THRESHOLD)  # Default 0.15
        self.rise_delta_slider.valueChanged.connect(self._handle_rise_delta_change)
        self.rise_delta_slider.setEnabled(False)  # Initially disabled
        self.rise_delta_slider.setToolTip(
            translate("ControlPanelQt", "How big the change must be to count as a press (only used when Delta Detection is enabled).")
        )
        self.rise_delta_label = QLabel("0.15")
        self.rise_delta_label.setMinimumWidth(40)
        self.rise_delta_label.setToolTip(
            translate("ControlPanelQt", "How big the change must be to count as a press (only used when Delta Detection is enabled).")
        )
        add_slider_row(delta_layout, translate("ControlPanelQt", "Rise:"), self.rise_delta_slider, self.rise_delta_label, indent=16)
        
        self.fall_delta_slider = QSlider(Qt.Horizontal)
        self.fall_delta_slider.setFixedWidth(110)
        self.fall_delta_slider.setRange(1, 50)  # 0.01 to 0.50
        self.fall_delta_slider.setValue(self.DEFAULT_FALL_DELTA_THRESHOLD)  # Default 0.05
        self.fall_delta_slider.valueChanged.connect(self._handle_fall_delta_change)
        self.fall_delta_slider.setEnabled(False)  # Initially disabled
        self.fall_delta_slider.setToolTip(
            translate("ControlPanelQt", "How big the change must be to count as a release (only used when Delta Detection is enabled).")
        )
        self.fall_delta_label = QLabel("0.05")
        self.fall_delta_label.setMinimumWidth(40)
        self.fall_delta_label.setToolTip(
            translate("ControlPanelQt", "How big the change must be to count as a release (only used when Delta Detection is enabled).")
        )
        add_slider_row(delta_layout, translate("ControlPanelQt", "Fall:"), self.fall_delta_slider, self.fall_delta_label, indent=16)
        
        # Black key filter with similarity ratio slider
        self.black_key_filter_cb = QCheckBox(translate("ControlPanelQt", "Enable Black Key Filter"))
        self.black_key_filter_cb.toggled.connect(self.winner_takes_black_changed.emit)
        self.black_key_filter_cb.toggled.connect(self._update_similarity_slider_state)
        self.black_key_filter_cb.setToolTip(
            translate("ControlPanelQt", "Reduces false black-key presses from nearby overlays.")
        )
        black_key_layout.addWidget(self.black_key_filter_cb)
        
        # Add similarity ratio slider
        self.similarity_ratio_slider = QSlider(Qt.Horizontal)
        self.similarity_ratio_slider.setFixedWidth(110)
        self.similarity_ratio_slider.setRange(10, 100)  # 0.1 to 1.0
        self.similarity_ratio_slider.setValue(self.DEFAULT_SIMILARITY_RATIO)  # Default 0.6
        self.similarity_ratio_slider.valueChanged.connect(self._handle_similarity_ratio_change)
        self.similarity_ratio_slider.setEnabled(False)  # Initially disabled
        self.similarity_ratio_slider.setToolTip(
            translate("ControlPanelQt", "Controls how strict black-key filtering is (only used when Black Key Filter is enabled).")
        )
        self.similarity_ratio_label = QLabel("0.60")
        self.similarity_ratio_label.setMinimumWidth(40)
        self.similarity_ratio_label.setToolTip(
            translate("ControlPanelQt", "Controls how strict black-key filtering is (only used when Black Key Filter is enabled).")
        )
        add_slider_row(black_key_layout, translate("ControlPanelQt", "Similarity:"), self.similarity_ratio_slider, self.similarity_ratio_label)

        slider_label_width = max(
            62,
            *(label.sizeHint().width() + indent for label, indent in advanced_slider_label_specs),
        )
        for label, indent in advanced_slider_label_specs:
            label.setFixedWidth(max(1, slider_label_width - indent))

        self.restore_detection_defaults_button = QPushButton(translate("ControlPanelQt", "Restore Defaults"))
        self.restore_detection_defaults_button.setToolTip(
            translate(
                "ControlPanelQt",
                "Reset detection threshold and detection mode parameter sliders to their defaults. Detection mode checkboxes stay unchanged.",
            )
        )
        self.restore_detection_defaults_button.clicked.connect(self._restore_detection_defaults)
        layout.addStretch()
        self._add_settings_section(tab, translate("ControlPanelQt", "Detection"))
    
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

        help_section = CollapsibleSection(translate("ControlPanelQt", "Help"), expanded=False)
        help_layout = help_section.content_layout()
        help_lines = [
            translate("ControlPanelQt", "Use Spark Detection only when:"),
            translate("ControlPanelQt", "1) Key overlays stay ON across repeated notes (false continuous press)."),
            translate("ControlPanelQt", "2) Key overlays are solid color (no fading or gradients)."),
        ]
        for line in help_lines:
            label = QLabel(line)
            label.setWordWrap(True)
            label.setStyleSheet("font-size: 9pt;")  # slightly smaller help text
            help_layout.addWidget(label)
        layout.addWidget(help_section)

        # Spark detection toggle
        main_group = QGroupBox(translate("ControlPanelQt", "Repeated Notes Fix"))
        main_group.setObjectName("first_in_tab")  # For CSS styling
        main_layout = QVBoxLayout(main_group)

        spark_guidance_label = QLabel(
            translate("ControlPanelQt", "Use this only if repeated notes merge into one long note.")
        )
        spark_guidance_label.setWordWrap(True)
        spark_guidance_label.setStyleSheet("color: #555;")
        main_layout.addWidget(spark_guidance_label)

        self.spark_detection_cb = QCheckBox(translate("ControlPanelQt", "Enable Repeated Notes Fix"))
        self.spark_detection_cb.toggled.connect(self.spark_detection_toggled.emit)
        self.spark_detection_cb.toggled.connect(self._update_spark_controls_state)
        self.spark_detection_cb.setToolTip(
            translate(
                "ControlPanelQt",
                "Use only when key overlays stay ON across repeated notes (false continuous press), and the overlays are solid color (no fading or gradients).",
            )
        )
        main_layout.addWidget(self.spark_detection_cb)

        # Sensitivity
        main_layout.addWidget(QLabel(translate("ControlPanelQt", "Sensitivity:")))
        self.spark_sensitivity_slider = QSlider(Qt.Horizontal)
        self.spark_sensitivity_slider.setMaximumWidth(150)  # Half of default width
        self.spark_sensitivity_slider.setRange(0, 100)
        self.spark_sensitivity_slider.setValue(50)
        self.spark_sensitivity_slider.valueChanged.connect(self._handle_spark_sensitivity_change)
        self.spark_sensitivity_slider.setToolTip(
            translate("ControlPanelQt", "Controls how aggressively Spark Detection splits false continuous notes.")
        )
        main_layout.addWidget(self.spark_sensitivity_slider)

        self.spark_sensitivity_label = QLabel("50%")
        self.spark_sensitivity_label.setToolTip(
            translate("ControlPanelQt", "Controls how aggressively Spark Detection splits false continuous notes.")
        )
        main_layout.addWidget(self.spark_sensitivity_label)

        layout.addWidget(main_group)

        # Spark calibration
        calibration_group = QGroupBox(translate("ControlPanelQt", "Repeated-Note Setup"))
        calibration_layout = QVBoxLayout(calibration_group)

        # ROI selection
        roi_layout = QVBoxLayout()
        roi_layout.setContentsMargins(0, 0, 0, 0)
        roi_layout.setSpacing(6)
        self.spark_roi_select_button = QPushButton(translate("ControlPanelQt", "Select Flash Area Above Keys"))
        self.spark_roi_select_button.clicked.connect(self.spark_roi_selection_requested.emit)
        self.spark_roi_select_button.setToolTip(
            translate("ControlPanelQt", "Select the area above the keys where spark bars and flashes appear.")
        )
        roi_layout.addWidget(self.spark_roi_select_button)

        # Add toggle button for showing/hiding spark overlays
        self.spark_roi_toggle_button = QPushButton(translate("ControlPanelQt", "Hide Spark Overlays"))
        self.spark_roi_toggle_button.setCheckable(True)
        self.spark_roi_toggle_button.clicked.connect(self._toggle_spark_roi_visibility)
        self.spark_roi_toggle_button.setToolTip(
            translate("ControlPanelQt", "Show or hide the spark area overlay on the video.")
        )
        roi_layout.addWidget(self.spark_roi_toggle_button)

        calibration_layout.addLayout(roi_layout)

        manual_section = CollapsibleSection(translate("ControlPanelQt", "Manual Calibration"), expanded=False)
        calib_buttons_layout = manual_section.content_layout()

        # Step 1: Calibrate Background
        step1_layout = QHBoxLayout()
        step1_label = QLabel(translate("ControlPanelQt", "Step 1)"))
        step1_label.setFixedWidth(60)  # Fixed width for alignment
        step1_layout.addWidget(step1_label)
        self.spark_bg_button = QPushButton(translate("ControlPanelQt", "Calibrate Background"))
        self.spark_bg_button.setFixedWidth(300)  # Fixed width for exact alignment
        self.spark_bg_button.clicked.connect(lambda: self.spark_calibration_requested.emit("background"))
        self.spark_bg_button.setToolTip(
            translate("ControlPanelQt", "Manual calibration: capture baseline brightness when there are no bars or sparks.")
        )
        step1_layout.addWidget(self.spark_bg_button)
        step1_layout.addStretch()
        calib_buttons_layout.addLayout(step1_layout)

        # Step 2: Calibrate Bar Only
        step2_layout = QHBoxLayout()
        step2_label = QLabel(translate("ControlPanelQt", "Step 2)"))
        step2_label.setFixedWidth(60)  # Fixed width for alignment
        step2_layout.addWidget(step2_label)
        self.spark_bar_button = QPushButton(translate("ControlPanelQt", "Calibrate Bar Only"))
        self.spark_bar_button.setFixedWidth(300)  # Fixed width for exact alignment
        self.spark_bar_button.clicked.connect(lambda: self.spark_calibration_requested.emit("bar_only"))
        self.spark_bar_button.setToolTip(
            translate("ControlPanelQt", "Manual calibration: click an overlay showing colored bars with no sparks.")
        )
        step2_layout.addWidget(self.spark_bar_button)
        step2_layout.addStretch()
        calib_buttons_layout.addLayout(step2_layout)

        # Step 3: Calibrate Dimmest Sparks
        step3_layout = QHBoxLayout()
        step3_label = QLabel(translate("ControlPanelQt", "Step 3)"))
        step3_label.setFixedWidth(60)  # Fixed width for alignment
        step3_layout.addWidget(step3_label)
        self.spark_brightest_button = QPushButton(translate("ControlPanelQt", "Calibrate Dimmest Sparks"))
        self.spark_brightest_button.setFixedWidth(300)  # Fixed width for exact alignment
        self.spark_brightest_button.clicked.connect(lambda: self.spark_calibration_requested.emit("dimmest_sparks"))
        self.spark_brightest_button.setToolTip(
            translate("ControlPanelQt", "Manual calibration: click an overlay where sparks are just barely visible.")
        )
        step3_layout.addWidget(self.spark_brightest_button)
        step3_layout.addStretch()
        calib_buttons_layout.addLayout(step3_layout)

        calibration_layout.addWidget(manual_section)

        # Auto calibration
        calibration_layout.addWidget(QLabel(translate("ControlPanelQt", "Auto Calibration:")))
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

            button = QPushButton(
                translate("ControlPanelQt", "Auto {key_type_label}").format(
                    key_type_label=_exemplar_display_label(key_type)
                )
            )
            button.clicked.connect(lambda checked=False, kt=key_type: self.auto_spark_calibration_requested.emit(kt))
            button.setToolTip(
                translate(
                    "ControlPanelQt",
                    "Recommended: auto-calibrate spark detection for this key type. Navigate to the frame where a key first turns ON, then click that overlay.",
                )
            )
            self.auto_calib_buttons[key_type] = button
            row.addWidget(button)

            status = QLabel(translate("ControlPanelQt", "Not Set"))
            status.setStyleSheet("color: #595959; font-style: italic;")
            self.auto_calib_status_labels[key_type] = status
            row.addWidget(status)

            auto_layout.addLayout(row)

        calibration_layout.addLayout(auto_layout)

        layout.addWidget(calibration_group)

        # Preview / status
        preview_group = QGroupBox(translate("ControlPanelQt", "Spark Preview / Status"))
        preview_layout = QVBoxLayout(preview_group)
        self.spark_preview_label = QLabel(translate("ControlPanelQt", "Preview will show spark calibration status here."))
        self.spark_preview_label.setWordWrap(True)
        preview_layout.addWidget(self.spark_preview_label)

        self.spark_status_label = QLabel(translate("ControlPanelQt", "Preview not available yet."))
        self.spark_status_label.setWordWrap(True)
        self.spark_status_label.setStyleSheet("color: #595959; font-style: italic;")
        preview_layout.addWidget(self.spark_status_label)

        layout.addWidget(preview_group)

        layout.addStretch()

        scroll_area.setWidget(content)
        tab_layout.addWidget(scroll_area)
        self.spark_settings_page = tab
        self.repeated_notes_tool_window = RepeatedNotesToolWindow(
            self, self.spark_settings_page
        )
    
    def _create_midi_settings_tab(self):
        """Tab 5: MIDI Settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)  # Make grey containers flush with tab header
        layout.setSpacing(5)  # Minimal spacing between grey containers
        
        # FPS Override
        fps_group = QGroupBox(translate("ControlPanelQt", "Frame Rate Override"))
        fps_group.setObjectName("first_in_tab")  # For CSS styling
        fps_layout = QVBoxLayout(fps_group)
        
        fps_button_layout = QHBoxLayout()
        fps_button_layout.setSpacing(5)  # Minimal spacing between buttons
        
        self.fps_30_button = QPushButton("30 FPS")
        self.fps_30_button.setMinimumWidth(self.fps_30_button.sizeHint().width())
        self.fps_30_button.setCheckable(True)
        self.fps_30_button.clicked.connect(lambda: self._set_fps_override(30))
        
        self.fps_60_button = QPushButton("60 FPS")
        self.fps_60_button.setMinimumWidth(self.fps_60_button.sizeHint().width())
        self.fps_60_button.setCheckable(True)
        self.fps_60_button.clicked.connect(lambda: self._set_fps_override(60))
        
        self.fps_auto_button = QPushButton(translate("ControlPanelQt", "Auto"))
        self.fps_auto_button.setMinimumWidth(self.fps_auto_button.sizeHint().width())
        self.fps_auto_button.setCheckable(True)
        self.fps_auto_button.setChecked(True)
        self.fps_auto_button.clicked.connect(lambda: self._set_fps_override(None))
        
        fps_button_layout.addWidget(self.fps_30_button)
        fps_button_layout.addWidget(self.fps_60_button)
        fps_button_layout.addWidget(self.fps_auto_button)
        fps_button_layout.addStretch()  # Push buttons to the left
        
        fps_layout.addLayout(fps_button_layout)
        
        # Current FPS display
        self.fps_display_label = QLabel(translate("ControlPanelQt", "Current FPS: Auto-detected"))
        fps_layout.addWidget(self.fps_display_label)
        
        layout.addWidget(fps_group)
        
        # Custom MIDI Processing Range
        processing_range_group = QGroupBox(translate("ControlPanelQt", "Convert Only Part of the Video"))
        processing_range_layout = QVBoxLayout(processing_range_group)

        processing_hint = QLabel(
            translate(
                "ControlPanelQt",
                "This affects MIDI creation only. It does not trim or change the video session.",
            )
        )
        processing_hint.setWordWrap(True)
        processing_hint.setStyleSheet("color: #555;")
        processing_range_layout.addWidget(processing_hint)
        
        # Frame controls in grid for alignment
        processing_grid = QGridLayout()
        processing_grid.setHorizontalSpacing(25)  # Increased spacing to move buttons to the right
        processing_grid.setColumnStretch(3, 1)  # Push everything to the left
        
        # Processing start frame
        processing_start_label = QLabel(translate("ControlPanelQt", "Start Frame:"))
        processing_start_label.setFixedWidth(144)
        processing_start_label.setWordWrap(True)
        processing_grid.addWidget(processing_start_label, 0, 0)
        
        self.processing_start_frame_spin = QSpinBox()
        self.processing_start_frame_spin.setMaximumWidth(180)  # widened for readability
        self.processing_start_frame_spin.setRange(0, 999999)
        self.processing_start_frame_spin.setValue(0)
        self.processing_start_frame_spin.valueChanged.connect(self.processing_start_frame_changed.emit)
        processing_grid.addWidget(self.processing_start_frame_spin, 0, 1)
        
        self.processing_start_set_button = QPushButton(translate("ControlPanelQt", "Set to Current"))
        self.processing_start_set_button.setMinimumWidth(
            self.processing_start_set_button.sizeHint().width()
        )
        self.processing_start_set_button.clicked.connect(self._set_processing_start_to_current)
        processing_grid.addWidget(self.processing_start_set_button, 0, 2)
        
        # Processing end frame
        processing_end_label = QLabel(translate("ControlPanelQt", "End Frame:"))
        processing_end_label.setFixedWidth(144)
        processing_end_label.setWordWrap(True)
        processing_grid.addWidget(processing_end_label, 1, 0)
        
        self.processing_end_frame_spin = QSpinBox()
        self.processing_end_frame_spin.setMaximumWidth(180)  # widened for readability
        self.processing_end_frame_spin.setRange(0, 999999)
        self.processing_end_frame_spin.setValue(0)
        self.processing_end_frame_spin.valueChanged.connect(self.processing_end_frame_changed.emit)
        processing_grid.addWidget(self.processing_end_frame_spin, 1, 1)
        
        self.processing_end_set_button = QPushButton(translate("ControlPanelQt", "Set to Current"))
        self.processing_end_set_button.setMinimumWidth(
            self.processing_end_set_button.sizeHint().width()
        )
        self.processing_end_set_button.clicked.connect(self._set_processing_end_to_current)
        processing_grid.addWidget(self.processing_end_set_button, 1, 2)
        
        processing_range_layout.addLayout(processing_grid)
        layout.addWidget(processing_range_group)
        
        # Octave transpose is configured in the Calibration tab.
        
        layout.addStretch()
        self._add_settings_section(tab, translate("ControlPanelQt", "MIDI"))
    
    def _create_video_trim_tab(self):
        """Tab 6: Video Trim Settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)  # Make grey containers flush with tab header
        layout.setSpacing(5)  # Minimal spacing between grey containers
        
        trim_group = QGroupBox(translate("ControlPanelQt", "Permanently Trim Project"))
        trim_group.setObjectName("first_in_tab")  # For CSS styling
        trim_layout = QVBoxLayout(trim_group)

        trim_warning = QLabel(
            translate(
                "ControlPanelQt",
                "Most users should use MIDI range instead. Trim changes the working video session, not the original video file.",
            )
        )
        trim_warning.setWordWrap(True)
        trim_warning.setStyleSheet("color: #8a4b00; font-weight: 600;")
        trim_layout.addWidget(trim_warning)
        
        # Frame controls in grid for alignment
        frame_grid = QGridLayout()
        frame_grid.setHorizontalSpacing(10)
        frame_grid.setColumnStretch(3, 1)  # Push everything to the left
        
        # Start frame
        start_label = QLabel(translate("ControlPanelQt", "Start Frame:"))
        start_label.setWordWrap(True)
        start_label.setMinimumWidth(120)
        frame_grid.addWidget(start_label, 0, 0)
        
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setRange(0, 999999)
        self.start_frame_spin.setMinimumWidth(self.start_frame_spin.sizeHint().width())
        self.start_frame_spin.setValue(0)
        self.start_frame_spin.valueChanged.connect(self.start_frame_changed.emit)
        frame_grid.addWidget(self.start_frame_spin, 0, 1)
        
        self.trim_start_set_button = QPushButton(translate("ControlPanelQt", "Set to Current"))
        self.trim_start_set_button.setMinimumWidth(
            self.trim_start_set_button.sizeHint().width()
        )
        self.trim_start_set_button.clicked.connect(self._set_trim_start_to_current)
        frame_grid.addWidget(self.trim_start_set_button, 0, 2)
        
        # End frame
        end_label = QLabel(translate("ControlPanelQt", "End Frame:"))
        end_label.setWordWrap(True)
        end_label.setMinimumWidth(120)
        frame_grid.addWidget(end_label, 1, 0)
        
        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setRange(-1, 999999)
        self.end_frame_spin.setMinimumWidth(self.end_frame_spin.sizeHint().width())
        self.end_frame_spin.setValue(-1)
        self.end_frame_spin.valueChanged.connect(self.end_frame_changed.emit)
        frame_grid.addWidget(self.end_frame_spin, 1, 1)
        
        self.trim_end_set_button = QPushButton(translate("ControlPanelQt", "Set to Current"))
        self.trim_end_set_button.setMinimumWidth(
            self.trim_end_set_button.sizeHint().width()
        )
        self.trim_end_set_button.clicked.connect(self._set_trim_end_to_current)
        frame_grid.addWidget(self.trim_end_set_button, 1, 2)
        
        trim_layout.addLayout(frame_grid)
        
        # Trim Video button
        self.trim_video_button = QPushButton(translate("ControlPanelQt", "Permanently Trim Project"))
        self.trim_video_button.clicked.connect(self._handle_trim_video_request)
        trim_layout.addWidget(self.trim_video_button)
        
        layout.addWidget(trim_group)
        self.trim_settings_group = trim_group
        
        layout.addStretch()
        self.trim_settings_page = tab

    def _create_advanced_settings_tab(self):
        """Compose specialist recovery controls under symptom-led sections."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        repeated_notes_launcher = QWidget()
        repeated_notes_layout = QVBoxLayout(repeated_notes_launcher)
        repeated_notes_layout.setContentsMargins(0, 0, 0, 0)
        repeated_notes_layout.setSpacing(6)
        repeated_notes_summary = QLabel(
            translate(
                "ControlPanelQt",
                "Open the dedicated setup window when repeated presses merge into one long note.",
            )
        )
        repeated_notes_summary.setWordWrap(True)
        repeated_notes_layout.addWidget(repeated_notes_summary)
        self.open_repeated_notes_tool_button = QPushButton(
            translate("ControlPanelQt", "Open Repeated Notes Tool")
        )
        self.open_repeated_notes_tool_button.clicked.connect(
            self.repeated_notes_tool_window.show_near_parent
        )
        repeated_notes_layout.addWidget(
            self.open_repeated_notes_tool_button, alignment=Qt.AlignLeft
        )

        section_specs = [
            (
                "histogram",
                translate("ControlPanelQt", "Gradient or uneven pressed colors"),
                self.histogram_advanced_widget,
            ),
            (
                "delta",
                translate("ControlPanelQt", "Pressed colors fade in or out"),
                self.delta_advanced_widget,
            ),
            (
                "black_keys",
                translate("ControlPanelQt", "False black-key notes"),
                self.black_key_advanced_widget,
            ),
            (
                "repeated_notes",
                translate("ControlPanelQt", "Repeated notes merge together"),
                repeated_notes_launcher,
            ),
            (
                "trim",
                translate("ControlPanelQt", "Permanently Trim Project"),
                self.trim_settings_group,
            ),
        ]
        self.advanced_sections = {}
        for key, title, content in section_specs:
            section = CollapsibleSection(title, expanded=False)
            section.content_layout().addWidget(content)
            layout.addWidget(section)
            self.advanced_sections[key] = section

        self.advanced_sections["black_keys"].content_layout().addWidget(
            self.restore_detection_defaults_button,
            alignment=Qt.AlignLeft,
        )

        glossary_section = CollapsibleSection(
            translate("ControlPanelQt", "Glossary"), expanded=False
        )
        self.ui_glossary = UiGlossary()
        glossary_section.content_layout().addWidget(self.ui_glossary)
        layout.addWidget(glossary_section)
        self.advanced_sections["glossary"] = glossary_section
        layout.addStretch(1)
        self._add_settings_section(tab, translate("ControlPanelQt", "Advanced"))
    
    def _create_optional_settings_tab(self):
        """Tab 7: Optional Settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)  # Make grey containers flush with tab header
        layout.setSpacing(5)  # Minimal spacing between grey containers
        
        optional_group = QGroupBox(translate("ControlPanelQt", "Optional Features"))
        optional_layout = QVBoxLayout(optional_group)
        
        # Hand assignment
        self.hand_assignment_cb = QCheckBox(
            translate("ControlPanelQt", "Put each Color family on a separate MIDI channel")
        )
        self.hand_assignment_cb.toggled.connect(self.hand_assignment_toggled.emit)
        optional_layout.addWidget(self.hand_assignment_cb)

        hand_assignment_hint = QLabel(
            translate(
                "ControlPanelQt",
                "Use this when the video contains more than one Color family.",
            )
        )
        hand_assignment_hint.setWordWrap(True)
        hand_assignment_hint.setStyleSheet("color: #555;")
        optional_layout.addWidget(hand_assignment_hint)

        
        # Add more optional settings here as needed
        
        layout.addWidget(optional_group)
        
        layout.addStretch()
        self._add_settings_section(tab, translate("ControlPanelQt", "Optional"))

    def _handle_language_changed(self, index: int):
        """Persist the selected UI language for the next app launch."""
        locale_name = self.language_combo.itemData(index)
        if not locale_name:
            return
        save_preferred_locale(str(locale_name), self.settings)
        QMessageBox.information(
            self,
            translate("ControlPanelQt", "Language"),
            translate("ControlPanelQt", "Restart Synthesia2MIDI to apply the selected language."),
        )
    
    def _handle_conversion_request(self):
        """Handle conversion button click."""
        self.convert_button.setText(translate("ControlPanelQt", "Converting..."))
        self.convert_button.setEnabled(False)
        self.conversion_status.setText(translate("ControlPanelQt", "Converting video to MIDI..."))
        self.conversion_requested.emit()

    def _handle_exemplar_key_type_presence_toggled(self, key_type: str, checked: bool):
        """Handle per-key-type exemplar availability toggle."""
        self._update_exemplar_key_type_ui_state(key_type)
        self.exemplar_key_type_enabled_changed.emit(key_type, checked)

    def _refresh_color_family_grid(self) -> None:
        detection = self.app_state.detection
        colors = detection.exemplar_lit_colors
        enabled = detection.exemplar_key_type_enabled
        self.color_family_grid.set_families(
            active_family_numbers(enabled, colors),
            colors=colors,
            enabled=enabled,
        )
        self.exemplar_buttons = {
            slot: row.set_button
            for slot, row in self.color_family_grid.rows.items()
            if row.set_button is not None
        }
        self.exemplar_swatches = {
            slot: row.swatch for slot, row in self.color_family_grid.rows.items()
        }
        self.exemplar_presence_checkboxes = {
            slot: row.present
            for slot, row in self.color_family_grid.rows.items()
            if row.present is not None
        }
        for slot in self.color_family_grid.rows:
            self._update_exemplar_key_type_ui_state(slot)

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
            swatch.setStyleSheet("border: 1px dashed #595959; background-color: #d0d0d0;")
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
        self.spark_roi_select_button.setEnabled(spark_enabled)
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
            QMessageBox.warning(
                self,
                translate("ControlPanelQt", "Invalid Trim Range"),
                translate("ControlPanelQt", "Start frame must be less than end frame."),
            )
            return
        
        # Keep the platform warning style so contrast and keyboard behavior remain native.
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(translate("ControlPanelQt", "Permanently Trim Project"))
        msg_box.setIcon(QMessageBox.Warning)
        
        end_text = (
            translate("ControlPanelQt", "frame {end_frame}").format(end_frame=end_frame)
            if end_frame != -1
            else translate("ControlPanelQt", "end of video")
        )
        msg_box.setText(
            translate(
                "ControlPanelQt",
                "<b>This will permanently trim the working video session.</b><br><br>"
                "Frames outside {start_frame} to {end_text} will be unavailable in this project session.<br><br>"
                "Most users should cancel and use the MIDI range controls instead.",
            ).format(start_frame=start_frame, end_text=end_text)
        )
        
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        msg_box.setDefaultButton(QMessageBox.Cancel)
        
        yes_button = msg_box.button(QMessageBox.Yes)
        yes_button.setText(translate("ControlPanelQt", "Trim Project"))
        
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

    def _overlay_adjustment_key(self, key_color: str, dimension: str) -> tuple[str, str]:
        return key_color, dimension

    def _current_overlay_adjustment_basis(self) -> tuple[tuple[int, float, float, float, float, float], ...] | None:
        overlays = getattr(self.app_state, "overlays", None)
        if not overlays:
            return None
        ordered_overlays = sorted(overlays, key=lambda overlay: int(getattr(overlay, "key_id", 0)))
        return tuple(
            (
                int(getattr(overlay, "key_id", 0)),
                float(getattr(overlay, "x", 0.0)),
                float(getattr(overlay, "y", 0.0)),
                float(getattr(overlay, "width", 0.0)),
                float(getattr(overlay, "height", 0.0)),
                float(getattr(overlay, "rotation_degrees", 0.0) or 0.0),
            )
            for overlay in ordered_overlays
        )

    def _set_overlay_adjustment_value(self, key_color: str, dimension: str, value: int | None) -> None:
        key = self._overlay_adjustment_key(key_color, dimension)
        self._overlay_adjustment_values[key] = value
        label = self._overlay_adjustment_value_labels.get(key)
        if label is not None:
            label.setText(self.OVERLAY_ADJUSTMENT_INDETERMINATE if value is None else str(value))
        reset_button = self._overlay_adjustment_reset_buttons.get(key)
        if reset_button is not None:
            reset_button.setEnabled(isinstance(value, int) and value != 0)

    def clear_overlay_adjustments(self) -> None:
        self._overlay_adjustment_pending_previous_values.clear()
        for key_color, dimension in self._overlay_adjustment_value_labels:
            self._set_overlay_adjustment_value(key_color, dimension, 0)

    def _sync_overlay_adjustment_state(self) -> None:
        current_basis = self._current_overlay_adjustment_basis()
        if current_basis is None:
            self.clear_overlay_adjustments()
            self._overlay_adjustment_basis = None
            return
        if self._overlay_adjustment_basis != current_basis:
            self.clear_overlay_adjustments()
            self._overlay_adjustment_basis = current_basis

    def _apply_overlay_adjustment(self, key_color: str, dimension: str, delta: int) -> None:
        key = self._overlay_adjustment_key(key_color, dimension)
        current_value = self._overlay_adjustment_values.get(key, 0)
        self._overlay_adjustment_pending_previous_values[key] = current_value
        requested_value = (current_value if isinstance(current_value, int) else 0) + delta
        self._set_overlay_adjustment_value(key_color, dimension, requested_value)
        self.overlay_size_adjustment_requested.emit(key_color, dimension, delta)
        self._overlay_adjustment_pending_previous_values.pop(key, None)
        self._overlay_adjustment_basis = self._current_overlay_adjustment_basis()

    def apply_overlay_adjustment_result(self, result) -> None:
        key = self._overlay_adjustment_key(result.key_color, result.dimension)
        previous_value = self._overlay_adjustment_pending_previous_values.pop(key, 0)
        if result.status == "none":
            self._set_overlay_adjustment_value(result.key_color, result.dimension, previous_value)
        elif result.status == "mixed":
            self._set_overlay_adjustment_value(result.key_color, result.dimension, None)
        self._overlay_adjustment_basis = self._current_overlay_adjustment_basis()

    def _reset_overlay_adjustment(self, key_color: str, dimension: str) -> None:
        key = self._overlay_adjustment_key(key_color, dimension)
        current_value = self._overlay_adjustment_values.get(key, 0)
        if not isinstance(current_value, int) or current_value == 0:
            return
        self._set_overlay_adjustment_value(key_color, dimension, 0)
        self.overlay_size_adjustment_requested.emit(key_color, dimension, -current_value)
        self._overlay_adjustment_basis = self._current_overlay_adjustment_basis()
    
    def _toggle_spark_roi_visibility(self):
        """Toggle spark ROI overlay visibility."""
        is_visible = not self.spark_roi_toggle_button.isChecked()
        self.spark_roi_visibility_toggled.emit(is_visible)
        # Update button text based on state
        if is_visible:
            self.spark_roi_toggle_button.setText(translate("ControlPanelQt", "Hide Spark Overlays"))
        else:
            self.spark_roi_toggle_button.setText(translate("ControlPanelQt", "Show Spark Overlays"))
    
    def _set_fps_override(self, fps, emit_signal=True):
        """Set FPS override and update button states."""
        # Update button states
        self.fps_30_button.setChecked(fps == 30)
        self.fps_60_button.setChecked(fps == 60)
        self.fps_auto_button.setChecked(fps is None)
        
        # Update display
        if fps is None:
            self.fps_display_label.setText(translate("ControlPanelQt", "Current FPS: Auto-detected"))
        else:
            self.fps_display_label.setText(
                translate("ControlPanelQt", "Current FPS: {fps} (override)").format(fps=fps)
            )
        
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
                self.fps_display_label.setText(
                    translate(
                        "ControlPanelQt",
                        "Current FPS: {fps_override} (override, detected: {detected_fps})",
                    ).format(fps_override=fps_override, detected_fps=f"{detected_fps:.2f}")
                )
            else:
                self.fps_display_label.setText(
                    translate("ControlPanelQt", "Current FPS: {detected_fps} (auto-detected)").format(
                        detected_fps=f"{detected_fps:.2f}"
                    )
                )
    
    def update_controls_from_state(self):
        """Update all controls to match the current app state."""
        if not self.app_state:
            return
        
        try:
            self._sync_overlay_adjustment_state()

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
                    self.unlit_status_label.setText(translate("ControlPanelQt", "Unlit State Calibrated"))
                    self.unlit_status_label.setStyleSheet("color: #2e7d32; font-style: italic;")
                else:
                    self.unlit_status_label.setText(translate("ControlPanelQt", "Not Set"))
                    self.unlit_status_label.setStyleSheet("color: #595959; font-style: italic;")
            
            # Rebuild the dynamic family rows from persisted calibration state.
            if hasattr(self.app_state, 'detection') and hasattr(self.app_state.detection, 'exemplar_lit_colors'):
                self._refresh_color_family_grid()
            
            # Update spark ROI visibility button state
            if hasattr(self.app_state, 'detection') and hasattr(self, 'spark_roi_toggle_button'):
                is_visible = self.app_state.detection.spark_roi_visible
                self.spark_roi_toggle_button.setChecked(not is_visible)
                if is_visible:
                    self.spark_roi_toggle_button.setText(translate("ControlPanelQt", "Hide Spark Overlays"))
                else:
                    self.spark_roi_toggle_button.setText(translate("ControlPanelQt", "Show Spark Overlays"))
            
            # Update auto calibration status indicators
            self._update_auto_calibration_status()

            # Update convert button availability based on prerequisites
            if hasattr(self, "convert_button"):
                self._update_conversion_readiness_display()

        except Exception as e:
            logging.warning(f"Error updating controls from state: {e}")
    
    def set_conversion_result(self, success: bool, message: str):
        """Update the conversion status."""
        self.convert_button.setText(translate("ControlPanelQt", "Convert"))
        self.convert_button.setEnabled(self._can_convert())
        
        if success:
            self.conversion_status.setText(
                translate("ControlPanelQt", "Success: {message}").format(message=message)
            )
        else:
            self.conversion_status.setText(
                translate("ControlPanelQt", "Error: {message}").format(message=message)
            )
    
    def update_selected_overlay(self, overlay_id: Optional[int]):
        """Update the selected overlay display."""
        if overlay_id is None:
            self.selected_overlay_label.setText(translate("ControlPanelQt", "None"))
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
                    label.setText(translate("ControlPanelQt", "Calibrated"))
                    label.setStyleSheet("color: #2e7d32; font-style: italic; font-size: 12px;")
                else:
                    label.setText(translate("ControlPanelQt", "Not Set"))
                    label.setStyleSheet("color: #595959; font-style: italic;")

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

    def _conversion_readiness(self) -> ConversionReadiness:
        """Return conversion availability plus the first user-actionable missing step."""
        if not self.app_state or not hasattr(self.app_state, "video"):
            return ConversionReadiness(
                False,
                translate("ControlPanelQt", "Load a video to convert."),
            )

        if not getattr(self.app_state.video, "filepath", None):
            return ConversionReadiness(
                False,
                translate("ControlPanelQt", "Load a video to convert."),
            )

        overlays = getattr(self.app_state, "overlays", None) or []
        if not overlays:
            return ConversionReadiness(
                False,
                translate("ControlPanelQt", "Create key overlays first."),
            )

        missing_unlit = [
            overlay.key_id
            for overlay in overlays
            if getattr(overlay, "unlit_reference_color", None) is None
        ]
        if missing_unlit:
            return ConversionReadiness(
                False,
                translate("ControlPanelQt", "Capture a no-key frame."),
            )

        if getattr(self.app_state.detection, "use_histogram_detection", False):
            missing_hist = [
                overlay.key_id
                for overlay in overlays
                if getattr(overlay, "unlit_hist", None) is None
            ]
            if missing_hist:
                return ConversionReadiness(
                    False,
                    translate("ControlPanelQt", "Capture a no-key frame."),
                )

        required_exemplars = self.app_state.detection.get_required_exemplar_types()
        if not required_exemplars:
            return ConversionReadiness(
                False,
                translate("ControlPanelQt", "Capture at least one pressed-key example."),
            )

        exemplar_colors = self.app_state.detection.get_effective_exemplar_lit_colors()
        for exemplar in required_exemplars:
            if exemplar_colors.get(exemplar) is None:
                exemplar_label = _exemplar_display_label(exemplar)
                return ConversionReadiness(
                    False,
                    translate(
                        "ControlPanelQt",
                        "Capture a pressed-key example for {label}.",
                    ).format(label=exemplar_label),
                )

        detection_threshold = getattr(self.app_state.detection, "detection_threshold", 0.0)
        if not 0.1 <= detection_threshold <= 0.99:
            return ConversionReadiness(
                False,
                translate("ControlPanelQt", "Check detection sensitivity."),
            )

        if getattr(self.app_state.midi, "tempo", 0) <= 0:
            return ConversionReadiness(
                False,
                translate("ControlPanelQt", "Check MIDI tempo."),
            )

        return ConversionReadiness(
            True,
            translate("ControlPanelQt", "Ready to create MIDI."),
        )

    def _update_conversion_readiness_display(self) -> None:
        readiness = self._conversion_readiness()
        self.convert_button.setEnabled(readiness.can_convert)
        self.conversion_status.setText(readiness.status_text)
        if hasattr(self, "guide_page"):
            self.guide_page.update_snapshot(derive_guide_snapshot(self.app_state, readiness.can_convert))

    def _can_convert(self) -> bool:
        """Return True if MIDI conversion prerequisites are satisfied."""
        return self._conversion_readiness().can_convert
    
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
