import pytest
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from synthesia2midi.gui.controls_qt import ControlPanelQt
from synthesia2midi.localization import install_translator, supported_user_locales


def test_settings_footer_uses_single_column_rows():
    app = QApplication.instance() or QApplication([])
    panel = ControlPanelQt()
    try:
        footer_layout = panel.settings_footer.layout()

        assert footer_layout.rowCount() == 3
        assert footer_layout.columnCount() == 1
    finally:
        panel.close()
        app.processEvents()


@pytest.mark.parametrize("locale_name", [*supported_user_locales(), "qps"])
@pytest.mark.parametrize("font_scale", [1.0, 1.25, 1.5])
def test_settings_rail_fits_every_label_at_shipped_font_scales(locale_name, font_scale):
    app = QApplication.instance() or QApplication([])
    original_font = QFont(app.font())
    scaled_font = QFont(original_font)
    base_size = original_font.pointSizeF() if original_font.pointSizeF() > 0 else 13.0
    scaled_font.setPointSizeF(base_size * font_scale)
    install_translator(app, locale_name)
    app.setFont(scaled_font)
    panel = ControlPanelQt()
    try:
        panel.show()
        app.processEvents()
        metrics = panel.settings_section_rail.fontMetrics()
        widest_text = max(
            metrics.horizontalAdvance(panel.settings_section_rail.item(index).text())
            for index in range(panel.settings_section_rail.count())
        )
        assert panel.settings_section_rail.width() == min(144, max(98, widest_text + 28))
        assert panel.settings_section_rail.horizontalScrollBar().maximum() == 0
        assert panel.settings_section_rail_container.width() == panel.settings_section_rail.width()
        assert panel.settings_footer.width() == panel.tab_widget.width()
        if font_scale == 1.5:
            for section in panel.advanced_sections.values():
                section._toggle.setChecked(True)
            app.processEvents()
            assert all(
                label.width() >= label.sizeHint().width()
                for label in panel.advanced_slider_labels
            )
    finally:
        panel.close()
        install_translator(app, "en")
        app.setFont(original_font)
