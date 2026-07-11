import pytest
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from synthesia2midi.gui.controls_qt import ControlPanelQt
from synthesia2midi.localization import install_translator, supported_user_locales


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
        metrics = panel.settings_section_rail.fontMetrics()
        widest_text = max(
            metrics.horizontalAdvance(panel.settings_section_rail.item(index).text())
            for index in range(panel.settings_section_rail.count())
        )
        assert panel.settings_section_rail.width() >= widest_text + 28
        assert panel.settings_section_rail_container.width() == panel.settings_section_rail.width()
        assert panel.settings_rail_actions.width() == panel.settings_section_rail.width()
    finally:
        panel.close()
        install_translator(app, "en")
        app.setFont(original_font)
