import ast
import inspect
import textwrap
from types import SimpleNamespace

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QAbstractScrollArea, QApplication, QWidget

from synthesia2midi.gui.color_family_grid import ColorFamilyGrid
from synthesia2midi.localization import install_translator
from synthesia2midi.tools.render_ui_matrix import _detect_clipping


def test_exemplar_labels_use_literal_extractable_translation_sources():
    source = textwrap.dedent(inspect.getsource(ColorFamilyGrid._add_exemplar_row))
    tree = ast.parse(source)
    translated_strings = {
        node.args[0].value
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "tr"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        )
    }

    assert {"Natural", "Sharp / Flat"} <= translated_strings


def _family_data():
    colors = {
        "LW": (30, 80, 220),
        "LB": (20, 45, 150),
        "COLOR_3_W": (230, 180, 30),
        "COLOR_3_B": (170, 120, 20),
    }
    enabled = {slot: True for slot in colors}
    return colors, enabled


def _four_family_data():
    colors = {
        "LW": (30, 80, 220),
        "LB": (20, 45, 150),
        "RW": (220, 60, 70),
        "RB": (160, 35, 50),
        "COLOR_3_W": (230, 180, 30),
        "COLOR_3_B": (170, 120, 20),
        "COLOR_4_W": (35, 190, 90),
        "COLOR_4_B": (25, 125, 60),
    }
    enabled = {slot: True for slot in colors}
    return colors, enabled


def test_calibration_grid_uses_compact_color_family_rows_and_emits_actions():
    QApplication.instance() or QApplication([])
    colors, enabled = _family_data()
    grid = ColorFamilyGrid(mode="calibration")
    requested = []
    enabled_changes = []
    added = []
    removed = []
    grid.exemplar_requested.connect(requested.append)
    grid.exemplar_enabled_changed.connect(
        lambda slot, is_enabled: enabled_changes.append((slot, is_enabled))
    )
    grid.family_add_requested.connect(lambda: added.append(True))
    grid.family_remove_requested.connect(removed.append)
    try:
        grid.set_families((1, 3), colors=colors, enabled=enabled)

        assert list(grid.rows) == ["LW", "LB", "COLOR_3_W", "COLOR_3_B"]
        assert grid.family_heading(1).text() == "Color 1"
        assert grid.family_heading(3).text() == "Color 3"
        assert grid.rows["LW"].label.text() == "Natural"
        assert grid.rows["LB"].label.text() == "Sharp / Flat"
        assert grid.rows["COLOR_3_W"].label.text() == "Natural"
        assert grid.rows["COLOR_3_B"].label.text() == "Sharp / Flat"

        grid.rows["COLOR_3_B"].set_button.click()
        grid.rows["LW"].present.click()
        grid.add_family_button.click()
        grid.remove_family_buttons[3].click()

        assert requested == ["COLOR_3_B"]
        assert enabled_changes == [("LW", False)]
        assert added == [True]
        assert removed == [3]
    finally:
        grid.close()
        grid.deleteLater()


def test_review_grid_replaces_editable_controls_with_found_and_missing_statuses():
    QApplication.instance() or QApplication([])
    colors, enabled = _family_data()
    assignments = {
        "LW": SimpleNamespace(rgb=colors["LW"]),
        "LB": SimpleNamespace(rgb=None),
    }
    grid = ColorFamilyGrid(mode="review")
    try:
        grid.set_families(
            (1,),
            colors=colors,
            enabled=enabled,
            assignments=assignments,
        )

        assert grid.rows["LW"].set_button is None
        assert grid.rows["LW"].present is None
        assert grid.rows["LW"].status.text() == "Found"
        assert grid.rows["LB"].status.text() == "Missing"
        assert not hasattr(grid, "add_family_button")
        assert grid.remove_family_buttons == {}
    finally:
        grid.close()
        grid.deleteLater()


def test_calibration_grid_rebuild_removes_stale_add_control_at_four_families():
    QApplication.instance() or QApplication([])
    colors, enabled = _four_family_data()
    grid = ColorFamilyGrid(mode="calibration")
    try:
        grid.set_families((1,), colors=colors, enabled=enabled)
        assert hasattr(grid, "add_family_button")

        grid.set_families(
            (1, 2, 3, 4),
            colors=colors,
            enabled=enabled,
        )

        assert not hasattr(grid, "add_family_button")
        assert list(grid.rows) == list(colors)
        assert set(grid.remove_family_buttons) == {2, 3, 4}
    finally:
        grid.close()
        grid.deleteLater()


def test_grid_stays_within_compact_geometry_at_150_percent_font_scale():
    app = QApplication.instance() or QApplication([])
    original_font = QFont(app.font())
    scaled_font = QFont(original_font)
    base_size = original_font.pointSizeF() if original_font.pointSizeF() > 0 else 13.0
    scaled_font.setPointSizeF(base_size * 1.5)
    app.setFont(scaled_font)
    colors, enabled = _four_family_data()
    grid = ColorFamilyGrid(mode="calibration")
    try:
        grid.set_families((1, 2, 3, 4), colors=colors, enabled=enabled)
        grid.show()
        app.processEvents()
        grid.resize(grid.sizeHint())
        app.processEvents()

        assert grid.minimumSizeHint().width() <= 760
        assert all(
            grid.rect().contains(child.geometry())
            for child in grid.findChildren(QWidget, options=Qt.FindDirectChildrenOnly)
        )
    finally:
        grid.close()
        grid.deleteLater()
        app.setFont(original_font)
        app.processEvents()


def test_family_grid_variants_render_without_clipping_at_default_and_150_percent():
    app = QApplication.instance() or QApplication([])
    original_font = QFont(app.font())
    colors, enabled = _four_family_data()

    try:
        for locale_name, font_scale in (("en", 1.0), ("qps", 1.5)):
            assert install_translator(app, locale_name) == locale_name
            scaled_font = QFont(original_font)
            base_size = original_font.pointSizeF() if original_font.pointSizeF() > 0 else 13.0
            scaled_font.setPointSizeF(base_size * font_scale)
            app.setFont(scaled_font)

            for family_count in (1, 2, 3, 4):
                grid = ColorFamilyGrid(mode="calibration")
                try:
                    family_numbers = tuple(range(1, family_count + 1))
                    grid.set_families(family_numbers, colors=colors, enabled=enabled)
                    grid.resize(max(420, grid.sizeHint().width()), grid.sizeHint().height())
                    grid.show()
                    app.processEvents()
                    grid.resize(max(420, grid.sizeHint().width()), grid.sizeHint().height())
                    app.processEvents()

                    scrollbars = [
                        area.horizontalScrollBar().maximum()
                        for area in grid.findChildren(QAbstractScrollArea)
                    ]
                    assert scrollbars == [] or max(scrollbars) == 0
                    assert _detect_clipping(grid) == []
                    assert all(
                        grid.rect().contains(child.geometry())
                        for child in grid.findChildren(QWidget, options=Qt.FindDirectChildrenOnly)
                        if child.isVisible()
                    )
                finally:
                    grid.close()
                    grid.deleteLater()
                    app.processEvents()
    finally:
        install_translator(app, "en")
        app.setFont(original_font)
        app.processEvents()
