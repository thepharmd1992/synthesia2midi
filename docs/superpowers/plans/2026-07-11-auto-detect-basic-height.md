# Auto-Detect Basic Height Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Open Auto-Detect Tuning with both Edge Drift sliders and Reset Section visible without scrolling the Basic tab.

**Architecture:** Keep the existing scroll-area structure, but retain the Basic scroll area as a dialog attribute and give it a minimum height equal to its expanded translated content plus the frame. Advanced keeps its existing scroll behavior and collapsed sections.

**Tech Stack:** Python 3, PySide6/Qt, pytest, existing qps pseudo-locale UI matrix.

## Global Constraints

- Work only on `codex/ux-phases-2-4`; do not create a worktree and do not push.
- Do not change auto-detect parameters, defaults, signals, persistence, width, or user-visible copy.
- Keep scrolling available as an operating-system fallback.
- Keep `uv.lock` untouched.

---

### Task 1: Fit the Expanded Basic Edge Controls on Open

**Files:**
- Modify: `tests/test_auto_detect_tuning_dialog.py`
- Modify: `synthesia2midi/synthesia2midi/gui/auto_detect_tuning_dialog.py`

**Interfaces:**
- Produces: `AutoDetectTuningDialog.basic_scroll_area: QScrollArea`.
- Preserves: `AutoDetectTuningDialog(...)`, parameter widgets, Basic/Advanced tabs, and save/cancel behavior.

- [ ] **Step 1: Write the failing opening-height test**

Add a helper that constructs and shows the dialog, then add an English/default-font assertion and a qps/150%-font assertion:

```python
def _assert_basic_edge_controls_fit(dialog):
    dialog.show()
    QApplication.processEvents()
    scroll_bar = dialog.basic_scroll_area.verticalScrollBar()
    reset_button = next(
        button
        for button in dialog.basic_scroll_area.findChildren(QPushButton)
        if button.text() == QCoreApplication.translate("AutoDetectTuningDialog", "Reset Section")
    )
    assert scroll_bar.maximum() == 0
    assert reset_button.isVisible()


def test_basic_edge_controls_fit_without_scrolling_at_default_font():
    dialog = _make_dialog()
    try:
        _assert_basic_edge_controls_fit(dialog)
    finally:
        dialog.close()


def test_basic_edge_controls_fit_without_scrolling_in_large_pseudo_locale():
    app = QApplication.instance() or QApplication([])
    original_font = QFont(app.font())
    install_translator(app, "qps")
    scaled_font = QFont(original_font)
    scaled_font.setPointSizeF(original_font.pointSizeF() * 1.5)
    app.setFont(scaled_font)
    dialog = _make_dialog()
    try:
        _assert_basic_edge_controls_fit(dialog)
    finally:
        dialog.close()
        install_translator(app, "en")
        app.setFont(original_font)
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest \
  tests/test_auto_detect_tuning_dialog.py::test_basic_edge_controls_fit_without_scrolling_at_default_font \
  tests/test_auto_detect_tuning_dialog.py::test_basic_edge_controls_fit_without_scrolling_in_large_pseudo_locale -q
```

Expected: FAIL because `basic_scroll_area` does not exist and the current Basic viewport has a nonzero scrollbar maximum.

- [ ] **Step 3: Implement responsive Basic viewport sizing**

In `_build_param_tab`, after `scroll_area.setWidget(sections_container)`, retain the Basic scroll area and set its minimum height from the content:

```python
if not expert:
    self.basic_scroll_area = scroll_area
    content_height = sections_container.sizeHint().height()
    scroll_area.setMinimumHeight(content_height + (2 * scroll_area.frameWidth()))
```

Do not set the minimum on the expert tab.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest \
  tests/test_auto_detect_tuning_dialog.py \
  tests/test_auto_detect_tuning_controller.py \
  tests/test_render_ui_matrix.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Run visual and full verification**

Run:

```bash
rm -rf logs/ux-audit/auto-detect-height-qps
PYTHONPATH=synthesia2midi QT_QPA_PLATFORM=offscreen \
  .venv/bin/python -m synthesia2midi.tools.render_ui_matrix \
  --locale qps --font-scale 1.5 --output logs/ux-audit/auto-detect-height-qps
git diff --check
.venv/bin/python -m compileall -q synthesia2midi
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest
```

Expected: matrix exits zero with no clipping/overlap findings, compileall succeeds, and the complete suite passes.

- [ ] **Step 6: Commit**

```bash
git add tests/test_auto_detect_tuning_dialog.py \
  synthesia2midi/synthesia2midi/gui/auto_detect_tuning_dialog.py
git commit -m "fix: show auto detect edge controls on open"
```

## Self-Review

- Spec coverage: Basic content fits; Advanced remains unchanged; width, behavior, copy, and persistence are untouched.
- Placeholder scan: no placeholders or deferred work.
- Type consistency: `basic_scroll_area` is a `QScrollArea` created by `_build_param_tab` before tests access it.
