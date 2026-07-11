# Auto-Detect Basic Height Design

## Goal

Open Auto-Detect Tuning with the complete Basic Edge Drift Correction section visible, including both directional sliders and Reset Section, without requiring users to discover the internal scrollbar.

## Design

- Keep the dialog width and the existing Basic/Advanced tab structure unchanged.
- After constructing the Basic tab content, set its scroll viewport minimum height from the expanded Basic content's actual size hint plus the scroll-area frame.
- Apply this minimum only to Basic. Advanced remains scrollable because its specialist sections are intentionally larger and collapsed by default.
- Let Qt grow the dialog from its existing opening size only as much as the current locale and font require.
- Preserve scrolling as a fallback if the operating system constrains the window on an unusually small display.

## Compatibility

- Do not change auto-detect parameters, defaults, signal wiring, saved settings, or tuning behavior.
- Do not change the dialog width or remove the scroll area.
- Keep all user-visible wording and translation catalogs unchanged.

## Verification

- Add a failing offscreen test proving the Basic tab opens without a vertical scrollbar and Reset Section is visible.
- Cover English at the default font and the qps pseudo-locale at 150% font size.
- Confirm Advanced remains scrollable and collapsed by default.
- Run the focused Auto-Detect tests, the qps UI matrix, and the complete test suite.
