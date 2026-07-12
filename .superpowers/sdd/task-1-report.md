# Task 1 Report: Canonical Color-Family Registry and State

## Status

Completed on `codex/four-color-families`.

## What Changed

- Added `synthesia2midi/synthesia2midi/core/color_families.py` as the sole
  canonical mapping for four color families, their Natural and Sharp / Flat
  slots, and zero-based MIDI channels.
- Expanded `DetectionConfig` to initialize colors, histograms, and enabled
  flags for all eight supported slots. Legacy Color 1 and Color 2 slots remain
  enabled by default; Color 3 and Color 4 default to disabled.
- Added `get_required_exemplar_types()` and retained
  `get_required_base_exemplar_types()` as a compatibility alias.
- Updated effective color and histogram maps to mask disabled supported slots
  while preserving pre-existing unknown dynamic entries for compatibility.
- Added focused registry and state tests.

## RED/GREEN Evidence

### RED

Command:

```bash
.venv/bin/python -m pytest tests/test_color_families.py tests/test_app_state.py -q
```

Result before implementation:

```text
ModuleNotFoundError: No module named 'synthesia2midi.core.color_families'
```

The failure occurred during collection of the new registry test, proving the
canonical registry did not yet exist.

### GREEN

Command:

```bash
.venv/bin/python -m pytest tests/test_color_families.py tests/test_app_state.py -q
```

Result after implementation:

```text
4 passed
```

Focused regression command:

```bash
.venv/bin/python -m pytest tests/test_color_families.py tests/test_app_state.py tests/test_bugfix_regressions.py -q
```

Result:

```text
36 passed
```

Repository default gate:

```bash
git diff --check
.venv/bin/python -m compileall -q synthesia2midi
.venv/bin/python -m pytest
```

Result:

```text
423 passed, 29 warnings
```

The warnings are existing PySide6 deprecations in manual-fit and font-database
test paths; they do not originate from this task.

## Files Changed

- `synthesia2midi/synthesia2midi/core/color_families.py`
- `synthesia2midi/synthesia2midi/core/app_state.py`
- `tests/test_color_families.py`
- `tests/test_app_state.py`
- `.superpowers/sdd/task-1-report.md`

## Self-Review

- The slot-to-family mapping exists in one new core module and is derived by
  `DetectionConfig` rather than duplicated there.
- Existing legacy slot identifiers and their channel identities are unchanged.
- Color 3 and Color 4 are present in state but disabled by default, so existing
  two-family videos retain their current required-exemplar behavior.
- The old required-exemplar method delegates directly to the new method.
- Effective maps retain unknown entries because prior persistence code accepts
  arbitrary dynamic keys, while the supported eight slots are always present
  and correctly masked.
- No persistence, scanner, UI, conversion, or MIDI-channel files were changed;
  those remain later-task ownership.

## Concerns

- `uv.lock` was already untracked and was intentionally left untouched.
- Task 2 must extend persistence to save and load the new enabled flags. This
  task deliberately does not alter config serialization.

## Review Finding: Legacy Enabled-Map Defaults

### RED

Command:

```bash
.venv/bin/python -m pytest tests/test_app_state.py::test_absent_legacy_enabled_flags_remain_effective_in_partial_map -q
```

Result before the fix:

```text
F                                                                        [100%]
_______ test_absent_legacy_enabled_flags_remain_effective_in_partial_map _______
E       AssertionError: assert [] == ['LB', 'RW', 'RB']

E         Right contains 3 more items, first extra item: 'LB'
E         Use -v to get more diff
```

The failure showed that absent legacy enabled flags were incorrectly treated
as disabled.

### GREEN

Focused regression command:

```bash
.venv/bin/python -m pytest tests/test_app_state.py::test_absent_legacy_enabled_flags_remain_effective_in_partial_map -q
```

Result:

```text
.                                                                        [100%]
```

Requested regression suite:

```bash
.venv/bin/python -m pytest tests/test_color_families.py tests/test_app_state.py tests/test_bugfix_regressions.py
```

Result:

```text
.....................................                                    [100%]
37 passed in 0.32s
```

The fix restores `True` fallback behavior for absent `LW`, `LB`, `RW`, and
`RB` flags while keeping absent `COLOR_3_*` and `COLOR_4_*` flags disabled in
required and effective exemplar results.
