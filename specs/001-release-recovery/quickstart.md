# Quickstart: v0.2.2 Release Recovery Verification

Run from the repository root.

## Focused regression gate

```bash
.venv/bin/python -m pytest \
  tests/test_runtime_paths.py \
  tests/test_package_self_check.py \
  tests/test_build_release.py \
  tests/test_release_workflow.py \
  tests/test_calibration_guide.py \
  tests/test_manual_keyboard_fit_controller.py \
  tests/test_auto_detect_tuning_controller.py \
  tests/test_bugfix_regressions.py \
  tests/test_midi_generator.py \
  tests/test_version.py
```

## Default local gate

```bash
git diff --check
.venv/bin/python -m compileall -q synthesia2midi
.venv/bin/python -m pytest
cargo fmt --manifest-path tools/midi_touchup_editor_rust/Cargo.toml --check
cargo test --manifest-path tools/midi_touchup_editor_rust/Cargo.toml
cargo check --manifest-path tools/midi_touchup_editor_rust/Cargo.toml
```

## Local package gate

```bash
.venv/bin/python packaging/build_release.py --version v0.2.2-dev
```

Expected order:

1. Pinned PyInstaller inputs install.
2. Real FFmpeg/ffprobe inputs pass `-version`.
3. The frozen application writes a passing package-self-check report.
4. GUI smoke survives eight seconds.
5. Only then is the portable zip created.

## Remote preflight and publication

1. Push the verified commit to a branch matching `codex/*-preflight`.
2. Require Windows x64 and macOS arm64 package jobs to pass.
3. Download both workflow artifacts and verify zip integrity and expected paths.
4. Merge the same commit to `main`.
5. Create and push annotated tag `v0.2.2`.
6. Require the tag workflow to pass and verify all public release assets.
7. Close GitHub issue #9 only after the public release is available.
8. Reconcile TASK-9 acceptance criteria and evidence.

The exact issue #9 YouTube URL remains a manual/network verification because the
baseline test suite intentionally does not depend on YouTube availability.
