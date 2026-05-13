# Agent Task Boundaries

Use this file to keep coding-agent work bounded and reviewable.

## Universal Preflight

Every implementation task starts with:

```bash
git status --short --branch
```

Record intended files before editing. Do not overwrite unrelated user/agent work.

## Ownership Table

| Area | Paths | Primary profile | Notes |
|---|---|---|---|
| Agent/project docs | `AGENTS.md`, `PROJECT_LOG.md`, `docs/*.md` | `pm`, `reviewer` | Keep source-of-truth concise and non-duplicative. |
| GUI/window/widgets | `synthesia2midi/synthesia2midi/gui/`, GUI-facing parts of `main.py` | `frontend-eng` | Avoid algorithm changes here. |
| Workflows/controllers | `synthesia2midi/synthesia2midi/workflows/` | `backend-eng` | Orchestration layer; keep UI rendering out when possible. |
| State/config | `synthesia2midi/synthesia2midi/core/`, `config_manager.py`, `app_config.py` | `backend-eng` | Backward-compatibility sensitive. |
| Detection | `synthesia2midi/synthesia2midi/detection/` | `backend-eng`, `researcher` | Require regression tests or synthetic fixtures. Keep `MonolithicPianoDetector` and auto-detect tuning parameter compatibility unless the task explicitly includes migration. |
| MIDI conversion | `midi_generator.py`, `midi_reader.py`, conversion workflows | `backend-eng` | Verify generated/parsed MIDI behavior. |
| Video/FFmpeg/YouTube | `video_loader.py`, `image_sequence_loader.py`, `youtube_downloader.py`, `utils/ffmpeg_helper.py` | `backend-eng`, `ops` | Platform/network sensitive. |
| Rust editor | `tools/midi_touchup_editor_rust/` | `backend-eng`, `frontend-eng` | Preserve CLI contract with Python host. |
| CI/setup/install | `.github/workflows/`, `setup_env.py`, `run.py` | `ops` | Test platform assumptions. |
| Reviews | Any changed files | `reviewer` | Check scope, behavior drift, generated files, and verification evidence. |

## Refactor Boundaries

- Extract one responsibility at a time.
- Keep wrappers when signal wiring still depends on old methods.
- Avoid mixing extraction with behavior changes.
- Add or update characterization tests before changing fragile orchestration.
- If a controller starts growing into a new god object, stop and split it before continuing.

## Handoff Template

When completing a card, report:

```text
Files changed:
Verification run:
Behavior changes: none / describe
Known risks:
Next recommended task:
```
