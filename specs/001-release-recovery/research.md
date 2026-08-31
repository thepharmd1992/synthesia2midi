# Research: v0.2.2 Release Recovery

## Decision 1: Resolve PyInstaller resources through `sys._MEIPASS`

**Decision**: Add `bundle_root` as the final optional `RuntimePaths` dataclass
field. `detect()` records `sys._MEIPASS` for frozen builds, and frozen lookup
prefers that root before executable-adjacent and existing macOS compatibility
roots.

**Rationale**: PyInstaller 6 one-folder builds put collected files in
`_internal` and set `sys._MEIPASS` to that directory. The released Windows app
therefore contains the helpers but searches only beside the top-level executable.
The field is last to preserve existing positional construction.

**Alternatives rejected**:

- `--contents-directory .`: restores an old layout but leaves runtime code
  dependent on a packager setting and does not protect later layout changes.
- Replace all roots with `_MEIPASS`: risks regressing existing macOS
  `Frameworks`/`Resources` and source-checkout fallbacks.

**Sources**:

- <https://pyinstaller.org/en/stable/runtime-information.html>
- <https://pyinstaller.org/en/stable/CHANGES.html>

## Decision 2: Pin the Python packager and its contributed hooks

**Decision**: Add `packaging/requirements-build.txt` with exact reviewed pins
for `pyinstaller==6.22.2` and `pyinstaller-hooks-contrib==2026.7`; install from
that file instead of floating `pip install pyinstaller`.

**Rationale**: The prior release silently moved from pre-6 layout assumptions to
PyInstaller 6 behavior. Exact inputs make future layout/tooling changes deliberate.
These were the current stable upstream releases reviewed on 2026-08-31.

**Alternatives rejected**:

- Pin only PyInstaller: its separately released contributed hooks could still
  change the collected package contents.
- Leave both floating and rely on the self-check: fail-closed verification helps,
  but does not make builds reproducible or changes reviewable.

**Sources**:

- <https://pypi.org/project/pyinstaller/6.22.2/>
- <https://pypi.org/project/pyinstaller-hooks-contrib/2026.7/>

## Decision 3: Resolve and probe the real Chocolatey FFmpeg tools

**Decision**: Pin the Windows Chocolatey FFmpeg package at `9.0.1`. When PATH
resolves to Chocolatey's global shim directory, locate the corresponding real
`ffmpeg.exe` or `ffprobe.exe` beneath the installed package, require an unambiguous
candidate, and execute `-version` before staging it.

**Rationale**: The `392,704` byte files in `v0.2.1` are ShimGen launchers whose
embedded targets exist only on the build agent. The installed package contains
the actual static tools. Resolving the target keeps the existing CI source while
preventing build-agent path leakage.

**Alternatives rejected**:

- Copy the PATH result: reproduces the released defect.
- Download another third-party archive in the Python builder: adds a second
  downloader, checksum registry, and licensing/supply-chain surface to this
  patch when the pinned CI package already provides the real files.

**Source**: <https://community.chocolatey.org/packages/ffmpeg/9.0.1>

## Decision 4: Make the packaged application self-check before archiving

**Decision**: Add a hidden `--package-self-check <report.json>` launcher mode.
The frozen app resolves FFmpeg, ffprobe, Deno, the Rust editor, SoundFont, and
license through `RuntimePaths`; requires them to be package-owned; runs bounded
non-interactive probes; and writes the versioned JSON report. The release builder
runs this mode before GUI smoke and creates the zip only after both pass.

**Rationale**: Archive inspection and an eight-second GUI survival test cannot
prove runtime resolution or executable viability. Calling the packaged entrypoint
tests the same frozen path environment users receive. A report file works with
the no-console Windows executable.

**Alternatives rejected**:

- Validate staged source files before PyInstaller: cannot catch collection layout
  or runtime resolver failures.
- Extend only the GUI smoke duration: the affected helpers are lazy and remain
  unused during startup.

## Decision 5: Track accepted alignment review as non-persisted session state

**Decision**: Add `CalibrationConfig.alignment_reviewed` with a default of
`False`, deliberately omit it from config serialization, set it only after an
accepted auto-tuning or Manual Fit dialog, and clear it for a new video or newly
generated overlay set. Existing downstream calibration remains valid evidence.

**Rationale**: The Guide currently infers review only from later no-key/exemplar
work, so accepting unchanged geometry cannot advance it. Review is about the
active overlay set, not durable calibration data. Keeping it session-only avoids
schema migration and stale confirmation in old projects.

**Alternatives rejected**:

- Treat opening the editor as review: cancel would incorrectly advance.
- Infer review from changed coordinates: correctly aligned overlays often require
  no changes.
- Persist a boolean: it can outlive the geometry it reviewed unless the project
  format also gains an overlay identity/version contract.

## Decision 6: Normalize MIDI text to MIDIUtil's supported encoding

**Decision**: Normalize track names with Unicode NFKC, then encode/decode with
ISO-8859-1 using deterministic `?` replacement for unsupported characters before
calling MIDIUtil.

**Rationale**: MIDIUtil 1.2.1 encodes text events as ISO-8859-1. NFKC maps the
reported fullwidth colon to ordinary `:`, preserves ASCII and supported accented
Latin text, and gives a stable fallback for CJK/emoji instead of aborting the
entire conversion.

**Alternatives rejected**:

- Strip all non-ASCII: unnecessarily loses supported accented text.
- Patch or replace MIDIUtil in this release: much larger dependency and file-
  compatibility blast radius.
- Omit the source-derived track name: prevents the crash but discards useful
  metadata for every international filename.

## Decision 7: Publish only after preflight and evidence reconciliation

**Decision**: Update the patch version to `0.2.2-dev`, pass all local gates, push
a `codex/*-preflight` ref, verify both package jobs and artifacts, then merge to
`main`, tag `v0.2.2`, verify the public assets, close GitHub issue #9, and finally
reconcile TASK-9 against actual evidence.

**Rationale**: TASK-9 describes the intended release contract but remained `To
Do`; `v0.2.1` proves several criteria only nominally. Status must follow a working
replacement, not the existence of files or a historical release run.

**Alternative rejected**: Mark TASK-9 done now because builds/releases already
exist. The broken helper resolution means acceptance criterion 2 is not yet true.
