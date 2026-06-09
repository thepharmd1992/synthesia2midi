# Packaged Release And YouTube Hardening Design

## Goal

Ship portable end-user builds for Windows x64 and macOS Apple Silicon from GitHub Releases without requiring Python, FFmpeg, Rust, or terminal setup, while keeping the YouTube downloader in the product and hardening it as far as practical.

## Non-Goals

- No Electron rewrite
- No installer-first rollout
- No signing, notarization, or auto-update work in this slice
- No promise that YouTube download becomes fully reliable against all upstream changes

## Product Scope

The end-user release surface moves from the source repo to GitHub Releases. Developers still use the repo and setup scripts. End users download zipped portable artifacts.

The packaged release keeps:

- local video loading
- image sequence loading
- MIDI export
- Rust touch-up editor
- YouTube download

The packaged release bundles:

- Python runtime
- PySide6 and Qt payloads
- FFmpeg
- ffprobe
- Deno
- Rust touch-up editor binary
- TouchUpPiano soundfont
- required notice and license files

## Core Problem

The app still assumes it is running from a repo checkout. Several code paths derive a repo root from `__file__`, write into code-relative folders, or instruct the user to run `setup_env.py` or `cargo build`. That is acceptable in developer mode and wrong in a packaged app.

The YouTube downloader is also upstream-dependent. It relies on `yt-dlp`, network conditions, site behavior, JavaScript challenge support, FFmpeg, and in some cases browser cookies. That path must be hardened and isolated, not treated like local file import.

## Architecture

### 1. Dual runtime model

Keep two launch surfaces:

- `run.py` at repo root remains the developer launcher
- a packaged entrypoint freezes `synthesia2midi/run.py` for end-user builds

Developer and packaged modes share app code but not runtime assumptions.

### 2. Runtime path layer

Add one small runtime module that owns environment detection and path resolution.

It answers:

- whether the app is running frozen or from source
- where bundled binaries live
- where bundled assets live
- where user-writable directories live

It provides stable resolvers for:

- `ffmpeg`
- `ffprobe`
- `deno`
- Rust touch-up editor binary
- Rust soundfont asset path
- default download directory
- default screenshot/debug output directory
- default log directory

Resolution order is:

1. packaged bundle paths when frozen
2. repo-relative developer paths when running from source
3. PATH/system fallback only when explicitly allowed

### 3. User-writable path policy

Packaged builds stop writing into repo-relative or code-relative folders.

Defaults move to user locations:

- downloads/videos: `QStandardPaths.MoviesLocation` or `DocumentsLocation`
- logs: per-user app log dir
- screenshots/debug output: user pictures dir, temp dir, or app data subdir

Developer mode may keep repo-local conveniences where they do not leak into packaged mode.

### 4. Companion tool integration

The packaged build bundles and launches helper tools by absolute resolved path.

That includes:

- FFmpeg and ffprobe for video probing and conversion
- Deno for yt-dlp EJS JavaScript challenge support
- Rust touch-up editor binary
- bundled soundfont and required license files

Touch-up editor launch must stop assuming `tools/midi_touchup_editor_rust/target/release/...` exists. Packaged mode resolves the bundled binary. Developer mode keeps the current release-binary fallback.

### 5. YouTube downloader hardening

The YouTube downloader becomes a managed subsystem with explicit retry behavior.

Primary flow:

1. attempt normal yt-dlp download
2. if failure matches auth, cookie, age, bot-check, or JS challenge categories, auto-retry with remembered browser cookies
3. if retry fails, show short, explicit error classification and point to manual local video loading

Supported browsers in v1:

- Chrome
- Edge
- Safari

Behavior:

- app remembers preferred browser
- app remembers whether auto-retry is enabled
- app first retries with the preferred browser
- on repeated auth/challenge failure, UI can offer alternate supported browsers

Hardening requirements:

- pass explicit `ffmpeg_location` into yt-dlp
- pass explicit JS runtime configuration instead of relying on PATH luck
- add browser-cookie support through yt-dlp browser-cookie features
- classify likely failure causes into actionable messages

Important boundary:

YouTube download remains upstream-dependent and can still fail after hardening. The app must frame this as a convenience path, not as the only supported way to load videos.

## Packaging Strategy

Use PyInstaller in `one-folder` mode.

Reasons:

- simpler sidecar staging than one-file mode
- easier debugging when packaged companion tools are missing
- better fit for large Qt/OpenCV payloads and helper binaries

The packaged app should be built from a dedicated packaging script plus a PyInstaller spec that stages:

- Python entrypoint
- package code
- non-Python data files
- Rust binary
- soundfont and license files
- FFmpeg
- ffprobe
- Deno

## Release Strategy

User builds are created from version tags only.

Flow:

1. push normal code to repo
2. create a tag like `v0.x.y`
3. GitHub Actions builds Windows x64 and macOS Apple Silicon artifacts
4. CI smoke-launches packaged builds
5. CI zips artifacts
6. CI creates or updates the GitHub Release for that tag

End users interact with Releases, not the source repo.

## Versioning

Add one explicit application version source in the repo.

Use:

- tag format: `v0.x.y`
- artifact naming:
  - `Synthesia2MIDI-windows-x64-v0.x.y.zip`
  - `Synthesia2MIDI-macos-arm64-v0.x.y.zip`

## Implementation Phases

### Phase 1. Runtime foundation

- add runtime-path/environment module
- add packaged entrypoint
- separate developer and packaged launch assumptions
- remove packaged writes to code-relative folders

### Phase 2. Companion tool routing

- route FFmpeg and ffprobe through runtime resolver
- route Rust editor through runtime resolver
- route soundfont through bundle-aware resolver
- add Deno resolver

### Phase 3. YouTube hardening

- explicit yt-dlp JS runtime wiring
- explicit `ffmpeg_location`
- preferred-browser persistence
- auto-retry with browser cookies for known failure classes
- better error classification and user messaging

### Phase 4. Packaging

- add PyInstaller spec
- add packaging script
- build local portable artifacts
- smoke-check packaged launch and companion binary discovery

### Phase 5. CI and releases

- add tag-triggered release workflow
- build both target platforms
- zip artifacts
- publish GitHub Release assets

### Phase 6. Docs

- add end-user download/run instructions
- add macOS Gatekeeper `Open Anyway` note
- add Windows SmartScreen `Run anyway` note
- keep developer setup docs separate

## Testing And Validation

Code-level validation:

- existing default gate still passes
- add unit tests for runtime path resolution
- add unit tests for YouTube retry policy and error classification
- add tests for preferred browser persistence

Packaged validation:

- packaged app launches on macOS Apple Silicon
- packaged app launches on Windows x64
- packaged app can resolve bundled FFmpeg and ffprobe
- packaged Rust editor launches
- packaged Rust editor finds its soundfont
- packaged YouTube path attempts plain download, retries with preferred browser cookies when appropriate, and emits explicit failure guidance when all attempts fail

Release validation:

- tag workflow builds both artifacts
- release assets are zipped and named correctly
- GitHub Release contains user-facing download assets, not source-only instructions

## Risks

- yt-dlp and YouTube behavior can still break after release
- browser-cookie access may vary by browser state, OS permissions, or upstream changes
- Safari cookie support is likely the highest-risk browser path
- bundling external binaries increases artifact size significantly
- Qt/OpenCV/PyInstaller integration may require iterative hook tuning

## Guardrails

- keep local video loading as the stable fallback
- do not block the rest of the app on YouTube success
- do not freeze the repo-root developer launcher
- do not let packaging logic leak repo-only error messages into packaged mode
- commit in small checkpoints after each coherent phase

## Decision Summary

- same repo, separate worktree/branch
- PyInstaller one-folder portable builds
- Windows x64 and macOS Apple Silicon first
- GitHub Releases from version tags only
- YouTube downloader stays in packaged v1
- bundle Deno and use browser-cookie auto-retry with remembered preferred browser
