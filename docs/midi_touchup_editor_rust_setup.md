# Rust MIDI Touch-Up Editor Setup

This project uses a standalone Rust executable for the MIDI touch-up editor.

## Binary path expected by Python host

- Linux/WSL: `tools/midi_touchup_editor_rust/target/release/midi-touchup-editor`
- macOS: `tools/midi_touchup_editor_rust/target/release/midi-touchup-editor`
- Windows: `tools\\midi_touchup_editor_rust\\target\\release\\midi-touchup-editor.exe`

## Rust toolchain install

- Windows (PowerShell):
  - `winget install --id Rustlang.Rustup -e`
- macOS/Linux/WSL:
  - `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
  - restart shell or run `source "$HOME/.cargo/env"`

## Build from source

```bash
cd tools/midi_touchup_editor_rust
cargo build --release
```

## SoundFont setup (audio playback)

The editor's built-in audio playback uses a SoundFont file.

Default expected filename:
- `TouchUpPiano.sf2`

Default search order:
1. CLI override `--sf2 /path/to/file.sf2`
2. Executable-relative bundled path:
   - `<binary_dir>/assets/soundfonts/TouchUpPiano.sf2`
   - `<binary_dir>/../assets/soundfonts/TouchUpPiano.sf2`
3. Repo dev path:
   - `tools/midi_touchup_editor_rust/assets/soundfonts/TouchUpPiano.sf2`

If no SoundFont is found or audio init fails, the editor stays open in muted fallback mode.

Current backend support:
- Windows/macOS: built-in audio playback enabled.
- Linux/WSL: muted fallback mode (editor remains fully usable for visual touch-up).

## CLI contract

```bash
tools/midi_touchup_editor_rust/target/release/midi-touchup-editor \
  --midi /path/to/file.mid \
  --result-json \
  --theme neothesia \
  --sf2 /path/to/TouchUpPiano.sf2
```

Stdout emits one JSON line before exit when `--result-json` is supplied:

```json
{"status":"saved|cancelled|error","source_path":"...","saved_path":"...|null","message":"..."}
```

Exit codes:
- `0` for `saved` or `cancelled`
- non-zero for `error`
