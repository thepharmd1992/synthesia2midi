# SoundFont Asset

The Rust touch-up editor looks for a default SoundFont file named:

- `TouchUpPiano.sf2`

Search order at runtime:

1. `--sf2 /path/to/file.sf2` CLI override
2. Executable-relative bundled path:
   - `<binary_dir>/assets/soundfonts/TouchUpPiano.sf2`
   - `<binary_dir>/../assets/soundfonts/TouchUpPiano.sf2`
3. Repo dev path:
   - `tools/midi_touchup_editor_rust/assets/soundfonts/TouchUpPiano.sf2`

If no SoundFont is found, the editor runs in muted fallback mode and remains fully usable for visual touch-up editing.

## Licensing

If you add or bundle a SoundFont in this folder, you must:

1. Verify redistribution rights for that SoundFont.
2. Add attribution and license details to `tools/midi_touchup_editor_rust/NOTICE.md`.
3. Include any required license files in release artifacts.
