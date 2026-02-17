# MIDI Touch-Up Rust Editor Notice

This Rust subproject is distributed under GPL-3.0-only.

## Provenance
- Visual direction and interaction patterns were inspired by Neothesia:
  - Repository: https://github.com/PolyMeilex/Neothesia
  - License: GPL-3.0

## Scope
- The Python host application in this repository remains separate and launches this Rust editor as a standalone process.
- The Rust editor implements synthesia2midi-specific touch-up workflow behavior (single-note edit/delete/resize/move, falling-bars UI, JSON exit contract).

## Attribution Notes
- If additional code is adapted from third-party GPL components in future changes, maintainers must document specific file-level provenance here.

## SoundFont Attribution Policy
- The editor supports bundled SoundFont playback through `TouchUpPiano.sf2`.
- Current local SoundFont used for development:
  - File: `tools/midi_touchup_editor_rust/assets/soundfonts/TouchUpPiano.sf2`
  - Origin package: Debian `timgm6mb-soundfont` (TimGM6mb.sf2)
  - Upstream source reference from Debian metadata:
    - https://github.com/musescore/musescore-old/commit/90c33ef9d87b3f5ff92efd3b07d89eb455fb1fef
  - Declared license for `TimGM6mb.sf2`: GPL-2
  - Attribution/copyright file:
    - `tools/midi_touchup_editor_rust/assets/soundfonts/TouchUpPiano_LICENSE.txt`
