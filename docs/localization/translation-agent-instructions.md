# Translation Agent Instructions

Use `docs/localization/translation-agent-packet.json` as the source of truth for
agent-assisted UI translation.

## App Context

Synthesia2MIDI is a PySide6 desktop app that analyzes Synthesia piano videos and
exports MIDI. Users load a local video or YouTube video, calibrate piano-key
overlays, tune detection settings, optionally use Spark Detection for repeated
notes, convert detected notes to MIDI, and may open a touch-up editor.

## Translation Rules

- Translate only user-visible UI text.
- Keep button, menu, label, and tooltip translations concise.
- Preserve placeholders exactly, including braces and names: `{filepath}`,
  `{error}`, `{count}`, `{key_type}`, and similar.
- Preserve product and technical terms unless the surrounding sentence needs
  normal grammatical treatment: `Synthesia2MIDI`, `Synthesia`, `MIDI`,
  `FFmpeg`, `FFprobe`, `YouTube`, `Rust`, `yt-dlp`, and `Deno`.
- Preserve URLs, paths, config keys, keyboard-note names, and file extensions.
- Preserve HTML tags, line breaks, and punctuation structure when they carry UI
  meaning.
- Add a translator note only when a source string is ambiguous.

## Required Output

Return a JSON array. Each item must contain:

```json
{
  "context": "Qt context name copied exactly",
  "source": "English source text copied exactly",
  "translation": "Translated text",
  "notes": ""
}
```

Do not return XML and do not edit repository files. The main implementation
applies translations mechanically to Qt `.ts` files.
