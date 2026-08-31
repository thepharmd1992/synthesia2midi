# Contract: Packaged Application Self-Check

## Invocation

```text
Synthesia2MIDI[.exe] --package-self-check <absolute-report-path>
```

This mode is internal to release verification. It must not create a Qt window,
prompt the user, or depend on stdout/stderr being available.

## Required checks

| Name | Resolution | Probe |
|---|---|---|
| `ffmpeg` | `RuntimePaths.ffmpeg_path()` | `-version` |
| `ffprobe` | `RuntimePaths.ffprobe_path()` | `-version` |
| `deno` | `RuntimePaths.deno_path()` | `--version` |
| `rust_editor` | `RuntimePaths.rust_editor_path()` | `--help` |
| `soundfont` | `RuntimePaths.rust_soundfont_path()` | readable file |
| `soundfont_license` | `RuntimePaths.rust_soundfont_license_path()` | readable file |

Each executable probe has a bounded timeout and captures only a short diagnostic
summary. A system PATH result is a failure even when executable because it does
not prove portability.

## JSON output

```json
{
  "schema_version": 1,
  "status": "passed",
  "frozen": true,
  "platform": "win32",
  "app_root": "C:\\...\\Synthesia2MIDI",
  "bundle_root": "C:\\...\\Synthesia2MIDI\\_internal",
  "checks": [
    {
      "name": "ffmpeg",
      "kind": "binary",
      "path": "C:\\...\\_internal\\bin\\ffmpeg.exe",
      "packaged": true,
      "probe": ["-version"],
      "returncode": 0,
      "status": "passed",
      "detail": "probe completed"
    }
  ],
  "errors": []
}
```

## Exit behavior

- `0`: report was written and every required check passed.
- nonzero: report was written with at least one failure, or the report itself
  could not be written.

The release builder must parse the report, require schema version `1` and status
`passed`, and reject the candidate before archive creation otherwise.
