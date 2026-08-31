# Data Model: v0.2.2 Release Recovery

## RuntimePaths

Existing immutable runtime-location value object.

### New field

- `bundle_root: Path | None = None`
  - Set from `sys._MEIPASS` only in a frozen process.
  - Declared after all existing fields for positional-call compatibility.
  - Preferred for bundled binary and asset lookup.

### Invariants

- Source mode continues to resolve from `repo_root` and PATH.
- Frozen Windows uses `_internal` as the primary bundle root.
- Frozen macOS retains `Contents/Frameworks` and `Contents/Resources`
  compatibility.
- Runtime lookup may fall back to PATH for normal app use, but release self-check
  accepts only package-owned paths.

## PackageSelfCheckReport

Ephemeral JSON evidence produced by the packaged executable.

### Fields

- `schema_version: 1`
- `status: "passed" | "failed"`
- `frozen: bool`
- `platform: str`
- `app_root: str`
- `bundle_root: str | null`
- `checks: list[PackageCheck]`
- `errors: list[str]`

### PackageCheck

- `name: str`
- `kind: "binary" | "asset"`
- `path: str | null`
- `packaged: bool`
- `probe: list[str] | null`
- `returncode: int | null`
- `status: "passed" | "failed"`
- `detail: str`

### State transitions

1. `unresolved` (implicit) -> `failed` when a path is absent.
2. `resolved` -> `failed` when the path is not package-owned.
3. Binary `resolved` -> `failed` on timeout, launch error, or nonzero exit.
4. All checks `passed` -> report `passed` and process exit `0`.
5. Any check `failed` -> report `failed` and process exit nonzero.

## AlignmentReviewState

Current-session evidence stored as `CalibrationConfig.alignment_reviewed: bool`.

### State transitions

- Initial app or new video: `False`.
- Newly generated manual or automatic overlay set: `False`.
- Manual Fit accepted: `True`.
- Auto-detect tuning accepted: `True`.
- Either editor canceled: unchanged (normally `False` for a pending review).
- Assisted calibration accepted, canceled, retried, or empty: unchanged.
- Config save/load: not serialized; a loaded session starts `False`, while
  downstream calibration can independently satisfy the Guide.

### Guide derivation

The overlay step is complete when overlays exist and either:

- `alignment_reviewed` is true, or
- existing no-key/exemplar calibration supplies legacy downstream evidence.

## MIDITrackName

Derived string stored in MIDIUtil's ISO-8859-1 text event.

### Transformation

1. Input: arbitrary Python Unicode string.
2. Normalize: Unicode NFKC.
3. Encode/decode: ISO-8859-1 with `errors="replace"`.
4. Output: safe string accepted by MIDIUtil.

### Invariants

- ASCII input is byte-for-byte unchanged.
- ISO-8859-1 characters remain recognizable.
- Compatibility characters such as `：` use their ordinary equivalent when
  NFKC supplies one.
- Unsupported code points become `?`; metadata conversion never raises a
  Unicode encoding exception.
