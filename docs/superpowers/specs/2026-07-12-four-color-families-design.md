# Four Color Families and Eight Lit Exemplars Design

## Goal

Support Synthesia videos containing up to four distinct note-color families on one rendered keyboard. Each family has a Natural-key exemplar and a Sharp / Flat-key exemplar, and each active family exports to its own MIDI channel.

The reference case is the dual-piano video at `https://www.youtube.com/watch?v=7i9ZcXGk4ZI`, which visibly uses orange, blue, yellow, and purple note families. The video is a local acceptance input only and must not be committed or packaged.

## Scope

- Support one through four color families.
- Support up to eight lit exemplars: Natural and Sharp / Flat for each family.
- Discover additional families automatically during assisted calibration.
- Provide a compact manual Add/Remove fallback when discovery misses a family.
- Preserve each family as a separate MIDI channel.
- Preserve existing calibration and per-video configuration compatibility.
- Update every shipped locale and the deterministic UI audit after user-visible changes.

This feature does not support more than four families, multiple independently detected keyboard regions, RTL layout work, or new release platforms.

## Canonical Family Model

The user-facing model is:

| Family | Natural exemplar | Sharp / Flat exemplar | MIDI channel |
| --- | --- | --- | --- |
| Color 1 | `LW` | `LB` | 1 |
| Color 2 | `RW` | `RB` | 2 |
| Color 3 | `COLOR_3_W` | `COLOR_3_B` | 3 |
| Color 4 | `COLOR_4_W` | `COLOR_4_B` | 4 |

The legacy identifiers remain internal compatibility keys. Existing INI and JSON data using `LW`, `LB`, `RW`, and `RB` must load without migration or reinterpretation. User-visible strings no longer describe these slots as Left White, Left Black, Right White, and Right Black.

Physical keyboard overlays remain unchanged. An overlay's Natural or Sharp / Flat morphology determines which exemplar subset it checks. It does not receive a permanent family assignment during overlay generation.

## Family Identity

Rescanning must not silently reorder channels.

- Existing calibrated family colors are identity anchors.
- New clusters first match the nearest compatible saved family within the family-distance threshold.
- Unmatched stable clusters fill the first unused family number.
- A new configuration uses deterministic clustering and ordering so identical evidence produces identical family numbers.
- Removing a family clears both exemplars, its enabled state, and its channel assignment only after confirmation.

## Calibration Interface

Replace the current large full-width exemplar buttons with a compact dynamic grid. Each family has a short heading followed by two inline rows:

```text
Color 1
Natural       [swatch]  [Set]  [Present]
Sharp / Flat  [swatch]  [Set]  [Present]
```

- Show only detected, enabled, or saved families, with Color 1 always available.
- Ordinary two-family videos show four exemplar rows.
- Scanner-discovered Color 3 and Color 4 rows appear automatically.
- `Add Color Family` enables the first unused family up to the four-family cap.
- Additional families have a compact remove action with a tooltip and destructive confirmation when calibrated data exists.
- `Set` retains the current manual flow: pause on a lit frame, start capture, then click the highlighted key overlay.
- Each row keeps its own Present checkbox because a video may never use one key morphology for a family.
- The layout must remain usable under every shipped locale and 150% font scaling without nested horizontal scrolling.

The assisted-calibration review dialog reuses the same family-grid component in review mode. Found samples show their swatches; missing samples remain visible and actionable.

## Hybrid Scanner

Assisted calibration separates family discovery from detailed exemplar refinement.

### Lightweight discovery

- Continue checking every 10 frames across the requested video range.
- Sample inexpensive RGB/HSV color evidence from overlays without computing detailed histograms for every overlay and neighboring frame.
- Cluster stable evidence into at most four circular-hue families.
- Require repeated evidence across temporally separated checkpoints and prefer evidence spanning multiple keys.
- A single flash, intro animation, or isolated saturated frame cannot create a family.
- After the usual two families are complete, lightweight discovery continues through the end to catch a late Color 3 or Color 4.
- Stop all scanning immediately once four families each have complete, confirmed Natural and Sharp / Flat evidence.

### Targeted refinement

- When discovery produces a promising new family or improves a missing morphology, refine only a small neighborhood around those candidate frames.
- Capture the best RGB and histogram exemplar for each family/morphology pair.
- Once a family's detailed samples are complete, do not keep refining it unless materially stronger or identity-correcting evidence appears.
- Finding a new family after detailed work has quiesced temporarily reactivates refinement around that event.

The implementation should use operation-count and refinement-count assertions rather than brittle wall-clock tests to prove that ordinary two-family videos avoid full-video detailed refinement.

## Assignment and Readiness

- Proposals may contain one through four families and incomplete morphology pairs.
- A family observed with only one morphology still appears in review.
- Applying a proposal is non-destructive until the user confirms Apply.
- Canceling or closing review restores the exact previous enabled flags, colors, histograms, family identities, and channel behavior.
- A row marked Present but lacking an exemplar blocks conversion with a direct readiness message.
- The user can resolve it by manually setting the exemplar or marking that morphology not present.
- Finding Color 3 or Color 4 automatically enables separate color-family MIDI assignment.
- With separate assignment enabled, Colors 1 through 4 map to MIDI channels 1 through 4. Internal zero-based channel values remain 0 through 3.

## Detection and Conversion

Natural overlays compare only with enabled Natural exemplars. Sharp / Flat overlays compare only with enabled Sharp / Flat exemplars. Detection selects the strongest valid matching family while retaining the existing detection threshold and histogram behavior.

MIDI channel assignment uses the winning family identity rather than recomputing a separate unrestricted nearest-color decision. This keeps note detection and channel selection consistent when colors are close.

Diagnostic and conversion metadata must serialize dynamic family keys instead of hard-coding only the four legacy keys.

## Persistence and Compatibility

- Preserve existing INI section names and legacy keys.
- Round-trip `COLOR_3_W/B` and `COLOR_4_W/B` colors, histograms, enabled flags, and family metadata.
- Loading an old file produces the same two-family behavior it has today.
- Loading a one-family file may disable or hide Color 2 without deleting its compatibility slots.
- Saving a current file must not introduce unrelated calibration churn.

## Error Handling

- More than four plausible families: keep the four strongest stable families and show a concise review warning.
- Nearby hues within the family clustering threshold remain one family. Evidence that conflicts with two already anchored families marks the proposal for review instead of silently changing either channel identity.
- Incomplete family: show the missing row and require manual capture or Present opt-out.
- Failed rescan or cancellation: preserve the previous calibration atomically.
- Family removal: confirm only when data would be lost.
- Invalid saved family data: ignore the invalid sample, retain valid siblings, and surface normal readiness guidance.

## Verification

### Deterministic tests

- Synthetic one-, two-, three-, and four-family candidate sets.
- Natural and Sharp / Flat assignment for every family.
- A third or fourth family appearing near the end of the scan.
- Short note events at the existing 10-frame interval.
- Intro animations, one-frame flashes, repeated sustained notes, and nearby hues.
- Four-family completion stopping before the configured end.
- Two-family detailed refinement becoming quiescent while lightweight discovery continues.
- Stable family identity and channel numbering across rescans.
- Incomplete and not-present morphology handling.
- Proposal apply, cancel, and rollback behavior.
- Old INI compatibility plus new dynamic-family round trips.
- Four-family MIDI output on four distinct channels.

### UI and localization tests

- Compact family grid with one through four families.
- Add and remove behavior at the cap and lower bound.
- Assisted-review missing/found states.
- Conversion readiness messaging.
- Every production locale plus qps at shipped font scales.
- Deterministic UI string manifest and screenshot matrix.

### Acceptance and release gates

- Run the linked dual-piano video locally without tracking the video or generated frames.
- Confirm all four visible families receive Natural and Sharp / Flat exemplars or an explicit not-present resolution.
- Confirm exported notes use four MIDI channels.
- Run the full Python and Rust gates.
- Run Windows x64 and Apple Silicon macOS package smokes before tagging `v0.2.0`.

## Release Boundary

`v0.2.0` remains untagged until this feature is implemented, reviewed, merged to main, and both release packages pass smoke verification. Release notes should describe support for up to four note-color families in user-facing language rather than internal exemplar identifiers.
