---
id: doc-2
title: Product Frame
type: guide
created_date: '2026-06-08 23:53'
---

# Product Frame

Synthesia2MIDI is a PySide6 desktop app that analyzes Synthesia piano videos and exports MIDI.

## Core User Flow

Users load a local video or download a YouTube video, calibrate visible piano overlays, tune detection, convert detected notes to MIDI, and optionally edit the MIDI output.

## Product Priorities

- Keep calibration and overlay fitting understandable for real users.
- Preserve MIDI correctness over visual cleverness.
- Make manual recovery workflows practical when auto-detection fails.
- Keep video, overlay, detection, and MIDI behavior compatible with existing per-video configuration unless a task explicitly includes migration.
- Keep UI modes explicit. Avoid ambiguous gesture inference.

## Current High-Value Surfaces

- Video source loading and recent-video workflow.
- YouTube download and quality selection.
- Calibration wizard and auto detector.
- Manual overlay generation and Manual Fit.
- Detection, spark detection, trim, MIDI conversion, and MIDI edit flow.

## Open Product Pressure

Manual Fit is becoming the main recovery path for difficult videos. Future work should reduce user guesswork, keep controls named in end-user language, and verify that visual overlay geometry matches detection geometry.
