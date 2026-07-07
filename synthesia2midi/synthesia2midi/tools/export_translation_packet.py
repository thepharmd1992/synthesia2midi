"""Export Qt source texts as a structured packet for translation agents."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence
import xml.etree.ElementTree as ET


DEFAULT_SOURCE_TS = Path("synthesia2midi/synthesia2midi/translations/synthesia2midi_es.ts")
DEFAULT_OUTPUT = Path("docs/localization/translation-agent-packet.json")
PLACEHOLDER_RE = re.compile(r"\{[^{}]+\}")


def _message_locations(message: ET.Element) -> list[dict[str, object]]:
    locations = []
    for location in message.findall("location"):
        item: dict[str, object] = {}
        filename = location.get("filename")
        line = location.get("line")
        if filename:
            item["filename"] = filename
        if line:
            try:
                item["line"] = int(line)
            except ValueError:
                item["line"] = line
        if item:
            locations.append(item)
    return locations


def build_packet(source_ts: Path) -> dict[str, object]:
    """Build the reusable translation packet from a Qt `.ts` file."""
    root = ET.parse(source_ts).getroot()
    entries = []
    for context in root.findall("context"):
        context_name = context.findtext("name") or ""
        for message in context.findall("message"):
            source = message.findtext("source") or ""
            if not source:
                continue
            translation = message.find("translation")
            if translation is not None and translation.get("type") == "vanished":
                continue
            entries.append(
                {
                    "context": context_name,
                    "source": source,
                    "placeholders": sorted(PLACEHOLDER_RE.findall(source)),
                    "locations": _message_locations(message),
                    "notes": "",
                }
            )

    return {
        "schema_version": 1,
        "app_context": (
            "Synthesia2MIDI is a PySide6 desktop app that analyzes Synthesia piano "
            "videos, calibrates piano-key overlays, detects notes, downloads optional "
            "YouTube videos, and exports MIDI files."
        ),
        "ui_context": [
            "startup video-source selection",
            "language selection",
            "keyboard calibration",
            "overlay alignment and Manual Fit",
            "detection and Spark Detection tuning",
            "MIDI conversion and touch-up editor launch",
            "YouTube download and fallback dialogs",
        ],
        "glossary_preserve": [
            "Synthesia2MIDI",
            "Synthesia",
            "MIDI",
            "FFmpeg",
            "FFprobe",
            "YouTube",
            "Rust",
            "yt-dlp",
            "Deno",
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm",
            ".mid",
            ".midi",
            "LW",
            "LB",
            "RW",
            "RB",
        ],
        "instructions": [
            "Translate user-visible UI text naturally for native speakers.",
            "Keep translations concise enough for buttons, labels, menus, and dialogs.",
            "Preserve placeholders exactly, including braces and names.",
            "Preserve URLs, file paths, file extensions, product names, and technical IDs.",
            "Keep HTML tags and line-break structure valid.",
            "Use notes only when a source string is ambiguous or needs product context.",
        ],
        "required_output_schema": {
            "context": "Qt context name copied from the packet entry",
            "source": "English source text copied exactly from the packet entry",
            "translation": "Translated UI text",
            "notes": "Optional translator note, otherwise empty string",
        },
        "entries": entries,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ts", type=Path, default=DEFAULT_SOURCE_TS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    packet = build_packet(args.source_ts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(packet['entries'])} translation entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
