"""Apply structured translation JSON to a Qt `.ts` catalog."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence
import xml.etree.ElementTree as ET


PLACEHOLDER_RE = re.compile(r"\{[^{}]+\}")


class TranslationCatalogError(ValueError):
    """Raised when translation JSON does not match the Qt source catalog."""


def _message_key(context_name: str, message: ET.Element) -> tuple[str, str] | None:
    source = message.findtext("source") or ""
    if not source:
        return None
    translation = message.find("translation")
    if translation is not None and translation.get("type") == "vanished":
        return None
    return context_name, source


def _source_keys(root: ET.Element) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for context in root.findall("context"):
        context_name = context.findtext("name") or ""
        for message in context.findall("message"):
            key = _message_key(context_name, message)
            if key is not None:
                keys.add(key)
    return keys


def _load_translations(translations_json: Path) -> dict[tuple[str, str], str]:
    raw_entries = json.loads(translations_json.read_text(encoding="utf-8"))
    if not isinstance(raw_entries, list):
        raise TranslationCatalogError("Translation JSON must be an array")

    translations: dict[tuple[str, str], str] = {}
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise TranslationCatalogError(f"Entry {index} must be an object")

        context_name = entry.get("context")
        source = entry.get("source")
        translation = entry.get("translation")
        if not isinstance(context_name, str) or not isinstance(source, str):
            raise TranslationCatalogError(f"Entry {index} must include string context and source")
        if not isinstance(translation, str) or not translation.strip():
            raise TranslationCatalogError(f"Entry {index} must include a non-empty translation")

        key = (context_name, source)
        if key in translations:
            raise TranslationCatalogError(f"Duplicate translation for {context_name!r}: {source!r}")

        source_placeholders = sorted(PLACEHOLDER_RE.findall(source))
        translated_placeholders = sorted(PLACEHOLDER_RE.findall(translation))
        if source_placeholders != translated_placeholders:
            raise TranslationCatalogError(
                "Placeholder mismatch for "
                f"{context_name!r}: {source!r} -> {translation!r}"
            )

        translations[key] = translation

    return translations


def _validate_coverage(
    source_keys: set[tuple[str, str]],
    translations: dict[tuple[str, str], str],
) -> None:
    translation_keys = set(translations)
    missing = sorted(source_keys - translation_keys)
    extra = sorted(translation_keys - source_keys)
    if missing:
        context_name, source = missing[0]
        raise TranslationCatalogError(
            f"Missing {len(missing)} translations; first missing is {context_name!r}: {source!r}"
        )
    if extra:
        context_name, source = extra[0]
        raise TranslationCatalogError(
            f"Found {len(extra)} extra translations; first extra is {context_name!r}: {source!r}"
        )


def apply_translation_json(
    *,
    source_ts: Path,
    translations_json: Path,
    output_ts: Path,
    language_code: str,
) -> None:
    """Write a translated Qt `.ts` file from source catalog and agent JSON."""
    tree = ET.parse(source_ts)
    root = tree.getroot()
    translations = _load_translations(translations_json)
    keys = _source_keys(root)
    _validate_coverage(keys, translations)

    root.set("language", language_code)
    for context in root.findall("context"):
        context_name = context.findtext("name") or ""
        for message in context.findall("message"):
            key = _message_key(context_name, message)
            if key is None:
                continue
            translation = message.find("translation")
            if translation is None:
                translation = ET.SubElement(message, "translation")
            translation.attrib.pop("type", None)
            translation.text = translations[key]

    ET.indent(tree, space="  ")
    output_ts.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_ts, encoding="utf-8", xml_declaration=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ts", type=Path, required=True)
    parser.add_argument("--translations-json", type=Path, required=True)
    parser.add_argument("--output-ts", type=Path, required=True)
    parser.add_argument("--language-code", required=True)
    args = parser.parse_args(argv)

    apply_translation_json(
        source_ts=args.source_ts,
        translations_json=args.translations_json,
        output_ts=args.output_ts,
        language_code=args.language_code,
    )
    print(f"Wrote {args.language_code} translations to {args.output_ts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
