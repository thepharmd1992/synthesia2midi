"""Audit app-visible Qt UI strings for localization coverage."""
from __future__ import annotations

import argparse
import ast
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QTabWidget,
    QWidget,
)


SCHEMA_VERSION = 1
DEFAULT_MANIFEST_PATH = Path("docs/localization/ui-string-manifest.json")

QT_CONSTRUCTORS = {
    "QAction",
    "QCheckBox",
    "QGroupBox",
    "QLabel",
    "QProgressDialog",
    "QPushButton",
    "QRadioButton",
    "QToolButton",
}
UI_METHODS = {
    "addAction",
    "addButton",
    "addItem",
    "addMenu",
    "addTab",
    "setCancelButtonText",
    "setDetailedText",
    "setInformativeText",
    "setLabelText",
    "setPlaceholderText",
    "setStatusTip",
    "setText",
    "setTitle",
    "setToolTip",
    "setWindowTitle",
}
MESSAGE_BOX_METHODS = {"about", "critical", "information", "question", "warning"}
FILE_DIALOG_METHODS = {
    "getExistingDirectory",
    "getOpenFileName",
    "getOpenFileNames",
    "getSaveFileName",
}
BRAND_TERMS = {
    "Chrome",
    "Edge",
    "FFmpeg",
    "MIDI",
    "PySide6",
    "Safari",
    "Synthesia2MIDI",
    "YouTube",
}
NON_TRANSLATABLE_TEXT = {
    "",
    "+",
    "-",
    "0",
    "30 FPS",
    "60 FPS",
    "1080p",
    "720p",
    "480p",
    "⚙",
}


@dataclass(frozen=True)
class UiStringCandidate:
    """One potentially user-visible string discovered by the audit."""

    text: str
    classification: str
    origin: str
    source: str
    line: int
    context: str
    role: str


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = _call_name(func.value)
        return f"{base}.{func.attr}" if base else func.attr
    return ""


def _text_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                parts.append("{...}")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _text_of(node.left)
        right = _text_of(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _is_qt_visible_call(name: str) -> str | None:
    leaf = name.rsplit(".", 1)[-1]
    if leaf in QT_CONSTRUCTORS:
        return leaf
    if leaf in UI_METHODS:
        return leaf
    if name.startswith("QMessageBox.") and leaf in MESSAGE_BOX_METHODS:
        return f"QMessageBox.{leaf}"
    if name.startswith("QFileDialog.") and leaf in FILE_DIALOG_METHODS:
        return f"QFileDialog.{leaf}"
    if name == "QCoreApplication.translate":
        return name
    return None


def classify_text(text: str, *, context: str = "", role: str = "", origin: str = "static") -> str:
    """Classify a candidate string for localization review."""
    stripped = text.strip()
    lowered = stripped.lower()
    if stripped in NON_TRANSLATABLE_TEXT:
        return "do_not_translate"
    if origin == "runtime" and (
        stripped.startswith(("/", "~"))
        or Path(stripped).suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".mid", ".midi"}
    ):
        return "dynamic_user_data"
    if "://" in stripped or stripped.startswith(("*.","/")):
        return "path_or_url"
    if role == "arg1" and context == "addItem":
        return "technical_id"
    if stripped in BRAND_TERMS:
        return "do_not_translate"
    if lowered in {"chrome", "edge", "safari", "black", "white", "all"}:
        return "technical_id"
    if stripped.startswith("<") and stripped.endswith(">"):
        return "technical_id"
    return "translate"


def _candidate(
    *,
    text: str,
    origin: str,
    source: str,
    line: int,
    context: str,
    role: str,
) -> UiStringCandidate | None:
    stripped = text.strip()
    if stripped in NON_TRANSLATABLE_TEXT:
        return None
    return UiStringCandidate(
        text=stripped,
        classification=classify_text(stripped, context=context, role=role, origin=origin),
        origin=origin,
        source=source,
        line=line,
        context=context,
        role=role,
    )


def collect_static_candidates(paths: Iterable[Path], *, root: Path) -> list[UiStringCandidate]:
    """Collect likely Qt-visible string literals from Python source files."""
    candidates: list[UiStringCandidate] = []
    for path in sorted(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        source = str(path.relative_to(root))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _call_name(node.func)
            context = _is_qt_visible_call(call_name)
            if context is None:
                continue
            for index, arg in enumerate(node.args):
                if context == "QCoreApplication.translate" and index != 1:
                    continue
                text = _text_of(arg)
                if text is None:
                    continue
                candidate = _candidate(
                    text=text,
                    origin="static",
                    source=source,
                    line=node.lineno,
                    context=context,
                    role=f"arg{index}",
                )
                if candidate is not None:
                    candidates.append(candidate)
            for keyword in node.keywords:
                text = _text_of(keyword.value)
                if text is None:
                    continue
                candidate = _candidate(
                    text=text,
                    origin="static",
                    source=source,
                    line=node.lineno,
                    context=context,
                    role=f"kw:{keyword.arg}",
                )
                if candidate is not None:
                    candidates.append(candidate)
    return _dedupe_candidates(candidates)


def _add_runtime_candidate(
    candidates: list[UiStringCandidate],
    *,
    text: str,
    source: str,
    context: str,
    role: str,
) -> None:
    candidate = _candidate(
        text=text,
        origin="runtime",
        source=source,
        line=0,
        context=context,
        role=role,
    )
    if candidate is not None:
        candidates.append(candidate)


def collect_widget_text(widget: QWidget) -> list[UiStringCandidate]:
    """Collect currently visible text from a Qt widget tree."""
    candidates: list[UiStringCandidate] = []
    source = type(widget).__name__
    _add_runtime_candidate(
        candidates,
        text=widget.windowTitle(),
        source=source,
        context=source,
        role="windowTitle",
    )
    for action in widget.findChildren(QAction):
        _add_runtime_candidate(
            candidates,
            text=action.text(),
            source=source,
            context=type(action).__name__,
            role="action.text",
        )
        _add_runtime_candidate(
            candidates,
            text=action.toolTip(),
            source=source,
            context=type(action).__name__,
            role="action.tooltip",
        )
    for child in widget.findChildren(QWidget):
        _add_runtime_candidate(
            candidates,
            text=child.toolTip(),
            source=source,
            context=type(child).__name__,
            role="tooltip",
        )
        if isinstance(child, QLabel):
            _add_runtime_candidate(
                candidates,
                text=child.text(),
                source=source,
                context=type(child).__name__,
                role="label.text",
            )
        if isinstance(child, QAbstractButton):
            _add_runtime_candidate(
                candidates,
                text=child.text(),
                source=source,
                context=type(child).__name__,
                role="button.text",
            )
        if isinstance(child, QGroupBox):
            _add_runtime_candidate(
                candidates,
                text=child.title(),
                source=source,
                context=type(child).__name__,
                role="group.title",
            )
        if isinstance(child, QLineEdit):
            _add_runtime_candidate(
                candidates,
                text=child.placeholderText(),
                source=source,
                context=type(child).__name__,
                role="placeholder",
            )
        if isinstance(child, QComboBox):
            for index in range(child.count()):
                _add_runtime_candidate(
                    candidates,
                    text=child.itemText(index),
                    source=source,
                    context=type(child).__name__,
                    role=f"combo.item{index}",
                )
        if isinstance(child, QTabWidget):
            for index in range(child.count()):
                _add_runtime_candidate(
                    candidates,
                    text=child.tabText(index),
                    source=source,
                    context=type(child).__name__,
                    role=f"tab.text{index}",
                )
    return _dedupe_candidates(candidates)


def _dedupe_candidates(candidates: Sequence[UiStringCandidate]) -> list[UiStringCandidate]:
    deduped: dict[tuple[str, str, str, str, str], UiStringCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.text,
            candidate.classification,
            candidate.origin,
            candidate.source,
            candidate.role,
        )
        deduped.setdefault(key, candidate)
    return sorted(
        deduped.values(),
        key=lambda item: (item.classification, item.text.lower(), item.source, item.line),
    )


def write_manifest(candidates: Sequence[UiStringCandidate], output_path: Path) -> None:
    """Write a stable JSON manifest for review and CI checks."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "counts": _counts(candidates),
        "candidates": [asdict(candidate) for candidate in _dedupe_candidates(candidates)],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _counts(candidates: Sequence[UiStringCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.classification] = counts.get(candidate.classification, 0) + 1
    return dict(sorted(counts.items()))


def _source_paths(root: Path) -> list[Path]:
    package_root = root / "synthesia2midi" / "synthesia2midi"
    return [
        path
        for path in package_root.rglob("*.py")
        if "__pycache__" not in path.parts and path.name != "audit_ui_strings.py"
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Synthesia2MIDI app-visible UI strings.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args(argv)

    root = args.root.resolve()
    output = args.output
    if not output.is_absolute():
        output = root / output

    candidates = collect_static_candidates(_source_paths(root), root=root)
    write_manifest(candidates, output)
    print(f"Wrote {len(candidates)} UI string candidates to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
