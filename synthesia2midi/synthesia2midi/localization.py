"""Qt translation loading for app-visible UI text."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

from PySide6.QtCore import QCoreApplication, QTranslator
from PySide6.QtWidgets import QApplication


_INSTALLED_TRANSLATOR: QTranslator | None = None


class _PseudoTranslator(QTranslator):
    """Development translator that makes untranslated UI text visible."""

    _CHAR_MAP = str.maketrans(
        {
            "a": "à",
            "e": "è",
            "i": "ì",
            "o": "ò",
            "u": "ù",
            "A": "À",
            "E": "È",
            "I": "Ì",
            "O": "Ò",
            "U": "Ù",
        }
    )

    def isEmpty(self) -> bool:  # noqa: N802 - Qt override
        return False

    def translate(  # noqa: D401 - Qt override
        self,
        context: str,
        sourceText: str,
        disambiguation: str | None = None,
        n: int = -1,
    ) -> str:
        """Return a visibly transformed pseudo translation."""
        if not sourceText:
            return ""
        placeholders: list[str] = []

        def preserve_placeholder(match: re.Match[str]) -> str:
            placeholders.append(match.group(0))
            return f"__S2M_PLACEHOLDER_{len(placeholders) - 1}__"

        protected = re.sub(r"\{[^{}]*\}", preserve_placeholder, sourceText)
        translated = protected.translate(self._CHAR_MAP)
        for index, placeholder in enumerate(placeholders):
            translated = translated.replace(f"__S2M_PLÀCÈHÒLDÈR_{index}__", placeholder)
        return f"[!! {translated} !!]"


def translation_dir() -> Path:
    """Return the package translation directory for source and packaged runs."""
    return Path(__file__).resolve().parent / "translations"


def available_locales() -> list[str]:
    """Return supported locale names known to the app."""
    locales = {"en", "qps"}
    translations = translation_dir()
    if translations.is_dir():
        for file_path in translations.glob("synthesia2midi_*.qm"):
            locales.add(file_path.stem.removeprefix("synthesia2midi_"))
    return sorted(locales)


def _candidate_translation_files(locale_name: str) -> Iterable[Path]:
    translations = translation_dir()
    yield translations / f"synthesia2midi_{locale_name}.qm"
    yield translations / f"{locale_name}.qm"


def install_translator(app: QApplication, locale_name: str | None = None) -> str:
    """Install a Qt translator and return the selected locale name."""
    global _INSTALLED_TRANSLATOR

    if _INSTALLED_TRANSLATOR is not None:
        app.removeTranslator(_INSTALLED_TRANSLATOR)
        _INSTALLED_TRANSLATOR = None

    requested = (locale_name or "en").strip() or "en"
    if requested == "en":
        return "en"

    if requested == "qps":
        translator: QTranslator = _PseudoTranslator(app)
        app.installTranslator(translator)
        _INSTALLED_TRANSLATOR = translator
        return "qps"

    for file_path in _candidate_translation_files(requested):
        if not file_path.is_file():
            continue
        translator = QTranslator(app)
        if translator.load(str(file_path)):
            app.installTranslator(translator)
            _INSTALLED_TRANSLATOR = translator
            return requested

    return "en"


def tr(context: str, source_text: str) -> str:
    """Translate text outside QObject subclasses."""
    return QCoreApplication.translate(context, source_text)
