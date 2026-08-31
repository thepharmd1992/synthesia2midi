"""Platform-native helper payload validation for packaged releases."""
from __future__ import annotations

from pathlib import Path


_MACHO_MAGICS = {
    b"\xca\xfe\xba\xbe",  # universal, big endian
    b"\xbe\xba\xfe\xca",  # universal, little endian
    b"\xca\xfe\xba\xbf",  # universal 64-bit, big endian
    b"\xbf\xba\xfe\xca",  # universal 64-bit, little endian
    b"\xfe\xed\xfa\xce",  # Mach-O 32-bit, big endian
    b"\xce\xfa\xed\xfe",  # Mach-O 32-bit, little endian
    b"\xfe\xed\xfa\xcf",  # Mach-O 64-bit, big endian
    b"\xcf\xfa\xed\xfe",  # Mach-O 64-bit, little endian
}
_INSPECTION_BYTES = 1024 * 1024


def native_binary_issue(path: Path, platform_name: str) -> str | None:
    """Return why ``path`` is not a redistributable native executable."""
    try:
        with path.open("rb") as handle:
            payload = handle.read(_INSPECTION_BYTES)
    except OSError as exc:
        return f"could not inspect binary payload: {exc}"

    if platform_name.startswith("win"):
        if not payload.startswith(b"MZ"):
            return "payload is not a Windows PE executable"
        searchable = payload.replace(b"\x00", b"").lower()
        if b"shimgen" in searchable and b"chocolatey" in searchable:
            return "payload is a Chocolatey ShimGen launcher, not the real executable"
        return None

    if platform_name == "darwin":
        if payload[:4] not in _MACHO_MAGICS:
            return "payload is not a Mach-O executable"
        return None

    if platform_name.startswith("linux"):
        if not payload.startswith(b"\x7fELF"):
            return "payload is not an ELF executable"
        return None

    return f"unsupported release platform: {platform_name}"
