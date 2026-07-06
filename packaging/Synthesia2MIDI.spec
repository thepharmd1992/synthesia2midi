from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


ROOT = Path(SPECPATH).resolve().parent
PACKAGE_ROOT = ROOT / "synthesia2midi"
sys.path.insert(0, str(PACKAGE_ROOT))

from synthesia2midi.version import RELEASE_APP_NAME  # noqa: E402


def _parse_build_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--deno", required=True)
    parser.add_argument("--rust-editor", required=True)
    parser.add_argument("--build-version-file", required=True)
    args = parser.parse_args()

    for attribute in ("ffmpeg", "ffprobe", "deno", "rust_editor", "build_version_file"):
        candidate = Path(getattr(args, attribute)).resolve()
        if not candidate.is_file():
            raise SystemExit(f"Missing required build input: {candidate}")
        setattr(args, attribute, candidate)
    return args


build_options = _parse_build_options()
is_macos = sys.platform == "darwin"

datas = collect_data_files("certifi")
datas.extend(collect_data_files("synthesia2midi", includes=["translations/*"]))
datas.extend(
    [
        (str(ROOT / "tools" / "midi_touchup_editor_rust" / "assets" / "soundfonts" / "TouchUpPiano.sf2"), "assets/soundfonts"),
        (str(ROOT / "tools" / "midi_touchup_editor_rust" / "assets" / "soundfonts" / "TouchUpPiano_LICENSE.txt"), "assets/soundfonts"),
        (str(build_options.build_version_file), "synthesia2midi"),
    ]
)

template_ini = ROOT / "synthesia2midi" / "my_immortal.ini"
if template_ini.is_file():
    datas.append((str(template_ini), "."))

binaries = [
    (str(build_options.ffmpeg), "bin"),
    (str(build_options.ffprobe), "bin"),
    (str(build_options.deno), "bin"),
    (str(build_options.rust_editor), "bin"),
]

hiddenimports = collect_submodules("yt_dlp")

a = Analysis(
    [str(ROOT / "synthesia2midi" / "run.py")],
    pathex=[str(PACKAGE_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=RELEASE_APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=RELEASE_APP_NAME,
)

if is_macos:
    app = BUNDLE(
        coll,
        name=f"{RELEASE_APP_NAME}.app",
        icon=None,
        bundle_identifier="com.synthesia2midi.app",
        info_plist={
            "CFBundleShortVersionString": build_options.version,
            "CFBundleVersion": build_options.version,
            "NSHighResolutionCapable": True,
        },
    )
