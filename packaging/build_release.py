#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "synthesia2midi"
BUILD_ROOT = ROOT / "build" / "release"
DIST_ROOT = ROOT / "dist" / "release"
TOOLS_ROOT = ROOT / ".release-tools"
VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")
RUST_EDITOR_DIR = ROOT / "tools" / "midi_touchup_editor_rust"
RUST_EDITOR_BINARY = RUST_EDITOR_DIR / "target" / "release" / ("midi-touchup-editor.exe" if sys.platform.startswith("win") else "midi-touchup-editor")
SOUNDFONT = RUST_EDITOR_DIR / "assets" / "soundfonts" / "TouchUpPiano.sf2"
SOUNDFONT_LICENSE = RUST_EDITOR_DIR / "assets" / "soundfonts" / "TouchUpPiano_LICENSE.txt"
THIRD_PARTY_NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"
LICENSE_FILE = ROOT / "LICENSE"
README_FILE = ROOT / "README.md"

sys.path.insert(0, str(PACKAGE_ROOT))
from synthesia2midi.version import DEFAULT_APP_VERSION, RELEASE_APP_NAME, normalize_release_version  # noqa: E402


class ReleaseBuildError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a portable Synthesia2MIDI release bundle")
    parser.add_argument("--version", help="Release tag or version, for example v0.1.0 or 0.1.0")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip launching the packaged app")
    parser.add_argument("--keep-pyinstaller-output", action="store_true", help="Keep raw PyInstaller dist output")
    return parser.parse_args()


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("$", " ".join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], cwd=cwd, env=env, check=True)


def require_file(path: Path, message: str) -> Path:
    if not path.is_file():
        raise ReleaseBuildError(message)
    return path


def normalize_build_version(raw_version: str | None) -> tuple[str, str]:
    candidates = (
        raw_version,
        os.getenv("S2M_RELEASE_VERSION"),
        os.getenv("GITHUB_REF_NAME"),
        os.getenv("GITHUB_REF"),
    )
    for candidate in candidates:
        normalized = normalize_release_version(candidate)
        if normalized:
            return normalized, f"v{normalized}"
    normalized = normalize_release_version(DEFAULT_APP_VERSION)
    if not normalized:
        raise ReleaseBuildError(f"Default version is invalid: {DEFAULT_APP_VERSION}")
    return normalized, "local-dev"


def platform_slug() -> str:
    if sys.platform.startswith("win"):
        return "windows-x64"
    if sys.platform == "darwin":
        return "macos-arm64"
    raise ReleaseBuildError(f"Unsupported release platform: {sys.platform}")


def archive_stem(tag_label: str) -> str:
    return f"{RELEASE_APP_NAME}-{platform_slug()}-{tag_label}"


def ensure_venv_python() -> Path:
    return require_file(VENV_PYTHON, "Missing .venv Python. Run setup_env.py first.")


def ensure_pyinstaller(venv_python: Path) -> None:
    run([str(venv_python), "-m", "pip", "install", "pyinstaller"])


def ensure_rust_editor() -> Path:
    run(["cargo", "build", "--release"], cwd=RUST_EDITOR_DIR)
    return require_file(RUST_EDITOR_BINARY, f"Rust editor binary was not built: {RUST_EDITOR_BINARY}")


def ensure_ffmpeg_binary(name: str) -> Path:
    resolved = shutil.which(name)
    if not resolved:
        raise ReleaseBuildError(f"Required binary `{name}` was not found on PATH.")
    return Path(resolved).resolve()


def _binary_name(stem: str) -> str:
    return f"{stem}.exe" if sys.platform.startswith("win") else stem


def stage_binary(source: Path, stem: str) -> Path:
    staged_dir = BUILD_ROOT / "staged-bin"
    staged_dir.mkdir(parents=True, exist_ok=True)
    destination = staged_dir / _binary_name(stem)
    shutil.copy2(source, destination)
    if not sys.platform.startswith("win"):
        destination.chmod(0o755)
    return destination


def install_deno_with_script(install_root: Path) -> Path:
    install_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["DENO_INSTALL"] = str(install_root)
    if sys.platform.startswith("win"):
        run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "iwr https://deno.land/install.ps1 -useb | iex",
            ],
            env=env,
        )
        deno_path = install_root / "bin" / "deno.exe"
    else:
        run(["sh", "-c", "curl -fsSL https://deno.land/install.sh | sh"], env=env)
        deno_path = install_root / "bin" / "deno"
        if deno_path.exists():
            deno_path.chmod(0o755)
    return require_file(deno_path, f"Deno install script completed but no binary was found at {deno_path}")


def ensure_deno() -> Path:
    resolved = shutil.which("deno")
    if resolved:
        return Path(resolved).resolve()
    install_root = TOOLS_ROOT / "deno"
    cached_binary = install_root / "bin" / _binary_name("deno")
    if cached_binary.is_file():
        return cached_binary
    return install_deno_with_script(install_root)


def write_build_version_file(version: str) -> Path:
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    build_version_file = BUILD_ROOT / "build_version.txt"
    build_version_file.write_text(f"{version}\n", encoding="utf-8")
    return build_version_file


def pyinstaller_dist_dir() -> Path:
    return BUILD_ROOT / "pyinstaller-dist"


def pyinstaller_work_dir() -> Path:
    return BUILD_ROOT / "pyinstaller-work"


def clean_output_dirs() -> None:
    shutil.rmtree(BUILD_ROOT, ignore_errors=True)
    DIST_ROOT.mkdir(parents=True, exist_ok=True)


def run_pyinstaller(
    venv_python: Path,
    *,
    version: str,
    ffmpeg: Path,
    ffprobe: Path,
    deno: Path,
    rust_editor: Path,
    build_version_file: Path,
) -> None:
    cmd = [
        str(venv_python),
        "-m",
        "PyInstaller",
        str(ROOT / "packaging" / "Synthesia2MIDI.spec"),
        "--noconfirm",
        "--clean",
        "--distpath",
        str(pyinstaller_dist_dir()),
        "--workpath",
        str(pyinstaller_work_dir()),
        "--",
        "--version",
        version,
        "--ffmpeg",
        str(ffmpeg),
        "--ffprobe",
        str(ffprobe),
        "--deno",
        str(deno),
        "--rust-editor",
        str(rust_editor),
        "--build-version-file",
        str(build_version_file),
    ]
    run(cmd, cwd=ROOT)


def raw_bundle_root() -> Path:
    return pyinstaller_dist_dir() / RELEASE_APP_NAME


def raw_mac_app() -> Path:
    return pyinstaller_dist_dir() / f"{RELEASE_APP_NAME}.app"


def stage_release_bundle(tag_label: str) -> tuple[Path, Path]:
    stem = archive_stem(tag_label)
    stage_dir = DIST_ROOT / stem
    shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    if sys.platform == "darwin":
        bundle_source = raw_mac_app()
        if not bundle_source.exists():
            raise ReleaseBuildError(f"Expected macOS app bundle was not created: {bundle_source}")
        shutil.copytree(bundle_source, stage_dir / bundle_source.name)
    else:
        bundle_source = raw_bundle_root()
        if not bundle_source.exists():
            raise ReleaseBuildError(f"Expected release folder was not created: {bundle_source}")
        shutil.copytree(bundle_source, stage_dir / bundle_source.name)

    shutil.copy2(LICENSE_FILE, stage_dir / LICENSE_FILE.name)
    shutil.copy2(THIRD_PARTY_NOTICES, stage_dir / THIRD_PARTY_NOTICES.name)
    shutil.copy2(README_FILE, stage_dir / README_FILE.name)

    archive_path = Path(shutil.make_archive(str(stage_dir), "zip", DIST_ROOT, stage_dir.name))
    return stage_dir, archive_path


def smoke_executable_path(stage_dir: Path) -> Path:
    if sys.platform == "darwin":
        return stage_dir / f"{RELEASE_APP_NAME}.app" / "Contents" / "MacOS" / RELEASE_APP_NAME
    return stage_dir / RELEASE_APP_NAME / f"{RELEASE_APP_NAME}.exe"


def smoke_launch(stage_dir: Path) -> None:
    executable = smoke_executable_path(stage_dir)
    require_file(executable, f"Smoke-launch target is missing: {executable}")
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    process = subprocess.Popen(
        [str(executable)],
        cwd=stage_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(8)
    exit_code = process.poll()
    if exit_code is not None and exit_code != 0:
        output = (process.stdout.read() or "").strip()
        raise ReleaseBuildError(f"Packaged app exited early with code {exit_code}.\n{output}")
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    args = parse_args()
    version, tag_label = normalize_build_version(args.version)
    venv_python = ensure_venv_python()
    clean_output_dirs()
    ensure_pyinstaller(venv_python)
    rust_editor = ensure_rust_editor()
    ffmpeg = stage_binary(ensure_ffmpeg_binary("ffmpeg"), "ffmpeg")
    ffprobe = stage_binary(ensure_ffmpeg_binary("ffprobe"), "ffprobe")
    deno = stage_binary(ensure_deno(), "deno")
    rust_editor = stage_binary(rust_editor, "midi-touchup-editor")
    build_version_file = write_build_version_file(version)
    require_file(SOUNDFONT, f"Bundled soundfont is missing: {SOUNDFONT}")
    require_file(SOUNDFONT_LICENSE, f"Bundled soundfont license is missing: {SOUNDFONT_LICENSE}")
    run_pyinstaller(
        venv_python,
        version=version,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        deno=deno,
        rust_editor=rust_editor,
        build_version_file=build_version_file,
    )
    stage_dir, archive_path = stage_release_bundle(tag_label)
    if not args.skip_smoke:
        smoke_launch(stage_dir)
    if not args.keep_pyinstaller_output:
        shutil.rmtree(pyinstaller_dist_dir(), ignore_errors=True)
    print(f"Release bundle: {stage_dir}")
    print(f"Release archive: {archive_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReleaseBuildError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
