#!/usr/bin/env python3
"""Cross-platform setup for Synthesia2MIDI.

This intentionally stays boring: create/update the repo-local virtual
environment, install Python dependencies, require FFmpeg, and build the optional
Rust touch-up editor when Cargo is available.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

MIN_PYTHON = (3, 10)
ROOT_DIR = Path(__file__).resolve().parent
VENV_DIR = ROOT_DIR / ".venv"
REQUIREMENTS = ROOT_DIR / "synthesia2midi" / "requirements.txt"
RUST_EDITOR_DIR = ROOT_DIR / "tools" / "midi_touchup_editor_rust"


class SetupError(RuntimeError):
    """Raised for setup failures with user-facing messages."""


def venv_python_path(venv_dir: Path, platform_name: str | None = None) -> Path:
    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def user_python_command(platform_name: str | None = None) -> str:
    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        return "py"
    return "python3"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up Synthesia2MIDI")
    parser.add_argument(
        "--recreate-venv",
        action="store_true",
        help="Delete and recreate .venv before installing dependencies.",
    )
    parser.add_argument(
        "--skip-rust",
        action="store_true",
        help="Skip building the Rust MIDI touch-up editor.",
    )
    parser.add_argument(
        "--strict-rust",
        action="store_true",
        help="Fail setup if Cargo is missing or the Rust touch-up build fails.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check the environment without installing Python packages or building Rust.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Install pytest and ruff for local verification.",
    )
    parser.set_defaults(require_ffmpeg=True)
    return parser.parse_args(argv)


def ffmpeg_install_hint(platform_name: str | None = None) -> str:
    platform_name = platform_name or sys.platform
    lines = [
        "FFmpeg is required for Synthesia2MIDI.",
        "Install FFmpeg and make sure the `ffmpeg` command is available on PATH.",
        "",
    ]
    if platform_name.startswith("win"):
        lines.extend(
            [
                "Windows:",
                "  winget install Gyan.FFmpeg",
                "  Then close/reopen your terminal and run setup again.",
            ]
        )
    elif platform_name == "darwin":
        lines.extend(
            [
                "macOS:",
                "  brew install ffmpeg",
            ]
        )
    else:
        lines.extend(
            [
                "Linux:",
                "  sudo apt install ffmpeg      # Debian/Ubuntu",
                "  sudo dnf install ffmpeg      # Fedora",
                "  sudo pacman -S ffmpeg        # Arch",
            ]
        )
    return "\n".join(lines)


def rust_install_hint(platform_name: str | None = None) -> str:
    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        return "Install Rust with: winget install --id Rustlang.Rustup -e"
    return "Install Rust with: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"


def run_command(cmd: Sequence[str | os.PathLike[str]], *, cwd: Path | None = None) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"$ {printable}")
    subprocess.run([str(part) for part in cmd], cwd=cwd, check=True)


def ensure_python_version() -> None:
    if sys.version_info < MIN_PYTHON:
        required = ".".join(str(part) for part in MIN_PYTHON)
        raise SetupError(f"Python {required}+ is required. Current Python: {sys.version.split()[0]}")


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        return
    raise SetupError(ffmpeg_install_hint())


def create_or_reuse_venv(*, recreate: bool) -> Path:
    venv_python = venv_python_path(VENV_DIR)

    if recreate and VENV_DIR.exists():
        current_python = Path(sys.executable).resolve()
        try:
            current_python.relative_to(VENV_DIR.resolve())
        except ValueError:
            pass
        else:
            raise SetupError("Cannot recreate .venv while running from inside that .venv.")
        print("Removing existing .venv...")
        shutil.rmtree(VENV_DIR)

    if not venv_python.exists():
        print("Creating .venv...")
        run_command([sys.executable, "-m", "venv", VENV_DIR])
    else:
        print("Using existing .venv.")

    return venv_python


def install_python_dependencies(venv_python: Path, *, dev: bool) -> None:
    if not REQUIREMENTS.exists():
        raise SetupError(f"Requirements file not found: {REQUIREMENTS}")

    print("Upgrading pip...")
    run_command([venv_python, "-m", "pip", "install", "--upgrade", "pip"])

    print("Installing Python dependencies...")
    run_command([venv_python, "-m", "pip", "install", "-r", REQUIREMENTS])

    if dev:
        print("Installing development verification tools...")
        run_command([venv_python, "-m", "pip", "install", "pytest", "ruff"])


def rust_binary_path(platform_name: str | None = None) -> Path:
    platform_name = platform_name or sys.platform
    binary_name = "midi-touchup-editor.exe" if platform_name.startswith("win") else "midi-touchup-editor"
    return RUST_EDITOR_DIR / "target" / "release" / binary_name


def build_rust_editor(*, skip_rust: bool, strict_rust: bool) -> list[str]:
    warnings: list[str] = []
    if skip_rust:
        warnings.append("Skipped Rust touch-up editor build (--skip-rust).")
        return warnings

    if not RUST_EDITOR_DIR.exists():
        warnings.append(f"Rust touch-up editor directory not found: {RUST_EDITOR_DIR}")
        return warnings

    cargo = shutil.which("cargo")
    if not cargo:
        message = "Cargo/Rust not found. MIDI touch-up editor was not built. " + rust_install_hint()
        if strict_rust:
            raise SetupError(message)
        warnings.append(message)
        return warnings

    print("Building Rust MIDI touch-up editor...")
    try:
        run_command([cargo, "build", "--release"], cwd=RUST_EDITOR_DIR)
    except subprocess.CalledProcessError as exc:
        message = f"Rust touch-up editor build failed with exit code {exc.returncode}."
        if strict_rust:
            raise SetupError(message) from exc
        warnings.append(message)
        return warnings

    binary = rust_binary_path()
    if not binary.exists():
        message = f"Rust build completed but expected binary was not found: {binary}"
        if strict_rust:
            raise SetupError(message)
        warnings.append(message)

    return warnings


def check_environment(*, skip_rust: bool) -> None:
    ensure_python_version()
    ensure_ffmpeg()

    venv_python = venv_python_path(VENV_DIR)
    if not venv_python.exists():
        raise SetupError(".venv is missing. Run `python3 setup_env.py` on macOS/Linux or `py setup_env.py` on Windows to create it.")

    if not skip_rust and RUST_EDITOR_DIR.exists() and not rust_binary_path().exists():
        print("Warning: Rust MIDI touch-up editor binary is missing.")

    print("Environment check passed.")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    warnings: list[str] = []

    try:
        if args.check:
            check_environment(skip_rust=args.skip_rust)
            return 0

        print("== Synthesia2MIDI setup ==")
        ensure_python_version()
        ensure_ffmpeg()
        venv_python = create_or_reuse_venv(recreate=args.recreate_venv)
        install_python_dependencies(venv_python, dev=args.dev)
        warnings.extend(build_rust_editor(skip_rust=args.skip_rust, strict_rust=args.strict_rust))

    except (SetupError, subprocess.CalledProcessError) as exc:
        print("\nSetup failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1

    print("\nSetup complete.")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    print("\nRun:")
    print(f"  {user_python_command()} run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
