#!/usr/bin/env python3
"""Repository-root launcher for Synthesia2MIDI.

Users run this from the repo root with:
  python3 run.py

The launcher keeps the virtual environment hidden: if `.venv` exists and the
current interpreter is not the venv interpreter, it re-execs itself with the venv
Python before launching the GUI.
"""
from __future__ import annotations

import os
import runpy
import shutil
import sys
from pathlib import Path


def venv_python_path(root_dir: Path, platform_name: str | None = None) -> Path:
    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        return root_dir / ".venv" / "Scripts" / "python.exe"
    return root_dir / ".venv" / "bin" / "python"


def user_python_command(platform_name: str | None = None) -> str:
    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        return "py"
    return "python3"


def find_venv_python(root_dir: Path, platform_name: str | None = None) -> Path | None:
    candidate = venv_python_path(root_dir, platform_name)
    if candidate.is_file():
        return candidate
    return None


def should_reexec_into_venv(current_executable: Path, venv_python: Path) -> bool:
    try:
        return current_executable.resolve() != venv_python.resolve()
    except OSError:
        return os.path.abspath(current_executable) != os.path.abspath(venv_python)


def setup_command_hint(platform_name: str | None = None) -> str:
    """Return a concise cross-platform setup command hint."""
    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        return "py setup_env.py (Windows) or python3 setup_env.py (macOS/Linux)"
    return "python3 setup_env.py (macOS/Linux) or py setup_env.py (Windows)"


def setup_required_message() -> str:
    return (
        "Synthesia2MIDI environment not found.\n\n"
        "Run setup from the repo root first:\n"
        f"  {setup_command_hint()}"
    )


def ffmpeg_required_message() -> str:
    py_cmd = user_python_command()
    return (
        "FFmpeg is required but was not found on PATH.\n\n"
        "Install FFmpeg, then run setup again:\n"
        f"  {py_cmd} setup_env.py"
    )


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    target = root_dir / "synthesia2midi" / "run.py"
    if not target.is_file():
        raise FileNotFoundError(f"Launcher target not found: {target}")

    venv_python = find_venv_python(root_dir)
    if venv_python is None:
        print(setup_required_message(), file=sys.stderr)
        raise SystemExit(1)

    if should_reexec_into_venv(Path(sys.executable), venv_python):
        os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])

    if shutil.which("ffmpeg") is None:
        print(ffmpeg_required_message(), file=sys.stderr)
        raise SystemExit(1)

    # Keep repo root importable for helper scripts that rely on cwd/pythonpath.
    root_str = str(root_dir)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
