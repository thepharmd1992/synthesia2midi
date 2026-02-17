#!/usr/bin/env python3
"""
Repository-root launcher for Synthesia2MIDI.

This delegates to synthesia2midi/run.py so users can run:
  python run.py
from the repo root on any platform.
"""
from __future__ import annotations

import os
import runpy
import sys


def main() -> None:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(root_dir, "synthesia2midi", "run.py")
    if not os.path.isfile(target):
        raise FileNotFoundError(f"Launcher target not found: {target}")

    # Keep repo root importable for helper scripts that rely on cwd/pythonpath.
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
