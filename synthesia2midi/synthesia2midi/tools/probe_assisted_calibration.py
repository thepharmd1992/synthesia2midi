"""Local assisted-calibration probe for real videos.

This is a developer tool. It is not part of the default pytest gate.
"""
from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path

import cv2

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.detection.assisted_calibration import (
    ExemplarScanSettings,
    build_assisted_calibration_proposal,
    capture_unlit_references_from_frame,
)
from synthesia2midi.video_loader import create_video_session


def _parse_rgb_triplet(raw_value: str, *, source: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 3 or any(part == "" for part in parts):
        raise ValueError(f"{source} must contain exactly three comma-separated integers")
    try:
        rgb = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{source} must contain exactly three comma-separated integers") from exc
    if any(component < 0 or component > 255 for component in rgb):
        raise ValueError(f"{source} must contain RGB values between 0 and 255")
    return rgb


def _load_exemplar_lit_color_targets(path: Path) -> dict[str, tuple[int, int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"INI file not found: {path}")

    parser = configparser.ConfigParser(interpolation=None)
    with path.open("r", encoding="utf-8") as handle:
        parser.read_file(handle)

    if not parser.has_section("ExemplarLitColors"):
        raise ValueError(f"{path} is missing required [ExemplarLitColors] section")

    targets: dict[str, tuple[int, int, int]] = {}
    for slot in ("lw", "lb", "rw", "rb"):
        raw_value = parser.get("ExemplarLitColors", slot, fallback="").strip()
        if not raw_value:
            continue
        targets[slot.upper()] = _parse_rgb_triplet(
            raw_value,
            source=f"{path} [ExemplarLitColors] {slot}",
        )
    return targets


def _format_target_comparison(
    assignment_rgb: tuple[int, int, int] | None,
    target_rgb: tuple[int, int, int] | None,
) -> str:
    proposed = f"proposed={assignment_rgb}" if assignment_rgb is not None else "proposed=None"
    if target_rgb is None:
        return f"{proposed} target=<missing>"
    if assignment_rgb is None:
        return f"{proposed} target={target_rgb} diff=<n/a>"
    diff = tuple(abs(left - right) for left, right in zip(assignment_rgb, target_rgb))
    return f"{proposed} target={target_rgb} diff={diff}"


def _load_overlays(path: Path) -> list[OverlayConfig]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_overlays = data if isinstance(data, list) else data.get("overlays", [])
    overlays: list[OverlayConfig] = []
    for item in raw_overlays:
        unlit_reference_color = item.get("unlit_reference_color")
        overlays.append(
            OverlayConfig(
                key_id=int(item["key_id"]),
                note_octave=int(item["note_octave"]),
                note_name_in_octave=str(item["note_name_in_octave"]),
                x=float(item["x"]),
                y=float(item["y"]),
                width=float(item["width"]),
                height=float(item["height"]),
                key_type=item.get("key_type"),
                unlit_reference_color=(
                    tuple(int(component) for component in unlit_reference_color)
                    if unlit_reference_color is not None
                    else None
                ),
            )
        )
    return overlays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--overlays", required=True)
    parser.add_argument("--ini", required=True)
    parser.add_argument("--baseline-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()

    session = create_video_session(args.video)
    overlays = _load_overlays(Path(args.overlays))
    if args.end_frame is not None:
        end_frame = args.end_frame
    else:
        total_frames = int(getattr(session, "total_frames", 0))
        end_frame = max(args.baseline_frame, total_frames - 1) if total_frames > 0 else args.baseline_frame

    def frame_provider(index: int):
        success, frame_bgr = session.get_frame(index)
        if not success or frame_bgr is None:
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    try:
        color_targets = _load_exemplar_lit_color_targets(Path(args.ini))
        baseline = frame_provider(args.baseline_frame)
        if baseline is None:
            raise SystemExit(f"Could not load baseline frame {args.baseline_frame}")
        capture_unlit_references_from_frame(baseline, overlays)
        proposal = build_assisted_calibration_proposal(
            frame_provider,
            overlays,
            baseline_frame_index=args.baseline_frame,
            end_frame=end_frame,
            settings=ExemplarScanSettings(coarse_stride=args.stride),
        )

        print(f"baseline_frame={proposal.baseline_frame_index}")
        print(f"unlit_status={proposal.unlit_assessment.status}")
        print(f"candidates={proposal.candidate_count}")
        print(f"families={proposal.assignment_result.family_count}")
        for slot, assignment in proposal.assignment_result.assignments.items():
            target = color_targets.get(slot)
            print(f"{slot}: enabled={assignment.enabled} {_format_target_comparison(assignment.rgb, target)}")
        return 0
    except (OSError, ValueError, configparser.Error) as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        release = getattr(session, "release", None)
        if callable(release):
            release()


if __name__ == "__main__":
    raise SystemExit(main())
