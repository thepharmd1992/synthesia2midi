"""Fail-closed verification for frozen release bundles."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

from synthesia2midi.binary_payload import native_binary_issue
from synthesia2midi.runtime_paths import RuntimePaths


PACKAGE_SELF_CHECK_FLAG = "--package-self-check"
PACKAGE_SELF_CHECK_SCHEMA_VERSION = 1
DEFAULT_PROBE_TIMEOUT_SECONDS = 15

ProbeRunner = Callable[..., Any]


def _package_ownership_root(runtime_paths: RuntimePaths) -> Path | None:
    if not runtime_paths.frozen or runtime_paths.bundle_root is None:
        return None
    bundle_root = Path(runtime_paths.bundle_root).resolve()
    if (
        runtime_paths.platform_name == "darwin"
        and bundle_root.name in {"Frameworks", "Resources"}
        and bundle_root.parent.name == "Contents"
    ):
        return bundle_root.parent
    return bundle_root


def _is_package_owned(path: Path, ownership_root: Path | None) -> bool:
    if ownership_root is None:
        return False
    try:
        path.resolve().relative_to(ownership_root)
    except (OSError, ValueError):
        return False
    return True


def _diagnostic_text(stdout: str | None, stderr: str | None) -> str:
    output = "\n".join(part.strip() for part in (stdout or "", stderr or "") if part.strip())
    if not output:
        return "probe completed"
    return output[:1000]


def _base_check(
    *,
    name: str,
    kind: str,
    path: Path | None,
    packaged: bool,
    probe: Sequence[str] | None,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "path": str(path) if path is not None else None,
        "packaged": packaged,
        "probe": list(probe) if probe is not None else None,
        "returncode": None,
        "status": "failed",
        "detail": "",
    }


def _check_binary(
    *,
    name: str,
    path: Path | None,
    probe: Sequence[str],
    ownership_root: Path | None,
    platform_name: str,
    run_probe: ProbeRunner,
    timeout_seconds: int,
) -> dict[str, Any]:
    packaged = path is not None and _is_package_owned(path, ownership_root)
    check = _base_check(
        name=name,
        kind="binary",
        path=path,
        packaged=packaged,
        probe=probe,
    )
    if path is None or not path.is_file():
        check["detail"] = "packaged binary was not resolved"
        return check
    if not packaged:
        check["detail"] = "resolved binary is outside the package ownership root"
        return check
    payload_issue = native_binary_issue(path, platform_name)
    if payload_issue is not None:
        check["detail"] = payload_issue
        return check

    command = [str(path), *probe]
    try:
        completed = run_probe(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        check["detail"] = f"probe timed out after {timeout_seconds} seconds"
        return check
    except OSError as exc:
        check["detail"] = f"probe could not start: {exc}"
        return check

    check["returncode"] = int(completed.returncode)
    check["detail"] = _diagnostic_text(
        getattr(completed, "stdout", ""),
        getattr(completed, "stderr", ""),
    )
    if completed.returncode == 0:
        check["status"] = "passed"
    return check


def _check_asset(
    *,
    name: str,
    path: Path | None,
    ownership_root: Path | None,
) -> dict[str, Any]:
    packaged = path is not None and _is_package_owned(path, ownership_root)
    check = _base_check(
        name=name,
        kind="asset",
        path=path,
        packaged=packaged,
        probe=None,
    )
    if path is None or not path.is_file():
        check["detail"] = "packaged asset was not resolved"
        return check
    if not packaged:
        check["detail"] = "resolved asset is outside the package ownership root"
        return check
    if not os.access(path, os.R_OK):
        check["detail"] = "packaged asset is not readable"
        return check
    check["status"] = "passed"
    check["detail"] = "asset is present and readable"
    return check


def build_package_self_check_report(
    runtime_paths: RuntimePaths,
    *,
    run_probe: ProbeRunner | None = None,
    timeout_seconds: int = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Resolve and exercise every required helper from the frozen bundle."""
    runner = run_probe or subprocess.run
    ownership_root = _package_ownership_root(runtime_paths)
    binary_specs = (
        ("ffmpeg", runtime_paths.ffmpeg_path(), ("-version",)),
        ("ffprobe", runtime_paths.ffprobe_path(), ("-version",)),
        ("deno", runtime_paths.deno_path(), ("--version",)),
        ("rust_editor", runtime_paths.rust_editor_path(), ("--help",)),
    )
    checks = [
        _check_binary(
            name=name,
            path=path,
            probe=probe,
            ownership_root=ownership_root,
            platform_name=runtime_paths.platform_name,
            run_probe=runner,
            timeout_seconds=timeout_seconds,
        )
        for name, path, probe in binary_specs
    ]
    checks.extend(
        [
            _check_asset(
                name="soundfont",
                path=runtime_paths.rust_soundfont_path(),
                ownership_root=ownership_root,
            ),
            _check_asset(
                name="soundfont_license",
                path=runtime_paths.rust_soundfont_license_path(),
                ownership_root=ownership_root,
            ),
        ]
    )
    errors = [
        f"{check['name']}: {check['detail']}"
        for check in checks
        if check["status"] != "passed"
    ]
    return {
        "schema_version": PACKAGE_SELF_CHECK_SCHEMA_VERSION,
        "status": "failed" if errors else "passed",
        "frozen": bool(runtime_paths.frozen),
        "platform": runtime_paths.platform_name,
        "app_root": str(runtime_paths.app_root),
        "bundle_root": (
            str(runtime_paths.bundle_root)
            if runtime_paths.bundle_root is not None
            else None
        ),
        "checks": checks,
        "errors": errors,
    }


def maybe_run_package_self_check(
    argv: Sequence[str],
    runtime_paths: RuntimePaths,
    *,
    run_probe: ProbeRunner | None = None,
    timeout_seconds: int = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> int | None:
    """Write a self-check report when the internal launcher flag is present."""
    try:
        flag_index = list(argv).index(PACKAGE_SELF_CHECK_FLAG)
    except ValueError:
        return None
    if flag_index + 1 >= len(argv):
        return 2

    report_path = Path(argv[flag_index + 1]).expanduser().resolve()
    report = build_package_self_check_report(
        runtime_paths,
        run_probe=run_probe,
        timeout_seconds=timeout_seconds,
    )
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        return 2
    return 0 if report["status"] == "passed" else 1
