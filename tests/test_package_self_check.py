import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from synthesia2midi.package_self_check import (
    build_package_self_check_report,
    maybe_run_package_self_check,
)
from synthesia2midi.runtime_paths import RuntimePaths


def _package_paths(tmp_path: Path) -> RuntimePaths:
    app_root = tmp_path / "app"
    bundle_root = app_root / "_internal"
    bin_root = bundle_root / "bin"
    asset_root = bundle_root / "assets" / "soundfonts"
    bin_root.mkdir(parents=True)
    asset_root.mkdir(parents=True)
    for name in ("ffmpeg", "ffprobe", "deno", "midi-touchup-editor"):
        binary = bin_root / name
        binary.write_bytes(b"executable")
        binary.chmod(0o755)
    (asset_root / "TouchUpPiano.sf2").write_bytes(b"soundfont")
    (asset_root / "TouchUpPiano_LICENSE.txt").write_text("license", encoding="utf-8")
    return RuntimePaths(
        frozen=True,
        app_root=app_root,
        repo_root=tmp_path / "repo",
        home_dir=tmp_path / "home",
        platform_name="linux",
        bundle_root=bundle_root,
    )


def _successful_runner(calls):
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="version output", stderr="")

    return run


def test_report_resolves_and_executes_every_packaged_helper(tmp_path):
    paths = _package_paths(tmp_path)
    calls = []

    report = build_package_self_check_report(
        paths,
        run_probe=_successful_runner(calls),
        timeout_seconds=3,
    )

    assert report["schema_version"] == 1
    assert report["status"] == "passed"
    assert report["frozen"] is True
    assert report["bundle_root"] == str(paths.bundle_root)
    assert [check["name"] for check in report["checks"]] == [
        "ffmpeg",
        "ffprobe",
        "deno",
        "rust_editor",
        "soundfont",
        "soundfont_license",
    ]
    assert all(check["packaged"] for check in report["checks"])
    assert all(check["status"] == "passed" for check in report["checks"])
    assert [command[1:] for command, _kwargs in calls] == [
        ["-version"],
        ["-version"],
        ["--version"],
        ["--help"],
    ]
    assert all(kwargs["timeout"] == 3 for _command, kwargs in calls)
    assert all(kwargs["check"] is False for _command, kwargs in calls)


def test_system_helper_cannot_make_frozen_package_self_check_pass(tmp_path):
    paths = _package_paths(tmp_path)
    external = tmp_path / "system" / "ffmpeg"
    external.parent.mkdir()
    external.write_bytes(b"system")
    external.chmod(0o755)
    fake_paths = SimpleNamespace(
        frozen=True,
        platform_name="linux",
        app_root=paths.app_root,
        bundle_root=paths.bundle_root,
        ffmpeg_path=lambda: external,
        ffprobe_path=paths.ffprobe_path,
        deno_path=paths.deno_path,
        rust_editor_path=paths.rust_editor_path,
        rust_soundfont_path=paths.rust_soundfont_path,
        rust_soundfont_license_path=paths.rust_soundfont_license_path,
    )
    calls = []

    report = build_package_self_check_report(
        fake_paths,
        run_probe=_successful_runner(calls),
    )

    ffmpeg_check = report["checks"][0]
    assert report["status"] == "failed"
    assert ffmpeg_check["status"] == "failed"
    assert ffmpeg_check["packaged"] is False
    assert "package" in ffmpeg_check["detail"].lower()
    assert all(command[0] != str(external) for command, _kwargs in calls)


def test_nonzero_and_timed_out_helpers_fail_report(tmp_path):
    paths = _package_paths(tmp_path)

    def failing_runner(command, **kwargs):
        if Path(command[0]).name == "ffmpeg":
            return SimpleNamespace(returncode=1, stdout="", stderr="broken")
        if Path(command[0]).name == "deno":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    report = build_package_self_check_report(paths, run_probe=failing_runner)
    checks = {check["name"]: check for check in report["checks"]}

    assert report["status"] == "failed"
    assert checks["ffmpeg"]["returncode"] == 1
    assert "broken" in checks["ffmpeg"]["detail"]
    assert checks["deno"]["returncode"] is None
    assert "timed out" in checks["deno"]["detail"].lower()


def test_missing_asset_fails_without_running_it(tmp_path):
    paths = _package_paths(tmp_path)
    paths.rust_soundfont_path().unlink()

    report = build_package_self_check_report(paths, run_probe=_successful_runner([]))
    checks = {check["name"]: check for check in report["checks"]}

    assert report["status"] == "failed"
    assert checks["soundfont"]["path"] is None
    assert checks["soundfont"]["probe"] is None


def test_cli_mode_writes_json_report_and_returns_status(tmp_path):
    paths = _package_paths(tmp_path)
    report_path = tmp_path / "reports" / "package.json"

    exit_code = maybe_run_package_self_check(
        ["--package-self-check", str(report_path)],
        paths,
        run_probe=_successful_runner([]),
    )

    assert exit_code == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "passed"
    assert maybe_run_package_self_check([], paths) is None
