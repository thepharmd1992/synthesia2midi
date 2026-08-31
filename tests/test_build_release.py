import importlib.util
from types import SimpleNamespace
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_install_deno_from_zip_extracts_windows_binary(monkeypatch, tmp_path):
    module = _load_module("build_release_under_test", ROOT / "packaging" / "build_release.py")

    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(module, "latest_deno_version", lambda: "9.9.9")

    seen = {}

    def fake_download(url: str, destination: Path) -> Path:
        seen["url"] = url
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("deno.exe", b"fake-deno")
        return destination

    monkeypatch.setattr(module, "download_to_file", fake_download)

    deno_path = module.install_deno_from_zip(tmp_path / "deno")

    assert seen["url"] == "https://dl.deno.land/release/v9.9.9/deno-x86_64-pc-windows-msvc.zip"
    assert deno_path.name == "deno.exe"
    assert deno_path.read_bytes() == b"fake-deno"


def test_urlopen_with_headers_sets_user_agent(monkeypatch):
    module = _load_module("build_release_headers_under_test", ROOT / "packaging" / "build_release.py")

    seen = {}

    def fake_urlopen(request):
        seen["url"] = request.full_url
        seen["user_agent"] = request.get_header("User-agent")
        class _Response:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def read(self):
                return b""
        return _Response()

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)

    with module.urlopen_with_headers("https://example.com/test"):
        pass

    assert seen["url"] == "https://example.com/test"
    assert seen["user_agent"] == "Mozilla/5.0"


def test_deno_release_url_normalizes_leading_v():
    module = _load_module("build_release_version_under_test", ROOT / "packaging" / "build_release.py")

    url = module.deno_release_url(version="v9.9.9", target_tuple="x86_64-pc-windows-msvc")

    assert url == "https://dl.deno.land/release/v9.9.9/deno-x86_64-pc-windows-msvc.zip"


def test_pyinstaller_install_uses_reviewed_build_requirements(monkeypatch, tmp_path):
    module = _load_module("build_release_pins_under_test", ROOT / "packaging" / "build_release.py")
    calls = []
    monkeypatch.setattr(module, "run", lambda command, **_kwargs: calls.append(command))
    venv_python = tmp_path / "python"

    module.ensure_pyinstaller(venv_python)

    assert calls == [
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--requirement",
            str(ROOT / "packaging" / "requirements-build.txt"),
        ]
    ]
    requirements = (ROOT / "packaging" / "requirements-build.txt").read_text(encoding="utf-8")
    assert "pyinstaller==6.22.2" in requirements.lower()
    assert "pyinstaller-hooks-contrib==2026.7" in requirements.lower()


def test_windows_chocolatey_shim_resolves_to_real_ffmpeg(monkeypatch, tmp_path):
    module = _load_module("build_release_choco_under_test", ROOT / "packaging" / "build_release.py")
    chocolatey_root = tmp_path / "chocolatey"
    shim = chocolatey_root / "bin" / "ffmpeg.exe"
    real = chocolatey_root / "lib" / "ffmpeg" / "tools" / "ffmpeg-9.0.1-essentials_build" / "bin" / "ffmpeg.exe"
    shim.parent.mkdir(parents=True)
    real.parent.mkdir(parents=True)
    shim.write_bytes(b"shim")
    real.write_bytes(b"MZreal")
    probes = []

    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setenv("ChocolateyInstall", str(chocolatey_root))
    monkeypatch.setattr(module.shutil, "which", lambda _name: str(shim))
    monkeypatch.setattr(module, "probe_binary", lambda path, *args: probes.append((path, args)))

    resolved = module.ensure_ffmpeg_binary("ffmpeg")

    assert resolved == real
    assert probes == [(real, ("-version",))]


def test_windows_chocolatey_target_resolution_fails_closed_when_ambiguous(monkeypatch, tmp_path):
    module = _load_module("build_release_choco_ambiguous", ROOT / "packaging" / "build_release.py")
    chocolatey_root = tmp_path / "chocolatey"
    shim = chocolatey_root / "bin" / "ffprobe.exe"
    shim.parent.mkdir(parents=True)
    shim.write_bytes(b"shim")
    for version in ("8.1.2", "9.0.1"):
        candidate = chocolatey_root / "lib" / "ffmpeg" / "tools" / version / "bin" / "ffprobe.exe"
        candidate.parent.mkdir(parents=True)
        candidate.write_bytes(version.encode("ascii"))

    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setenv("ChocolateyInstall", str(chocolatey_root))
    monkeypatch.setattr(module.shutil, "which", lambda _name: str(shim))

    with pytest.raises(module.ReleaseBuildError, match="unambiguous"):
        module.ensure_ffmpeg_binary("ffprobe")


def test_windows_chocolatey_shim_payload_is_rejected(monkeypatch, tmp_path):
    module = _load_module("build_release_choco_payload", ROOT / "packaging" / "build_release.py")
    shim = tmp_path / "ffmpeg.exe"
    marker = "ShimGen generated shim - Chocolatey Shim".encode("utf-16-le")
    shim.write_bytes(b"MZ" + marker)

    monkeypatch.setattr(module.sys, "platform", "win32")

    with pytest.raises(module.ReleaseBuildError, match="ShimGen"):
        module.validate_native_binary(shim)


def test_macos_ffmpeg_python_launcher_is_rejected_even_when_it_runs_locally(
    monkeypatch,
    tmp_path,
):
    module = _load_module("build_release_macos_wrapper", ROOT / "packaging" / "build_release.py")
    wrapper = tmp_path / "ffmpeg"
    wrapper.write_text("#!/build-machine/python\nprint('ffmpeg version')\n", encoding="utf-8")
    wrapper.chmod(0o755)

    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.shutil, "which", lambda _name: str(wrapper))
    monkeypatch.setattr(
        module,
        "probe_binary",
        lambda *_args, **_kwargs: pytest.fail("a non-native wrapper must not be probed"),
    )

    with pytest.raises(module.ReleaseBuildError, match="Mach-O"):
        module.ensure_ffmpeg_binary("ffmpeg")


def test_macos_ffmpeg_explicit_native_override_is_accepted(monkeypatch, tmp_path):
    module = _load_module("build_release_macos_override", ROOT / "packaging" / "build_release.py")
    native = tmp_path / "ffmpeg-native"
    native.write_bytes(b"\xcf\xfa\xed\xfe" + b"native")
    native.chmod(0o755)
    probes = []

    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setenv("S2M_FFMPEG_PATH", str(native))
    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda _name: pytest.fail("the explicit override should be preferred"),
    )
    monkeypatch.setattr(module, "probe_binary", lambda path, *args: probes.append((path, args)))

    resolved = module.ensure_ffmpeg_binary("ffmpeg")

    assert resolved == native
    assert probes == [(native, ("-version",))]


def test_probe_binary_rejects_nonzero_helper(monkeypatch, tmp_path):
    module = _load_module("build_release_probe_under_test", ROOT / "packaging" / "build_release.py")
    binary = tmp_path / "ffmpeg"
    binary.write_bytes(b"broken")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr="cannot start"),
    )

    with pytest.raises(module.ReleaseBuildError, match="cannot start"):
        module.probe_binary(binary, "-version")


def test_finalize_release_validates_before_archiving(monkeypatch, tmp_path):
    module = _load_module("build_release_finalize_under_test", ROOT / "packaging" / "build_release.py")
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    calls = []
    archive = tmp_path / "release.zip"
    monkeypatch.setattr(module, "package_self_check", lambda path: calls.append(("self-check", path)))
    monkeypatch.setattr(module, "smoke_launch", lambda path: calls.append(("smoke", path)))
    monkeypatch.setattr(module, "archive_release_bundle", lambda path: calls.append(("archive", path)) or archive)

    result = module.finalize_release_bundle(stage_dir, skip_smoke=False)

    assert result == archive
    assert calls == [
        ("self-check", stage_dir),
        ("smoke", stage_dir),
        ("archive", stage_dir),
    ]


def test_finalize_release_does_not_archive_failed_self_check(monkeypatch, tmp_path):
    module = _load_module("build_release_finalize_failure", ROOT / "packaging" / "build_release.py")
    archived = []
    monkeypatch.setattr(
        module,
        "package_self_check",
        lambda _path: (_ for _ in ()).throw(module.ReleaseBuildError("bad helper")),
    )
    monkeypatch.setattr(module, "archive_release_bundle", lambda path: archived.append(path))

    with pytest.raises(module.ReleaseBuildError, match="bad helper"):
        module.finalize_release_bundle(tmp_path, skip_smoke=True)

    assert archived == []


def test_package_self_check_requires_passing_report(monkeypatch, tmp_path):
    module = _load_module("build_release_self_check_under_test", ROOT / "packaging" / "build_release.py")
    stage_dir = tmp_path / "stage"
    executable = stage_dir / "Synthesia2MIDI.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"app")
    monkeypatch.setattr(module, "smoke_executable_path", lambda _stage: executable)
    monkeypatch.setattr(module, "platform_slug", lambda: "test-platform")

    def fake_run(command, **_kwargs):
        report_path = Path(command[-1])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            '{"schema_version": 1, "status": "failed", "errors": ["ffmpeg failed"]}',
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(module.ReleaseBuildError, match="ffmpeg failed"):
        module.package_self_check(stage_dir)
