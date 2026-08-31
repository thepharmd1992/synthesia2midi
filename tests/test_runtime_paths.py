import sys
from pathlib import Path

from synthesia2midi.runtime_paths import RuntimePaths


def test_ffmpeg_path_prefers_bundled_binary_in_frozen_mode(tmp_path):
    bundle_root = tmp_path / "bundle"
    ffmpeg = bundle_root / "bin" / "ffmpeg"
    ffmpeg.parent.mkdir(parents=True)
    ffmpeg.write_text("", encoding="utf-8")
    ffmpeg.chmod(0o755)

    paths = RuntimePaths(
        frozen=True,
        app_root=bundle_root,
        repo_root=tmp_path / "repo",
        home_dir=tmp_path / "home",
        platform_name="darwin",
    )

    assert paths.ffmpeg_path() == ffmpeg


def test_detect_uses_meipass_bundle_root_for_pyinstaller_onedir(monkeypatch, tmp_path):
    executable_root = tmp_path / "Synthesia2MIDI"
    bundle_root = executable_root / "_internal"
    ffmpeg = bundle_root / "bin" / "ffmpeg.exe"
    ffmpeg.parent.mkdir(parents=True)
    ffmpeg.write_bytes(b"real-ffmpeg")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_root), raising=False)
    monkeypatch.setattr(sys, "executable", str(executable_root / "Synthesia2MIDI.exe"))
    monkeypatch.setattr(sys, "platform", "win32")

    paths = RuntimePaths.detect()

    assert paths.app_root == executable_root
    assert paths.bundle_root == bundle_root
    assert paths.ffmpeg_path() == ffmpeg


def test_frozen_lookup_prefers_meipass_over_executable_adjacent_workaround(tmp_path):
    app_root = tmp_path / "app"
    bundle_root = app_root / "_internal"
    workaround = app_root / "bin" / "ffmpeg.exe"
    bundled = bundle_root / "bin" / "ffmpeg.exe"
    workaround.parent.mkdir(parents=True)
    bundle_root.joinpath("bin").mkdir(parents=True)
    workaround.write_bytes(b"stale-workaround")
    bundled.write_bytes(b"packaged-binary")

    paths = RuntimePaths(
        frozen=True,
        app_root=app_root,
        repo_root=tmp_path / "repo",
        home_dir=tmp_path / "home",
        platform_name="win32",
        bundle_root=bundle_root,
    )

    assert paths.ffmpeg_path() == bundled


def test_frozen_macos_bundle_finds_frameworks_and_resources_assets(tmp_path):
    app_root = tmp_path / "Synthesia2MIDI.app" / "Contents" / "MacOS"
    frameworks_root = tmp_path / "Synthesia2MIDI.app" / "Contents" / "Frameworks"
    resources_root = tmp_path / "Synthesia2MIDI.app" / "Contents" / "Resources"
    ffprobe = resources_root / "bin" / "ffprobe"
    soundfont = frameworks_root / "assets" / "soundfonts" / "TouchUpPiano.sf2"

    ffprobe.parent.mkdir(parents=True)
    ffprobe.write_text("", encoding="utf-8")
    ffprobe.chmod(0o755)

    soundfont.parent.mkdir(parents=True)
    soundfont.write_text("", encoding="utf-8")

    paths = RuntimePaths(
        frozen=True,
        app_root=app_root,
        repo_root=frameworks_root,
        home_dir=tmp_path / "home",
        platform_name="darwin",
    )

    assert paths.ffprobe_path() == ffprobe
    assert paths.rust_soundfont_path() == soundfont


def test_rust_editor_path_uses_repo_release_binary_in_source_mode(tmp_path):
    repo_root = tmp_path / "repo"
    rust_binary = (
        repo_root
        / "tools"
        / "midi_touchup_editor_rust"
        / "target"
        / "release"
        / "midi-touchup-editor"
    )
    rust_binary.parent.mkdir(parents=True)
    rust_binary.write_text("", encoding="utf-8")
    rust_binary.chmod(0o755)

    paths = RuntimePaths(
        frozen=False,
        app_root=repo_root,
        repo_root=repo_root,
        home_dir=tmp_path / "home",
        platform_name="darwin",
    )

    assert paths.rust_editor_path() == rust_binary


def test_default_video_dir_prefers_movies_then_documents(tmp_path):
    home_dir = tmp_path / "home"
    movies_dir = home_dir / "Movies"
    documents_dir = home_dir / "Documents"
    documents_dir.mkdir(parents=True)

    paths = RuntimePaths(
        frozen=True,
        app_root=tmp_path / "bundle",
        repo_root=tmp_path / "repo",
        home_dir=home_dir,
        platform_name="darwin",
    )

    assert paths.default_video_dir() == documents_dir

    movies_dir.mkdir(parents=True)
    assert paths.default_video_dir() == movies_dir


def test_user_output_dirs_are_dedicated_folders(tmp_path):
    paths = RuntimePaths(
        frozen=False,
        app_root=tmp_path / "repo",
        repo_root=tmp_path / "repo",
        home_dir=tmp_path,
        platform_name="darwin",
    )

    assert paths.midi_exports_dir() == tmp_path / "Desktop" / "Synthesia2MIDI MIDI Files"
    assert paths.default_download_dir() == tmp_path / "Downloads" / "Synthesia2MIDI"


def test_project_paths_are_app_managed_and_stable(tmp_path):
    paths = RuntimePaths(
        frozen=False,
        app_root=tmp_path / "repo",
        repo_root=tmp_path / "repo",
        home_dir=tmp_path,
        platform_name="darwin",
    )
    video_path = "/Users/jeff/Movies/Game of Thrones Main Theme.mp4"

    project_dir = paths.project_dir_for_video(video_path)

    assert project_dir.parent.name == "projects"
    assert "game-of-thrones-main-theme" in project_dir.name
    assert paths.project_ini_path(video_path).parent == project_dir
    assert paths.project_overlay_json_path(video_path).parent == project_dir
    assert paths.project_frames_dir(video_path).parent == project_dir


def test_project_data_dir_uses_macos_application_support(tmp_path):
    paths = RuntimePaths(
        frozen=False,
        app_root=tmp_path / "repo",
        repo_root=tmp_path / "repo",
        home_dir=tmp_path,
        platform_name="darwin",
    )

    assert paths.project_data_dir() == (
        tmp_path / "Library" / "Application Support" / "Synthesia2MIDI" / "projects"
    )


def test_log_dir_uses_platform_specific_user_path(tmp_path):
    home_dir = tmp_path / "home"
    local_app_data = tmp_path / "localappdata"

    paths = RuntimePaths(
        frozen=True,
        app_root=tmp_path / "bundle",
        repo_root=tmp_path / "repo",
        home_dir=home_dir,
        platform_name="win32",
    )

    assert "synthesia2midi" in str(paths.log_dir()).lower()
    assert paths.binary_name("ffmpeg") == "ffmpeg.exe"
    assert isinstance(local_app_data, Path)
