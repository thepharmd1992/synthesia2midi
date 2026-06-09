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
