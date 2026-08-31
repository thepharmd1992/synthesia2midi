from synthesia2midi.version import (
    DEFAULT_APP_VERSION,
    normalize_release_version,
    resolve_app_version,
)


def test_default_app_version_tracks_next_release():
    assert DEFAULT_APP_VERSION == "0.2.2-dev"


def test_normalize_release_version_accepts_tag_and_version():
    assert normalize_release_version("v1.2.3") == "1.2.3"
    assert normalize_release_version("1.2.3-beta.1") == "1.2.3-beta.1"
    assert normalize_release_version("refs/tags/v2.0.0") == "2.0.0"


def test_normalize_release_version_rejects_non_release_strings():
    assert normalize_release_version("main") is None
    assert normalize_release_version("release-candidate") is None


def test_resolve_app_version_prefers_explicit_release_env(monkeypatch, tmp_path):
    monkeypatch.setenv("S2M_RELEASE_VERSION", "v9.8.7")
    monkeypatch.setattr("synthesia2midi.version._read_build_version_file", lambda: "1.2.3")

    assert resolve_app_version("0.1.0-dev") == "9.8.7"
