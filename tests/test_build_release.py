import importlib.util
import zipfile
from pathlib import Path


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
