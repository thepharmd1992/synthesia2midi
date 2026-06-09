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
