from pathlib import Path

from synthesia2midi import youtube_downloader
from synthesia2midi.youtube_downloader import YouTubeDownloader


def test_get_video_info_enables_js_challenge_support(monkeypatch, tmp_path):
    captured_opts = []

    class FakeYoutubeDL:
        def __init__(self, opts):
            captured_opts.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=False):
            return {
                "title": "Mary had a little lamb",
                "duration": 123,
                "uploader": "Piano",
            }

    monkeypatch.setattr("shutil.which", lambda name: "/fake/node" if name == "node" else None)
    monkeypatch.setattr(youtube_downloader.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    info = YouTubeDownloader(str(tmp_path)).get_video_info("https://www.youtube.com/watch?v=SFFSZQCnU_M")

    assert info["title"] == "Mary had a little lamb"
    assert captured_opts[0]["js_runtimes"] == {"node": {"path": "/fake/node"}}
    assert captured_opts[0]["remote_components"] == ["ejs:github"]


def test_download_video_only_enables_js_challenge_support_for_info_and_download(monkeypatch, tmp_path):
    captured_opts = []

    class FakeYoutubeDL:
        def __init__(self, opts):
            captured_opts.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=False):
            return {"title": "Mary had a little lamb", "id": "SFFSZQCnU_M"}

        def prepare_filename(self, info):
            return str(Path(captured_opts[-1]["outtmpl"]).with_suffix(".mp4"))

    monkeypatch.setattr("shutil.which", lambda name: "/fake/node" if name == "node" else None)
    monkeypatch.setattr(youtube_downloader.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    output_path = YouTubeDownloader(str(tmp_path)).download_video_only(
        "https://www.youtube.com/watch?v=SFFSZQCnU_M"
    )

    assert output_path.endswith(".mp4")
    assert captured_opts[0]["js_runtimes"] == {"node": {"path": "/fake/node"}}
    assert captured_opts[0]["remote_components"] == ["ejs:github"]
    assert captured_opts[1]["js_runtimes"] == {"node": {"path": "/fake/node"}}
    assert captured_opts[1]["remote_components"] == ["ejs:github"]


def test_get_video_info_explains_missing_js_runtime_for_misleading_unavailable_error(
    monkeypatch, tmp_path
):
    class FakeYoutubeDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=False):
            raise Exception("ERROR: [youtube] SFFSZQCnU_M: This video is not available")

    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(youtube_downloader, "_COMMON_RUNTIME_DIRS", (tmp_path / "missing",))
    monkeypatch.setattr(youtube_downloader.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    try:
        YouTubeDownloader(str(tmp_path)).get_video_info(
            "https://www.youtube.com/watch?v=SFFSZQCnU_M"
        )
    except Exception as exc:
        message = str(exc)
    else:
        raise AssertionError("expected get_video_info to fail")

    assert "No JavaScript runtime was found" in message
    assert "Node.js, Deno, Bun, or QuickJS" in message
