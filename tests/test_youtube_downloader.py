from pathlib import Path

from synthesia2midi import youtube_downloader
from synthesia2midi.youtube_downloader import (
    YouTubeDownloader,
    _discover_js_runtimes,
    _find_js_runtime_path,
    _youtube_ydl_opts,
    browser_cookie_args,
    _format_youtube_error,
    _format_download_status,
    _progress_percentage,
    should_retry_with_browser_cookies,
)


def test_runtime_discovery_skips_inaccessible_common_directory_and_finds_later_runtime(
    monkeypatch, tmp_path
):
    inaccessible_dir = tmp_path / "inaccessible"
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    working_runtime = working_dir / ("deno.exe" if youtube_downloader.os.name == "nt" else "deno")
    working_runtime.write_text("runtime")
    working_runtime.chmod(0o755)

    original_is_file = Path.is_file

    def permission_aware_is_file(path):
        if path.parent == inaccessible_dir:
            raise PermissionError(13, "Permission denied", str(path))
        return original_is_file(path)

    class FakeRuntimePaths:
        def deno_path(self):
            return None

    monkeypatch.setattr(Path, "is_file", permission_aware_is_file)
    monkeypatch.setattr(youtube_downloader, "detect_runtime_paths", lambda: FakeRuntimePaths())
    monkeypatch.setattr(youtube_downloader.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        youtube_downloader,
        "_COMMON_RUNTIME_DIRS",
        (inaccessible_dir, working_dir),
    )

    assert _find_js_runtime_path("deno") == str(working_runtime)


def test_runtime_discovery_keeps_bundled_deno_when_other_runtime_locations_are_inaccessible(
    monkeypatch, tmp_path
):
    bundled_deno = tmp_path / "bundle" / "bin" / "deno"
    inaccessible_dir = tmp_path / "inaccessible"

    original_is_file = Path.is_file

    def permission_aware_is_file(path):
        if path.parent == inaccessible_dir:
            raise PermissionError(13, "Permission denied", str(path))
        return original_is_file(path)

    class FakeRuntimePaths:
        def deno_path(self):
            return bundled_deno

    monkeypatch.setattr(Path, "is_file", permission_aware_is_file)
    monkeypatch.setattr(youtube_downloader, "detect_runtime_paths", lambda: FakeRuntimePaths())
    monkeypatch.setattr(youtube_downloader.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        youtube_downloader,
        "_COMMON_RUNTIME_DIRS",
        (inaccessible_dir,),
    )

    assert _discover_js_runtimes() == {"deno": {"path": str(bundled_deno)}}


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


def test_get_video_info_includes_available_qualities(monkeypatch, tmp_path):
    class FakeYoutubeDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=False):
            return {
                "title": "Mary had a little lamb",
                "duration": 123,
                "uploader": "Piano",
                "formats": [
                    {"height": 360, "vcodec": "h264", "acodec": "none", "ext": "mp4"},
                    {"height": 720, "vcodec": "h264", "acodec": "none", "ext": "mp4"},
                ],
            }

    monkeypatch.setattr(youtube_downloader.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    info = YouTubeDownloader(str(tmp_path)).get_video_info("https://www.youtube.com/watch?v=SFFSZQCnU_M")

    assert info["available_qualities"]["720p"]["actual_height"] == 720
    assert info["available_qualities"]["1080p"]["actual_height"] == 720
    assert info["available_qualities"]["480p"]["actual_height"] == 360


def test_youtube_opts_include_bundled_ffmpeg_and_cookie_browser(monkeypatch, tmp_path):
    ffmpeg_path = tmp_path / "bin" / "ffmpeg"
    ffmpeg_path.parent.mkdir()
    ffmpeg_path.write_text("ffmpeg")

    class FakeRuntimePaths:
        def ffmpeg_path(self):
            return ffmpeg_path

        def deno_path(self):
            return None

    monkeypatch.setattr(youtube_downloader, "detect_runtime_paths", lambda: FakeRuntimePaths())
    monkeypatch.setattr("shutil.which", lambda name: "/fake/node" if name == "node" else None)

    opts = _youtube_ydl_opts({"quiet": True}, browser_cookie="chrome")

    assert opts["ffmpeg_location"] == str(ffmpeg_path.parent)
    assert opts["cookiesfrombrowser"] == ("chrome",)
    assert opts["js_runtimes"] == {"node": {"path": "/fake/node"}}


def test_get_video_info_retries_with_preferred_browser_cookies(monkeypatch, tmp_path):
    captured_opts = []
    statuses = []

    class FakeYoutubeDL:
        def __init__(self, opts):
            self.opts = opts
            captured_opts.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=False):
            if self.opts.get("cookiesfrombrowser") is None:
                raise Exception("Sign in to confirm your age")
            return {
                "title": "Mary had a little lamb",
                "duration": 123,
                "uploader": "Piano",
            }

    monkeypatch.setattr(youtube_downloader.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    info = YouTubeDownloader(
        str(tmp_path),
        preferred_browser="safari",
        auto_cookie_retry=True,
        status_callback=statuses.append,
    ).get_video_info("https://www.youtube.com/watch?v=SFFSZQCnU_M")

    assert info["title"] == "Mary had a little lamb"
    assert captured_opts[0].get("cookiesfrombrowser") is None
    assert captured_opts[1]["cookiesfrombrowser"] == ("safari",)
    assert statuses == ["Video info request failed. Retrying with Safari browser cookies..."]


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


def test_download_retries_media_403_with_embedded_youtube_client(monkeypatch, tmp_path):
    captured_opts = []
    statuses = []

    class FakeYoutubeDL:
        def __init__(self, opts):
            self.opts = opts
            captured_opts.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=False):
            if not download:
                return {
                    "title": "Mary had a little lamb",
                    "id": "SFFSZQCnU_M",
                    "formats": [
                        {"height": 480, "vcodec": "h264", "acodec": "none", "ext": "mp4"},
                    ],
                }
            if self.opts.get("extractor_args") != {
                "youtube": {"player_client": ["web_embedded"]}
            }:
                raise Exception("ERROR: unable to download video data: HTTP Error 403: Forbidden")
            return {"title": "Mary had a little lamb", "id": "SFFSZQCnU_M"}

        def prepare_filename(self, info):
            return str(Path(self.opts["outtmpl"]).with_suffix(".mp4"))

    monkeypatch.setattr(youtube_downloader.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    output_path = YouTubeDownloader(
        str(tmp_path),
        status_callback=statuses.append,
    ).download_video_only("https://www.youtube.com/watch?v=SFFSZQCnU_M")

    assert output_path.endswith("mary_had_a_little_lamb_480p.mp4")
    assert captured_opts[1].get("extractor_args") is None
    assert captured_opts[2]["extractor_args"] == {
        "youtube": {"player_client": ["web_embedded"]}
    }
    assert statuses == [
        "YouTube rejected the initial media URL. Retrying with an alternate YouTube client..."
    ]


def test_download_reuses_cookie_browser_after_info_retry(monkeypatch, tmp_path):
    captured_opts = []
    phase = {"downloads": 0}

    class FakeYoutubeDL:
        def __init__(self, opts):
            self.opts = opts
            captured_opts.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=False):
            if not download:
                if self.opts.get("cookiesfrombrowser") is None:
                    raise Exception("Sign in to confirm your age")
                return {"title": "Mary had a little lamb", "id": "SFFSZQCnU_M", "formats": []}
            phase["downloads"] += 1
            return {"title": "Mary had a little lamb", "id": "SFFSZQCnU_M"}

        def prepare_filename(self, info):
            return str(Path(captured_opts[-1]["outtmpl"]).with_suffix(".mp4"))

    monkeypatch.setattr(youtube_downloader.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    output_path = YouTubeDownloader(
        str(tmp_path),
        preferred_browser="edge",
        auto_cookie_retry=True,
    ).download_video_only("https://www.youtube.com/watch?v=SFFSZQCnU_M")

    assert output_path.endswith(".mp4")
    assert captured_opts[0].get("cookiesfrombrowser") is None
    assert captured_opts[1]["cookiesfrombrowser"] == ("edge",)
    assert captured_opts[2]["cookiesfrombrowser"] == ("edge",)
    assert phase["downloads"] == 1


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


def test_normalizes_common_youtube_url_forms(tmp_path):
    downloader = YouTubeDownloader(str(tmp_path))

    urls = [
        "https://www.youtube.com/watch?v=SFFSZQCnU_M&list=RDSFFSZQCnU_M",
        "https://youtu.be/SFFSZQCnU_M?si=abc",
        "https://www.youtube.com/shorts/SFFSZQCnU_M",
        "https://www.youtube.com/live/SFFSZQCnU_M?feature=share",
        "https://music.youtube.com/watch?v=SFFSZQCnU_M&list=RDAMVM",
    ]

    for url in urls:
        assert downloader.validate_url(url)
        assert downloader.normalize_url(url) == "https://www.youtube.com/watch?v=SFFSZQCnU_M"


def test_download_uses_quality_suffix_and_selected_height(monkeypatch, tmp_path):
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

    monkeypatch.setattr(youtube_downloader.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    output_path = YouTubeDownloader(str(tmp_path)).download_video_only(
        "https://youtu.be/SFFSZQCnU_M?si=abc",
        quality="720p",
    )

    assert output_path.endswith("mary_had_a_little_lamb_720p.mp4")
    assert captured_opts[-1]["format"] == "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]"


def test_download_falls_back_to_closest_lower_quality(monkeypatch, tmp_path):
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
                "id": "SFFSZQCnU_M",
                "formats": [
                    {"height": 360, "vcodec": "h264", "acodec": "none"},
                    {"height": 720, "vcodec": "h264", "acodec": "none"},
                ],
            }

        def prepare_filename(self, info):
            return str(Path(captured_opts[-1]["outtmpl"]).with_suffix(".mp4"))

    monkeypatch.setattr(youtube_downloader.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    output_path = YouTubeDownloader(str(tmp_path)).download_video_only(
        "https://www.youtube.com/watch?v=SFFSZQCnU_M",
        quality="1080p",
    )

    assert output_path.endswith("mary_had_a_little_lamb_720p.mp4")
    assert captured_opts[-1]["format"] == "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]"


def test_download_retries_with_lower_quality_when_requested_format_is_unavailable(monkeypatch, tmp_path):
    captured_opts = []
    statuses = []

    class FakeYoutubeDL:
        def __init__(self, opts):
            self.opts = opts
            captured_opts.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=False):
            if not download:
                return {"title": "Mary had a little lamb", "id": "SFFSZQCnU_M", "formats": []}
            if "height<=1080" in self.opts["format"]:
                raise Exception(
                    "ERROR: [youtube] SFFSZQCnU_M: Requested format is not available. "
                    "Use --list-formats for a list of available formats"
                )
            return {"title": "Mary had a little lamb", "id": "SFFSZQCnU_M"}

        def prepare_filename(self, info):
            return str(Path(captured_opts[-1]["outtmpl"]).with_suffix(".mp4"))

    monkeypatch.setattr(youtube_downloader.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    output_path = YouTubeDownloader(
        str(tmp_path),
        status_callback=statuses.append,
    ).download_video_only("https://www.youtube.com/watch?v=SFFSZQCnU_M", quality="1080p")

    assert output_path.endswith("mary_had_a_little_lamb_720p.mp4")
    assert captured_opts[1]["format"] == "bestvideo[height<=1080][ext=mp4]/bestvideo[height<=1080]"
    assert captured_opts[2]["format"] == "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]"
    assert statuses == ["1080p is unavailable for this video. Trying a lower quality..."]


def test_progress_status_includes_download_metrics():
    progress = {
        "status": "downloading",
        "downloaded_bytes": 52 * 1024 * 1024,
        "total_bytes": 100 * 1024 * 1024,
        "speed": 2.5 * 1024 * 1024,
        "eta": 19,
    }

    assert _progress_percentage(progress) == 52
    assert _format_download_status(progress) == "Downloading: 52% - 52.0 MB - 2.5 MB/s - ETA 00:19"


def test_progress_without_total_stays_indeterminate_but_informative():
    progress = {
        "status": "downloading",
        "downloaded_bytes": 8 * 1024 * 1024,
        "speed": 1024 * 1024,
    }

    assert _progress_percentage(progress) == -1
    assert _format_download_status(progress) == "Downloading: 8.0 MB - 1.0 MB/s"


def test_youtube_error_messages_distinguish_common_failure_modes():
    assert "sign-in or cookies" in _format_youtube_error(Exception("Sign in to confirm your age"))
    assert "private" in _format_youtube_error(Exception("Private video"))
    assert "region restricted" in _format_youtube_error(
        Exception("This video is not available in your country")
    )
    assert "network or proxy" in _format_youtube_error(Exception("Unable to download webpage"))
    forbidden_message = _format_youtube_error(Exception("HTTP Error 403: Forbidden"))
    assert "YouTube rejected" in forbidden_message
    assert "network or proxy" not in forbidden_message


def test_cookie_browser_helpers_cover_supported_policy():
    assert browser_cookie_args("Chrome") == ("chrome",)
    assert should_retry_with_browser_cookies("Sign in to confirm your age")
    assert should_retry_with_browser_cookies("nsig extraction failed")
    assert not should_retry_with_browser_cookies("temporary socket timeout")
