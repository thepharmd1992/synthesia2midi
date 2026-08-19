"""YouTube video downloader module for Synthesia2MIDI"""

import logging
import os
import re
import shutil
from typing import Optional, Dict, Any, Tuple, Callable
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from PySide6.QtCore import QObject, Signal, QThread

import yt_dlp

from synthesia2midi.runtime_paths import detect_runtime_paths


_SUPPORTED_JS_RUNTIMES = ("node", "deno", "bun", "quickjs")
SUPPORTED_COOKIE_BROWSERS = ("chrome", "edge", "safari")
QUALITY_ORDER = ("1080p", "720p", "480p")
_COMMON_RUNTIME_DIRS = (
    Path.home() / ".local" / "bin",
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
    Path("/usr/bin"),
)


def _ensure_cert_store():
    """
    On some macOS Python installs the default cert store is missing, which causes
    yt-dlp SSL failures. Point OpenSSL to certifi's bundle if not already set.
    """
    try:
        import certifi

        ca_bundle = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", ca_bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)
    except Exception as exc:  # pragma: no cover - defensive only
        logging.warning("Could not set cert bundle for yt-dlp: %s", exc)


def _find_js_runtime_path(runtime: str) -> Optional[str]:
    if runtime == "deno":
        try:
            deno_path = detect_runtime_paths().deno_path()
        except OSError as exc:
            logging.debug("Skipping inaccessible Deno runtime candidate: %s", exc)
            deno_path = None
        if deno_path is not None:
            return str(deno_path)

    try:
        runtime_path = shutil.which(runtime)
    except OSError as exc:
        logging.debug("Could not search PATH for JavaScript runtime %s: %s", runtime, exc)
        runtime_path = None
    if runtime_path:
        return runtime_path

    executable_name = f"{runtime}.exe" if os.name == "nt" else runtime
    for directory in _COMMON_RUNTIME_DIRS:
        candidate = directory / executable_name
        try:
            is_executable = candidate.is_file() and os.access(candidate, os.X_OK)
        except OSError as exc:
            logging.debug("Skipping inaccessible JavaScript runtime candidate %s: %s", candidate, exc)
            continue
        if is_executable:
            return str(candidate)

    return None


def _discover_js_runtimes() -> Dict[str, Dict[str, str]]:
    runtimes = {}
    for runtime in _SUPPORTED_JS_RUNTIMES:
        runtime_path = _find_js_runtime_path(runtime)
        if runtime_path:
            runtimes[runtime] = {"path": runtime_path}
    return runtimes


def browser_cookie_args(browser: str) -> Tuple[str]:
    normalized = (browser or "").strip().lower()
    if normalized not in SUPPORTED_COOKIE_BROWSERS:
        raise ValueError(f"Unsupported browser for cookies-from-browser: {browser}")
    return (normalized,)


def should_retry_with_browser_cookies(error: str | Exception) -> bool:
    normalized = str(error).lower()
    return any(
        token in normalized
        for token in (
            "sign in",
            "cookies",
            "confirm your age",
            "age-restricted",
            "members only",
            "private video",
            "bot",
            "challenge",
            "javascript runtime",
            "nsig",
        )
    )


def should_retry_with_lower_quality(error: str | Exception) -> bool:
    return "requested format is not available" in str(error).lower()


def should_retry_with_embedded_client(error: str | Exception) -> bool:
    normalized = str(error).lower()
    return "http error 403" in normalized or "403: forbidden" in normalized


def _youtube_ydl_opts(base_opts: Dict[str, Any], *, browser_cookie: str | None = None) -> Dict[str, Any]:
    opts = dict(base_opts)
    ffmpeg_path = detect_runtime_paths().ffmpeg_path()
    if ffmpeg_path is not None:
        opts["ffmpeg_location"] = str(ffmpeg_path.parent)
    js_runtimes = _discover_js_runtimes()
    if js_runtimes:
        opts["js_runtimes"] = js_runtimes
        remote_components = set(opts.get("remote_components", []))
        remote_components.add("ejs:github")
        opts["remote_components"] = sorted(remote_components)
    if browser_cookie:
        opts["cookiesfrombrowser"] = browser_cookie_args(browser_cookie)
    return opts


def _format_youtube_error(error: Exception) -> str:
    message = str(error)
    normalized = message.lower()
    if "requested format is not available" in normalized:
        return f"{message}. This video does not offer that quality."
    if should_retry_with_embedded_client(error):
        return (
            f"{message}. YouTube rejected the media request. "
            "This can happen when YouTube requires additional playback authorization."
        )
    if (
        "sign in" in normalized
        or "cookies" in normalized
        or "age-restricted" in normalized
        or "confirm your age" in normalized
    ):
        return f"{message}. YouTube requires sign-in or cookies for this video."
    if "private video" in normalized or "this video is private" in normalized:
        return f"{message}. This video is private."
    if "not available in your country" in normalized or "region" in normalized:
        return f"{message}. This video is region restricted."
    if "cookies from browser" in normalized or "browser cookies" in normalized:
        return f"{message}. Browser cookie access failed."
    if "challenge" in normalized or "javascript runtime" in normalized or "nsig" in normalized:
        return f"{message}. YouTube challenge solving failed."
    if (
        "unable to download webpage" in normalized
        or "http error" in normalized
        or "network" in normalized
        or "proxy" in normalized
    ):
        return f"{message}. This looks like a network or proxy failure."
    if "This video is not available" in message and not _discover_js_runtimes():
        return (
            f"{message}. No JavaScript runtime was found for YouTube challenge solving. "
            "Install Node.js, Deno, Bun, or QuickJS, then retry."
        )
    return message


def _format_eta(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    try:
        seconds = int(seconds)
    except (TypeError, ValueError):
        return None

    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _format_mb(byte_count: Optional[float]) -> Optional[str]:
    if not byte_count:
        return None
    return f"{byte_count / 1024 / 1024:.1f} MB"


def _progress_percentage(progress: Dict[str, Any]) -> int:
    total_bytes = progress.get("total_bytes") or progress.get("total_bytes_estimate")
    downloaded_bytes = progress.get("downloaded_bytes")
    if not total_bytes or not downloaded_bytes:
        return -1
    return max(0, min(100, int(downloaded_bytes * 100 / total_bytes)))


def _format_download_status(progress: Dict[str, Any]) -> str:
    percentage = _progress_percentage(progress)
    parts = [f"Downloading: {percentage}%" if percentage >= 0 else "Downloading"]

    downloaded_mb = _format_mb(progress.get("downloaded_bytes"))
    if downloaded_mb:
        parts.append(downloaded_mb)

    speed_mb = _format_mb(progress.get("speed"))
    if speed_mb:
        parts.append(f"{speed_mb}/s")

    eta = _format_eta(progress.get("eta"))
    if eta:
        parts.append(f"ETA {eta}")

    if len(parts) == 1:
        return "Downloading..."
    if percentage < 0:
        return f"Downloading: {' - '.join(parts[1:])}"
    return " - ".join(parts)


class DownloadProgress(QObject):
    """Signals for download progress updates"""
    progress = Signal(int)  # Percentage
    status = Signal(str)    # Status message
    finished = Signal(str)  # Finished with file path
    error = Signal(str)     # Error message
    
class YouTubeDownloaderThread(QThread):
    """Thread for downloading YouTube videos without blocking UI"""
    
    def __init__(self, url: str, output_dir: str, quality: str = '1080p', overwrite: bool = False):
        super().__init__()
        self.url = url
        self.output_dir = output_dir
        self.quality = quality
        self.overwrite = overwrite
        self.progress_handler = DownloadProgress()
        self._cancel_requested = False
        
    def run(self):
        """Run the download in a separate thread"""
        try:
            downloader = YouTubeDownloader(
                self.output_dir,
                preferred_browser=getattr(self, "preferred_browser", None),
                auto_cookie_retry=getattr(self, "auto_cookie_retry", True),
                status_callback=self.progress_handler.status.emit,
            )
            self.progress_handler.progress.emit(-1)
            self.progress_handler.status.emit("Resolving video...")
            
            # Connect progress hooks
            def progress_hook(d):
                if self._cancel_requested:
                    raise Exception("Download cancelled")
                    
                if d['status'] == 'downloading':
                    self.progress_handler.progress.emit(_progress_percentage(d))
                    self.progress_handler.status.emit(_format_download_status(d))
                elif d['status'] == 'finished':
                    self.progress_handler.progress.emit(-1)
                    self.progress_handler.status.emit("Processing video...")
            
            # Download the video
            file_path = downloader.download_video_only(
                self.url, 
                quality=self.quality,
                progress_hook=progress_hook,
                overwrite=self.overwrite,
            )
            
            if file_path:
                self.progress_handler.finished.emit(file_path)
            else:
                self.progress_handler.error.emit("Download failed")
                
        except Exception as e:
            self.progress_handler.error.emit(str(e))
    
    def cancel(self):
        """Request cancellation of download"""
        self._cancel_requested = True

class YouTubeDownloader:
    """YouTube video downloader for Synthesia2MIDI"""
    
    QUALITY_PRESETS = {
        '480p': {'height': 480, 'note': 'Fastest processing, highest calibration risk'},
        '720p': {'height': 720, 'note': 'Faster processing, higher calibration risk'},
        '1080p': {'height': 1080, 'note': 'Highest detail'},
    }
    
    def __init__(
        self,
        output_dir: str = 'videos',
        *,
        preferred_browser: str | None = None,
        auto_cookie_retry: bool = True,
        status_callback: Callable[[str], None] | None = None,
    ):
        """Initialize downloader with output directory"""
        _ensure_cert_store()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preferred_browser = self._normalize_browser(preferred_browser)
        self.auto_cookie_retry = bool(auto_cookie_retry)
        self.status_callback = status_callback

    @staticmethod
    def _normalize_browser(browser: str | None) -> str | None:
        if not browser:
            return None
        normalized = browser.strip().lower()
        return normalized if normalized in SUPPORTED_COOKIE_BROWSERS else None

    def _emit_status(self, message: str) -> None:
        if self.status_callback:
            self.status_callback(message)

    def _with_cookie_retry(self, callback, *, status_message: str):
        try:
            return callback(None), None
        except Exception as exc:
            if not self.auto_cookie_retry or not self.preferred_browser:
                raise
            if not should_retry_with_browser_cookies(exc):
                raise
            self._emit_status(
                f"{status_message} Retrying with {self.preferred_browser.title()} browser cookies..."
            )
            return callback(self.preferred_browser), self.preferred_browser
        
    def validate_url(self, url: str) -> bool:
        """Validate if URL is a valid YouTube URL"""
        try:
            self.normalize_url(url)
        except ValueError:
            return False
        return True

    def normalize_url(self, url: str) -> str:
        """Normalize supported YouTube URL forms to a single-video watch URL."""
        url = url.strip()
        if not url:
            raise ValueError("Invalid YouTube URL")
        if not re.match(r"^https?://", url):
            url = f"https://{url}"

        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]

        video_id = None
        path_parts = [part for part in parsed.path.split("/") if part]

        if host == "youtu.be" and path_parts:
            video_id = path_parts[0]
        elif host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
            if parsed.path == "/watch":
                video_id = parse_qs(parsed.query).get("v", [None])[0]
            elif len(path_parts) >= 2 and path_parts[0] in {"embed", "shorts", "live"}:
                video_id = path_parts[1]

        if not video_id or not re.match(r"^[A-Za-z0-9_-]{11}$", video_id):
            raise ValueError("Invalid YouTube URL")

        return f"https://www.youtube.com/watch?v={video_id}"
    
    def get_video_info(self, url: str) -> Optional[Dict[str, Any]]:
        """Get video information without downloading"""
        url = self.normalize_url(url)

        def fetch_info(browser_cookie: str | None) -> Dict[str, Any]:
            ydl_opts = _youtube_ydl_opts({
                'quiet': True,
                'no_warnings': True,
                'noplaylist': True,  # Avoid pulling entire mixes/radio playlists
            }, browser_cookie=browser_cookie)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'upload_date': info.get('upload_date', 'Unknown'),
                    'description': info.get('description', ''),
                    'thumbnail': info.get('thumbnail', ''),
                    'available_qualities': self._available_qualities_for_formats(info.get('formats', [])),
                }

        try:
            info, _cookie_browser = self._with_cookie_retry(
                fetch_info,
                status_message="Video info request failed.",
            )
            return info
        except Exception as e:
            raise Exception(f"Failed to get video info: {_format_youtube_error(e)}")
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file system usage"""
        # Remove invalid characters including brackets and other problematic chars
        # This includes: < > : " / \ | ? * [ ] { } ( ) ' ` ! @ # $ % ^ & + = ; ,
        invalid_chars = r'[<>:"/\\|?*\[\]{}()\'`!@#$%^&+=;,]'
        filename = re.sub(invalid_chars, '_', filename)

        # Replace multiple spaces or underscores with single underscore
        filename = re.sub(r'[\s_]+', '_', filename)

        # Remove leading/trailing underscores and dots
        filename = filename.strip('_.')

        # Ensure filename doesn't start with a dot (hidden file on Unix)        
        if filename.startswith('.'):
            filename = filename[1:]

        # Strip non-ASCII characters (e.g., emojis) to avoid path issues on Windows/OpenCV
        filename = filename.encode("ascii", "ignore").decode()

        # Limit length
        max_length = 200
        if len(filename) > max_length:
            filename = filename[:max_length]
            
        # Final cleanup: ensure no trailing underscore after truncation
        filename = filename.rstrip('_')
        
        # If filename is empty after sanitization, use a default
        if not filename:
            filename = 'video'
            
        return filename
    
    def _folder_name_for_title(self, video_title: str) -> str:
        folder_name = self.sanitize_filename(video_title)
        folder_name = folder_name.lower().replace(' ', '_').replace('-', '_')
        return re.sub(r'_+', '_', folder_name)

    def get_download_path(self, video_title: str, quality: str = "1080p") -> Path:
        if quality not in self.QUALITY_PRESETS:
            quality = "1080p"
        folder_name = self._folder_name_for_title(video_title)
        return self.output_dir / folder_name / f"{folder_name}_{quality}.mp4"

    @classmethod
    def _available_qualities_for_formats(cls, formats: list) -> Dict[str, Dict[str, Any]]:
        video_formats = [
            fmt
            for fmt in formats
            if fmt.get("vcodec") != "none" and fmt.get("acodec") == "none"
        ]
        max_available_height = max(
            ((fmt.get("height", 0) or 0) for fmt in video_formats),
            default=0,
        )
        available_qualities = {}

        for preset, details in cls.QUALITY_PRESETS.items():
            height = details["height"]
            matching_formats = [
                fmt for fmt in video_formats if (fmt.get("height", 0) or 0) <= height
            ]

            if matching_formats:
                best = max(
                    matching_formats,
                    key=lambda fmt: ((fmt.get("height", 0) or 0), (fmt.get("filesize", 0) or 0)),
                )
                available_qualities[preset] = {
                    "available": True,
                    "actual_height": best.get("height"),
                    "filesize_mb": (
                        (best.get("filesize", 0) or 0) / 1024 / 1024
                        if best.get("filesize")
                        else None
                    ),
                    "format": best.get("ext", "unknown"),
                    "note": details["note"],
                }
            else:
                available_qualities[preset] = {
                    "available": False,
                    "note": f"Not available (max available: {max_available_height}p)",
                }

        return available_qualities

    def _quality_for_available_formats(self, requested_quality: str, formats: list) -> str:
        requested_height = self.QUALITY_PRESETS[requested_quality]["height"]
        video_heights = {
            fmt.get("height")
            for fmt in formats
            if fmt.get("vcodec") != "none" and fmt.get("acodec") == "none" and fmt.get("height")
        }

        for quality, details in sorted(
            self.QUALITY_PRESETS.items(),
            key=lambda item: item[1]["height"],
            reverse=True,
        ):
            height = details["height"]
            if height <= requested_height and any(candidate >= height for candidate in video_heights):
                return quality

        return requested_quality

    @staticmethod
    def _quality_fallback_chain(requested_quality: str) -> list[str]:
        if requested_quality not in QUALITY_ORDER:
            return list(QUALITY_ORDER)
        start_index = QUALITY_ORDER.index(requested_quality)
        return list(QUALITY_ORDER[start_index:])

    def download_video_only(self, url: str, quality: str = '1080p',
                          progress_hook=None, overwrite: bool = False) -> Optional[str]:
        """Download video-only stream from YouTube
        
        Args:
            url: YouTube video URL
            quality: Quality preset (480p, 720p, 1080p, 1440p, 2160p)
            progress_hook: Optional callback for progress updates
            
        Returns:
            Path to downloaded file or None if failed
        """
        url = self.normalize_url(url)
            
        if quality not in self.QUALITY_PRESETS:
            quality = '1080p'
            
        def fetch_info(browser_cookie: str | None):
            ydl_opts_info = _youtube_ydl_opts({
                'quiet': True,
                'no_warnings': True,
                'noplaylist': True,  # Avoid accidentally downloading full mixes
            }, browser_cookie=browser_cookie)
            with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
                return ydl.extract_info(url, download=False)

        try:
            info, info_browser = self._with_cookie_retry(
                fetch_info,
                status_message="Video info request failed.",
            )
            video_title = info.get('title', 'Unknown')
            quality = self._quality_for_available_formats(quality, info.get("formats", []))
        except Exception as e:
            raise Exception(f"Failed to get video info: {_format_youtube_error(e)}")

        folder_name = self._folder_name_for_title(video_title)
        
        # Create subfolder in videos directory
        video_folder = self.output_dir / folder_name
        video_folder.mkdir(exist_ok=True)
        target_path = self.get_download_path(video_title, quality)
        if target_path.exists() and not overwrite:
            return str(target_path)
        
        def perform_download(
            attempt_quality: str,
            browser_cookie: str | None,
            *,
            player_client: str | None = None,
        ) -> str:
            attempt_height = self.QUALITY_PRESETS[attempt_quality]['height']
            base_opts = {
                'outtmpl': str(video_folder / f'{folder_name}_{attempt_quality}.%(ext)s'),
                # Select best video format up to specified quality, prefer mp4
                'format': f'bestvideo[height<={attempt_height}][ext=mp4]/bestvideo[height<={attempt_height}]',
                'quiet': False,
                'no_warnings': False,
                'noplaylist': True,  # Do not expand mixes/playlist links
                'postprocessors': [{
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4',  # Convert to mp4 if needed
                }],
            }
            if player_client:
                base_opts['extractor_args'] = {
                    'youtube': {'player_client': [player_client]},
                }
            ydl_opts = _youtube_ydl_opts(base_opts, browser_cookie=browser_cookie)

            if progress_hook:
                ydl_opts['progress_hooks'] = [progress_hook]

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                if not filename.endswith('.mp4'):
                    base = os.path.splitext(filename)[0]
                    mp4_path = f"{base}.mp4"
                    if os.path.exists(mp4_path):
                        filename = mp4_path
                return filename

        def attempt_download_with_quality_fallback(
            browser_cookie: str | None,
            *,
            player_client: str | None = None,
        ) -> str:
            last_error: Exception | None = None
            for attempt_quality in self._quality_fallback_chain(quality):
                try:
                    return perform_download(
                        attempt_quality,
                        browser_cookie,
                        player_client=player_client,
                    )
                except Exception as exc:
                    last_error = exc
                    if not should_retry_with_lower_quality(exc) or attempt_quality == QUALITY_ORDER[-1]:
                        raise
                    self._emit_status(
                        f"{attempt_quality} is unavailable for this video. Trying a lower quality..."
                    )

            if last_error is not None:
                raise last_error
            raise Exception("Download failed")

        def attempt_download_with_client_fallback(browser_cookie: str | None) -> str:
            try:
                return attempt_download_with_quality_fallback(browser_cookie)
            except Exception as exc:
                if browser_cookie is not None or not should_retry_with_embedded_client(exc):
                    raise
                self._emit_status(
                    "YouTube rejected the initial media URL. "
                    "Retrying with an alternate YouTube client..."
                )
                return attempt_download_with_quality_fallback(
                    None,
                    player_client="web_embedded",
                )

        try:
            if info_browser is not None:
                return attempt_download_with_quality_fallback(info_browser)
            filename, _cookie_browser = self._with_cookie_retry(
                attempt_download_with_client_fallback,
                status_message="Download failed.",
            )
            return filename
        except Exception as e:
            raise Exception(f"Download failed: {_format_youtube_error(e)}")
    
    def get_available_qualities(self, url: str) -> Dict[str, Dict[str, Any]]:
        """Get available quality options for a video"""
        url = self.normalize_url(url)

        def fetch_qualities(browser_cookie: str | None):
            ydl_opts = _youtube_ydl_opts({
                'quiet': True,
                'no_warnings': True,
            }, browser_cookie=browser_cookie)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return self._available_qualities_for_formats(info.get('formats', []))

        try:
            qualities, _cookie_browser = self._with_cookie_retry(
                fetch_qualities,
                status_message="Quality lookup failed.",
            )
            return qualities
        except Exception as e:
            raise Exception(f"Failed to get quality options: {_format_youtube_error(e)}")


# Example usage
if __name__ == "__main__":
    downloader = YouTubeDownloader()
    
    # Test URL
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    # Get video info
    info = downloader.get_video_info(test_url)
    
    # Get available qualities
    qualities = downloader.get_available_qualities(test_url)
