"""YouTube video downloader module for Synthesia2MIDI"""

import logging
import os
import re
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from PySide6.QtCore import QObject, Signal, QThread

import yt_dlp


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

class DownloadProgress(QObject):
    """Signals for download progress updates"""
    progress = Signal(int)  # Percentage
    status = Signal(str)    # Status message
    finished = Signal(str)  # Finished with file path
    error = Signal(str)     # Error message
    
class YouTubeDownloaderThread(QThread):
    """Thread for downloading YouTube videos without blocking UI"""
    
    def __init__(self, url: str, output_dir: str, quality: str = '1080p'):
        super().__init__()
        self.url = url
        self.output_dir = output_dir
        self.quality = quality
        self.progress_handler = DownloadProgress()
        self._cancel_requested = False
        
    def run(self):
        """Run the download in a separate thread"""
        try:
            downloader = YouTubeDownloader(self.output_dir)
            
            # Connect progress hooks
            def progress_hook(d):
                if self._cancel_requested:
                    raise Exception("Download cancelled")
                    
                if d['status'] == 'downloading':
                    if 'total_bytes' in d:
                        percentage = int(d['downloaded_bytes'] * 100 / d['total_bytes'])
                        self.progress_handler.progress.emit(percentage)
                        speed = d.get('speed', 0)
                        if speed:
                            speed_mb = speed / 1024 / 1024
                            self.progress_handler.status.emit(f"Downloading: {percentage}% ({speed_mb:.1f} MB/s)")
                    else:
                        self.progress_handler.status.emit("Downloading...")
                elif d['status'] == 'finished':
                    self.progress_handler.status.emit("Processing...")
            
            # Download the video
            file_path = downloader.download_video_only(
                self.url, 
                quality=self.quality,
                progress_hook=progress_hook
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
        '480p': {'height': 480, 'note': 'Fast download, lower quality'},
        '720p': {'height': 720, 'note': 'Good balance of quality and size'},
        '1080p': {'height': 1080, 'note': 'High quality (recommended)'},
        '1440p': {'height': 1440, 'note': 'Very high quality'},
        '2160p': {'height': 2160, 'note': '4K quality (large files)'},
    }
    
    def __init__(self, output_dir: str = 'videos'):
        """Initialize downloader with output directory"""
        _ensure_cert_store()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def validate_url(self, url: str) -> bool:
        """Validate if URL is a valid YouTube URL"""
        youtube_regex = r'(https?://)?(www\.)?(youtube\.com/(watch\?v=|embed/)|youtu\.be/)[\w-]+'
        return bool(re.match(youtube_regex, url))
    
    def get_video_info(self, url: str) -> Optional[Dict[str, Any]]:
        """Get video information without downloading"""
        if not self.validate_url(url):
            raise ValueError("Invalid YouTube URL")
            
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'upload_date': info.get('upload_date', 'Unknown'),
                    'description': info.get('description', ''),
                    'thumbnail': info.get('thumbnail', ''),
                }
            except Exception as e:
                raise Exception(f"Failed to get video info: {str(e)}")
    
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
    
    def download_video_only(self, url: str, quality: str = '1080p', 
                          progress_hook=None) -> Optional[str]:
        """Download video-only stream from YouTube
        
        Args:
            url: YouTube video URL
            quality: Quality preset (480p, 720p, 1080p, 1440p, 2160p)
            progress_hook: Optional callback for progress updates
            
        Returns:
            Path to downloaded file or None if failed
        """
        if not self.validate_url(url):
            raise ValueError("Invalid YouTube URL")
            
        if quality not in self.QUALITY_PRESETS:
            quality = '1080p'
            
        height = self.QUALITY_PRESETS[quality]['height']
        
        # First get video info to determine folder name
        ydl_opts_info = {
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'Unknown')
            
        # Create sanitized folder name
        folder_name = self.sanitize_filename(video_title)
        # Convert to lowercase and replace spaces with underscores for consistency
        folder_name = folder_name.lower().replace(' ', '_').replace('-', '_')
        # Remove multiple underscores
        folder_name = re.sub(r'_+', '_', folder_name)
        
        # Create subfolder in videos directory
        video_folder = self.output_dir / folder_name
        video_folder.mkdir(exist_ok=True)
        
        # Configure download options
        ydl_opts = {
            'outtmpl': str(video_folder / f'{folder_name}.%(ext)s'),
            # Select best video format up to specified quality, prefer mp4
            'format': f'bestvideo[height<={height}][ext=mp4]/bestvideo[height<={height}]',
            'quiet': False,
            'no_warnings': False,
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',  # Convert to mp4 if needed
            }],
        }
        
        if progress_hook:
            ydl_opts['progress_hooks'] = [progress_hook]
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
                
                # Get the actual filename
                filename = ydl.prepare_filename(info)
                # Handle potential format conversion
                if not filename.endswith('.mp4'):
                    base = os.path.splitext(filename)[0]
                    mp4_path = f"{base}.mp4"
                    if os.path.exists(mp4_path):
                        filename = mp4_path
                        
                return filename
                
            except Exception as e:
                raise Exception(f"Download failed: {str(e)}")
    
    def get_available_qualities(self, url: str) -> Dict[str, Dict[str, Any]]:
        """Get available quality options for a video"""
        if not self.validate_url(url):
            raise ValueError("Invalid YouTube URL")
            
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                formats = info.get('formats', [])
                
                available_qualities = {}
                
                for preset, details in self.QUALITY_PRESETS.items():
                    height = details['height']
                    # Find best format for this quality
                    matching_formats = [
                        f for f in formats 
                        if f.get('vcodec') != 'none' 
                        and f.get('acodec') == 'none'  # Video only
                        and f.get('height', 0) <= height
                    ]
                    
                    if matching_formats:
                        # Sort by height (descending) then by filesize
                        best = max(matching_formats, 
                                 key=lambda x: (x.get('height', 0), x.get('filesize', 0)))
                        
                        available_qualities[preset] = {
                            'available': True,
                            'actual_height': best.get('height'),
                            'filesize_mb': best.get('filesize', 0) / 1024 / 1024 if best.get('filesize') else None,
                            'format': best.get('ext', 'unknown'),
                            'note': details['note']
                        }
                    else:
                        available_qualities[preset] = {
                            'available': False,
                            'note': f"Not available (max available: {max([f.get('height', 0) for f in formats if f.get('vcodec') != 'none'])}p)"
                        }
                
                return available_qualities
                
            except Exception as e:
                raise Exception(f"Failed to get quality options: {str(e)}")


# Example usage
if __name__ == "__main__":
    downloader = YouTubeDownloader()
    
    # Test URL
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    # Get video info
    info = downloader.get_video_info(test_url)
    
    # Get available qualities
    qualities = downloader.get_available_qualities(test_url)
