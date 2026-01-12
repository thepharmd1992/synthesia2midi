"""
Cross-platform FFmpeg detection and execution utilities.

Provides reliable FFmpeg command execution across Windows, macOS, and Linux.
"""
import os
import sys
import subprocess
import shutil
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


def find_ffmpeg() -> Optional[str]:
    """
    Find FFmpeg executable path on the system.
    
    Returns:
        Path to ffmpeg executable, or None if not found
    """
    # First, check if ffmpeg is in PATH
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    
    # Platform-specific common locations
    if sys.platform == "win32":
        # Common Windows locations
        common_paths = [
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe",
            os.path.expanduser(r"~\ffmpeg\bin\ffmpeg.exe"),
        ]
        
        # Check if user has ffmpeg in the same directory as the script
        script_dir = os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else __file__)
        common_paths.append(os.path.join(script_dir, "ffmpeg.exe"))
        
        # Check project's ffmpeg directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        common_paths.append(os.path.join(project_root, "ffmpeg", "ffmpeg.exe"))
        
    elif sys.platform == "darwin":  # macOS
        # Common macOS locations
        common_paths = [
            "/usr/local/bin/ffmpeg",
            "/opt/homebrew/bin/ffmpeg",  # Apple Silicon Macs
            "/usr/bin/ffmpeg",
            os.path.expanduser("~/bin/ffmpeg"),
        ]
    else:  # Linux and other Unix-like
        common_paths = [
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/snap/bin/ffmpeg",
            os.path.expanduser("~/bin/ffmpeg"),
        ]
    
    # Check each path
    for path in common_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    
    return None


def check_ffmpeg_available() -> Tuple[bool, str]:
    """
    Check if FFmpeg is available and working.
    
    Returns:
        (is_available, message) tuple
    """
    ffmpeg_path = find_ffmpeg()
    
    if not ffmpeg_path:
        return False, "FFmpeg not found. Please install FFmpeg and ensure it's in your PATH."
    
    try:
        # Test if ffmpeg actually runs
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            # Extract version info
            version_line = result.stdout.split('\n')[0]
            return True, f"FFmpeg found: {version_line}"
        else:
            return False, f"FFmpeg found at {ffmpeg_path} but failed to run"
            
    except subprocess.TimeoutExpired:
        return False, f"FFmpeg found at {ffmpeg_path} but timed out"
    except Exception as e:
        return False, f"Error testing FFmpeg: {e}"


def run_ffmpeg_command(args: List[str], **kwargs) -> subprocess.CompletedProcess:
    """
    Run an FFmpeg command with proper cross-platform handling.
    
    Args:
        args: Command arguments (without 'ffmpeg' prefix)
        **kwargs: Additional arguments for subprocess.run
        
    Returns:
        CompletedProcess instance
        
    Raises:
        FileNotFoundError: If FFmpeg is not found
        subprocess.CalledProcessError: If FFmpeg command fails
    """
    ffmpeg_path = find_ffmpeg()
    
    if not ffmpeg_path:
        raise FileNotFoundError(
            "FFmpeg not found. Please install FFmpeg:\n"
            "- Windows: Download from https://ffmpeg.org/download.html\n"
            "- macOS: Install with 'brew install ffmpeg'\n"
            "- Linux: Install with your package manager (e.g., 'apt install ffmpeg')"
        )
    
    # Build full command
    cmd = [ffmpeg_path] + args

    logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")
    
    # Set default kwargs
    defaults = {
        'capture_output': True,
        'text': True,
        'check': True,
        'shell': False,
    }
    defaults.update(kwargs)
    
    try:
        return subprocess.run(cmd, **defaults)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg command failed: {e.stderr}")
        raise


def convert_video_to_frames(video_path: str, output_dir: str, 
                          quality: int = 2, format_filter: str = "bgr24") -> bool:
    """
    Convert video to image sequence using FFmpeg.
    
    Args:
        video_path: Path to input video
        output_dir: Directory for output frames
        quality: JPEG quality (1-31, lower is better)
        format_filter: Pixel format filter
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Build output pattern
        output_pattern = os.path.join(output_dir, "frame_%06d.jpg")
        
        # Build FFmpeg arguments
        args = [
            '-i', video_path,
            '-q:v', str(quality),
            '-vf', f'format={format_filter}',
            output_pattern
        ]
        
        # Run conversion
        run_ffmpeg_command(args, check=False)
        
        # Verify frames were created
        import glob
        frames = glob.glob(os.path.join(output_dir, "frame_*.jpg"))
        
        if frames:
            logger.info(f"Successfully converted video to {len(frames)} frames")
            return True
        else:
            logger.error("No frames were created")
            return False
            
    except Exception as e:
        logger.error(f"Failed to convert video to frames: {e}")
        return False


def get_video_info(video_path: str) -> Optional[dict]:
    """
    Get video information using FFmpeg probe.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video info or None if failed
    """
    try:
        args = [
            '-i', video_path,
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
            '-of', 'json'
        ]
        
        # Use ffprobe if available, otherwise parse ffmpeg output
        ffprobe_path = shutil.which("ffprobe")
        if ffprobe_path:
            cmd = [ffprobe_path] + args
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                stream = data.get('streams', [{}])[0]
                return {
                    'width': stream.get('width'),
                    'height': stream.get('height'),
                    'fps': eval(stream.get('r_frame_rate', '0/1')),  # Convert fraction to float
                    'frame_count': stream.get('nb_frames')
                }
        
        # Fallback: parse ffmpeg stderr
        result = run_ffmpeg_command(['-i', video_path], check=False)
        
        # Parse output for video info
        # This is a basic parser - might need refinement
        info = {}
        for line in result.stderr.split('\n'):
            if 'Video:' in line:
                # Extract resolution
                import re
                res_match = re.search(r'(\d+)x(\d+)', line)
                if res_match:
                    info['width'] = int(res_match.group(1))
                    info['height'] = int(res_match.group(2))
                
                # Extract fps
                fps_match = re.search(r'(\d+(?:\.\d+)?)\s*fps', line)
                if fps_match:
                    info['fps'] = float(fps_match.group(1))
        
        return info if info else None
        
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return None
