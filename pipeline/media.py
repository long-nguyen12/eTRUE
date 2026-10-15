"""Prepare supplied media without audio analysis or dataset writes."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from stages.vision import evenly_spaced
from utils.ytdlp import cookie_file_from_env


def _download_video(url, work_dir):
    import yt_dlp

    options = {
        "quiet": True,
        "no_warnings": True,
        "no_color": True,
        "noplaylist": True,
        "format": "bestvideo/best",
        "outtmpl": str(work_dir / "video.%(ext)s"),
        "socket_timeout": 20,
        "retries": 1,
        "fragment_retries": 1,
        "extractor_retries": 1,
    }
    cookie_file = cookie_file_from_env()
    if cookie_file:
        options["cookiefile"] = cookie_file
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=True)
        path = Path(downloader.prepare_filename(info)).resolve()
    return path


def _sample_frames(video_path, work_dir):
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "format=duration:stream=duration", "-of", "json", str(video_path),
        ],
        capture_output=True, text=True, check=True, timeout=60,
    )
    details = json.loads(probe.stdout)
    duration = details.get("format", {}).get("duration")
    if not duration:
        duration = next((stream.get("duration") for stream in details.get("streams", [])
                         if stream.get("duration")), None)
    duration = float(duration)
    # Midpoints avoid seeking beyond the last frame on short videos.
    timestamps = [duration * (index + 0.5) / 4 for index in range(4)]
    frames, errors, extracted_timestamps = [], [], []
    for index, timestamp in enumerate(timestamps, 1):
        frame = work_dir / ("frame_%02d.jpg" % index)
        try:
            subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", "%.6f" % timestamp,
                    "-i", str(video_path), "-map", "0:v:0", "-frames:v", "1", "-an",
                    "-q:v", "2", str(frame),
                ],
                capture_output=True, text=True, check=True, timeout=60,
            )
            frames.append(frame)
            extracted_timestamps.append(round(timestamp, 6))
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            errors.append({"stage": "frames", "message": str(error)[:500]})
    return frames, extracted_timestamps, errors


def prepare_media(payload: dict, work_dir: Path) -> dict:
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    result = {"frames": [], "video_path": None, "frame_timestamps": [], "errors": []}
    if payload.get("keyframes"):
        result["frames"] = evenly_spaced([Path(frame) for frame in payload["keyframes"]], 4)
        return result
    video_path = payload.get("video_path")
    if video_path:
        video_path = Path(video_path).resolve()
    else:
        try:
            video_path = _download_video(payload["post_url"], work_dir)
        except Exception as error:
            print(f"Error downloading video: {error}")
            return result
    result["video_path"] = str(video_path)
    try:
        frames, timestamps, errors = _sample_frames(video_path, work_dir)
        result.update(frames=frames, frame_timestamps=timestamps, errors=errors)
    except (OSError, subprocess.SubprocessError, ValueError, TypeError) as error:
        print(f"Error sampling frames: {error}")
    return result
