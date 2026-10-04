import os
from pathlib import Path
from urllib.parse import urlsplit

from utils.json_ld import _absolute_url, _clean_text, _date_value, filled


def cookie_file_from_env():
    value = os.environ.get("YTDLP_COOKIE_FILE")
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError("YTDLP_COOKIE_FILE does not point to a readable file")
    try:
        with path.open("rb"):
            pass
    except OSError as error:
        raise FileNotFoundError("YTDLP_COOKIE_FILE does not point to a readable file") from error
    return str(path.resolve())


VIDEO_EXTENSIONS = {"mp4", "webm", "mov", "mkv", "flv", "avi", "m4v", "ts"}


def _format_has_video(item):
    if not isinstance(item, dict) or not item.get("url"):
        return False
    vcodec = item.get("vcodec")
    if isinstance(vcodec, str) and vcodec.casefold() == "none":
        return False
    if vcodec or item.get("height") or item.get("width") or item.get("fps"):
        return True
    extension = str(item.get("ext") or "").casefold()
    if extension in VIDEO_EXTENSIONS:
        return True
    path = urlsplit(str(item.get("url") or "")).path.casefold()
    return any(path.endswith("." + extension) for extension in VIDEO_EXTENSIONS)


def has_video_stream(info):
    if not isinstance(info, dict):
        return False
    if any(has_video_stream(entry) for entry in info.get("entries") or []):
        return True
    if any(_format_has_video(item) for item in info.get("formats") or []):
        return True
    if any(_format_has_video(item) for item in info.get("requested_formats") or []):
        return True
    return _format_has_video(info)


def _merged_info(info):
    if not isinstance(info, dict):
        return {}
    entry = next((item for item in info.get("entries") or [] if isinstance(item, dict)), None)
    merged = {key: value for key, value in info.items() if filled(value)}
    if entry:
        merged.update({key: value for key, value in entry.items() if filled(value)})
    return merged


def metadata_from_info(info, fallback_url):
    value = _merged_info(info)
    upload_date = _date_value(value.get("upload_date")) or _date_value(value.get("timestamp"))
    metadata = {
        "title": _clean_text(value.get("title"), 1000),
        "description": _clean_text(value.get("description"), 10000),
        "uploader_name": _clean_text(value.get("uploader") or value.get("channel"), 300),
        "uploader_profile": _absolute_url(
            value.get("uploader_url") or value.get("channel_url"), fallback_url
        ),
        "upload_date": upload_date,
        "webpage_url": _absolute_url(value.get("webpage_url"), fallback_url) or fallback_url,
        "extractor": _clean_text(value.get("extractor"), 200),
    }
    return {key: item for key, item in metadata.items() if filled(item)}


def extract_info(url, cookie_file=None):
    import yt_dlp

    options = {
        "quiet": True,
        "no_warnings": True,
        "no_color": True,
        "skip_download": True,
        "ignore_no_formats_error": True,
        "socket_timeout": 20,
        "retries": 1,
        "fragment_retries": 1,
        "extractor_retries": 1,
        "playlist_items": "1",
    }
    if cookie_file:
        options["cookiefile"] = cookie_file
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=False)
    return info
