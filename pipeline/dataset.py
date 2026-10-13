"""Read-only adapter for TRUE_Dataset's test split, without fact-check labels."""

import json
import re
from pathlib import Path

from utils.json_ld import public_http_url
from utils.records import normalize_date, normalize_platform

from .pipeline import validate_payload
from .retrieval import is_factcheck_text, is_factcheck_url

FRAME_EXTENSIONS = {".jpeg", ".jpg", ".png"}
VIDEO_EXTENSIONS = {
    ".mp4",
    ".webm",
    ".mov",
    ".mkv",
    ".avi",
    ".m4v",
    ".flv",
    ".ts",
    ".mpeg",
    ".mpg",
}


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", value
    ):
        raise ValueError("Test identifiers must be safe filenames")
    return value


def list_test_ids(dataset_root, limit=None):
    root = Path(dataset_root).resolve()
    manifest = root / "test_set.txt"
    if manifest.is_file():
        identifiers = [
            line.strip()
            for line in manifest.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    else:
        identifiers = [path.stem for path in sorted((root / "test").glob("*.json"))]
    seen = set()
    for identifier in identifiers:
        _identifier(identifier)
        if identifier.casefold() in seen:
            raise ValueError("Duplicate test identifier: " + identifier)
        seen.add(identifier.casefold())
    return identifiers if limit is None else identifiers[:limit]


def _safe_text(value):
    if not isinstance(value, str) or not value.strip() or is_factcheck_text(value):
        return None
    return value.strip()


def load_test_request(dataset_root, claim_id):
    root = Path(dataset_root).resolve()
    claim_id = _identifier(claim_id)
    with (root / "test" / (claim_id + ".json")).open(encoding="utf-8-sig") as handle:
        data = json.load(handle)
    video = data.get("video_information") or {}
    payload = {"claim": data.get("claim")}
    frame_dir = root / "test_output" / claim_id
    frames = (
        sorted(
            (
                path
                for path in frame_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in FRAME_EXTENSIONS
                and path.stem.isdecimal()
            ),
            key=lambda path: (int(path.stem), path.name),
        )
        if frame_dir.is_dir()
        else []
    )
    if frames:
        payload["keyframes"] = [str(path.resolve()) for path in frames]
    else:
        video_path = root / "test_video" / (claim_id + ".mp4")
        if not video_path.is_file():
            video_dir = root / "test_video" / claim_id
            videos = (
                sorted(
                    path
                    for path in video_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
                )
                if video_dir.is_dir()
                else []
            )
            video_path = videos[0] if videos else None
        if video_path:
            payload["video_path"] = str(video_path.resolve())

    post_url = _safe_text(video.get("video_url"))
    if post_url and public_http_url(post_url) and not is_factcheck_url(post_url):
        payload["post_url"] = post_url

    metadata = {}
    for source_field, target_field in (
        ("video_headline", "title"),
        ("video_description", "description"),
        ("platform", "platform"),
    ):
        value = _safe_text(video.get(source_field))
        if value:
            metadata[target_field] = (
                normalize_platform(value) if target_field == "platform" else value
            )
    upload_date = normalize_date(video.get("video_date"))
    if upload_date:
        metadata["upload_date"] = upload_date
    if is_factcheck_text(" ".join(metadata.values())):
        metadata = {"upload_date": upload_date} if upload_date else {}
    if metadata:
        payload["metadata"] = metadata
    transcript = _safe_text(video.get("video_transcript"))
    if transcript:
        payload["transcript"] = transcript
    return validate_payload(payload)
