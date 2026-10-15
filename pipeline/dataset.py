"""Read-only adapter for TRUE_Dataset's test split, without fact-check labels."""

import json
import re
from pathlib import Path

from utils.json_ld import public_http_url
from utils.records import normalize_date, normalize_platform

from .pipeline import validate_payload

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


def list_test_ids(dataset_root, limit=None):
    root = Path(dataset_root).resolve()
    file_set = root.parent / "test_set.txt"
    items = [
        line.strip()
        for line in file_set.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    print(f"Found {len(items)} test records in {file_set}", flush=True)
    return items if limit is None else items[:limit]


def load_record(dataset_root, claim_id):
    root = Path(dataset_root).resolve()
    with (root / (claim_id + ".json")).open(encoding="utf-8-sig") as handle:
        data = json.load(handle)
    video = data.get("video_information") or {}
    payload = {"claim": data.get("claim")}
    frame_dir = root.parent / "test_output" / claim_id
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
        video_path = root.parent / "test_video" / (claim_id + ".mp4")
        if video_path:
            payload["video_path"] = str(video_path.resolve())

    post_url = video.get("video_url")
    if post_url and public_http_url(post_url):
        payload["post_url"] = post_url

    metadata = {}
    for source_field, target_field in (
        ("video_headline", "title"),
        ("video_description", "description"),
        ("platform", "platform"),
    ):
        value = video.get(source_field)
        if value:
            metadata[target_field] = (
                normalize_platform(value) if target_field == "platform" else value
            )
    upload_date = normalize_date(video.get("video_date"))
    if upload_date:
        metadata["upload_date"] = upload_date
    if metadata:
        payload["metadata"] = metadata
    transcript = video.get("video_transcript")
    if transcript:
        payload["transcript"] = transcript
    return validate_payload(payload)
