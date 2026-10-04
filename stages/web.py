from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import re
from urllib.parse import urlsplit

from pillars.source import get_platform_source
from utils.evidence import add_evidence, remove_evidence_type
from utils.errors import optional_result
from utils.files import now, read_json, write_json
from utils.json_ld import public_http_url
from utils.media_metadata import (
    CORE_METADATA_FIELDS,
    extract_media,
)
from utils.records import normalize_date, normalize_platform, sidecar_path
from utils.utils import normalize_social_url
from utils.ytdlp import cookie_file_from_env


METADATA_FIELD_PATHS = {
    "uploader_name": "verification.source.uploader_name",
    "uploader_profile": "verification.source.uploader_profile",
    "upload_date": "verification.date.video_upload_date",
    "title": "verification.motivation.original_caption",
    "description": "verification.motivation.original_description",
}


def wayback_earliest(url, session):
    url = normalize_social_url(url)["url"]
    endpoint = "https://web.archive.org/cdx/search/cdx"
    response = session.get(
        endpoint,
        params={
            "url": url,
            "output": "json",
            "fl": "timestamp,original,statuscode",
            "filter": "statuscode:200",
            "limit": 1,
            "from": 1990,
        },
        timeout=20,
    )
    response.raise_for_status()
    rows = response.json()
    if len(rows) < 2:
        return None
    timestamp, original, status = rows[1]
    return {
        "date": datetime.strptime(timestamp[:8], "%Y%m%d").date().isoformat(),
        "url": original,
        "snapshot_url": "https://web.archive.org/web/" + timestamp + "/" + original,
        "status_code": status,
    }


def restore_true_metadata(record, extra):
    video = record["data"].get("video_information") or {}
    verification = extra["verification"]
    platform = normalize_platform(
        (extra.get("normalized_video_information") or {}).get("platform")
        or video.get("platform")
    )
    official_source = get_platform_source(platform)
    verification["source"]["uploader_name"] = official_source["name"] if official_source else None
    verification["source"]["uploader_profile"] = None
    verification["date"]["video_upload_date"] = normalize_date(video.get("video_date"))
    verification["motivation"]["original_caption"] = video.get("video_headline")
    verification["motivation"]["original_description"] = video.get("video_description")


def extract_uploader_from_url(url):
    """Infer an account identifier only when the URL path contains one."""
    parsed = urlsplit(str(url or "").strip())
    host = (parsed.hostname or "").lower()
    parts = [part for part in parsed.path.split("/") if part]
    if not parts or parsed.scheme not in {"http", "https"}:
        return None, None
    if host == "youtube.com" or host.endswith(".youtube.com"):
        handle = re.match(r"^/@([^/]+)(?:/|$)", parsed.path)
        if handle:
            name = handle.group(1)
            return name, "https://www.youtube.com/@" + name
        channel = re.match(r"^/(channel|user|c)/([^/]+)(?:/|$)", parsed.path)
        if channel:
            return channel.group(2), "https://www.youtube.com/" + channel.group(1) + "/" + channel.group(2)
    if host == "facebook.com" or host.endswith(".facebook.com"):
        if parts[0].lower() not in {
            "video", "videos", "watch", "reel", "share", "groups", "pages",
            "plugins", "login", "profile.php", "story.php", "permalink.php", "l.php",
        }:
            return parts[0], "https://www.facebook.com/" + parts[0]
    if host in {"twitter.com", "x.com"} or host.endswith((".twitter.com", ".x.com")):
        if parts[0].lower() not in {"i", "status", "intent", "share", "home", "search", "explore", "hashtag", "login"}:
            return "@" + parts[0], "https://x.com/" + parts[0]
    return None, None


def apply_metadata(extra, extraction, retrieved_at, source, original_url=None):
    remove_evidence_type(extra, "platform_metadata")
    remove_evidence_type(extra, "uploader_url_pattern")
    metadata = extraction["metadata"]
    verification = extra["verification"]

    if metadata.get("uploader_name"):
        verification["source"]["uploader_name"] = metadata["uploader_name"]
    if metadata.get("uploader_profile"):
        verification["source"]["uploader_profile"] = metadata["uploader_profile"]
    if metadata.get("upload_date"):
        verification["date"]["video_upload_date"] = metadata["upload_date"]
    if metadata.get("title"):
        verification["motivation"]["original_caption"] = metadata["title"]
    if metadata.get("description"):
        verification["motivation"]["original_description"] = metadata["description"]

    fields = [
        METADATA_FIELD_PATHS[field]
        for field in CORE_METADATA_FIELDS
        if metadata.get(field)
    ]
    add_evidence(
        extra,
        "platform_metadata",
        source,
        "Platform metadata retrieved for the available fields.",
        fields=fields,
        retrieved_at=retrieved_at,
        metadata=metadata,
        field_sources=extraction["field_sources"],
        missing_fields=extraction["missing_fields"],
        errors=extraction["errors"],
    )

    uploader_url = original_url or extraction.get("url") or source
    uploader_name, uploader_profile = extract_uploader_from_url(uploader_url)
    inferred_fields = []
    for field, value in (("uploader_name", uploader_name), ("uploader_profile", uploader_profile)):
        if value and not verification["source"].get(field):
            verification["source"][field] = value
            inferred_fields.append(METADATA_FIELD_PATHS[field])
    if inferred_fields:
        add_evidence(
            extra,
            "uploader_url_pattern",
            uploader_url,
            "Uploader account identifier inferred from the video URL path.",
            fields=inferred_fields,
            retrieved_at=retrieved_at,
        )


def run(records, output):
    import requests
    from dotenv import load_dotenv

    load_dotenv()
    cookie_file = cookie_file_from_env()
    metadata_session = requests.Session()
    archive_session = requests.Session()
    metadata_session.headers["User-Agent"] = "eTRUE-research-dataset/1.0"
    archive_session.headers["User-Agent"] = "eTRUE-research-dataset/1.0"

    try:
        for number, record in enumerate(records, 1):
            original_url = (record["data"].get("video_information") or {}).get("video_url")
            url = normalize_social_url(original_url)["url"]
            retrieved_at = now()
            archive_errors = []
            with ThreadPoolExecutor(max_workers=2) as executor:
                metadata_future = executor.submit(
                    extract_media, url, metadata_session, cookie_file
                )
                archive_future = (
                    executor.submit(
                        optional_result,
                        archive_errors,
                        "wayback",
                        lambda: wayback_earliest(url, archive_session),
                    )
                    if public_http_url(url)
                    else None
                )
                extraction = metadata_future.result()
                archive = archive_future.result() if archive_future else None

            errors = list(extraction["errors"])
            errors.extend(
                {
                    "provider": "wayback",
                    "message": message.split(": ", 1)[-1][:500],
                }
                for message in archive_errors
            )
            result = {
                "retrieved_at": retrieved_at,
                "url": extraction["url"],
                "metadata": extraction["metadata"],
                "field_sources": extraction["field_sources"],
                "missing_fields": extraction["missing_fields"],
                "errors": errors,
            }
            if archive is not None:
                result["wayback"] = archive

            sidecar_file = sidecar_path(output, record["claim_id"])
            extra = read_json(sidecar_file)
            restore_true_metadata(record, extra)
            evidence_source = extraction["url"] or record["path"].as_posix()
            apply_metadata(extra, extraction, retrieved_at, evidence_source, original_url=original_url)

            if archive:
                provenance = extra["verification"]["provenance"]
                if not provenance.get("earliest_known_date") or archive["date"] < provenance["earliest_known_date"]:
                    provenance["earliest_known_date"] = archive["date"]
                    provenance["earliest_known_url"] = archive["url"]
                add_evidence(
                    extra,
                    "archive_record",
                    archive["snapshot_url"],
                    "Earliest successful Wayback snapshot for the video URL.",
                    fields=(
                        "verification.provenance.earliest_known_url",
                        "verification.provenance.earliest_known_date",
                        "verification.date.earliest_online_date",
                    ),
                    retrieved_at=retrieved_at,
                    archive=archive,
                )
                extra["verification"]["date"]["earliest_online_date"] = archive["date"]
            extra.setdefault("automation", {})["web"] = result
            write_json(sidecar_file, extra)
            if number % 10 == 0:
                print("web", number, "/", len(records), flush=True)
    finally:
        metadata_session.close()
        archive_session.close()
