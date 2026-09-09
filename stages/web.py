"""Retrieve public platform metadata and archive records."""

from datetime import datetime

from build_etrue import normalize_date
from utils.evidence import add_evidence, mark_automated
from utils.files import now, read_json, write_json
from utils.records import sidecar_path


WEB_CACHE_VERSION = 2


def wayback_earliest(url, session):
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


def platform_metadata(url):
    import yt_dlp

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignore_no_formats_error": True,
        "socket_timeout": 15,
        "retries": 1,
        "playlist_items": "1",
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=False)
    if info and info.get("entries"):
        info = next((entry for entry in info["entries"] if entry), info)
    fields = (
        "uploader",
        "uploader_url",
        "channel",
        "channel_url",
        "upload_date",
        "timestamp",
        "title",
        "description",
        "webpage_url",
        "extractor",
    )
    extracted = {field: info.get(field) for field in fields if info and info.get(field) is not None}
    return extracted


def run(records, output, force=False):
    """Retrieve current platform metadata and the first Wayback snapshot."""
    import requests

    cache = output / "cache" / "web"
    cache.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "eTRUE-research-dataset/1.0"
    for number, record in enumerate(records, 1):
        cached = cache / (record["claim_id"] + ".json")
        if cached.exists() and not force:
            result = read_json(cached)
        else:
            result = None
        if not result or result.get("cache_version") != WEB_CACHE_VERSION:
            url = (record["data"].get("video_information") or {}).get("video_url")

            if url and "https://www.youtube.com/embed/" in url:
                url = url.replace("https://www.youtube.com/embed/", "https://www.youtube.com/watch?v=")

            result = {
                "status": "ok",
                "cache_version": WEB_CACHE_VERSION,
                "retrieved_at": now(),
                "url": url,
                "errors": [],
            }
            if url:
                try:
                    result["platform_metadata"] = platform_metadata(url)
                except Exception as error:
                    result["errors"].append("platform_metadata: " + str(error))
                try:
                    result["wayback"] = wayback_earliest(url, session)
                except Exception as error:
                    result["errors"].append("wayback: " + str(error))
            write_json(cached, result)

        sidecar_file = sidecar_path(output, record["claim_id"])
        extra = read_json(sidecar_file)
        metadata = result.get("platform_metadata") or {}
        if metadata:
            source_fields = extra["verification"]["source"]
            source_fields["uploader_name"] = metadata.get("uploader") or metadata.get("channel")
            source_fields["uploader_profile"] = metadata.get("uploader_url") or metadata.get("channel_url")
            date = normalize_date(metadata.get("upload_date"))
            if date:
                extra["verification"]["date"]["video_upload_date"] = date
            motivation = extra["verification"]["motivation"]
            motivation["original_caption"] = metadata.get("title") or motivation.get("original_caption")
            motivation["original_description"] = metadata.get("description") or motivation.get(
                "original_description"
            )
            add_evidence(
                extra,
                "platform_metadata",
                result.get("url"),
                "Uploader, profile, title, description, and upload date retrieved from the platform.",
                fields=(
                    "verification.source.uploader_name",
                    "verification.source.uploader_profile",
                    "verification.date.video_upload_date",
                    "verification.motivation.original_caption",
                    "verification.motivation.original_description",
                ),
                retrieved_at=result.get("retrieved_at"),
                metadata=metadata,
            )
        archive = result.get("wayback")
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
                retrieved_at=result.get("retrieved_at"),
                archive=archive,
            )
            extra["verification"]["date"]["earliest_online_date"] = archive["date"]
        mark_automated(extra, "web", result)
        write_json(sidecar_file, extra)
        if number % 10 == 0:
            print("web", number, "/", len(records), flush=True)
