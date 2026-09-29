from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from pillars.source import OFFICIAL_SOURCES
from utils.evidence import add_evidence, remove_evidence_type
from utils.errors import optional_result
from utils.files import now, read_json, write_json
from utils.media_metadata import (
    CORE_METADATA_FIELDS,
    cookie_file_from_env,
    extract_media,
    public_http_url,
)
from utils.records import normalize_date, normalize_platform, sidecar_path
from utils.utils import normalize_social_url


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
    verification["source"]["uploader_name"] = OFFICIAL_SOURCES.get(platform)
    verification["source"]["uploader_profile"] = None
    verification["date"]["video_upload_date"] = normalize_date(video.get("video_date"))
    verification["motivation"]["original_caption"] = video.get("video_headline")
    verification["motivation"]["original_description"] = video.get("video_description")


def apply_metadata(extra, extraction, retrieved_at, source):
    remove_evidence_type(extra, "platform_metadata")
    metadata = extraction["metadata"]
    verification = extra["verification"]
    verification["source"]["metadata_status"] = extraction["metadata_status"]

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

    fields = ["verification.source.metadata_status"]
    fields.extend(
        METADATA_FIELD_PATHS[field]
        for field in CORE_METADATA_FIELDS
        if metadata.get(field)
    )
    add_evidence(
        extra,
        "platform_metadata",
        source,
        "Current video metadata extraction status: " + extraction["metadata_status"] + ".",
        fields=fields,
        retrieved_at=retrieved_at,
        metadata=metadata,
        field_sources=extraction["field_sources"],
        missing_fields=extraction["missing_fields"],
        errors=extraction["errors"],
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
                "metadata_status": extraction["metadata_status"],
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
            apply_metadata(extra, extraction, retrieved_at, evidence_source)

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
