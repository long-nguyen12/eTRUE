import re

from utils.json_ld import fetch_page_metadata, filled, public_http_url
from utils.utils import normalize_social_url
from utils.ytdlp import extract_info, has_video_stream, metadata_from_info


CORE_METADATA_FIELDS = (
    "title",
    "description",
    "uploader_name",
    "uploader_profile",
    "upload_date",
)


def metadata_status(metadata):
    count = sum(filled((metadata or {}).get(field)) for field in CORE_METADATA_FIELDS)
    if count == len(CORE_METADATA_FIELDS):
        return "complete"
    return "partial" if count else "missing"


def extract_media(url, session, cookie_file=None):
    normalized_url = normalize_social_url(url)["url"]
    result = {
        "url": normalized_url or None,
        "metadata": {},
        "field_sources": {},
        "errors": [],
    }
    if not normalized_url:
        result["video"] = {"decision": "discard", "reason": "missing_url", "error": None}
    elif not public_http_url(normalized_url):
        result["video"] = {"decision": "discard", "reason": "invalid_url", "error": None}
    else:
        try:
            info = extract_info(normalized_url, cookie_file)
            result["metadata"] = metadata_from_info(info, normalized_url)
            result["field_sources"] = {
                field: "yt_dlp"
                for field in CORE_METADATA_FIELDS
                if filled(result["metadata"].get(field))
            }
            if has_video_stream(info):
                result["video"] = {"decision": "keep", "reason": "video_format_found", "error": None}
            else:
                result["video"] = {"decision": "unknown", "reason": "no_video_format", "error": None}
        except Exception as error:
            message = re.sub(r"\s+", " ", str(error)).strip()[:500]
            result["errors"].append({"provider": "yt_dlp", "message": message})
            result["video"] = {"decision": "unknown", "reason": "probe_failed", "error": message}

        if metadata_status(result["metadata"]) != "complete":
            try:
                page = fetch_page_metadata(session, normalized_url)
                for field in CORE_METADATA_FIELDS:
                    if not filled(result["metadata"].get(field)) and filled(page.get(field)):
                        result["metadata"][field] = page[field]
                        result["field_sources"][field] = "html"
                if not filled(result["metadata"].get("webpage_url")) and page.get("webpage_url"):
                    result["metadata"]["webpage_url"] = page["webpage_url"]
                if not filled(result["metadata"].get("extractor")):
                    result["metadata"]["extractor"] = "html"
            except Exception as error:
                message = re.sub(r"\s+", " ", str(error)).strip()[:500]
                result["errors"].append({"provider": "html", "message": message})

    result["metadata_status"] = metadata_status(result["metadata"])
    result["missing_fields"] = [
        field for field in CORE_METADATA_FIELDS if not filled(result["metadata"].get(field))
    ]
    return result
