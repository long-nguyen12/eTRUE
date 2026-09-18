"""Build bounded evidence payloads for text extraction."""

import json

from stages.text_fields import TEXT_OUTPUT_FIELDS
from utils.files import trim
from utils.records import normalize_date


def bounded_value(value, length=1000):
    if isinstance(value, str):
        return trim(value, length)
    if isinstance(value, list):
        return value[:10]
    if isinstance(value, dict):
        return {key: bounded_value(item, length) for key, item in value.items()}
    return value


def bounded_web_retrieval(value):
    if not isinstance(value, dict):
        return None
    metadata = value.get("platform_metadata") or {}
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
    return {
        "url": value.get("url"),
        "retrieved_at": value.get("retrieved_at"),
        "platform_metadata": {
            key: trim(metadata[key], 1000) if key == "description" else metadata[key]
            for key in fields
            if metadata.get(key) is not None
        },
        "wayback": value.get("wayback"),
        "errors": (value.get("errors") or [])[:5],
    }


def text_input(record, extra):
    data = record["data"]
    video = data.get("video_information") or {}
    transcript = video.get("video_transcript")
    verification = extra.get("verification") or {}
    candidate_facts = {}
    for pillar, values in verification.items():
        for field, value in values.items():
            if field in TEXT_OUTPUT_FIELDS or "mismatch" in field or value is None or value == []:
                continue
            if field != "near_duplicate_matches":
                candidate_facts[pillar + "." + field] = bounded_value(value)
    vision = (extra.get("automation") or {}).get("vision") or {}
    if vision.get("status") != "ok":
        vision = None
    search = (extra.get("automation") or {}).get("search") or {}
    image_search = search.get("image_search") or {}
    reverse_image_candidates = None
    if image_search:
        pages = []
        fields = (
            "url",
            "canonical_url",
            "title",
            "description",
            "context_excerpt",
            "author",
            "site_name",
            "published_at",
            "archive_first_seen",
            "match_types",
            "matched_frames",
            "match_count",
        )
        for page in (image_search.get("pages") or [])[:5]:
            pages.append(
                {
                    key: trim(page[key], 400)
                    if key in {"description", "context_excerpt"}
                    else page[key]
                    for key in fields
                    if page.get(key) is not None
                }
            )
        reverse_image_candidates = {
            "status": "unverified_candidates",
            "web_entities": (image_search.get("web_entities") or [])[:10],
            "best_guess_labels": (image_search.get("best_guess_labels") or [])[:10],
            "pages": pages,
        }
    web_search_candidates = None
    if search.get("results"):
        fields = (
            "url",
            "canonical_url",
            "title",
            "description",
            "context_excerpt",
            "author",
            "site_name",
            "published_at",
            "archive_first_seen",
            "query",
            "query_type",
            "rank",
        )
        web_search_candidates = {
            "status": "unverified_candidates",
            "provider": search.get("search_provider"),
            "results": [
                {
                    key: trim(item[key], 400)
                    if key in {"description", "context_excerpt"}
                    else item[key]
                    for key in fields
                    if item.get(key) is not None
                }
                for item in search.get("results", [])[:5]
            ],
        }
    return {
        "claim": data.get("claim"),
        "fact_check_publisher": {"name": "Snopes", "url": data.get("url")},
        "fact_check_article": trim(data.get("content"), 5000),
        "current_video": {
            "url": video.get("video_url"),
            "platform": video.get("platform"),
            "upload_date": normalize_date(video.get("video_date")),
            "title": video.get("video_headline"),
            "description": trim(video.get("video_description"), 2000),
        },
        "transcript": trim(transcript, 3500),
        "candidate_facts": candidate_facts,
        "visual_analysis": vision,
        "web_retrieval": bounded_web_retrieval(
            (extra.get("automation") or {}).get("web")
        ),
        "web_search_candidates": web_search_candidates,
        "reverse_image_candidates": reverse_image_candidates,
        "fact_check_evidence": [
            {
                "id": evidence.get("id"),
                "type": evidence.get("type"),
                "source": evidence.get("source"),
                "observation": trim(evidence.get("observation"), 1200),
                "references": (evidence.get("references") or [])[:5],
            }
            for evidence in extra.get("evidence", [])
            if evidence.get("type") == "fact_check_evidence"
        ][:3],
    }


def build_text_grounding(record, extra):
    """Return the full prompt evidence and the stricter event-only evidence."""
    supplied_evidence = text_input(record, extra)
    grounding_text = json.dumps(supplied_evidence, ensure_ascii=False)
    current_video = supplied_evidence.get("current_video") or {}
    event_evidence = {
        "claim": supplied_evidence.get("claim"),
        "fact_check_article": supplied_evidence.get("fact_check_article"),
        "fact_check_evidence": supplied_evidence.get("fact_check_evidence"),
        "transcript": supplied_evidence.get("transcript"),
        "visual_analysis": supplied_evidence.get("visual_analysis"),
        "title": current_video.get("title"),
        "description": current_video.get("description"),
        "reverse_image_candidates": supplied_evidence.get("reverse_image_candidates"),
        "retrieved_event_candidates": [
            item
            for item in (
                (supplied_evidence.get("web_search_candidates") or {}).get("results")
                or []
            )
            if item.get("query_type") in {"transcript", "caption"}
        ],
    }
    event_grounding_text = json.dumps(event_evidence, ensure_ascii=False)
    return grounding_text, event_grounding_text
