"""Find possible earlier versions of a video on the web."""

import os
import re

from pillars.provenance import apply_search_provenance
from stages.search_clients import (
    brave_results,
    ddgs_results,
    google_image_matches,
    select_keyframes,
)
from stages.search_metadata import (
    candidate_page_metadata,
    public_http_url,
)
from stages.web import wayback_earliest
from utils.evidence import add_evidence, remove_evidence_type
from utils.errors import optional_result
from utils.files import now, read_json, trim, write_json
from utils.records import frame_paths, sidecar_path


def quoted_phrase(value):
    text = re.sub(r"\s+", " ", str(value or "")).strip().replace('"', "")
    words = text.split()
    return '"' + " ".join(words[:18]) + '"' if len(words) >= 4 else None


def build_query_specs(record, extra):
    """Use transcript, caption, claim, and credited accounts."""
    data = record["data"]
    video = data.get("video_information") or {}
    verification = extra["verification"]
    values = [
        (
            "transcript",
            extra["normalized_video_information"].get("video_transcript"),
        ),
        (
            "caption",
            verification["motivation"].get("original_caption")
            or video.get("video_headline"),
        ),
        ("claim", data.get("claim")),
    ]

    source = verification.get("source") or {}
    accounts = [source.get("uploader_name"), source.get("original_source_name")]
    accounts += [
        handle
        for evidence in extra.get("evidence") or []
        for handle in re.findall(
            r"@[A-Za-z0-9_.]+", str(evidence.get("observation") or "")
        )
    ]

    queries = [
        {"query": quoted_phrase(value), "query_type": query_type}
        for query_type, value in values
    ]
    queries += [
        {
            "query": '"' + str(account).strip().replace('"', "") + '"',
            "query_type": "account",
        }
        for account in accounts
        if account
    ]
    unique = []
    for item in queries:
        query = item["query"]
        if query and query.casefold() not in {
            existing["query"].casefold() for existing in unique
        }:
            unique.append(item)
    return unique[:5]


def enrich_candidates(session, candidates, errors):
    """Add page metadata and archive dates without failing the search stage."""
    for item in candidates:
        metadata = optional_result(
            errors,
            item["url"] + " metadata",
            lambda item=item: candidate_page_metadata(session, item["url"]),
        )
        if metadata:
            item.update(metadata)

        archive_target = (
            item.get("canonical_url") or item.get("final_url") or item["url"]
        )
        if not public_http_url(archive_target):
            continue
        archive = optional_result(
            errors,
            item["url"] + " Wayback",
            lambda archive_target=archive_target: wayback_earliest(
                archive_target, session
            ),
        )
        if archive:
            item["archive_first_seen"] = archive["date"]
            item["archive_url"] = archive["snapshot_url"]


def enrich_image_pages(session, image_search):
    enrich_candidates(
        session,
        image_search.get("pages") or [],
        image_search.setdefault("errors", []),
    )
    return image_search


def enrich_text_results(session, result, maximum=5):
    """Attach page and archive metadata to a bounded set of search hits."""
    enrich_candidates(
        session,
        (result.get("results") or [])[:maximum],
        result["errors"],
    )
    return result


def search_record(record, extra, source, session, brave_key, vision_key):
    result = {
        "status": "ok",
        "retrieved_at": now(),
        "results": [],
        "errors": [],
    }
    result["search_provider"] = "Brave Search" if brave_key else "DDGS"
    seen = set()
    for query_spec in build_query_specs(record, extra):
        query = query_spec["query"]
        items = optional_result(
            result["errors"],
            result["search_provider"] + " " + query,
            lambda query=query: brave_results(session, brave_key, query)
            if brave_key
            else ddgs_results(query),
        )
        for item in items or []:
            if item["url"] not in seen:
                seen.add(item["url"])
                item["query_type"] = query_spec["query_type"]
                result["results"].append(item)
    if session is not None:
        enrich_text_results(session, result)
    frames = frame_paths(source, record)
    if vision_key and frames:
        selected_frames = select_keyframes(frames)
        image_search = optional_result(
            result["errors"],
            "Google Vision",
            lambda: google_image_matches(
                session,
                vision_key,
                selected_frames,
                source=source,
            ),
        )
        if image_search is not None:
            result["image_search"] = enrich_image_pages(session, image_search)
    elif not vision_key:
        result["image_search_status"] = (
            "skipped: GOOGLE_CLOUD_VISION_API_KEY is not set"
        )
    else:
        result["image_search_status"] = "skipped: no keyframes"
    return result


def add_search_evidence(extra, result):
    """Search hits are review candidates, never verified field evidence."""
    remove_evidence_type(extra, "provenance_search_candidate")
    remove_evidence_type(extra, "provenance_image_candidate")
    evidence_by_url = {}
    for item in result.get("results") or []:
        observation = ". ".join(
            value for value in (item.get("title"), item.get("description")) if value
        )
        evidence_id = add_evidence(
            extra,
            "provenance_search_candidate",
            item["url"],
            trim(observation, 700),
            provider=result.get("search_provider") or "Brave Search",
            query=item.get("query"),
            query_type=item.get("query_type"),
            rank=item.get("rank"),
            retrieved_at=result.get("retrieved_at"),
            canonical_url=item.get("canonical_url"),
            author=item.get("author"),
            site_name=item.get("site_name"),
            published_at=item.get("published_at"),
            archive_first_seen=item.get("archive_first_seen"),
            archive_url=item.get("archive_url"),
        )
        evidence_by_url[item["url"]] = evidence_id
        if item.get("canonical_url"):
            evidence_by_url[item["canonical_url"]] = evidence_id

    image_search = result.get("image_search") or {}
    for page in image_search.get("pages") or []:
        image_urls = page.get("image_urls") or (
            (page.get("full_images") or []) + (page.get("partial_images") or [])
        )
        observation = ". ".join(
            value
            for value in (
                page.get("title"),
                page.get("description"),
                page.get("context_excerpt"),
            )
            if value
        )
        evidence_id = add_evidence(
            extra,
            "provenance_image_candidate",
            page["url"],
            trim(observation, 1200)
            or "Page containing a matching keyframe image.",
            provider="Google Cloud Vision Web Detection",
            match_types=page.get("match_types"),
            matched_frames=page.get("matched_frames"),
            match_count=page.get("match_count"),
            image_urls=image_urls,
            canonical_url=page.get("canonical_url"),
            author=page.get("author"),
            site_name=page.get("site_name"),
            published_at=page.get("published_at"),
            archive_first_seen=page.get("archive_first_seen"),
            archive_url=page.get("archive_url"),
            retrieved_at=result.get("retrieved_at"),
        )
        evidence_by_url[page["url"]] = evidence_id
        if page.get("canonical_url"):
            evidence_by_url[page["canonical_url"]] = evidence_id
    return evidence_by_url


def run(records, source, output):
    import requests
    from dotenv import load_dotenv

    load_dotenv()
    brave_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    vision_key = os.environ.get("GOOGLE_CLOUD_VISION_API_KEY")
    session = requests.Session()
    session.headers["User-Agent"] = "eTRUE-research-dataset/1.0"

    for number, record in enumerate(records, 1):
        sidecar_file = sidecar_path(output, record["claim_id"])
        extra = read_json(sidecar_file)
        result = search_record(
            record, extra, source, session, brave_key, vision_key
        )

        if result.get("status") == "ok":
            evidence_by_url = add_search_evidence(extra, result)
            apply_search_provenance(
                extra,
                result.get("image_search") or {},
                evidence_by_url,
                result.get("results") or [],
            )
        extra.setdefault("automation", {})["search"] = result
        write_json(sidecar_file, extra)
        if number % 10 == 0:
            print("search", number, "/", len(records), flush=True)
