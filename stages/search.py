"""Find possible earlier versions of a video on the web."""

import base64
import os
import re

from utils.evidence import add_evidence, mark_automated, remove_evidence_type
from utils.files import now, read_json, trim, write_json
from utils.records import frame_paths, sidecar_path


BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
GOOGLE_VISION_URL = "https://vision.googleapis.com/v1/images:annotate"


def quoted_phrase(value):
    text = re.sub(r"\s+", " ", str(value or "")).strip().replace('"', "")
    words = text.split()
    return '"' + " ".join(words[:18]) + '"' if len(words) >= 4 else None


def build_queries(record, sidecar):
    """Use transcript, caption, claim, and credited accounts."""
    data = record["data"]
    video = data.get("video_information") or {}
    verification = sidecar["verification"]
    values = [
        sidecar["normalized_video_information"].get("video_transcript"),
        verification["motivation"].get("original_caption") or video.get("video_headline"),
        data.get("claim"),
    ]

    source = verification.get("source") or {}
    accounts = [source.get("uploader_name"), source.get("original_source_name")]
    accounts += [
        handle
        for evidence in sidecar.get("evidence") or []
        for handle in re.findall(r"@[A-Za-z0-9_.]+", str(evidence.get("observation") or ""))
    ]

    queries = [quoted_phrase(value) for value in values]
    queries += ['"' + str(account).strip().replace('"', "") + '"' for account in accounts if account]
    unique = []
    for query in queries:
        if query and query.casefold() not in {item.casefold() for item in unique}:
            unique.append(query)
    return unique[:5]


def brave_results(session, api_key, query):
    response = session.get(
        BRAVE_URL,
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        params={"q": query, "count": 5},
        timeout=20,
    )
    response.raise_for_status()
    results = []
    for rank, item in enumerate((response.json().get("web") or {}).get("results") or [], 1):
        if item.get("url"):
            results.append(
                {
                    "url": item["url"],
                    "title": trim(item.get("title"), 300),
                    "description": trim(item.get("description"), 500),
                    "rank": rank,
                    "query": query,
                }
            )
    return results


def google_image_matches(session, api_key, frame):
    """Return source pages containing a matching keyframe image."""
    response = session.post(
        GOOGLE_VISION_URL,
        params={"key": api_key},
        json={
            "requests": [
                {
                    "image": {"content": base64.b64encode(frame.read_bytes()).decode("ascii")},
                    "features": [{"type": "WEB_DETECTION", "maxResults": 10}],
                }
            ]
        },
        timeout=30,
    )
    response.raise_for_status()
    annotation = (response.json().get("responses") or [{}])[0]
    if annotation.get("error"):
        raise ValueError(annotation["error"].get("message") or str(annotation["error"]))

    pages = []
    web = annotation.get("webDetection") or {}
    for item in web.get("pagesWithMatchingImages") or []:
        if not item.get("url"):
            continue
        images = (item.get("fullMatchingImages") or []) + (item.get("partialMatchingImages") or [])
        pages.append(
            {
                "url": item["url"],
                "title": trim(item.get("pageTitle"), 300),
                "image_urls": [image["url"] for image in images if image.get("url")],
            }
        )
    return {"pages": pages}


def search_record(record, sidecar, source, session, brave_key, vision_key):
    result = {"status": "ok", "retrieved_at": now(), "results": [], "errors": []}

    if brave_key:
        seen = set()
        for query in build_queries(record, sidecar):
            try:
                for item in brave_results(session, brave_key, query):
                    if item["url"] not in seen:
                        seen.add(item["url"])
                        result["results"].append(item)
            except Exception as error:
                result["errors"].append("Brave " + query + ": " + str(error))
    else:
        result["brave_status"] = "skipped: BRAVE_SEARCH_API_KEY is not set"

    frames = frame_paths(source, record)
    if vision_key and frames:
        frame = frames[len(frames) // 2]
        try:
            result["image_search"] = google_image_matches(session, vision_key, frame)
            result["image_search"]["frame"] = frame.relative_to(source).as_posix()
        except Exception as error:
            result["errors"].append("Google Vision: " + str(error))
    elif not vision_key:
        result["image_search_status"] = "skipped: GOOGLE_CLOUD_VISION_API_KEY is not set"
    else:
        result["image_search_status"] = "skipped: no keyframes"
    return result


def add_search_evidence(sidecar, result):
    """Search hits are review candidates, never verified field evidence."""
    remove_evidence_type(sidecar, "provenance_search_candidate")
    remove_evidence_type(sidecar, "provenance_image_candidate")
    for item in result.get("results") or []:
        observation = ". ".join(
            value for value in (item.get("title"), item.get("description")) if value
        )
        add_evidence(
            sidecar,
            "provenance_search_candidate",
            item["url"],
            trim(observation, 700),
            provider="Brave Search",
            query=item.get("query"),
            rank=item.get("rank"),
            retrieved_at=result.get("retrieved_at"),
        )

    image_search = result.get("image_search") or {}
    for page in image_search.get("pages") or []:
        image_urls = page.get("image_urls") or (
            (page.get("full_images") or []) + (page.get("partial_images") or [])
        )
        add_evidence(
            sidecar,
            "provenance_image_candidate",
            page["url"],
            page.get("title") or "Page containing a matching keyframe image.",
            provider="Google Cloud Vision Web Detection",
            frame=image_search.get("frame"),
            image_urls=image_urls,
            retrieved_at=result.get("retrieved_at"),
        )


def run(records, source, output, force=False):
    import requests

    cache = output / "cache" / "search"
    cache.mkdir(parents=True, exist_ok=True)
    brave_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    vision_key = os.environ.get("GOOGLE_CLOUD_VISION_API_KEY")
    session = requests.Session()
    session.headers["User-Agent"] = "eTRUE-research-dataset/1.0"

    for number, record in enumerate(records, 1):
        cached = cache / (record["claim_id"] + ".json")
        sidecar_file = sidecar_path(output, record["claim_id"])
        sidecar = read_json(sidecar_file)

        if cached.exists() and not force:
            result = read_json(cached)
        elif brave_key or vision_key:
            result = search_record(record, sidecar, source, session, brave_key, vision_key)
            write_json(cached, result)
        else:
            result = {
                "status": "skipped",
                "reason": "BRAVE_SEARCH_API_KEY and GOOGLE_CLOUD_VISION_API_KEY are not set",
            }

        if result.get("status") == "ok":
            add_search_evidence(sidecar, result)
        mark_automated(sidecar, "search", result)
        write_json(sidecar_file, sidecar)
        if number % 10 == 0:
            print("search", number, "/", len(records), flush=True)
