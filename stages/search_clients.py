"""Search provider and reverse-image-search clients."""

import base64
import re
from html import unescape

from utils.files import trim

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
GOOGLE_VISION_URL = "https://vision.googleapis.com/v1/images:annotate"


def brave_results(session, api_key, query):
    response = session.get(
        BRAVE_URL,
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        params={"q": query, "count": 5},
        timeout=20,
    )
    response.raise_for_status()
    results = []
    for rank, item in enumerate(
        (response.json().get("web") or {}).get("results") or [], 1
    ):
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


def ddgs_results(query):
    from ddgs import DDGS

    results = []
    for rank, item in enumerate(DDGS(timeout=20).text(query, max_results=5), 1):
        url = item.get("href") or item.get("url")
        if url:
            results.append(
                {
                    "url": url,
                    "title": trim(item.get("title"), 300),
                    "description": trim(item.get("body"), 500),
                    "rank": rank,
                    "query": query,
                }
            )
    return results


def select_keyframes(frames, maximum=3):
    if len(frames) <= maximum:
        return list(frames)
    indexes = [
        round(index * (len(frames) - 1) / (maximum - 1))
        for index in range(maximum)
    ]
    return [frames[index] for index in indexes]


def clean_html_text(value, length):
    text = re.sub(r"<[^>]+>", " ", unescape(str(value or "")))
    return trim(re.sub(r"\s+", " ", text), length)


def frame_name(frame, source):
    if source and frame.is_relative_to(source):
        return frame.relative_to(source).as_posix()
    return frame.name


def add_image_match(matches, item, matched_frame):
    url = item.get("url")
    if not url:
        return
    match = matches.setdefault(url, {"url": url, "matched_frames": []})
    if matched_frame not in match["matched_frames"]:
        match["matched_frames"].append(matched_frame)
    score = item.get("score")
    if isinstance(score, (int, float)) and score > match.get("score", -1):
        match["score"] = score


def image_match_list(matches):
    return sorted(
        matches.values(),
        key=lambda item: (-item.get("score", 0), item["url"]),
    )


def google_image_matches(session, api_key, frames, source=None):
    """Return and merge Web Detection results for up to three keyframes."""
    if hasattr(frames, "read_bytes"):
        frames = [frames]
    else:
        frames = list(frames)
    names = [frame_name(frame, source) for frame in frames]
    response = session.post(
        GOOGLE_VISION_URL,
        params={"key": api_key},
        json={
            "requests": [
                {
                    "image": {
                        "content": base64.b64encode(frame.read_bytes()).decode("ascii")
                    },
                    "features": [{"type": "WEB_DETECTION", "maxResults": 10}],
                }
                for frame in frames
            ]
        },
        timeout=30,
    )
    response.raise_for_status()
    annotations = response.json().get("responses") or []
    pages = {}
    entities = {}
    labels = {}
    full_matches = {}
    partial_matches = {}
    similar_images = {}
    errors = []

    for index, matched_frame in enumerate(names):
        annotation = annotations[index] if index < len(annotations) else {}
        if annotation.get("error"):
            message = annotation["error"].get("message") or str(annotation["error"])
            errors.append(matched_frame + ": " + message)
            continue
        if index >= len(annotations):
            errors.append(matched_frame + ": Google Vision returned no response")
            continue

        web = annotation.get("webDetection") or {}
        for item in web.get("webEntities") or []:
            description = clean_html_text(item.get("description"), 300)
            key = item.get("entityId") or description.casefold()
            if not key:
                continue
            entity = entities.setdefault(
                key,
                {
                    "entity_id": item.get("entityId"),
                    "description": description,
                    "matched_frames": [],
                },
            )
            if matched_frame not in entity["matched_frames"]:
                entity["matched_frames"].append(matched_frame)
            score = item.get("score")
            if isinstance(score, (int, float)) and score > entity.get("score", -1):
                entity["score"] = score

        for item in web.get("bestGuessLabels") or []:
            label = clean_html_text(item.get("label"), 300)
            key = (label.casefold(), item.get("languageCode"))
            if not label:
                continue
            value = labels.setdefault(
                key,
                {
                    "label": label,
                    "language_code": item.get("languageCode"),
                    "matched_frames": [],
                },
            )
            if matched_frame not in value["matched_frames"]:
                value["matched_frames"].append(matched_frame)

        for item in web.get("fullMatchingImages") or []:
            add_image_match(full_matches, item, matched_frame)
        for item in web.get("partialMatchingImages") or []:
            add_image_match(partial_matches, item, matched_frame)
        for item in web.get("visuallySimilarImages") or []:
            add_image_match(similar_images, item, matched_frame)

        for item in web.get("pagesWithMatchingImages") or []:
            url = item.get("url")
            if not url:
                continue
            page = pages.setdefault(
                url,
                {
                    "url": url,
                    "title": "",
                    "match_types": [],
                    "matched_frames": [],
                    "_full_images": {},
                    "_partial_images": {},
                },
            )
            title = clean_html_text(item.get("pageTitle"), 300)
            if title and not page["title"]:
                page["title"] = title
            if matched_frame not in page["matched_frames"]:
                page["matched_frames"].append(matched_frame)
            for match_type, field, target in (
                ("full", "fullMatchingImages", page["_full_images"]),
                ("partial", "partialMatchingImages", page["_partial_images"]),
            ):
                images = item.get(field) or []
                if images and match_type not in page["match_types"]:
                    page["match_types"].append(match_type)
                for image in images:
                    add_image_match(target, image, matched_frame)

    normalized_pages = []
    for page in pages.values():
        full_images = image_match_list(page.pop("_full_images"))
        partial_images = image_match_list(page.pop("_partial_images"))
        page["match_types"] = [
            match_type
            for match_type in ("full", "partial")
            if match_type in page["match_types"]
        ] or ["unspecified"]
        page["full_images"] = full_images
        page["partial_images"] = partial_images
        page["image_urls"] = list(
            dict.fromkeys(item["url"] for item in full_images + partial_images)
        )
        page["match_count"] = len(page["matched_frames"])
        normalized_pages.append(page)
    normalized_pages.sort(
        key=lambda page: (
            0 if "full" in page["match_types"] else 1,
            -page["match_count"],
            page["url"],
        )
    )
    for rank, page in enumerate(normalized_pages, 1):
        page["rank"] = rank

    return {
        "frames": names,
        "pages": normalized_pages[:10],
        "web_entities": sorted(
            entities.values(),
            key=lambda item: (-item.get("score", 0), item.get("description") or ""),
        ),
        "best_guess_labels": list(labels.values()),
        "full_matching_images": image_match_list(full_matches),
        "partial_matching_images": image_match_list(partial_matches),
        "visually_similar_images": image_match_list(similar_images),
        "errors": errors,
    }
