"""Find possible earlier versions of a video on the web."""

import base64
import ipaddress
import json
import os
import re
from html import unescape
from urllib.parse import urljoin, urlsplit

from dotenv import load_dotenv

# load_dotenv()

from build_etrue import normalize_date
from stages.web import wayback_earliest
from utils.evidence import add_evidence, mark_automated, remove_evidence_type
from utils.files import now, read_json, trim, write_json
from utils.records import frame_paths, sidecar_path

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
GOOGLE_VISION_URL = "https://vision.googleapis.com/v1/images:annotate"
SEARCH_CACHE_VERSION = 2


def quoted_phrase(value):
    text = re.sub(r"\s+", " ", str(value or "")).strip().replace('"', "")
    words = text.split()
    return '"' + " ".join(words[:18]) + '"' if len(words) >= 4 else None


def build_queries(record, extra):
    """Use transcript, caption, claim, and credited accounts."""
    data = record["data"]
    video = data.get("video_information") or {}
    verification = extra["verification"]
    values = [
        extra["normalized_video_information"].get("video_transcript"),
        verification["motivation"].get("original_caption")
        or video.get("video_headline"),
        data.get("claim"),
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

    queries = [quoted_phrase(value) for value in values]
    queries += [
        '"' + str(account).strip().replace('"', "") + '"'
        for account in accounts
        if account
    ]
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
    if source:
        try:
            return frame.relative_to(source).as_posix()
        except ValueError:
            pass
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
            match_type for match_type in ("full", "partial") if match_type in page["match_types"]
        ] or ["unspecified"]
        page["full_images"] = full_images
        page["partial_images"] = partial_images
        page["image_urls"] = list(
            dict.fromkeys(
                item["url"] for item in full_images + partial_images
            )
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


def public_http_url(url):
    try:
        parsed = urlsplit(str(url or "").strip())
        hostname = parsed.hostname
        if parsed.scheme not in {"http", "https"} or not hostname:
            return False
        if parsed.username or parsed.password or hostname.casefold() == "localhost":
            return False
        try:
            return ipaddress.ip_address(hostname).is_global
        except ValueError:
            return True
    except ValueError:
        return False


def json_ld_nodes(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from json_ld_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from json_ld_nodes(child)


def candidate_page_metadata(session, url):
    """Retrieve bounded public HTML and extract structured provenance metadata."""
    if not public_http_url(url):
        raise ValueError("candidate URL is not a public HTTP(S) URL")

    response = session.get(
        url,
        headers={"Range": "bytes=0-999999"},
        timeout=15,
        allow_redirects=True,
        stream=True,
    )
    try:
        response.raise_for_status()
        final_url = getattr(response, "url", None) or url
        if not public_http_url(final_url):
            raise ValueError("candidate redirected to a non-public URL")
        content_type = (response.headers.get("Content-Type") or "").lower()
        if content_type and "html" not in content_type:
            return {"final_url": final_url, "content_type": trim(content_type, 100)}

        content = bytearray()
        if hasattr(response, "iter_content"):
            for chunk in response.iter_content(chunk_size=65536):
                content.extend(chunk)
                if len(content) >= 1000000:
                    break
        else:
            raw = getattr(response, "content", b"")
            content.extend(raw[:1000000])
        encoding = getattr(response, "encoding", None) or "utf-8"
        html = bytes(content[:1000000]).decode(encoding, errors="replace")
    finally:
        if hasattr(response, "close"):
            response.close()

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    def meta_content(*names):
        for name in names:
            for attribute in ("property", "name", "itemprop"):
                tag = soup.find("meta", attrs={attribute: name})
                if tag and tag.get("content"):
                    return clean_html_text(tag["content"], 1000)
        return None

    title_tag = soup.find("title")
    title = meta_content("og:title", "twitter:title") or clean_html_text(
        title_tag.get_text(" ", strip=True) if title_tag else None,
        300,
    )
    description = meta_content(
        "og:description",
        "description",
        "twitter:description",
    )
    author = meta_content("author", "article:author")
    published_value = meta_content(
        "article:published_time",
        "datePublished",
        "date",
        "pubdate",
    )
    site_name = meta_content("og:site_name")

    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        try:
            structured = json.loads(script.string or script.get_text())
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        for node in json_ld_nodes(structured):
            published_value = published_value or node.get("datePublished") or node.get("uploadDate")
            if not author and node.get("author"):
                author_value = node["author"]
                if isinstance(author_value, list):
                    author_value = author_value[0] if author_value else None
                if isinstance(author_value, dict):
                    author_value = author_value.get("name")
                author = clean_html_text(author_value, 300)

    canonical = soup.find("link", rel="canonical")
    canonical_url = (
        urljoin(final_url, canonical.get("href"))
        if canonical and canonical.get("href")
        else final_url
    )
    if not public_http_url(canonical_url):
        canonical_url = final_url

    context = soup.find("article") or soup.find("main")
    context_excerpt = clean_html_text(
        context.get_text(" ", strip=True) if context else None,
        1200,
    )
    date_match = re.search(
        r"(?<!\d)(?:19|20)\d{2}(?:-\d{2}-\d{2}|\d{4})(?!\d)",
        str(published_value or ""),
    )
    published_at = normalize_date(date_match.group(0)) if date_match else None

    metadata = {
        "final_url": final_url,
        "canonical_url": canonical_url,
        "title": title,
        "description": description,
        "author": author,
        "site_name": site_name,
        "published_at": published_at,
        "context_excerpt": context_excerpt,
    }
    return {key: value for key, value in metadata.items() if value}


def enrich_image_pages(session, image_search):
    errors = image_search.setdefault("errors", [])
    for page in image_search.get("pages") or []:
        try:
            metadata = candidate_page_metadata(session, page["url"])
            page.update(metadata)
        except Exception as error:
            errors.append(page["url"] + " metadata: " + str(error))

        archive_target = page.get("canonical_url") or page.get("final_url") or page["url"]
        if not public_http_url(archive_target):
            continue
        try:
            archive = wayback_earliest(archive_target, session)
            if archive:
                page["archive_first_seen"] = archive["date"]
                page["archive_url"] = archive["snapshot_url"]
        except Exception as error:
            errors.append(page["url"] + " Wayback: " + str(error))
    return image_search


def search_record(record, extra, source, session, brave_key, vision_key):
    result = {
        "status": "ok",
        "cache_version": SEARCH_CACHE_VERSION,
        "retrieved_at": now(),
        "results": [],
        "errors": [],
    }
    if brave_key:
        result["search_provider"] = "Brave Search"
        seen = set()
        for query in build_queries(record, extra):
            try:
                for item in brave_results(session, brave_key, query):
                    if item["url"] not in seen:
                        seen.add(item["url"])
                        result["results"].append(item)
            except Exception as error:
                result["errors"].append("Brave " + query + ": " + str(error))
    elif not vision_key:
        result["search_provider"] = "DDGS"
        seen = set()
        for query in build_queries(record, extra):
            try:
                for item in ddgs_results(query):
                    if item["url"] not in seen:
                        seen.add(item["url"])
                        result["results"].append(item)
            except Exception as error:
                result["errors"].append("DDGS " + query + ": " + str(error))
    else:
        result["brave_status"] = "skipped: BRAVE_SEARCH_API_KEY is not set"
    print(result["results"])
    frames = frame_paths(source, record)
    if vision_key and frames:
        selected_frames = select_keyframes(frames)
        try:
            result["image_search"] = google_image_matches(
                session,
                vision_key,
                selected_frames,
                source=source,
            )
            enrich_image_pages(session, result["image_search"])
        except Exception as error:
            result["errors"].append("Google Vision: " + str(error))
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
    for item in result.get("results") or []:
        observation = ". ".join(
            value for value in (item.get("title"), item.get("description")) if value
        )
        add_evidence(
            extra,
            "provenance_search_candidate",
            item["url"],
            trim(observation, 700),
            provider=result.get("search_provider") or "Brave Search",
            query=item.get("query"),
            rank=item.get("rank"),
            retrieved_at=result.get("retrieved_at"),
        )

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
        add_evidence(
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
        extra = read_json(sidecar_file)

        if cached.exists() and not force:
            result = read_json(cached)
        else:
            result = None
        if not result or result.get("cache_version") != SEARCH_CACHE_VERSION:
            result = search_record(
                record, extra, source, session, brave_key, vision_key
            )
            write_json(cached, result)

        if result.get("status") == "ok":
            add_search_evidence(extra, result)
        mark_automated(extra, "search", result)
        write_json(sidecar_file, extra)
        if number % 10 == 0:
            print("search", number, "/", len(records), flush=True)
