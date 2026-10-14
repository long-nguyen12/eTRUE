import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

from stages.search_clients import (
    brave_results,
    ddgs_results,
    google_image_matches,
    select_keyframes,
)
from stages.web import wayback_earliest
from utils.json_ld import _date_value, json_ld_nodes, public_http_url
from utils.media_metadata import CORE_METADATA_FIELDS, extract_media

FACTCHECK_DOMAINS = {
    "snopes.com",
    "politifact.com",
    "factcheck.org",
    "fullfact.org",
    "checkyourfact.com",
    "leadstories.com",
    "factcheck.afp.com",
    "factcheckni.org",
    "factly.in",
    "boomlive.in",
    "altnews.in",
    "logicallyfacts.com",
    "maldita.es",
    "newtral.es",
    "chequeado.com",
    "correctiv.org",
    "pesacheck.org",
    "africacheck.org",
    "factcheck.ge",
    "factcrescendo.com",
    "verafiles.org",
    "verificat.cat",
    "science.feedback.org",
    "climatefeedback.org",
    "healthfeedback.org",
    "facta.news",
}
FACTCHECK_LABEL = re.compile(
    r"\bfact[\s_-]?check(?:ing|ed|er|ers)?\b|\bclaim\s*review\b",
    re.I,
)
FACTCHECK_CONCLUSION = re.compile(
    r"\b(?:snopes|politifact|full\s*fact|factcheck\.org|lead\s*stories|check\s*your\s*fact)\b",
    re.I,
)
RATING_CUE = re.compile(
    r"\b(?:rated|rating|verdict|conclud(?:ed|es)|debunk(?:ed|s)|determined|false|true|misleading|pants on fire)\b",
    re.I,
)
RATING_MARKER = re.compile(
    r"\b(?:our\s+)?(?:rating|verdict)\s*[:\-\u2013\u2014]\s*(?:(?:mostly|partly|partially)\s+)?"
    r"(?:true|false|misleading|unverified|mixture|misattributed|pants\s+on\s+fire)\b|"
    r"\bour\s+rating\s+(?:is\s+)?(?:(?:mostly|partly|partially)\s+)?(?:true|false|misleading)\b|"
    r"\bwe\s+rate(?:d)?\s+(?:this|the)\s+claim\s+(?:as\s+)?"
    r"(?:(?:mostly|partly|partially)\s+)?(?:true|false|misleading)\b",
    re.I,
)


def is_factcheck_url(url):
    try:
        parsed = urlsplit(str(url or ""))
        host = (parsed.hostname or "").lower().rstrip(".")
        return any(
            host == domain or host.endswith("." + domain)
            for domain in FACTCHECK_DOMAINS
        ) or bool(FACTCHECK_LABEL.search(unquote(parsed.path)))
    except ValueError:
        return True


def is_factcheck_text(value):
    text = str(value or "")
    return bool(
        FACTCHECK_LABEL.search(text)
        or RATING_MARKER.search(text)
        or (FACTCHECK_CONCLUSION.search(text) and RATING_CUE.search(text))
    )


def _url_key(url):
    parsed = urlsplit(url)
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path or "/",
            parsed.query,
            "",
        )
    )


def _clean(value, maximum=12000):
    return re.sub(r"\s+", " ", str(value or "")).strip()[:maximum]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _error(errors, stage, error):
    message = _clean(error, 500)
    for name in (
        "BRAVE_SEARCH_API_KEY",
        "GOOGLE_CLOUD_VISION_API_KEY",
        "YTDLP_COOKIE_FILE",
    ):
        secret = os.environ.get(name)
        if secret:
            message = message.replace(secret, "[redacted]")
    message = re.sub(
        r"([?&](?:key|api_key|token)=)[^&\s]+", r"\1[redacted]", message, flags=re.I
    )
    errors.append({"stage": stage, "message": message})


def inspect_page(session, url):
    if not public_http_url(url) or is_factcheck_url(url):
        return None
    response = session.get(url, timeout=20, allow_redirects=True, stream=True)
    try:
        response.raise_for_status()
        final_url = getattr(response, "url", None) or url
        redirects = [
            getattr(item, "url", "") for item in getattr(response, "history", [])
        ]
        if any(
            not public_http_url(item) or is_factcheck_url(item)
            for item in redirects + [final_url]
        ):
            return None
        content_type = (response.headers.get("Content-Type") or "").lower()
        if (
            content_type
            and "html" not in content_type
            and "text/plain" not in content_type
        ):
            return None
        content = bytearray()
        for chunk in response.iter_content(chunk_size=65536):
            content.extend(chunk)
            if len(content) >= 1000000:
                break
        # An incomplete page cannot be screened for markup later in the document.
        if len(content) >= 1000000:
            return None
        html = bytes(content).decode(
            getattr(response, "encoding", None) or "utf-8", errors="replace"
        )
    finally:
        response.close()

    soup = BeautifulSoup(html, "html.parser")
    canonical_tag = soup.find("link", rel=lambda value: value and "canonical" in value)
    canonical_url = (
        urljoin(final_url, canonical_tag.get("href"))
        if canonical_tag and canonical_tag.get("href")
        else final_url
    )
    og_url = soup.find("meta", attrs={"property": "og:url"})
    source_urls = [canonical_url]
    if og_url and og_url.get("content"):
        source_urls.append(urljoin(final_url, og_url["content"]))
    if any(not public_http_url(item) or is_factcheck_url(item) for item in source_urls):
        return None

    def meta(*names):
        for name in names:
            for attribute in ("property", "name", "itemprop"):
                node = soup.find("meta", attrs={attribute: name})
                if node and node.get("content"):
                    return _clean(node["content"])
        return None

    title_node = soup.find("title")
    title = meta("og:title", "twitter:title") or _clean(
        title_node.get_text(" ", strip=True) if title_node else "", 1000
    )
    headers = " ".join(
        tag.get_text(" ", strip=True) for tag in soup.find_all(["h1", "h2"])
    )
    if is_factcheck_text(title + " " + headers):
        return None
    if re.search(
        r"access denied|captcha|just a moment|attention required|sign in to continue",
        title,
        re.I,
    ):
        return None
    published_at = _date_value(
        meta("article:published_time", "datePublished", "uploadDate", "date", "pubdate")
    )
    for script in soup.find_all(
        "script", attrs={"type": re.compile(r"ld\+json", re.I)}
    ):
        try:
            structured = json.loads(script.string or script.get_text())
        except (TypeError, ValueError):
            # Also reject malformed ClaimReview markup rather than losing the filter.
            if re.search(r"ClaimReview", script.get_text(), re.I):
                return None
            continue
        for node in json_ld_nodes(structured):
            types = node.get("@type") or []
            types = [types] if isinstance(types, str) else types
            if any(
                str(value).rsplit("/", 1)[-1].rsplit(":", 1)[-1].lower()
                == "claimreview"
                for value in types
            ):
                return None
            published_at = published_at or _date_value(
                node.get("datePublished") or node.get("uploadDate")
            )
    description = meta("og:description", "description", "twitter:description")
    if is_factcheck_text(description):
        return None
    for tag in soup.find_all(
        ["script", "style", "noscript", "nav", "footer", "header", "form"]
    ):
        tag.decompose()
    context = soup.find("article") or soup.find("main") or soup.body or soup
    body_text = context.get_text(" ", strip=True)
    if is_factcheck_text(body_text):
        return None
    body = _clean(body_text)
    if not body and not title and not description:
        return None
    return {
        "source": _url_key(canonical_url),
        "original_url": url,
        "final_url": final_url,
        "canonical_url": canonical_url,
        "title": title,
        "description": description,
        "body": body,
        "published_at": published_at,
        "raw_html": html,
    }


class _ReplayResponse:
    def __init__(self, page):
        self.url = page["final_url"]
        self.content = page["raw_html"].encode("utf-8")
        self.encoding = "utf-8"
        self.headers = {"Content-Type": "text/html; charset=utf-8"}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.content

    def close(self):
        pass


class _MetadataSession:
    def __init__(self, page):
        self.page = page

    def get(self, url, **kwargs):
        if _url_key(url) not in {
            _url_key(self.page["final_url"]),
            _url_key(self.page["canonical_url"]),
        }:
            raise ValueError("Metadata requested an unscreened URL")
        return _ReplayResponse(self.page)


def build_queries(payload, components, media):
    values = [
        item.get("text")
        or " ".join(str(item.get(field) or "") for field in ("subject", "action"))
        for item in components[:2]
    ]
    if not any(values):
        values = [payload["claim"]]
    values += [(payload.get("metadata") or {}).get("title"), payload.get("transcript")]
    analysis = media.get("analysis") or {}
    values += [analysis.get("ocr_text"), analysis.get("scene_summary")]
    queries = []
    for value in values:
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        text = " ".join(_clean(value).replace('"', "").split()[:30])
        if text and text.casefold() not in {query.casefold() for query in queries}:
            queries.append(text)
        if len(queries) == 4:
            break
    basis = queries[0] if queries else _clean(payload["claim"], 300)
    return queries + [
        basis + " site:youtube.com",
        basis + " (site:x.com OR site:facebook.com OR site:tiktok.com)",
    ]


def collect_evidence(payload, components, media, store, errors):
    """Collect into the caller's store without consulting dataset labels."""
    from dotenv import load_dotenv

    load_dotenv()
    brave_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    vision_key = os.environ.get("GOOGLE_CLOUD_VISION_API_KEY")
    session = requests.Session()
    session.headers["User-Agent"] = "eTRUE-verification/1.0"
    pages = {}
    inspected = set()
    archive_urls = []

    def page_for(url):
        if not public_http_url(url) or is_factcheck_url(url):
            return None
        key = _url_key(url)
        if key in inspected:
            return pages.get(key)
        if len(inspected) >= 10:
            return None
        inspected.add(key)
        try:
            page = inspect_page(session, url)
        except Exception as error:
            _error(errors, "retrieval.page", error)
            return None
        if not page:
            return None
        pages[key] = page
        source = page["source"]
        if source not in archive_urls:
            archive_urls.append(source)
            store.add(
                "retrieved_page",
                source,
                page["body"] or page["description"] or page["title"],
                method="http",
                published_at=page["published_at"],
                verified=True,
                title=page["title"],
                final_url=page["final_url"],
                canonical_url=page["canonical_url"],
            )
        return page

    try:
        post_url = payload.get("post_url")
        post_page = page_for(post_url) if post_url else None
        if post_page:
            try:
                extraction = extract_media(
                    post_page["final_url"], _MetadataSession(post_page)
                )
                metadata = extraction.get("metadata") or {}
                targets = [
                    metadata.get(field)
                    for field in ("canonical_url", "webpage_url", "final_url")
                    if metadata.get(field)
                ]
                cleared = all(
                    public_http_url(url) and not is_factcheck_url(url) and page_for(url)
                    for url in targets
                )
                safe_metadata = {
                    field: metadata[field]
                    for field in CORE_METADATA_FIELDS
                    if metadata.get(field)
                }
                if (
                    cleared
                    and safe_metadata
                    and not is_factcheck_text(
                        " ".join(str(value) for value in safe_metadata.values())
                    )
                ):
                    store.add(
                        "platform_metadata",
                        post_page["source"],
                        json.dumps(safe_metadata, ensure_ascii=False),
                        method="platform_metadata",
                        verified=True,
                        metadata=safe_metadata,
                        field_sources=extraction.get("field_sources") or {},
                    )
                for error in extraction.get("errors") or []:
                    _error(
                        errors,
                        "retrieval.metadata",
                        error.get("message", "Metadata retrieval failed"),
                    )
            except Exception as error:
                _error(errors, "retrieval.metadata", error)

        for query in build_queries(payload, components, media):
            try:
                results = (
                    brave_results(session, brave_key, query)
                    if brave_key
                    else ddgs_results(query)
                )
            except Exception as error:
                _error(errors, "retrieval.search", error)
                continue
            for item in results or []:
                page = page_for(item.get("url"))
                observation = _clean(
                    " ".join(
                        str(item.get(field) or "") for field in ("title", "description")
                    ),
                    1000,
                )
                if page and observation and not is_factcheck_text(observation):
                    store.add(
                        "search_candidate",
                        page["source"],
                        observation,
                        method="brave" if brave_key else "ddgs",
                        verified=False,
                        query=query,
                        rank=item.get("rank"),
                        canonical_url=page["canonical_url"],
                    )

        frames = [Path(frame) for frame in media.get("frames") or []]
        if vision_key and frames:
            try:
                image_search = google_image_matches(
                    session, vision_key, select_keyframes(frames, maximum=3)
                )
                for message in image_search.get("errors") or []:
                    _error(errors, "retrieval.image", message)
                for item in image_search.get("pages") or []:
                    page = page_for(item.get("url"))
                    if page and not is_factcheck_text(item.get("title")):
                        store.add(
                            "image_candidate",
                            page["source"],
                            _clean(item.get("title"), 1000)
                            or "Page containing an image similar to a supplied keyframe.",
                            method="google_vision",
                            verified=False,
                            matched_frames=item.get("matched_frames") or [],
                            match_types=item.get("match_types") or [],
                            match_count=item.get("match_count"),
                            image_urls=[
                                url
                                for url in item.get("image_urls") or []
                                if public_http_url(url) and not is_factcheck_url(url)
                            ],
                        )
            except Exception as error:
                _error(errors, "retrieval.image", error)
        elif not vision_key:
            _error(
                errors,
                "retrieval.image",
                "GOOGLE_CLOUD_VISION_API_KEY is not set; reverse image retrieval skipped",
            )

        for url in archive_urls[:5]:
            try:
                archive = wayback_earliest(url, session)
                if (
                    archive
                    and public_http_url(archive.get("url"))
                    and not is_factcheck_url(archive["url"])
                ):
                    store.add(
                        "archive_record",
                        archive["snapshot_url"],
                        "Earliest successful Wayback snapshot of "
                        + archive["url"]
                        + " was recorded on "
                        + archive["date"]
                        + "; this is an online availability date, not a capture date.",
                        method="wayback",
                        verified=True,
                        archive=archive,
                    )
            except Exception as error:
                _error(errors, "retrieval.archive", error)
    finally:
        session.close()
