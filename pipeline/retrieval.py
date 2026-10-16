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
from stages.web import extract_uploader_from_url, wayback_earliest
from utils.json_ld import _date_value, json_ld_nodes, public_http_url
from utils.media_metadata import CORE_METADATA_FIELDS, extract_media
from utils.ytdlp import cookie_file_from_env

FACTCHECK_DOMAINS = {
    "snopes.com",
    "politifact.com",
    "factcheck.org",
    "fullfact.org",
    "factcheck.afp.com",
    "mythdetector.com",
    "mythdetector.ge",
    "leadstories.com",
    "checkyourfact.com",
}
FACTCHECK_HANDLES = {
    "snopes", "politifact", "fullfact", "factcheckorg", "factcheckdotorg",
    "afpfactcheck", "leadstories", "checkyourfact", "mythdetector",
}
FACTCHECK_LABEL = re.compile(
    r"\bfact[\s_-]?check(?:ing|ed|er|ers)?\b|\bclaim\s*review\b",
    re.I,
)
FACTCHECK_CONCLUSION = re.compile(
    r"\b(?:snopes|politifact|full\s*fact|factcheck\.org|lead\s*stories|"
    r"check\s*your\s*fact|myth\s*detector)\b",
    re.I,
)
RATING_CUE = re.compile(
    r"\b(?:rated|rating|verdict|conclud(?:ed|es)|debunk(?:ed|s)|determined|"
    r"false|true|misleading|pants on fire|fake|hoax|CGI|computer[\s-]+generated|fabricated)\b",
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
FACTCHECK_ATTRIBUTION = re.compile(
    r"\b(?:sources?|via|published\s+by|originally\s+(?:published|reported)\s+(?:by|at))"
    r"\s*[:\-]?\s*(?:https?://)?(?:www\.)?"
    r"(?:snopes|politifact|full\s*fact|factcheck\.org|lead\s*stories|"
    r"check\s*your\s*fact|myth\s*detector)\b",
    re.I,
)
PAGE_SHELL = re.compile(
    r"\b(?:this\s+post\s+is\s+only\s+available\s+in\s+the\s+x\s+app|"
    r"scan\s+to\s+get\s+app|sensitive\s+content|log\s+in|sign\s+in|sign\s+up|"
    r"post|youtube|msn|tiktok(?:\s*[-\u2013\u2014]\s*make\s+your\s+day)?)\b",
    re.I,
)


def is_factcheck_url(url):
    try:
        parsed = urlsplit(str(url or ""))
        host = (parsed.hostname or "").lower().rstrip(".")
        path = unquote(parsed.path)
        if host == "web.archive.org":
            original = re.match(r"/web/[^/]+/(https?://.+)", path)
            if original:
                return is_factcheck_url(original.group(1))
        if host.removeprefix("www.").removeprefix("m.") in {
            "facebook.com", "x.com", "twitter.com", "tiktok.com",
            "youtube.com", "instagram.com",
        }:
            parts = path.strip("/").split("/")
            handle = (
                parts[1] if len(parts) > 1 and parts[0].lower() in {"c", "user", "pages"}
                else parts[0]
            )
            handle = re.sub(r"[^a-z0-9]", "", handle.lower())
            if handle in FACTCHECK_HANDLES:
                return True
        return any(
            host == domain or host.endswith("." + domain)
            for domain in FACTCHECK_DOMAINS
        ) or bool(FACTCHECK_LABEL.search(path))
    except ValueError:
        return True


def is_factcheck_text(value):
    text = str(value or "")
    return bool(
        FACTCHECK_LABEL.search(text)
        or RATING_MARKER.search(text)
        or FACTCHECK_ATTRIBUTION.search(text)
        or (FACTCHECK_CONCLUSION.search(text) and RATING_CUE.search(text))
    )


def _page_content(soup, title, description):
    """Prefer the article matching the page, rather than a related-story card."""
    ignored = {
        "this", "that", "with", "from", "have", "will", "when", "what",
        "news", "video", "watch", "breaking", "latest", "live", "updates",
    }

    def words(value):
        return set(re.findall(r"[^\W\d_]{4,}", value.casefold())) - ignored

    targets = words(title) or words(description or "")
    candidates = soup.find_all(["article", "main"])
    candidates += soup.find_all(attrs={"itemprop": "articleBody"})
    if not candidates:
        candidates = [soup.body or soup]

    def score(node):
        text = node.get_text(" ", strip=True)
        headings = node.find_all(["h1", "h2"])
        heading_match = max(
            (len(targets & words(tag.get_text(" ", strip=True))) for tag in headings),
            default=0,
        )
        return heading_match, len(targets & words(text[:1500])), -len(text)

    context = max(candidates, key=score)
    body = context.get_text(" ", strip=True)
    # Short unrelated cards are not evidence for the requested page.
    if (
        targets and len(body) < 300 and context.find(["h1", "h2"])
        and not targets & words(body)
    ):
        return None
    return body


def _meaningful_body(body, title):
    if not body or _clean(body).casefold() == title.casefold():
        return False
    text = PAGE_SHELL.sub(" ", body)
    text = re.sub(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|"
        r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b",
        " ", text, flags=re.I,
    )
    return (
        len(set(re.findall(r"[^\W\d_]{2,}", text.casefold()))) >= 4
        or len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text)) >= 12
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


def inspect_page(session, url, *, require_body=True):
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
    access_challenge = re.search(
        r"access denied|captcha|just a moment|attention required|sign in to continue",
        title,
        re.I,
    )
    if access_challenge and require_body:
        return None
    published_at = _date_value(
        meta("article:published_time", "datePublished", "uploadDate", "date", "pubdate")
    )
    author = meta("author", "article:author")
    site_name = meta("og:site_name")
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
            node_author = node.get("author")
            if not author and isinstance(node_author, str):
                author = _clean(node_author)
            elif not author and isinstance(node_author, dict):
                author = _clean(node_author.get("name")) or None
    description = meta("og:description", "description", "twitter:description")
    if is_factcheck_text(description):
        return None
    for publisher in (site_name, author, meta("article:author")):
        if publisher:
            publisher_url = publisher if public_http_url(publisher) else "https://" + publisher
            if (
                re.sub(r"[^a-z0-9]", "", publisher.lower()) in FACTCHECK_HANDLES
                or is_factcheck_url(publisher_url)
            ):
                return None
    if is_factcheck_text(soup.get_text(" ", strip=True)):
        return None
    for tag in soup.find_all(
        ["script", "style", "noscript", "nav", "footer", "header", "form"]
    ):
        tag.decompose()
    body_text = _page_content(soup, title, description)
    if is_factcheck_text(body_text):
        return None
    body = _clean(body_text) if not access_challenge and _meaningful_body(body_text, title) else ""
    if require_body and not body:
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
        "author": author,
        "site_name": site_name,
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
        self.page = page or {}
        from utils.utils import normalize_social_url

        allowed = set()
        for key in ("final_url", "canonical_url", "original_url"):
            val = self.page.get(key)
            if val:
                allowed.add(_url_key(val))
                normalized = normalize_social_url(val).get("url")
                if normalized:
                    allowed.add(_url_key(normalized))
        self.allowed = allowed

    def get(self, url, **kwargs):
        from utils.utils import normalize_social_url

        normalized = normalize_social_url(url).get("url") or url
        if (
            _url_key(url) not in self.allowed
            and _url_key(normalized) not in self.allowed
        ):
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
            if not value:
                continue
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
    failed = set()
    attempts = {"metadata": 0, "web": 0, "social": 0, "image": 0}
    limits = {"metadata": 4, "web": 6, "social": 4, "image": 4}
    archive_urls = []

    def page_for(url, *, purpose="web"):
        if not public_http_url(url) or is_factcheck_url(url):
            print(f"Skipping page_for({url}): not public or is factcheck", flush=True)
            return None
        key = _url_key(url)
        if key in inspected:
            print(f"Skipping page_for({url}): already inspected", flush=True)
            return pages.get(key)
        if attempts[purpose] >= limits[purpose]:
            print(f"Skipping page_for({url}): inspected limit reached", flush=True)
            return None
        inspected.add(key)
        attempts[purpose] += 1
        try:
            page = inspect_page(session, url, require_body=False)
        except Exception as error:
            failed.add(key)
            print(f"Error inspecting page {url}: {error}", flush=True)
            _error(errors, "retrieval.page", error)
            return None
        if not page:
            return None
        pages[key] = page
        source = page["source"]
        if page["body"]:
            if source not in archive_urls:
                archive_urls.append(source)
            observation = page["body"]
            for label, field in (
                ("Page author", "author"),
                ("Site name", "site_name"),
                ("Publication date (not capture date)", "published_at"),
            ):
                if page.get(field):
                    observation += "\n" + label + ": " + str(page[field])
            store.add(
                "retrieved_page",
                source,
                observation,
                method="http",
                published_at=page["published_at"],
                verified=True,
                title=page["title"],
                final_url=page["final_url"],
                canonical_url=page["canonical_url"],
                author=page.get("author"),
                site_name=page.get("site_name"),
            )
        return page

    try:
        post_url = payload.get("post_url")
        post_page = page_for(post_url, purpose="metadata") if post_url else None
        print(f"Post page for {post_url}: {post_page}", flush=True)
        post_allowed = bool(post_page) or (
            public_http_url(post_url) and not is_factcheck_url(post_url)
            and _url_key(post_url) in failed
        )
        uploader_metadata = {}
        if post_allowed:
            if post_url not in archive_urls:
                archive_urls.insert(0, post_url)
            uploader_name, uploader_profile = extract_uploader_from_url(post_url)
            uploader_metadata = {
                field: value
                for field, value in (
                    ("uploader_name", uploader_name),
                    ("uploader_profile", uploader_profile),
                )
                if value
            }
            metadata_url = post_page["final_url"] if post_page else post_url
            print(f"Extracting metadata from {metadata_url}", flush=True)
            try:
                extraction = extract_media(
                    metadata_url,
                    _MetadataSession(post_page),
                    cookie_file=cookie_file_from_env(),
                )
                metadata = dict(extraction.get("metadata") or {})
                field_sources = dict(extraction.get("field_sources") or {})
                acquired_metadata = any(
                    metadata.get(field) and field_sources.get(field) != "url_pattern"
                    for field in CORE_METADATA_FIELDS
                )
                for field, value in uploader_metadata.items():
                    if value and not metadata.get(field):
                        metadata[field] = value
                        field_sources[field] = "url_pattern"
                targets = [
                    metadata.get(field)
                    for field in ("canonical_url", "webpage_url", "final_url")
                    if metadata.get(field)
                ]
                cleared = bool(post_page) and all(
                    public_http_url(url) and not is_factcheck_url(url)
                    and page_for(url, purpose="metadata")
                    for url in targets
                )
                safe_metadata = {
                    field: metadata[field]
                    for field in CORE_METADATA_FIELDS
                    if metadata.get(field)
                }
                uploader = str(metadata.get("uploader_name") or "")
                uploader_url = uploader if public_http_url(uploader) else "https://" + uploader
                if (
                    cleared
                    and acquired_metadata
                    and safe_metadata
                    and not is_factcheck_url(metadata.get("uploader_profile"))
                    and not is_factcheck_url(uploader_url)
                    and re.sub(r"[^a-z0-9]", "", uploader.lower()) not in FACTCHECK_HANDLES
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
                        field_sources=field_sources,
                        requested_url=post_url,
                        final_url=post_page["final_url"],
                        canonical_url=post_page["canonical_url"],
                    )
                    uploader_metadata = {}
                for error in extraction.get("errors") or []:
                    _error(
                        errors,
                        "retrieval.metadata",
                        error.get("message", "Metadata retrieval failed"),
                    )
            except Exception as error:
                _error(errors, "retrieval.metadata", error)
        if uploader_metadata:
            store.add(
                "uploader_url_pattern",
                post_url,
                json.dumps(uploader_metadata, ensure_ascii=False),
                method="url_pattern",
                verified=False,
                metadata=uploader_metadata,
                field_sources={field: "url_pattern" for field in uploader_metadata},
                requested_url=post_url,
            )

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
                observation = _clean(
                    " ".join(
                        str(item.get(field) or "") for field in ("title", "description")
                    ),
                    1000,
                )
                if is_factcheck_text(observation):
                    continue
                page = page_for(
                    item.get("url"), purpose="social" if "site:" in query else "web"
                )
                if page and observation:
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
                    if is_factcheck_text(item.get("title")):
                        continue
                    page = page_for(item.get("url"), purpose="image")
                    if page:
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
                        requested_url=url,
                    )
            except Exception as error:
                _error(errors, "retrieval.archive", error)
    finally:
        session.close()
