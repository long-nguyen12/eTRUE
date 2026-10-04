import ipaddress
import json
import re
from datetime import datetime, timezone
from urllib.parse import urljoin, urlsplit

from utils.records import normalize_date


def filled(value):
    return value is not None and str(value).strip() != ""


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


def _clean_text(value, maximum):
    from html import unescape

    text = re.sub(r"<[^>]+>", " ", unescape(str(value or "")))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:maximum] if text else None


def _absolute_url(value, base_url):
    if isinstance(value, list):
        for item in value:
            url = _absolute_url(item, base_url)
            if url:
                return url
        return None
    if not filled(value):
        return None
    url = urljoin(base_url, str(value).strip())
    return url if public_http_url(url) else None


def _date_value(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        timestamp = value / 1000 if value > 100000000000 else value
        try:
            return datetime.fromtimestamp(timestamp, timezone.utc).date().isoformat()
        except (OSError, OverflowError, ValueError):
            return None
    text = str(value).strip()
    normalized = normalize_date(text)
    if normalized:
        return normalized
    match = re.search(r"(?<!\d)((?:19|20)\d{2}-\d{2}-\d{2})(?!\d)", text)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d").date().isoformat()
        except ValueError:
            return None
    return None


def _entity(entity, base_url):
    if isinstance(entity, list):
        entity = next((item for item in entity if item), None)
    if isinstance(entity, dict):
        return (
            _clean_text(entity.get("name"), 300),
            _absolute_url(entity.get("url") or entity.get("sameAs"), base_url),
        )
    if isinstance(entity, str):
        if public_http_url(entity):
            return None, entity
        return _clean_text(entity, 300), None
    return None, None


def _node_types(node):
    values = node.get("@type") or []
    if isinstance(values, str):
        values = [values]
    return {str(value).casefold() for value in values}


def fetch_page_metadata(session, url):
    if not public_http_url(url):
        raise ValueError("URL is not a public HTTP(S) URL")

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
            raise ValueError("URL redirected to a non-public address")
        content_type = (response.headers.get("Content-Type") or "").lower()
        if content_type and "html" not in content_type:
            return {"final_url": final_url, "content_type": content_type[:100]}

        content = bytearray()
        if hasattr(response, "iter_content"):
            for chunk in response.iter_content(chunk_size=65536):
                content.extend(chunk)
                if len(content) >= 1000000:
                    break
        else:
            content.extend((getattr(response, "content", b"") or b"")[:1000000])
        encoding = getattr(response, "encoding", None) or "utf-8"
        html = bytes(content[:1000000]).decode(encoding, errors="replace")
    finally:
        response.close()

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    def meta_content(*names):
        for name in names:
            for attribute in ("property", "name", "itemprop"):
                tag = soup.find("meta", attrs={attribute: name})
                if tag and tag.get("content"):
                    return _clean_text(tag["content"], 10000)
        return None

    nodes = []
    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        try:
            structured = json.loads(script.string or script.get_text())
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        nodes.extend(json_ld_nodes(structured))

    preferred_types = {"videoobject", "newsarticle", "article", "socialmediaposting"}
    content_nodes = [node for node in nodes if _node_types(node).intersection(preferred_types)]
    content_nodes.sort(key=lambda node: 0 if "videoobject" in _node_types(node) else 1)

    json_title = json_description = json_date = None
    author = author_url = publisher = publisher_url = None
    for node in content_nodes:
        json_title = json_title or node.get("headline") or node.get("name")
        json_description = json_description or node.get("description")
        json_date = json_date or node.get("uploadDate") or node.get("datePublished")
        node_author, node_author_url = _entity(node.get("author"), final_url)
        node_publisher, node_publisher_url = _entity(node.get("publisher"), final_url)
        author = author or node_author
        author_url = author_url or node_author_url
        publisher = publisher or node_publisher
        publisher_url = publisher_url or node_publisher_url

    title_tag = soup.find("title")
    title = _clean_text(json_title, 1000) or _clean_text(
        meta_content("og:title", "twitter:title"), 1000
    )
    title = title or _clean_text(title_tag.get_text(" ", strip=True) if title_tag else None, 1000)
    if title and re.search(r"access denied|captcha|just a moment|attention required", title, re.I):
        raise ValueError("Page returned an access challenge instead of metadata")
    description = _clean_text(json_description, 10000) or meta_content(
        "og:description", "description", "twitter:description"
    )
    meta_author = meta_content("author", "article:author")
    if not author and meta_author and not public_http_url(meta_author):
        author = _clean_text(meta_author, 300)
    if not author_url and meta_author and public_http_url(meta_author):
        author_url = meta_author
    site_name = meta_content("og:site_name")
    published_value = json_date or meta_content(
        "article:published_time", "datePublished", "uploadDate", "date", "pubdate"
    )

    canonical = soup.find("link", rel=lambda value: value and "canonical" in value)
    canonical_url = _absolute_url(
        canonical.get("href") if canonical else None,
        final_url,
    ) or final_url
    context = soup.find("article") or soup.find("main")
    context_excerpt = _clean_text(
        context.get_text(" ", strip=True) if context else None,
        1200,
    )
    uploader_name = author or publisher or site_name
    uploader_profile = author_url or publisher_url
    published_at = _date_value(published_value)
    metadata = {
        "final_url": final_url,
        "canonical_url": canonical_url,
        "webpage_url": canonical_url,
        "title": title,
        "description": description,
        "author": author,
        "author_url": author_url,
        "publisher": publisher,
        "publisher_url": publisher_url,
        "site_name": site_name,
        "uploader_name": uploader_name,
        "uploader_profile": uploader_profile,
        "published_at": published_at,
        "upload_date": published_at,
        "context_excerpt": context_excerpt,
        "content_type": content_type[:100] if content_type else None,
        "extractor": "html",
    }
    return {key: value for key, value in metadata.items() if filled(value)}
