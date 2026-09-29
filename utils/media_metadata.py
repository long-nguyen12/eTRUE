import ipaddress
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from utils.records import normalize_date
from utils.utils import normalize_social_url


CORE_METADATA_FIELDS = (
    "title",
    "description",
    "uploader_name",
    "uploader_profile",
    "upload_date",
)


def filled(value):
    return value is not None and str(value).strip() != ""


def metadata_status(metadata):
    count = sum(filled((metadata or {}).get(field)) for field in CORE_METADATA_FIELDS)
    if count == len(CORE_METADATA_FIELDS):
        return "complete"
    return "partial" if count else "missing"


def missing_metadata_fields(metadata):
    return [field for field in CORE_METADATA_FIELDS if not filled((metadata or {}).get(field))]


def cookie_file_from_env():
    value = os.environ.get("YTDLP_COOKIE_FILE")
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError("YTDLP_COOKIE_FILE does not point to a readable file")
    try:
        with path.open("rb"):
            pass
    except OSError as error:
        raise FileNotFoundError("YTDLP_COOKIE_FILE does not point to a readable file") from error
    return str(path.resolve())


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


VIDEO_EXTENSIONS = {"mp4", "webm", "mov", "mkv", "flv", "avi", "m4v", "ts"}


def _format_has_video(item):
    if not isinstance(item, dict) or not item.get("url"):
        return False
    vcodec = item.get("vcodec")
    if isinstance(vcodec, str) and vcodec.casefold() == "none":
        return False
    if vcodec or item.get("height") or item.get("width") or item.get("fps"):
        return True
    extension = str(item.get("ext") or "").casefold()
    if extension in VIDEO_EXTENSIONS:
        return True
    path = urlsplit(str(item.get("url") or "")).path.casefold()
    return any(path.endswith("." + extension) for extension in VIDEO_EXTENSIONS)


def has_video_stream(info):
    if not isinstance(info, dict):
        return False
    if any(has_video_stream(entry) for entry in info.get("entries") or []):
        return True
    if any(_format_has_video(item) for item in info.get("formats") or []):
        return True
    if any(_format_has_video(item) for item in info.get("requested_formats") or []):
        return True
    return _format_has_video(info)


def _merged_info(info):
    if not isinstance(info, dict):
        return {}
    entry = next((item for item in info.get("entries") or [] if isinstance(item, dict)), None)
    merged = {key: value for key, value in info.items() if filled(value)}
    if entry:
        merged.update({key: value for key, value in entry.items() if filled(value)})
    return merged


def _yt_dlp_metadata(info, fallback_url):
    value = _merged_info(info)
    upload_date = _date_value(value.get("upload_date")) or _date_value(value.get("timestamp"))
    metadata = {
        "title": _clean_text(value.get("title"), 1000),
        "description": _clean_text(value.get("description"), 10000),
        "uploader_name": _clean_text(value.get("uploader") or value.get("channel"), 300),
        "uploader_profile": _absolute_url(
            value.get("uploader_url") or value.get("channel_url"), fallback_url
        ),
        "upload_date": upload_date,
        "webpage_url": _absolute_url(value.get("webpage_url"), fallback_url) or fallback_url,
        "extractor": _clean_text(value.get("extractor"), 200),
    }
    return {key: item for key, item in metadata.items() if filled(item)}


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
            import yt_dlp

            options = {
                "quiet": True,
                "no_warnings": True,
                "no_color": True,
                "skip_download": True,
                "ignore_no_formats_error": True,
                "socket_timeout": 20,
                "retries": 1,
                "fragment_retries": 1,
                "extractor_retries": 1,
                "playlist_items": "1",
            }
            if cookie_file:
                options["cookiefile"] = cookie_file
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(normalized_url, download=False)
            result["metadata"] = _yt_dlp_metadata(info, normalized_url)
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

        if missing_metadata_fields(result["metadata"]):
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
    result["missing_fields"] = missing_metadata_fields(result["metadata"])
    return result
