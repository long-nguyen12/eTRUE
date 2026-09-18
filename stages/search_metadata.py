"""Extract provenance metadata from public candidate pages."""

import ipaddress
import json
import re
from urllib.parse import urljoin, urlsplit

from stages.search_clients import clean_html_text
from utils.files import trim
from utils.records import normalize_date


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

    for script in soup.find_all(
        "script", attrs={"type": re.compile(r"ld\+json", re.I)}
    ):
        try:
            structured = json.loads(script.string or script.get_text())
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        for node in json_ld_nodes(structured):
            published_value = (
                published_value or node.get("datePublished") or node.get("uploadDate")
            )
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
