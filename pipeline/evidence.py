"""Evidence shared by all components of one verification request."""

import hashlib
import json
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from utils.files import now


def canonical_source(source):
    source = str(source).strip()
    parsed = urlsplit(source)
    if parsed.scheme not in {"http", "https"}:
        return source
    query = [(k, v) for k, v in parse_qsl(parsed.query)
             if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}]
    return urlunsplit(("https", parsed.netloc.lower(), parsed.path.rstrip("/"),
                       urlencode(sorted(query)), ""))


def normalized_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


class EvidenceStore:
    def __init__(self):
        self._items = {}

    @property
    def items(self):
        return list(self._items.values())

    def add(self, type, source, observation, *, method, retrieved_at=None,
            published_at=None, verified=False, **details):
        source = canonical_source(source)
        identity = json.dumps([source, normalized_text(observation)], ensure_ascii=False)
        evidence_id = "ev-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        if evidence_id not in self._items:
            self._items[evidence_id] = {
                "id": evidence_id, "type": type, "source": source,
                "observation": str(observation), "method": method,
                "retrieved_at": retrieved_at or now(), "published_at": published_at,
                "verified": bool(verified), **details,
            }
        elif verified and not self._items[evidence_id]["verified"]:
            self._items[evidence_id].update(type=type, method=method, verified=True,
                                            published_at=published_at, **details)
        return evidence_id


def validate_citations(citations, evidence, *, decisive=False):
    """Only accept actual quotes from evidence supplied to the current prompt."""
    by_id = {item["id"]: item for item in evidence}
    valid = []
    if not isinstance(citations, list):
        return valid
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        item = by_id.get(citation.get("evidence_id"))
        quote = citation.get("quote")
        if not item or not isinstance(quote, str) or len(quote.strip()) < 8:
            continue
        if decisive and (not item.get("verified") or item.get("type") not in {
            "retrieved_page", "platform_metadata", "archive_record", "geocoder_result",
        }):
            continue
        if normalized_text(quote) not in normalized_text(item.get("observation")):
            continue
        row = {"evidence_id": item["id"], "quote": quote.strip()}
        if row not in valid:
            valid.append(row)
    return valid
