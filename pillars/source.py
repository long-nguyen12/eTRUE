"""Source and uploader validation rules."""

import re
import unicodedata
from urllib.parse import unquote, urlparse

from utils.evidence import add_field_evidence


OFFICIAL_SOURCES = {
    "cnbc": "CNBC",
    "foxnews": "Fox News",
    "nbcnews": "NBC News",
    "nytimes": "The New York Times",
    "today": "TODAY",
}

SOURCE_TYPES = {
    "eyewitness",
    "news outlet",
    "news agency",
    "official account",
    "political actor",
    "activist group",
    "entertainment source",
    "satire source",
    "unknown",
}

GENERIC_SOURCE_NAMES = {
    "facebook",
    "instagram",
    "reddit",
    "snopes",
    "telegram",
    "tiktok",
    "twitter",
    "x",
    "youtube",
}


SOURCE_ALIAS_GROUPS = {
    "agence france presse": {"afp", "agence france-presse", "agence france presse"},
    "associated press": {"ap", "ap news", "associated press"},
    "bbc news": {"bbc", "bbc news"},
    "c span": {"c-span", "cspan", "c span"},
    "cnbc": {"cnbc", "cnbc television"},
    "fox news": {"foxnews", "fox news"},
    "nbc news": {"nbcnews", "nbc news"},
    "new york times": {"nyt", "nytimes", "ny times", "the new york times"},
    "reuters": {"reuters", "reuters news"},
    "today": {"today", "today show", "the today show"},
}


def _normalize_source_text(value):
    text = unquote(str(value or "").strip()).casefold()
    if re.match(r"^(?:https?://|www\.)", text):
        parsed = urlparse(text if "://" in text else "https://" + text)
        path_parts = [part for part in parsed.path.split("/") if part]
        text = path_parts[0] if path_parts else parsed.netloc.split(".")[0]
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"\s*\(@?[\w.-]+\)\s*$", "", text)
    text = text.replace("&", " and ")
    text = re.sub(r"^@", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_NORMALIZED_SOURCE_ALIASES = {
    canonical: {_normalize_source_text(alias) for alias in aliases | {canonical}}
    for canonical, aliases in SOURCE_ALIAS_GROUPS.items()
}
_SOURCE_ALIAS_LOOKUP = {
    alias: canonical
    for canonical, aliases in _NORMALIZED_SOURCE_ALIASES.items()
    for alias in aliases
}


def normalize_source_name(value):
    """Return a conservative canonical form for a source name, handle, or profile URL."""
    normalized = _normalize_source_text(value)
    normalized = re.sub(r"^the\s+", "", normalized)
    return _SOURCE_ALIAS_LOOKUP.get(normalized, normalized)


def source_entities_match(left, right):
    """Compare uploader/source entities after handle and known-alias normalization."""
    left_name = normalize_source_name(left)
    right_name = normalize_source_name(right)
    if not left_name or not right_name:
        return False
    if left_name == right_name:
        return True
    left_compact = left_name.replace(" ", "")
    right_compact = right_name.replace(" ", "")
    return len(left_compact) >= 5 and left_compact == right_compact


def _source_aliases(value):
    canonical = normalize_source_name(value)
    return _NORMALIZED_SOURCE_ALIASES.get(canonical, {canonical})


def _source_name_is_known(value):
    normalized = normalize_source_name(value)
    unknown_names = {"", "n a", "none", "unknown", "unidentified"}
    generic_names = {normalize_source_name(name) for name in GENERIC_SOURCE_NAMES}
    return normalized not in unknown_names | generic_names


def source_name_is_grounded(value, grounding_text):
    if not _source_name_is_known(value):
        return False
    evidence = _normalize_source_text(grounding_text)
    return any(
        alias and re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", evidence)
        for alias in _source_aliases(value)
    )


def apply_source_rules(extra, grounding_text, protected_source, evidence_ids):
    source = extra["verification"]["source"]
    original_source = source.get("original_source_name")
    uploader = source.get("uploader_name")
    field_evidence = extra.get("field_evidence", {})
    source_evidence = (
        field_evidence.get("verification.source.original_source_name", [])
        + field_evidence.get("verification.source.uploader_name", [])
    )
    supporting_evidence = source_evidence or evidence_ids

    source_type = source.get("source_type")
    if source_type not in SOURCE_TYPES:
        source_type = None
        source["source_type"] = None

    if original_source and not protected_source and source_type is None:
        normalized_evidence = _normalize_source_text(grounding_text)
        source_windows = []
        for alias in _source_aliases(original_source):
            for match in re.finditer(
                r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])",
                normalized_evidence,
            ):
                source_windows.append(
                    normalized_evidence[max(0, match.start() - 400) : match.end() + 400]
                )

        satire_terms = r"\b(satire|satirical|fiction|fabricated|fake news)\b"
        canonical_source = normalize_source_name(original_source)
        if any(re.search(satire_terms, window) for window in source_windows):
            source["source_type"] = "satire source"
        elif canonical_source in {
            normalize_source_name(name) for name in OFFICIAL_SOURCES.values()
        }:
            source["source_type"] = "news outlet"
        else:
            source["source_type"] = "unknown"
        add_field_evidence(extra, "verification.source.source_type", supporting_evidence)

    if not original_source:
        source["source_type"] = None

    if not original_source:
        source["source_mismatch"] = None

    if not _source_name_is_known(original_source) or not _source_name_is_known(uploader):
        source["source_is_uploader"] = None
        return

    source["source_is_uploader"] = source_entities_match(original_source, uploader)
    add_field_evidence(extra, "verification.source.source_is_uploader", supporting_evidence)
