"""Source and uploader validation rules."""

import re

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


def source_name_is_grounded(value, grounding_text):
    name = re.sub(r"\s+", " ", str(value or "").strip().lower())
    evidence = re.sub(r"\s+", " ", str(grounding_text or "").lower())
    return bool(name and name not in GENERIC_SOURCE_NAMES and name in evidence)


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

    if original_source and not protected_source:
        lowered_evidence = grounding_text.lower()
        source_windows = []
        search_from = 0
        while True:
            index = lowered_evidence.find(original_source.lower(), search_from)
            if index < 0:
                break
            start = max(0, index - 400)
            end = index + len(original_source) + 400
            source_windows.append(lowered_evidence[start:end])
            search_from = index + len(original_source)

        satire_terms = r"\b(satire|satirical|fiction|fabricated|fake news)\b"
        source["source_type"] = (
            "satire source"
            if any(re.search(satire_terms, window) for window in source_windows)
            else "unknown"
        )
        add_field_evidence(extra, "verification.source.source_type", supporting_evidence)

    if not original_source:
        source["source_type"] = None

    if not original_source or not uploader:
        source["source_is_uploader"] = None
        source["source_mismatch"] = None
        return
    if original_source.strip().lower() != uploader.strip().lower():
        return

    source["source_is_uploader"] = True
    source["source_mismatch"] = False
    add_field_evidence(extra, "verification.source.source_mismatch", supporting_evidence)
    add_field_evidence(extra, "verification.source.source_is_uploader", supporting_evidence)
