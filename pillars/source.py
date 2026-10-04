import re
import unicodedata
from urllib.parse import unquote, urlparse

from utils.evidence import add_field_evidence

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


KNOWN_SOURCES = {
    "agence france presse": {
        "name": "AFP", "platform": "afp", "source_type": "news outlet",
        "aliases": {"afp", "agence france-presse", "agence france presse"},
    },
    "associated press": {
        "name": "AP News", "platform": "apnews", "source_type": "news outlet",
        "aliases": {"ap", "apnews", "ap news", "associated press"},
    },
    "bbc news": {
        "name": "BBC News", "platform": "bbc", "source_type": "news outlet",
        "aliases": {"bbc", "bbc news"},
    },
    "c span": {
        "name": "C-SPAN", "platform": "cspan", "source_type": "news outlet",
        "aliases": {"c-span", "cspan", "c span"},
    },
    "cnn": {
        "name": "CNN", "platform": "cnn", "source_type": "news outlet",
        "aliases": {"cnn", "cnn news"},
    },
    "cnbc": {
        "name": "CNBC", "platform": "cnbc", "source_type": "news outlet",
        "aliases": {"cnbc", "cnbc television"},
    },
    "guardian": {
        "name": "The Guardian", "platform": "theguardian", "source_type": "news outlet",
        "aliases": {"theguardian", "the guardian", "guardian"},
    },
    "dw news": {
        "name": "DW News", "platform": "dw", "source_type": "news outlet",
        "aliases": {"dw", "dw news", "deutsche welle"},
    },
    "abc news": {
        "name": "ABC News", "platform": "abc", "source_type": "news outlet",
        "aliases": {"abc", "abc news", "abcnews"},
    },
    "cbs news": {
        "name": "CBS News", "platform": "cbsnews", "source_type": "news outlet",
        "aliases": {"cbsnews", "cbs news"},
    },
    "pbs newshour": {
        "name": "PBS NewsHour", "platform": "pbs", "source_type": "news outlet",
        "aliases": {"pbs", "pbs news", "pbs newshour"},
    },
    "fox news": {
        "name": "Fox News", "platform": "foxnews", "source_type": "news outlet",
        "aliases": {"foxnews", "fox news"},
    },
    "nbc news": {
        "name": "NBC News", "platform": "nbcnews", "source_type": "news outlet",
        "aliases": {"nbcnews", "nbc news"},
    },
    "new york times": {
        "name": "The New York Times", "platform": "nytimes", "source_type": "news outlet",
        "aliases": {"nyt", "nytimes", "ny times", "the new york times"},
    },
    "reuters": {
        "name": "Reuters", "platform": "reuters", "source_type": "news outlet",
        "aliases": {"reuters", "reuters news"},
    },
    "washington post": {
        "name": "The Washington Post", "platform": "washingtonpost", "source_type": "news outlet",
        "aliases": {"washingtonpost", "washington post", "the washington post"},
    },
    "usa today": {
        "name": "USA Today", "platform": "usatoday", "source_type": "news outlet",
        "aliases": {"usatoday", "usa today"},
    },
    "sky news": {
        "name": "Sky News", "platform": "sky", "source_type": "news outlet",
        "aliases": {"sky", "sky news", "skynews"},
    },
    "france 24": {
        "name": "France 24", "platform": "france24", "source_type": "news outlet",
        "aliases": {"france24", "france 24"},
    },
    "al jazeera": {
        "name": "Al Jazeera", "platform": "aljazeera", "source_type": "news outlet",
        "aliases": {"aljazeera", "al jazeera"},
    },
    "today": {
        "name": "TODAY", "platform": "today", "source_type": "news outlet",
        "aliases": {"today", "today show", "the today show"},
    },
}


def _normalize_source_text(value):
    text = unquote(str(value or "").strip()).casefold()
    if re.match(r"^(?:https?://|www\.)", text):
        parsed = urlparse(text if "://" in text else "https://" + text)
        path_parts = [part for part in parsed.path.split("/") if part]
        text = path_parts[0] if path_parts else parsed.netloc.split(".")[0]
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        character for character in text if not unicodedata.combining(character)
    )
    text = re.sub(r"\s*\(@?[\w.-]+\)\s*$", "", text)
    text = text.replace("&", " and ")
    text = re.sub(r"^@", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_NORMALIZED_SOURCE_ALIASES = {
    canonical: {_normalize_source_text(alias) for alias in source["aliases"] | {canonical}}
    for canonical, source in KNOWN_SOURCES.items()
}
_SOURCE_ALIAS_LOOKUP = {
    alias: canonical
    for canonical, aliases in _NORMALIZED_SOURCE_ALIASES.items()
    for alias in aliases
}


def normalize_source_name(value):
    normalized = _normalize_source_text(value)
    normalized = re.sub(r"^the\s+", "", normalized)
    return _SOURCE_ALIAS_LOOKUP.get(normalized, normalized)


def get_known_source(value):
    return KNOWN_SOURCES.get(normalize_source_name(value))


def get_platform_source(platform):
    return next(
        (source for source in KNOWN_SOURCES.values() if source["platform"] == platform),
        None,
    )


def source_entities_match(left, right):
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
        alias
        and re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", evidence)
        for alias in _source_aliases(value)
    )


def apply_source_rules(extra, grounding_text, protected_source, evidence_ids):
    source = extra["verification"]["source"]
    original_source = source.get("original_source_name")
    uploader = source.get("uploader_name")
    field_evidence = extra.get("field_evidence", {})
    source_evidence = field_evidence.get(
        "verification.source.original_source_name", []
    ) + field_evidence.get("verification.source.uploader_name", [])
    supporting_evidence = source_evidence or evidence_ids

    source_type = source.get("source_type")
    if source_type not in SOURCE_TYPES:
        source_type = None
        source["source_type"] = None

    if original_source and source_type in {None, "unknown"}:
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
        known_source = get_known_source(original_source)
        if not protected_source and any(re.search(satire_terms, window) for window in source_windows):
            source["source_type"] = "satire source"
        elif known_source:
            source["source_type"] = known_source["source_type"]
        else:
            source["source_type"] = "unknown"
        add_field_evidence(
            extra, "verification.source.source_type", supporting_evidence
        )

    if not original_source:
        source["source_type"] = None

    if not original_source:
        source["source_mismatch"] = None

    if not _source_name_is_known(original_source) or not _source_name_is_known(
        uploader
    ):
        source["source_is_uploader"] = None
        return

    source["source_is_uploader"] = source_entities_match(original_source, uploader)
    add_field_evidence(
        extra, "verification.source.source_is_uploader", supporting_evidence
    )
