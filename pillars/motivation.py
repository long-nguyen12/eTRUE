"""Original-context and motivation mismatch rules."""

from pillars.date import first_year
from utils.text import word_overlap


CONTEXT_CATEGORIES = {
    "news report",
    "eyewitness",
    "official record",
    "campaign",
    "activism",
    "entertainment",
    "satire",
    "advertisement",
    "archive",
    "unknown",
}

MOTIVATION_MISMATCHES = {
    "same",
    "satire as real",
    "entertainment as news",
    "old news as current",
    "political reframing",
    "unknown",
}


def apply_motivation_rules(verification, protected_source):
    motivation = verification["motivation"]
    source = verification["source"]

    if protected_source:
        motivation["original_context_category"] = "news report"
    elif source.get("source_type") == "satire source":
        motivation["original_context_category"] = "satire"
    elif motivation.get("original_context_category"):
        motivation["original_context_category"] = "unknown"

    context = motivation.get("original_context_category")
    mismatch = motivation.get("motivation_mismatch_type")
    if not context:
        motivation["motivation_mismatch_type"] = None
        return
    if mismatch == "satire as real" and context != "satire":
        motivation["motivation_mismatch_type"] = None
    if mismatch == "entertainment as news" and context != "entertainment":
        motivation["motivation_mismatch_type"] = None
    if mismatch == "old news as current":
        claimed_year = first_year(verification["date"].get("claimed_date"))
        estimated_year = first_year(verification["date"].get("estimated_date"))
        if not claimed_year or not estimated_year or estimated_year >= claimed_year:
            motivation["motivation_mismatch_type"] = None
    if mismatch == "same":
        original_text = " ".join(
            str(motivation.get(field) or "")
            for field in ("original_caption", "original_description")
        )
        if word_overlap(motivation.get("claimed_framing"), original_text) < 0.25:
            motivation["motivation_mismatch_type"] = None
