"""Original-context and motivation mismatch rules."""

from pillars.date import compare_date_values
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

    context = motivation.get("original_context_category")
    if context not in CONTEXT_CATEGORIES:
        context = None
        motivation["original_context_category"] = None

    if context is None and protected_source:
        motivation["original_context_category"] = "news report"
    elif context is None and source.get("source_type") == "satire source":
        motivation["original_context_category"] = "satire"

    context = motivation.get("original_context_category")
    mismatch = motivation.get("motivation_mismatch_type")
    claimed_framing = str(motivation.get("claimed_framing") or "").strip()
    if not context or not claimed_framing:
        motivation["motivation_mismatch_type"] = None
        return
    if mismatch not in MOTIVATION_MISMATCHES:
        motivation["motivation_mismatch_type"] = None
        return
    if context == "unknown" and mismatch != "unknown":
        motivation["motivation_mismatch_type"] = None
        return
    if mismatch == "satire as real" and context != "satire":
        motivation["motivation_mismatch_type"] = None
    if mismatch == "entertainment as news" and context != "entertainment":
        motivation["motivation_mismatch_type"] = None
    if mismatch == "old news as current":
        comparison = compare_date_values(
            verification["date"].get("estimated_date"),
            verification["date"].get("claimed_date"),
        )
        if comparison != -1:
            motivation["motivation_mismatch_type"] = None
    if mismatch == "same":
        original_text = " ".join(
            str(motivation.get(field) or "")
            for field in ("original_caption", "original_description")
        )
        if word_overlap(motivation.get("claimed_framing"), original_text) < 0.25:
            motivation["motivation_mismatch_type"] = None
