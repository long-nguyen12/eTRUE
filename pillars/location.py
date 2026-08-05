"""Visual candidates and claimed-versus-verified location rules."""

import re

from utils.evidence import add_field_evidence
from utils.files import trim
from utils.text import text_value_is_grounded


LOCATION_MISMATCHES = {
    "same",
    "different city",
    "different region",
    "different country",
    "unknown",
}


def location_candidate_is_plausible(value):
    text = str(value or "").strip()
    words = re.findall(r"[A-Za-z]+", text)
    lowered = text.lower()
    if not text or len(text) > 80 or len(words) > 8:
        return False
    if lowered in {"front", "school district", "wooden surface"}:
        return False
    if any(phrase in lowered for phrase in (" on board", "wearing ", "holding ", " on their ")):
        return False

    indicators = (
        "city",
        "county",
        "state",
        "province",
        "country",
        "district",
        "park",
        "island",
        "airport",
        "station",
        "school",
        "house",
    )
    if any(re.search(r"\b" + indicator + r"\b", lowered) for indicator in indicators):
        return True
    significant = [word for word in words if word.lower() not in {"of", "the", "and"}]
    if len(significant) == 1 and len(significant[0]) < 3:
        return False
    return bool(significant) and all(word[0].isupper() for word in significant)


def extract_location_contrast(grounding_text):
    text = str(grounding_text or "")
    place = r"[A-Z][A-Za-z'-]*(?:\s+(?:of\s+|the\s+)?[A-Z][A-Za-z'-]*){0,4}"
    actual_first = re.search(
        r"\b(?:in|at)\s+(" + place + r")\s*,\s*not\s+(?:in\s+)?(?:the\s+)?(" + place + r")",
        text,
    )
    if actual_first:
        return actual_first.group(2).strip(), actual_first.group(1).strip()

    claimed_first = re.search(
        r"\bnot\s+(?:in\s+)?(?:the\s+)?(" + place + r")\s*,?\s+but\s+(?:in\s+)?(?:the\s+)?(" + place + r")",
        text,
    )
    if claimed_first:
        return claimed_first.group(1).strip(), claimed_first.group(2).strip()
    return None


def apply_location_rules(extra, event_grounding_text, evidence_ids):
    location = extra["verification"]["location"]
    contrast = extract_location_contrast(event_grounding_text)
    if contrast:
        location["claimed_location"], location["verified_location"] = contrast
        add_field_evidence(extra, "verification.location.claimed_location", evidence_ids)
        add_field_evidence(extra, "verification.location.verified_location", evidence_ids)

    claimed_location = location.get("claimed_location")
    verified_location = location.get("verified_location")
    if not claimed_location or not verified_location:
        location["location_mismatch_type"] = None
    elif claimed_location.lower() == verified_location.lower():
        location["location_mismatch_type"] = "same"
    elif location.get("location_mismatch_type") == "same":
        location["location_mismatch_type"] = None


def clean_visual_location_candidates(verification):
    cleaned = []
    for candidate in verification["location"]["candidate_locations"]:
        if isinstance(candidate, dict):
            candidate = candidate.get("name") or candidate.get("location") or candidate.get("value")
        candidate = trim(candidate, 200)
        if location_candidate_is_plausible(candidate) and candidate not in cleaned:
            cleaned.append(candidate)
    verification["location"]["candidate_locations"] = cleaned
