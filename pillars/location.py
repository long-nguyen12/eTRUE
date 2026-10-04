import json
import re

from utils.evidence import add_field_evidence
from utils.files import trim


LOCATION_MISMATCHES = {
    "same",
    "different city",
    "different region",
    "different country",
    "unknown",
}


NON_ADMIN_LOCATION_TERMS = {
    "airport",
    "avenue",
    "bridge",
    "building",
    "church",
    "highway",
    "hospital",
    "house",
    "lake",
    "landmark",
    "museum",
    "park",
    "river",
    "road",
    "school",
    "square",
    "stadium",
    "station",
    "street",
    "tower",
    "university",
}


def location_candidate_is_plausible(value):
    text = str(value or "").strip()
    words = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
    lowered = text.casefold()
    if not text or len(text) > 120 or len(words) > 12:
        return False
    if any(
        phrase in lowered
        for phrase in (" on board", "wearing ", "holding ", " on their ")
    ):
        return False

    indicators = (
        "city",
        "county",
        "state",
        "province",
        "country",
        "region",
        "district",
    )
    components = [component.strip() for component in lowered.split(",")]
    if any(
        re.search(r"\b" + indicator + r"\s*$", component)
        for indicator in indicators
        for component in components
    ):
        return True
    if any(
        re.search(r"\b" + term + r"\b", lowered)
        for term in NON_ADMIN_LOCATION_TERMS
    ):
        return False
    significant = [word for word in words if word.lower() not in {"of", "the", "and"}]
    if len(significant) == 1 and len(significant[0]) < 3:
        return False
    return bool(significant) and all(word[0].isupper() for word in significant)


def extract_location_contrast(grounding_text):
    text = str(grounding_text or "")
    place = r"[A-Z][A-Za-z'-]*(?:\s+(?:of\s+|the\s+)?[A-Z][A-Za-z'-]*){0,4}"
    actual_first = re.search(
        r"\b(?:in|at)\s+("
        + place
        + r")\s*,\s*not\s+(?:in\s+)?(?:the\s+)?("
        + place
        + r")",
        text,
    )
    if actual_first:
        return actual_first.group(2).strip(), actual_first.group(1).strip()

    claimed_first = re.search(
        r"\bnot\s+(?:in\s+)?(?:the\s+)?("
        + place
        + r")\s*,?\s+but\s+(?:in\s+)?(?:the\s+)?("
        + place
        + r")",
        text,
    )
    if claimed_first:
        return claimed_first.group(1).strip(), claimed_first.group(2).strip()

    try:
        evidence = json.loads(text)
    except ValueError:
        evidence = None
    if isinstance(evidence, dict):
        text = json.dumps({
            key: evidence.get(key)
            for key in ("fact_check_article", "fact_check_evidence")
        }, ensure_ascii=False)
    verified_patterns = (
        r"(?i:\bactually\s+(?:filmed|taken|recorded|shot)\s+in\s+)",
        r"(?i:\bvideo\s+(?:is|was)\s+from\s+)",
        r"(?i:\bshows?\s+(?:scenes?\s+from|footage\s+(?:from|of))\s+)",
        r"(?i:\boriginated?\s+(?:from|in)\s+)",
        r"(?i:\bclip\s+(?:is|was)\s+(?:from|filmed\s+in)\s+)",
    )
    for pattern in verified_patterns:
        for match in re.finditer(pattern + "(" + place + ")", text):
            if re.search(r"\bnot\s*$", text[:match.start()], flags=re.I):
                continue
            if location_candidate_is_plausible(match.group(1)):
                return None, match.group(1).strip()
    return None


def apply_location_rules(extra, event_grounding_text, evidence_ids):
    location = extra["verification"]["location"]
    contrast = extract_location_contrast(event_grounding_text)
    if contrast:
        claimed, location["verified_location"] = contrast
        if claimed:
            location["claimed_location"] = claimed
            add_field_evidence(
                extra, "verification.location.claimed_location", evidence_ids
            )
        add_field_evidence(
            extra, "verification.location.verified_location", evidence_ids
        )

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
            candidate = (
                candidate.get("name")
                or candidate.get("location")
                or candidate.get("value")
            )
        candidate = trim(candidate, 200)
        if location_candidate_is_plausible(candidate) and candidate not in cleaned:
            cleaned.append(candidate)
    verification["location"]["candidate_locations"] = cleaned
