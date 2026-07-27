"""Provenance-specific grounding and consistency rules."""

import re

from utils.text import containment_overlap


def previous_context_is_useful(value, claimed_framing, grounding_text):
    summary = re.sub(r"\s+", " ", str(value or "").strip())
    lowered = summary.lower()
    boilerplate = (
        "previous context summary",
        "not provided",
        "not relevant",
        "does not match",
        "provided information",
    )
    if len(summary) < 25 or any(phrase in lowered for phrase in boilerplate):
        return False
    if containment_overlap(summary, claimed_framing) >= 0.65:
        return False

    context_cue = re.search(
        r"\b(actually|originally|previously|filmed|recorded|depicts|footage|newsreel|archive)\b",
        lowered,
    )
    if not context_cue:
        return False

    summary_words = set(re.findall(r"[a-z0-9]+", lowered))
    evidence_words = set(re.findall(r"[a-z0-9]+", str(grounding_text or "").lower()))
    return bool(summary_words) and len(summary_words & evidence_words) / len(summary_words) >= 0.7


def apply_provenance_rules(verification):
    provenance = verification["provenance"]
    previous_context = provenance.get("previous_context_summary")
    claimed_framing = verification["motivation"].get("claimed_framing")

    if not previous_context or previous_context == claimed_framing:
        provenance["previous_context_summary"] = None
        provenance["provenance_mismatch"] = None
