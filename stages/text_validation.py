
import json

from pillars.date import DATE_GRANULARITIES, DATE_MISMATCHES, apply_date_rules, date_is_grounded
from pillars.location import LOCATION_MISMATCHES, apply_location_rules, clean_visual_location_candidates
from pillars.motivation import CONTEXT_CATEGORIES, MOTIVATION_MISMATCHES, apply_motivation_rules
from pillars.provenance import PROVENANCE_STATUSES, apply_provenance_rules, previous_context_is_useful
from pillars.source import SOURCE_TYPES, apply_source_rules, get_known_source, get_platform_source, source_name_is_grounded
from stages.text_fields import PILLAR_EVIDENCE_TYPES
from utils.evidence import add_field_evidence
from utils.model_output import enum_value
from utils.text import containment_overlap, text_value_is_grounded


TEXT_OUTPUT_FIELDS = {
    "provenance_status",
    "previous_context_summary",
    "provenance_mismatch",
    "original_source_name",
    "source_type",
    "source_is_uploader",
    "source_mismatch",
    "claimed_date",
    "estimated_date",
    "capture_date_granularity",
    "date_mismatch_type",
    "claimed_location",
    "verified_location",
    "location_mismatch_type",
    "original_context_category",
    "motivation_mismatch_type",
}

TEXT_FIELD_MAP = {
    "provenance_status": ("provenance", "provenance_status"),
    "previous_context_summary": ("provenance", "previous_context_summary"),
    "provenance_mismatch": ("provenance", "provenance_mismatch"),
    "original_source_name": ("source", "original_source_name"),
    "source_type": ("source", "source_type"),
    "source_is_uploader": ("source", "source_is_uploader"),
    "source_mismatch": ("source", "source_mismatch"),
    "claimed_date": ("date", "claimed_date"),
    "estimated_date": ("date", "estimated_date"),
    "capture_date_granularity": ("date", "capture_date_granularity"),
    "date_mismatch_type": ("date", "date_mismatch_type"),
    "claimed_location": ("location", "claimed_location"),
    "verified_location": ("location", "verified_location"),
    "location_mismatch_type": ("location", "location_mismatch_type"),
    "claimed_framing": ("motivation", "claimed_framing"),
    "original_context_category": ("motivation", "original_context_category"),
    "motivation_mismatch_type": ("motivation", "motivation_mismatch_type"),
}

TEXT_ENUM_FIELDS = {
    "provenance_status": PROVENANCE_STATUSES,
    "source_type": SOURCE_TYPES,
    "capture_date_granularity": DATE_GRANULARITIES,
    "date_mismatch_type": DATE_MISMATCHES,
    "location_mismatch_type": LOCATION_MISMATCHES,
    "original_context_category": CONTEXT_CATEGORIES,
    "motivation_mismatch_type": MOTIVATION_MISMATCHES,
}

TEXT_BOOLEAN_FIELDS = {"provenance_mismatch", "source_is_uploader", "source_mismatch"}

TEXT_RESET_FIELDS = (
    ("provenance", "provenance_mismatch"),
    ("source", "source_mismatch"),
    ("date", "claimed_date"),
    ("date", "estimated_date"),
    ("date", "capture_date_granularity"),
    ("date", "date_mismatch_type"),
    ("location", "claimed_location"),
    ("location", "verified_location"),
    ("location", "location_mismatch_type"),
    ("motivation", "original_context_category"),
    ("motivation", "motivation_mismatch_type"),
)


def pillar_evidence_ids(extra, pillar):
    allowed = PILLAR_EVIDENCE_TYPES[pillar]
    return [
        evidence["id"]
        for evidence in extra.get("evidence", [])
        if evidence.get("id") and evidence.get("type") in allowed
    ]


def field_has_evidence_type(extra, field, evidence_types):
    linked = set((extra.get("field_evidence") or {}).get(field) or [])
    return any(
        evidence.get("id") in linked and evidence.get("type") in evidence_types
        for evidence in extra.get("evidence", [])
    )


def field_evidence_ids(extra, key, pillar, value):
    field = "verification." + pillar + "." + TEXT_FIELD_MAP[key][1]
    existing = list((extra.get("field_evidence") or {}).get(field) or [])
    allowed = PILLAR_EVIDENCE_TYPES[pillar]
    candidates = [
        evidence
        for evidence in extra.get("evidence", [])
        if evidence.get("id") and evidence.get("type") in allowed
    ]

    if key in TEXT_ENUM_FIELDS or key in TEXT_BOOLEAN_FIELDS or pillar == "date":
        selected = [evidence["id"] for evidence in candidates]
    else:
        needle = " ".join(str(value or "").casefold().split())
        selected = []
        for evidence in candidates:
            haystack = " ".join(
                json.dumps(evidence, ensure_ascii=False, sort_keys=True)
                .casefold()
                .split()
            )
            if needle and needle in haystack:
                selected.append(evidence["id"])
            elif key == "previous_context_summary" and containment_overlap(
                value, haystack
            ) >= 0.5:
                selected.append(evidence["id"])
    return list(dict.fromkeys(existing + selected))


def copy_grounded_text_fields(
    extra,
    analysis,
    grounding_text,
    event_grounding_text,
    protected_source,
    protected_context,
):
    verification = extra["verification"]

    for key, (pillar, field) in TEXT_FIELD_MAP.items():
        value = analysis.get(key)

        if key == "previous_context_summary" and protected_context:
            continue
        if (
            key == "provenance_status"
            and verification["provenance"].get("provenance_status")
            == "earlier version found"
        ):
            continue
        if key == "claimed_framing":
            continue
        if key == "original_source_name" and protected_source:
            continue
        if key in {"source_type", "source_is_uploader"}:
            known_source = get_known_source(protected_source)
            if known_source and protected_source == known_source["name"]:
                continue

        if key == "previous_context_summary" and not previous_context_is_useful(
            value,
            verification["motivation"].get("claimed_framing"),
            grounding_text,
        ):
            value = None
        if key == "original_source_name" and not source_name_is_grounded(value, grounding_text):
            value = None
        if key in {"claimed_date", "estimated_date"} and not date_is_grounded(
            value, event_grounding_text
        ):
            value = None
        if key in {"claimed_location", "verified_location"} and not text_value_is_grounded(
            value, event_grounding_text
        ):
            value = None

        if key in TEXT_ENUM_FIELDS:
            value = enum_value(value, TEXT_ENUM_FIELDS[key])
        if key in TEXT_BOOLEAN_FIELDS and value is not None and not isinstance(value, bool):
            value = None

        if value is not None:
            verification[pillar][field] = value
            add_field_evidence(
                extra,
                "verification." + pillar + "." + field,
                field_evidence_ids(extra, key, pillar, value),
            )


def apply_text_analysis(extra, analysis, grounding_text="", event_grounding_text=""):
    verification = extra["verification"]
    protected_status = (
        verification["provenance"].get("provenance_status")
        == "earlier version found"
    )
    protected_context = protected_status and bool(
        verification["provenance"].get("previous_context_summary")
    )
    source_name = verification["source"].get("original_source_name")
    known_source = get_known_source(source_name)
    official_source = source_name if known_source and source_name == known_source["name"] else None
    protected_source = source_name
    if not official_source and not field_has_evidence_type(
        extra,
        "verification.source.original_source_name",
        {"provenance_image_candidate", "provenance_search_candidate"},
    ):
        protected_source = None

    copy_grounded_text_fields(
        extra,
        analysis,
        grounding_text,
        event_grounding_text,
        protected_source,
        protected_context,
    )
    apply_source_rules(
        extra,
        grounding_text,
        protected_source,
        pillar_evidence_ids(extra, "source"),
    )
    apply_provenance_rules(verification)
    apply_date_rules(verification)
    apply_location_rules(
        extra,
        event_grounding_text,
        pillar_evidence_ids(extra, "location"),
    )
    apply_motivation_rules(verification, official_source)
    clean_visual_location_candidates(verification)


def reset_text_analysis(extra):
    verification = extra["verification"]
    links = extra.setdefault("field_evidence", {})
    for pillar, field in TEXT_RESET_FIELDS:
        verification[pillar][field] = None
        links.pop("verification." + pillar + "." + field, None)
    protected_status = (
        verification["provenance"].get("provenance_status")
        == "earlier version found"
    )
    if not protected_status:
        verification["provenance"]["previous_context_summary"] = None
        verification["provenance"]["provenance_status"] = "unknown"
        links.pop("verification.provenance.previous_context_summary", None)
    platform = extra["normalized_video_information"].get("platform")
    official_source = get_platform_source(platform)
    protected_search_source = field_has_evidence_type(
        extra,
        "verification.source.original_source_name",
        {"provenance_image_candidate", "provenance_search_candidate"},
    )
    if official_source:
        verification["source"]["original_source_name"] = official_source["name"]
        verification["source"]["source_type"] = official_source["source_type"]
        verification["source"]["source_is_uploader"] = True
    else:
        if not protected_search_source:
            verification["source"]["original_source_name"] = None
            links.pop("verification.source.original_source_name", None)
        verification["source"]["source_type"] = None
        verification["source"]["source_is_uploader"] = None
        links.pop("verification.source.source_type", None)
        links.pop("verification.source.source_is_uploader", None)
    vision = (extra.get("automation") or {}).get("vision") or {}
    verification["location"]["candidate_locations"] = list(vision.get("candidate_locations") or [])
    extra.pop("rationales", None)
