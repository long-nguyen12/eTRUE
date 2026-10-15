from copy import deepcopy
import json

from pillars.date import DATE_GRANULARITIES, DATE_MISMATCHES
from pillars.location import LOCATION_MISMATCHES
from pillars.motivation import CONTEXT_CATEGORIES, MOTIVATION_MISMATCHES
from pillars.provenance import PROVENANCE_STATUSES
from pillars.source import SOURCE_TYPES
from utils.schema import empty_verification

from .text import plain_explanation

ENUMS = {
    "provenance": {"provenance_status": PROVENANCE_STATUSES},
    "source": {"source_type": SOURCE_TYPES},
    "date": {
        "capture_date_granularity": DATE_GRANULARITIES,
        "date_mismatch_type": DATE_MISMATCHES,
    },
    "location": {"location_mismatch_type": LOCATION_MISMATCHES},
    "motivation": {
        "original_context_category": CONTEXT_CATEGORIES,
        "motivation_mismatch_type": MOTIVATION_MISMATCHES,
    },
}


def run_pillar(
    runtime, pillar, *, rules, component, applicable, claimed_fields, media, evidence
):
    name = pillar + "_agent"
    defaults = empty_verification()[pillar]
    assessment = {
        "status": "unknown" if applicable else "not_applicable",
        "explanation": "Insufficient independently grounded evidence."
        if applicable else "The claim does not assert this dimension.",
        "supporting_citations": [],
        "refuting_citations": [],
    }
    example = {
        "fields": {pillar: defaults},
        "field_citations": {pillar + "." + field: [] for field in defaults},
        "assessments": {pillar: assessment},
    }
    if pillar == "provenance":
        example["core_assessment"] = deepcopy(assessment)
        example["core_assessment"]["status"] = "unknown"
        example["core_assessment"]["explanation"] = "Insufficient independently grounded evidence."
    allowed = {field: sorted(values) for field, values in ENUMS[pillar].items()}
    instructions = (
        rules
        + "\nYou are "
        + name
        + ". "
        + PILLAR_PROMPTS[pillar]
        + "\nAssess only "
        + pillar
        + " independently; derive prerequisites from raw evidence, never other agents."
        + " Media paths are references; use supplied vision observations."
        + " Follow applicable_pillars: false requires not_applicable; otherwise supported/refuted/unknown."
        + " Copy claimed fields from claimed_fields. Unknown facts are null; absent lists are []."
        + " Cite observed fields and decisive findings under the shared rules."
        + " Fill response_template exactly; keep every field and root key, including core_assessment when present."
        + " field_citations uses the literal dotted keys in the template, never nested pillar objects."
        + "\nAllowed categorical values:\n"
        + json.dumps(allowed)
    )
    if pillar == "provenance":
        instructions += "\ncore_assessment is always applicable, separately from provenance applicability."
    result = runtime.generate(
        instructions,
        {
            "agent": name,
            "component": deepcopy(component),
            "applicable_pillars": {pillar: applicable},
            "claimed_fields": {pillar: deepcopy(claimed_fields)},
            "supplied_media": deepcopy(media),
            "evidence": deepcopy(evidence),
            "response_template": deepcopy(example),
        },
        max_new_tokens=2048,
    )
    if not isinstance(result, dict):
        raise ValueError("Pillar agent response must be an object")
    result.setdefault("field_citations", {})
    if pillar == "provenance":
        result.setdefault("core_assessment", deepcopy(example["core_assessment"]))
    if set(result) != set(example):
        raise ValueError("Pillar agent response must contain exactly " + ", ".join(example))
    for key in ("fields", "field_citations", "assessments"):
        if not isinstance(result[key], dict):
            raise ValueError(key + " must be an object")
    if set(result["fields"]) != {pillar} or set(result["assessments"]) != {pillar}:
        raise ValueError("fields and assessments must contain only " + pillar)
    fields = result["fields"][pillar]
    if not isinstance(fields, dict) or set(fields) != set(defaults):
        raise ValueError(pillar + " fields must contain every defined field and no others")
    for field, value in fields.items():
        if field == "verified_coordinates" and value is not None:
            raise ValueError("verified_coordinates must remain null for geocoding")
        if value is None:
            if isinstance(defaults[field], list):
                raise ValueError(pillar + "." + field + " must be a list")
            continue
        if isinstance(defaults[field], list):
            valid = isinstance(value, list) and all(isinstance(v, (str, dict)) for v in value)
        elif field in {"provenance_mismatch", "source_is_uploader", "source_mismatch"}:
            valid = isinstance(value, bool)
        else:
            valid = isinstance(value, str)
        if not valid:
            raise ValueError(pillar + "." + field + " has an invalid type")
    nested = result["field_citations"].pop(pillar, {})
    if isinstance(nested, dict):
        for field, citations in nested.items():
            path = pillar + "." + field
            if not result["field_citations"].get(path):
                result["field_citations"][path] = citations
    for path, citations in result["field_citations"].items():
        if path not in {pillar + "." + field for field in defaults}:
            raise ValueError("Unknown field citation path: " + str(path))
        result["field_citations"][path] = _validate_citation_shape(citations)
    _validate_assessment(result["assessments"][pillar], applicable)
    if pillar == "provenance":
        _validate_assessment(result["core_assessment"], True)
    return result


def _validate_citation_shape(citations):
    if not isinstance(citations, list):
        raise ValueError("Citations must be a list")
    valid = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        if all(
            isinstance(citation.get(key), str) and citation[key].strip()
            for key in ("evidence_id", "quote")
        ):
            valid.append({key: citation[key].strip() for key in ("evidence_id", "quote")})
    return valid


def _validate_assessment(assessment, applicable):
    keys = {"status", "explanation", "supporting_citations", "refuting_citations"}
    if not isinstance(assessment, dict) or set(assessment) != keys:
        raise ValueError("Assessment requires status, explanation and both citation lists")
    if not applicable:
        assessment.update(
            status="not_applicable",
            explanation="The claim does not assert this dimension.",
            supporting_citations=[], refuting_citations=[],
        )
    statuses = {"supported", "refuted", "unknown"} if applicable else {"not_applicable"}
    if not isinstance(assessment["status"], str) or assessment["status"] not in statuses:
        raise ValueError("Invalid assessment status for pillar applicability")
    assessment["explanation"] = plain_explanation(assessment["explanation"])
    for key in ("supporting_citations", "refuting_citations"):
        assessment[key] = _validate_citation_shape(assessment[key])


PILLAR_PROMPTS = {
    "provenance": (
        "Establish this video's documented origin, earlier context and reuse. original requires positive "
        "creation/publication evidence identifying this footage; absence of earlier results is insufficient. "
        "Distinguish media identity from event occurrence. Report only documented earliest publication. "
        "Set provenance_mismatch true/false for documented contradiction/agreement, otherwise null. "
        "core_assessment checks every asserted subject/action/event from independent event reports; "
        "visual depiction and verified video provenance are not prerequisites."
    ),
    "source": (
        "Separate current uploader from original creator/publisher/broadcaster. Platform metadata may identify "
        "the uploader; original_source_name and source_type require retrieved_page proof about THIS footage/source. "
        "Authority, search order and platform names do not establish authorship. Normalize aliases only when documented. "
        "Set source_is_uploader and source_mismatch only from documented identity and attribution comparisons: "
        "true/false requires evidence, otherwise null."
    ),
    "date": (
        "Compare claimed_date with the documented date of the asserted event, or capture if explicitly claimed. "
        "A report identifying the same event can suffice; the date need not appear in keyframes. "
        "Extract video_upload_date from platform_metadata and earliest_online_date from archive_record observations; "
        "these and publication dates establish availability, not capture. Preserve actual capture_date_granularity; "
        "invent no precision. date_mismatch_type=same requires evidence at least as precise as the claim; "
        "coarser compatible dates remain unknown."
    ),
    "location": (
        "Verify the depicted event's location from linked retrieved_page passages; choose only supported specificity. "
        "Seek independent sources, not different evidence types or copied reports; one direct documented source "
        "can suffice. Broad compatibility cannot confirm a specific claim. candidate_locations are place-name leads; "
        "visual_location_clues require explicit keyframe_analysis observations, never guessed terrain/weather. "
        "Compare with claimed_location; leave verified_coordinates null for geocoding."
    ),
    "motivation": (
        "Compare documented original framing with claimed_framing; never infer psychological intent. Extract "
        "original_caption/original_description verbatim only from retrieved_page observations identifying THIS video's "
        "original context; without either, original_context_category is unknown. Reframing requires documented "
        "change, not merely a political topic. Satire/entertainment categories require that original context; "
        "old news as current requires an independently established older date and current-event framing here."
    ),
}
