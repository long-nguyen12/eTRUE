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
        "field_citations": {},
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
        + "\nAssess only the "
        + pillar
        + " pillar independently. Other agents' conclusions are not evidence."
        + " Derive any prerequisite from the supplied raw evidence yourself."
        + " Media paths are references, not images you can inspect; use only provided vision observations."
        + "\nReturn fields and assessments keyed ONLY by '"
        + pillar
        + "'."
        + "\nUse not_applicable exactly when applicable_pillars is false; otherwise use supported, refuted or unknown."
        + " Missing evidence is unknown, not support or refutation. Keep unknown facts null and absent lists []."
        + " Return every field shown in the example; do not add fields or other pillar keys."
        + "\nfield_citations maps each non-null observed field's exact 'pillar.field' path to"
        + ' [{"evidence_id":"ev-...","quote":"exact excerpt from observation"}].'
        + " Do not cite claimed fields copied from the component. Lists need citations when nonempty."
        + " Only text inside evidence.observation is quotable; published_at or other metadata outside it is not."
        + " supported/refuted requires a directly relevant quote in the corresponding citation list."
        + " Keep explanations to one or two short prose sentences; IDs, URLs and quotations belong in citation fields."
        + "\nReturn exactly this complete JSON structure with grounded values replacing defaults:\n"
        + json.dumps(example)
        + "\nAllowed categorical values:\n"
        + json.dumps(allowed)
    )
    if pillar == "provenance":
        instructions += "\ncore_assessment is always applicable and checks the asserted subject/action/event, separately from media provenance."
    result = runtime.generate(
        instructions,
        {
            "agent": name,
            "component": deepcopy(component),
            "applicable_pillars": {pillar: applicable},
            "claimed_fields": {pillar: deepcopy(claimed_fields)},
            "supplied_media": deepcopy(media),
            "evidence": deepcopy(evidence),
        },
        max_new_tokens=2048,
    )
    if not isinstance(result, dict):
        raise ValueError("Pillar agent response must be an object")
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
    for path, citations in result["field_citations"].items():
        if path not in {pillar + "." + field for field in defaults}:
            raise ValueError("Unknown field citation path: " + str(path))
        _validate_citation_shape(citations)
    _validate_assessment(result["assessments"][pillar], applicable)
    if pillar == "provenance":
        _validate_assessment(result["core_assessment"], True)
    return result


def _validate_citation_shape(citations):
    if not isinstance(citations, list):
        raise ValueError("Citations must be a list")
    for citation in citations:
        if not isinstance(citation, dict) or set(citation) != {"evidence_id", "quote"}:
            raise ValueError("Each citation requires evidence_id and quote")
        if any(not isinstance(value, str) or not value.strip() for value in citation.values()):
            raise ValueError("Citation evidence_id and quote must be nonempty strings")


def _validate_assessment(assessment, applicable):
    keys = {"status", "explanation", "supporting_citations", "refuting_citations"}
    if not isinstance(assessment, dict) or set(assessment) != keys:
        raise ValueError("Assessment requires status, explanation and both citation lists")
    statuses = {"supported", "refuted", "unknown"} if applicable else {"not_applicable"}
    if not isinstance(assessment["status"], str) or assessment["status"] not in statuses:
        raise ValueError("Invalid assessment status for pillar applicability")
    assessment["explanation"] = plain_explanation(assessment["explanation"])
    for key in ("supporting_citations", "refuting_citations"):
        _validate_citation_shape(assessment[key])


PILLAR_PROMPTS = {
    "provenance": (
        "Establish the supplied video's documented history and whether it depicts the asserted event/context.\n"
        "1. Treat search_candidate, image_candidate and duplicate matches as leads only; a matching image does not "
        "prove origin, time, place or event identity. Do not invent similarity scores or visual comparisons.\n"
        "2. Use retrieved_page passages explicitly linking this footage to its creator, an earlier publication, "
        "reuse or editing. Report the earliest documented URL/date only within supplied evidence; "
        "the date must occur in the cited observation.\n"
        "3. original requires positive documentation of original creation/publication and identity with this video. "
        "Finding no earlier version is not proof of originality. Use unknown when origin is unresolved. "
        "Use other allowed history categories only when documented.\n"
        "4. Extract previous_context_summary from the documented earlier event/context. "
        "Set provenance_mismatch=true only for a direct contextual contradiction; "
        "false requires positive agreement, otherwise null.\n"
        "5. For core_assessment, verify the asserted subject/action/event using retrieved_page quotes. "
        "A genuine event does not alone establish that the supplied video depicts it. "
        "For provenance, distinguish event evidence from evidence identifying this media.\n"
        "Only retrieved_page evidence is decisive here. Archive dates establish availability, not capture dates."
    ),
    "source": (
        "Identify who uploaded the video and who originally created, published, or broadcast it.\n"
        "1. Extract uploader_name and uploader_profile from platform_metadata or retrieved_page evidence.\n"
        "2. Set original_source_name only when a retrieved_page passage documents creation, original publication, "
        "broadcast origin or credit for THIS footage. Being authoritative or the earliest search result "
        "does not establish original authorship. Current uploader metadata alone does not identify the creator.\n"
        "3. Normalize aliases, channel names and handles only when identity is documented. "
        "A platform name such as YouTube or Facebook is not a source entity.\n"
        "4. Classify source_type using retrieved_page evidence about that source entity and the allowed values.\n"
        "5. Set source_is_uploader=true only when documented original source and uploader are the same entity; "
        "false requires evidence they differ, otherwise null.\n"
        "6. Compare the claim's attribution to the recovered original source. Set source_mismatch=true "
        "for documented contradiction, false for documented agreement, otherwise null.\n"
        "Only retrieved_page evidence establishes the original creator or source type. Verified platform_metadata "
        "can establish the current uploader; supplied metadata or the claim's attribution is not independent evidence."
    ),
    "date": (
        "Establish the capture date, or a documented event date tied to THIS video, and compare to the claim.\n"
        "1. Copy claimed_date from claimed_fields. Extract video_upload_date from a platform_metadata "
        "observation and earliest_online_date from an archive_record observation when explicitly present. "
        "They describe online availability only; neither proves when footage was captured.\n"
        "2. Set estimated_date only from retrieved_page passages dating the depicted event or recording "
        "and establishing its identity with this video. An article publication date is not automatically an event date. "
        "If evidence dates only an unlinked event, leave estimated_date null.\n"
        "3. Use capture_date_granularity to preserve actual precision: day, month, year, range or unknown. "
        "Never invent day/month details.\n"
        "4. Set date_mismatch_type from explicit comparison with claimed_date using the allowed values. "
        "same requires evidence at least as precise as the claim; coarser compatible dates remain unknown.\n"
        "Only retrieved_page evidence is decisive for the date assessment. claimed_date, upload, archive and "
        "publication dates do not prove capture time. Metadata published_at outside observation is not citable."
    ),
    "location": (
        "Locate the recording or depicted event using evidence linked to THIS footage/event.\n"
        "1. Copy claimed_location from claimed_fields. Gather candidate_locations from page text, metadata "
        "or supplied transcript as leads, not verified facts. Candidate locations must be place names.\n"
        "2. Extract visual_location_clues only from explicit keyframe_analysis observations, such as a "
        "transcribed sign or identified landmark. You cannot inspect image files from paths.\n"
        "3. Choose verified_location only from retrieved_page passages establishing the depicted event's place. "
        "Use the most specific place directly supported. Seek corroboration from independent sources, "
        "not merely different evidence types or copies of one report. A direct documented source can suffice; "
        "unresolved disagreement remains unknown.\n"
        "4. Compare the grounded location to claimed_location using allowed location_mismatch_type values. "
        "A broad compatible location does not prove a more specific claim. Leave verified_coordinates null; "
        "the pipeline assigns coordinates through geocoding after location verification.\n"
        "Only retrieved_page evidence is decisive. Geocoding normalizes names, not event identity. "
        "Do not infer place from terrain/weather, the claim alone, or unconfirmed visual resemblance."
    ),
    "motivation": (
        "Recover documented original framing and compare it with the current claim; never infer psychological intent.\n"
        "1. Copy claimed_framing from claimed_fields. Extract original_caption and original_description "
        "verbatim from retrieved_page observations documenting the original context of THIS video. "
        "Uploader metadata, supplied text, earliest search rank or outlet authority alone does not prove original context.\n"
        "2. Classify original_context_category using those passages and the allowed values. If neither "
        "original_caption nor original_description is available, use unknown.\n"
        "3. Compare documented original framing to claimed_framing using allowed motivation_mismatch_type values. "
        "satire as real requires documented satirical context; entertainment as news requires documented "
        "entertainment context; political reframing requires a documented change, not the topic being political.\n"
        "4. old news as current requires retrieved_page evidence establishing an older event/video date "
        "and a current-event framing in the claim. Derive those facts here; do not assume another agent verified them.\n"
        "Only retrieved_page evidence is decisive. Missing original context is unknown, not evidence of reframing."
    ),
}
