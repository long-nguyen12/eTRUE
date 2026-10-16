"""Pydantic contracts for Qwen responses; evidence grounding is checked separately."""

from functools import lru_cache
import json
from typing import Annotated, Literal

from pydantic import (
    BaseModel, ConfigDict, Field, JsonValue, RootModel, StringConstraints,
    ValidationError, create_model,
)

from pillars.date import DATE_GRANULARITIES, DATE_MISMATCHES
from pillars.location import LOCATION_MISMATCHES
from pillars.motivation import CONTEXT_CATEGORIES, MOTIVATION_MISMATCHES
from pillars.provenance import PROVENANCE_STATUSES
from pillars.source import SOURCE_TYPES

NonemptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
VerdictLabel = Literal["TRUE", "FALSE", "NEI"]
PillarName = Literal["provenance", "source", "date", "location", "motivation"]
ReviewName = Literal["core", "provenance", "source", "date", "location", "motivation"]


class OutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class JsonObject(RootModel[dict[str, JsonValue]]):
    model_config = ConfigDict(strict=True)


class ClaimComponent(OutputModel):
    text: NonemptyString
    subject: NonemptyString | None
    action: NonemptyString | None
    time: NonemptyString | None
    location: NonemptyString | None
    attribution: NonemptyString | None
    context: NonemptyString | None


class ClaimNormalizationOutput(OutputModel):
    normalized_claim: NonemptyString
    claim_components: Annotated[list[ClaimComponent], Field(min_length=1)]


class VisionOutput(OutputModel):
    scene_summary: str
    ocr_text: list[str]
    landmarks: list[str]
    signs_and_logos: list[str]
    languages: list[str]
    terrain_and_weather: list[str]
    candidate_locations: list[str]


class Citation(OutputModel):
    evidence_id: NonemptyString
    quote: NonemptyString


class Assessment(OutputModel):
    status: Literal["supported", "refuted", "unknown", "not_applicable"]
    explanation: str
    supporting_citations: list[Citation]
    refuting_citations: list[Citation]


class ApplicableAssessment(Assessment):
    status: Literal["supported", "refuted", "unknown"]


class ProvenanceFields(OutputModel):
    provenance_status: Literal[tuple(sorted(PROVENANCE_STATUSES))] | None
    earliest_known_url: str | None
    earliest_known_date: str | None
    near_duplicate_matches: list[str | dict[str, JsonValue]]
    previous_context_summary: str | None
    provenance_mismatch: bool | None


class SourceFields(OutputModel):
    uploader_name: str | None
    uploader_profile: str | None
    original_source_name: str | None
    source_type: Literal[tuple(sorted(SOURCE_TYPES))] | None
    source_is_uploader: bool | None
    source_mismatch: bool | None


class DateFields(OutputModel):
    claimed_date: str | None
    video_upload_date: str | None
    earliest_online_date: str | None
    estimated_date: str | None
    capture_date_granularity: Literal[tuple(sorted(DATE_GRANULARITIES))] | None
    date_mismatch_type: Literal[tuple(sorted(DATE_MISMATCHES))] | None


class LocationFields(OutputModel):
    claimed_location: str | None
    candidate_locations: list[str | dict[str, JsonValue]]
    visual_location_clues: list[str | dict[str, JsonValue]]
    verified_location: str | None
    verified_coordinates: None
    location_mismatch_type: Literal[tuple(sorted(LOCATION_MISMATCHES))] | None


class MotivationFields(OutputModel):
    claimed_framing: str | None
    original_caption: str | None
    original_description: str | None
    original_context_category: Literal[tuple(sorted(CONTEXT_CATEGORIES))] | None
    motivation_mismatch_type: Literal[tuple(sorted(MOTIVATION_MISMATCHES))] | None


PILLAR_FIELDS = {
    "provenance": ProvenanceFields,
    "source": SourceFields,
    "date": DateFields,
    "location": LocationFields,
    "motivation": MotivationFields,
}


@lru_cache(maxsize=5)
def pillar_output_model(pillar: PillarName) -> type[BaseModel]:
    """Require exactly the named pillar in each envelope, without duplicating schemas."""
    fields = PILLAR_FIELDS[pillar]
    name = pillar.capitalize()
    field_group = create_model(name + "FieldGroup", __base__=OutputModel,
                               **{pillar: (fields, ...)})
    assessment_group = create_model(
        name + "AssessmentGroup", __base__=OutputModel,
        **{pillar: (ApplicableAssessment if pillar == "provenance" else Assessment, ...)},
    )
    citation_key = Literal[tuple(pillar + "." + field for field in fields.model_fields)]
    definitions = {
        "fields": (field_group, ...),
        "field_citations": (dict[citation_key, list[Citation]], ...),
        "assessments": (assessment_group, ...),
    }
    if pillar == "provenance":
        definitions["core_assessment"] = (ApplicableAssessment, ...)
    return create_model(name + "Output", __base__=OutputModel, **definitions)


class ComponentVerdict(OutputModel):
    component_id: NonemptyString
    verdict: VerdictLabel
    explanation: str
    provenance_required: bool = True
    assessment_reviews: dict[ReviewName, ApplicableAssessment]


class VerdictOutput(OutputModel):
    verdict: VerdictLabel
    explanation: str
    open_questions: list[str]
    component_verdicts: Annotated[list[ComponentVerdict], Field(min_length=1)]


def _validation_message(model, error):
    errors = error.errors(include_input=False, include_url=False)
    details = []
    for item in errors[:3]:
        path = ".".join(map(str, item["loc"])) or "$"
        details.append(path + ": " + item["msg"])
    if len(errors) > 3:
        details.append(str(len(errors) - 3) + " more validation errors")
    return model.__name__ + ": " + "; ".join(details)


def validate_output(model: type[BaseModel], value: dict) -> dict:
    try:
        value = JsonObject.model_validate(value).root
        _check_finite_numbers(value)
        return model.model_validate(value).model_dump(mode="json")
    except ValidationError as error:
        raise ValueError(_validation_message(model, error)) from None


def parse_json_object(text: str) -> dict:
    """Reject malformed JSON, prose, code fences and non-object roots."""
    try:
        value = JsonObject.model_validate_json(text).root
        _check_finite_numbers(value)
        return value
    except ValidationError as error:
        raise ValueError(_validation_message(JsonObject, error)) from None


def _check_finite_numbers(value):
    # Pydantic's JSON parser accepts NaN/Infinity; JSON output must not contain them.
    try:
        json.dumps(value, allow_nan=False)
    except ValueError:
        raise ValueError("Model output contains a non-finite JSON number") from None
