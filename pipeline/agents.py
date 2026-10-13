"""Independent pillar roles sharing one local inference runtime."""

from copy import deepcopy
from dataclasses import dataclass
import json

from pillars.date import DATE_GRANULARITIES, DATE_MISMATCHES
from pillars.location import LOCATION_MISMATCHES
from pillars.motivation import CONTEXT_CATEGORIES, MOTIVATION_MISMATCHES
from pillars.provenance import PROVENANCE_STATUSES
from pillars.source import SOURCE_TYPES
from utils.schema import empty_verification

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


@dataclass(frozen=True)
class PillarAgent:
    name: str
    pillar: str
    responsibility: str

    def run(
        self, runtime, *, rules, component, applicable, claimed_fields, media, evidence
    ):
        schema = {self.pillar: empty_verification()[self.pillar]}
        allowed = {
            field: sorted(values) for field, values in ENUMS[self.pillar].items()
        }
        instructions = (
            rules
            + "\nYou are "
            + self.name
            + ". "
            + self.responsibility
            + "\nAssess only the "
            + self.pillar
            + " pillar independently. Other agents' conclusions are not evidence."
            + "\nReturn fields and assessments keyed ONLY by '"
            + self.pillar
            + "'."
            + "\nUse not_applicable when this dimension is not asserted; never treat missing evidence as support."
            + "\nExtract fields using this schema:\n"
            + json.dumps(schema)
            + "\nAllowed categorical values:\n"
            + json.dumps(allowed)
        )
        if self.pillar == "provenance":
            instructions += "\nAlso return core_assessment with the same assessment structure, checking the asserted subject/action/event."
        return runtime.generate(
            instructions,
            {
                "agent": self.name,
                "component": deepcopy(component),
                "applicable_pillars": {self.pillar: applicable},
                "claimed_fields": {self.pillar: deepcopy(claimed_fields)},
                "supplied_media": deepcopy(media),
                "evidence": deepcopy(evidence),
            },
            max_new_tokens=2048,
        )


PILLAR_AGENTS = (
    PillarAgent(
        "provenance_agent",
        "provenance",
        "Identify footage history, earlier versions, reuse and context; verify that the supplied media depicts the asserted event.",
    ),
    PillarAgent(
        "source_agent",
        "source",
        "Identify current uploader and original creator/publisher and compare the claim's attribution with the recovered source.",
    ),
    PillarAgent(
        "date_agent",
        "date",
        "Establish capture/event time from dated event evidence and compare it with claimed time; keep upload and archive dates separate.",
    ),
    PillarAgent(
        "location_agent",
        "location",
        "Establish the event location from textual/visual evidence and compare it with the claimed place; a map result alone is not confirmation.",
    ),
    PillarAgent(
        "motivation_agent",
        "motivation",
        "Recover the original publication context and assess reframing; do not guess the creator's psychological intention.",
    ),
)


@dataclass(frozen=True)
class AggregatorAgent:
    name: str = "verdict_agent"

    def run(self, runtime, *, instructions, data):
        return runtime.generate(
            "You are "
            + self.name
            + ". Synthesize the validated findings of the five pillar agents.\n"
            + instructions,
            {"agent": self.name, **deepcopy(data)},
            max_new_tokens=2048,
        )


VERDICT_AGENT = AggregatorAgent()
