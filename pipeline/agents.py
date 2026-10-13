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
        "Identify footage history, earlier versions, reuse and context; verify that the supplied media depicts the asserted event.\n"
        "WORKFLOW:\n"
        "1. Check near_duplicate_matches and image_candidate evidence for visually matching pages. Use match_types (full/partial) and clip_similarity scores when available.\n"
        "2. Compare published_at dates across duplicate evidence to identify the earliest_known_url and earliest_known_date.\n"
        "3. Determine provenance_status: original (no earlier version found), earlier version found (same content predates claim), "
        "repost (verbatim re-upload), edited excerpt (cropped/cut version), compilation (combined with other footage), "
        "screen recording (recorded from another screen), or unknown.\n"
        "4. If an earlier version exists, extract previous_context_summary from its page title/description.\n"
        "5. Set provenance_mismatch=true only if the earlier context contradicts the claim's event, date, location, source, or framing.\n"
        "CONSTRAINTS:\n"
        "- An archive/wayback date is an online availability date, not a capture date.\n"
        "- A search_candidate is not independently verified; only retrieved_page evidence is decisive.\n"
        "- Do not mark provenance_mismatch based on missing evidence alone.",
    ),
    PillarAgent(
        "source_agent",
        "source",
        "Identify who uploaded the video and who originally created, published, or broadcast it.\n"
        "WORKFLOW:\n"
        "1. Extract uploader_name and uploader_profile from platform_metadata or retrieved_page evidence.\n"
        "2. Inspect earliest known pages and duplicate evidence to identify the original_source_name "
        "(the earliest or most authoritative creator, outlet, agency, or account).\n"
        "3. Normalize source names: resolve aliases, channel names, outlet names, and @handles to canonical names.\n"
        "4. Classify source_type: eyewitness, news outlet, news agency, official account, political actor, "
        "activist group, entertainment source, satire source, or unknown.\n"
        "5. Set source_is_uploader=true only if normalized uploader and original source refer to the same entity.\n"
        "6. Set source_mismatch=true only if the claim's attribution contradicts the recovered original source.\n"
        "CONSTRAINTS:\n"
        "- A platform name (YouTube, Facebook) is not a source name.\n"
        "- Do not infer source_type from the claim alone; use evidence about the source entity.\n"
        "- Do not guess the original source from metadata that only identifies the current uploader.",
    ),
    PillarAgent(
        "date_agent",
        "date",
        "Establish when the video was captured or when the depicted event occurred, and compare with the claimed date.\n"
        "WORKFLOW:\n"
        "1. Extract video_upload_date from platform_metadata evidence.\n"
        "2. Extract earliest_online_date from archive_record evidence (Wayback snapshot dates).\n"
        "3. Search event keywords, actor names, location names in retrieved_page evidence to find the event date.\n"
        "4. Estimate the most plausible capture/event date (estimated_date) by combining event evidence. "
        "Prefer event-page evidence over upload/archive dates.\n"
        "5. Set capture_date_granularity: day, month, year, range, or unknown.\n"
        "6. Compare claimed_date with estimated_date to set date_mismatch_type: "
        "same, older video (video predates claim), newer video (video postdates claim), wrong event date, or unknown.\n"
        "CONSTRAINTS:\n"
        "- Upload date ≠ capture date. A video can be uploaded long after being recorded.\n"
        "- Archive/Wayback date = first online snapshot date, not capture or event date.\n"
        "- Do not use the claimed_date as evidence for estimated_date.\n"
        "- estimated_date must be grounded in retrieved_page evidence; platform_metadata dates alone are insufficient.\n"
        "- A capture date less precise than the claimed date cannot support a 'same' assessment.",
    ),
    PillarAgent(
        "location_agent",
        "location",
        "Determine where the video was captured or where the depicted event occurred.\n"
        "WORKFLOW:\n"
        "1. Extract candidate_locations from retrieved_page evidence, transcript, and metadata "
        "(place names mentioned in titles, descriptions, and page text).\n"
        "2. Use keyframe_analysis evidence for visual_location_clues: visible signs, road markings, "
        "landmarks, terrain, license plates, uniforms, language on signs, shop names, weather.\n"
        "3. Cross-reference textual candidates with visual clues to select the verified_location.\n"
        "4. The verified_location must be the most specific place supported by multiple independent evidence types.\n"
        "5. Compare claimed_location with verified_location to set location_mismatch_type: "
        "same, different city, different region, different country, or unknown.\n"
        "CONSTRAINTS:\n"
        "- Do not guess a location from terrain or weather appearance alone.\n"
        "- Geocoding/map lookup alone does not verify an event location; it only normalizes a place name.\n"
        "- A location mentioned in a claim is NOT evidence for verified_location.\n"
        "- Visual clues must be explicitly visible in keyframe analysis, not inferred.\n"
        "- candidate_locations must be place names (city, region, country), not descriptions of objects.",
    ),
    PillarAgent(
        "motivation_agent",
        "motivation",
        "Infer the original purpose or framing of the video and determine whether the current claim reframes it.\n"
        "WORKFLOW:\n"
        "1. Extract original_caption and original_description from the earliest known page or most authoritative source page "
        "(retrieved_page or platform_metadata evidence).\n"
        "2. Classify original_context_category based on original caption, description, source type, and event context: "
        "news report, eyewitness, official record, campaign, activism, entertainment, satire, advertisement, archive, or unknown.\n"
        "3. Compare claimed_framing with original_caption and original_description to detect reframing.\n"
        "4. Set motivation_mismatch_type: same (framing matches), satire as real (satire presented as genuine), "
        "entertainment as news (entertainment presented as real news), old news as current (old footage presented as current event), "
        "political reframing (context altered for political messaging), or unknown.\n"
        "CONSTRAINTS:\n"
        "- Do not guess the creator's psychological intention or motive.\n"
        "- original_context_category must be supported by evidence about the original context, not inferred from the claim.\n"
        "- satire as real requires the original source to be a known satire source.\n"
        "- old news as current requires a date mismatch (video significantly predates claimed event).\n"
        "- If original_caption and original_description are unavailable, set original_context_category to unknown.",
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
