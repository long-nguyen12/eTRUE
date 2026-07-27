"""Extract text candidates and apply conservative pillar validation."""

import json

from build_etrue import normalize_date
from pillars.date import DATE_GRANULARITIES, DATE_MISMATCHES, apply_date_rules, date_is_grounded
from pillars.location import LOCATION_MISMATCHES, apply_location_rules, clean_visual_location_candidates
from pillars.motivation import CONTEXT_CATEGORIES, MOTIVATION_MISMATCHES, apply_motivation_rules
from pillars.provenance import apply_provenance_rules, previous_context_is_useful
from pillars.source import OFFICIAL_SOURCES, SOURCE_TYPES, apply_source_rules, source_name_is_grounded
from stages import MODELS
from utils.evidence import add_field_evidence, mark_automated
from utils.files import read_json, trim, write_json
from utils.model_output import enum_value, parse_json_output, release_models
from utils.records import sidecar_path
from utils.text import text_value_is_grounded


TEXT_OUTPUT_FIELDS = {
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

TEXT_PROMPT = """You are annotating a video verification dataset. Use only the supplied evidence.
Return one JSON object and no prose. Use null when evidence is insufficient. Do not infer facts from
the fact-check rating, and do not treat a model observation as stronger than a quoted source.
Keep every returned string under 60 words so the JSON object remains compact and complete.
The fact-check publisher reports on the claim; it is not the original video source unless evidence
explicitly says it created or uploaded the video. Boolean fields must be only true, false, or null.
Do not use a platform name such as Reddit, YouTube, Facebook, or X as the original source.
`previous_context_summary` must describe the video's different, earlier context. It must not repeat
the claim and must not contain meta commentary such as "not provided" or "not relevant".
`claimed_location` is the place asserted by the circulating claim; `verified_location` is the
actual place. For evidence saying "in Brazil, not the United States", claimed is United States and
verified is Brazil.
Set a mismatch field to null when either side of its comparison is missing. `estimated_date` means
the capture date or depicted event date of the current video, not the date when the claim circulated.
Ignore unrelated dates in the fact-check article. The Boolean keys provenance_mismatch,
source_is_uploader, and source_mismatch may never contain objects or explanations.

Allowed categorical values:
- source_type: eyewitness, news outlet, news agency, official account, political actor,
  activist group, entertainment source, satire source, unknown
- capture_date_granularity: day, month, year, range, unknown
- date_mismatch_type: same, older video, newer video, wrong event date, unknown
- location_mismatch_type: same, different city, different region, different country, unknown
- original_context_category: news report, eyewitness, official record, campaign, activism,
  entertainment, satire, advertisement, archive, unknown
- motivation_mismatch_type: same, satire as real, entertainment as news, old news as current,
  political reframing, unknown

Return exactly these keys:
{
  "previous_context_summary": null,
  "provenance_mismatch": null,
  "original_source_name": null,
  "source_type": null,
  "source_is_uploader": null,
  "source_mismatch": null,
  "claimed_date": null,
  "estimated_date": null,
  "capture_date_granularity": null,
  "date_mismatch_type": null,
  "claimed_location": null,
  "candidate_locations": [],
  "verified_location": null,
  "location_mismatch_type": null,
  "claimed_framing": null,
  "original_context_category": null,
  "motivation_mismatch_type": null
}

EVIDENCE:
"""


def text_input(record, sidecar):
    data = record["data"]
    video = data.get("video_information") or {}
    transcript = video.get("video_transcript")
    verification = sidecar.get("verification") or {}
    candidate_facts = {}
    for pillar, values in verification.items():
        for field, value in values.items():
            if field in TEXT_OUTPUT_FIELDS or "mismatch" in field or value is None or value == []:
                continue
            candidate_facts[pillar + "." + field] = value
    vision = (sidecar.get("automation") or {}).get("vision") or {}
    if vision.get("status") != "ok":
        vision = None
    return {
        "claim": data.get("claim"),
        "fact_check_publisher": {"name": "Snopes", "url": data.get("url")},
        "fact_check_article": trim(data.get("content"), 6000),
        "current_video": {
            "url": video.get("video_url"),
            "platform": video.get("platform"),
            "upload_date": normalize_date(video.get("video_date")),
            "title": video.get("video_headline"),
            "description": trim(video.get("video_description"), 2000),
        },
        "transcript": trim(transcript, 4000),
        "candidate_facts": candidate_facts,
        "visual_analysis": vision,
        "web_retrieval": (sidecar.get("automation") or {}).get("web"),
        "fact_check_evidence": [
            evidence
            for evidence in sidecar.get("evidence", [])
            if evidence.get("type") == "fact_check_evidence"
        ][:10],
    }


def copy_grounded_text_fields(
    sidecar,
    analysis,
    grounding_text,
    event_grounding_text,
    protected_source,
    protected_context,
    evidence_ids,
):
    """Copy model fields that pass their basic type and evidence checks."""
    verification = sidecar["verification"]

    for key, (pillar, field) in TEXT_FIELD_MAP.items():
        value = analysis.get(key)

        if key == "previous_context_summary" and protected_context:
            continue
        if key in {"original_source_name", "source_type", "source_is_uploader"} and protected_source:
            continue

        if key == "previous_context_summary" and not previous_context_is_useful(
            value,
            verification["motivation"].get("claimed_framing"),
            grounding_text,
        ):
            value = None
        if key == "original_source_name" and not source_name_is_grounded(value, grounding_text):
            value = None
        if key == "estimated_date" and not date_is_grounded(value, event_grounding_text):
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
            add_field_evidence(sidecar, "verification." + pillar + "." + field, evidence_ids)


def apply_text_analysis(sidecar, analysis, grounding_text="", event_grounding_text=""):
    """Copy model output, then validate each verification pillar in turn."""
    verification = sidecar["verification"]
    protected_source = verification["source"].get("original_source_name")
    if protected_source not in OFFICIAL_SOURCES.values():
        protected_source = None
    protected_context = verification["provenance"].get("provenance_status") == "earlier version found"
    evidence_ids = [evidence["id"] for evidence in sidecar.get("evidence", [])]

    copy_grounded_text_fields(
        sidecar,
        analysis,
        grounding_text,
        event_grounding_text,
        protected_source,
        protected_context,
        evidence_ids,
    )
    apply_source_rules(sidecar, grounding_text, protected_source, evidence_ids)
    apply_provenance_rules(verification)
    apply_date_rules(verification)
    apply_location_rules(sidecar, event_grounding_text, evidence_ids)
    apply_motivation_rules(verification, protected_source)
    clean_visual_location_candidates(verification)


def reset_text_analysis(sidecar):
    """Remove prior text-derived values while preserving stronger evidence."""
    verification = sidecar["verification"]
    for pillar, field in TEXT_RESET_FIELDS:
        verification[pillar][field] = None
    if verification["provenance"].get("provenance_status") != "earlier version found":
        verification["provenance"]["previous_context_summary"] = None
    platform = sidecar["normalized_video_information"].get("platform")
    official_source = OFFICIAL_SOURCES.get(platform)
    if official_source:
        verification["source"]["original_source_name"] = official_source
        verification["source"]["source_type"] = "news outlet"
        verification["source"]["source_is_uploader"] = True
    else:
        verification["source"]["original_source_name"] = None
        verification["source"]["source_type"] = None
        verification["source"]["source_is_uploader"] = None
    vision = (sidecar.get("automation") or {}).get("vision") or {}
    verification["location"]["candidate_locations"] = list(vision.get("candidate_locations") or [])
    sidecar.pop("rationales", None)


def build_text_grounding(record, sidecar):
    """Return the full prompt evidence and the stricter event-only evidence."""
    supplied_evidence = text_input(record, sidecar)
    grounding_text = json.dumps(supplied_evidence, ensure_ascii=False)
    current_video = supplied_evidence.get("current_video") or {}
    event_evidence = {
        "fact_check_article": supplied_evidence.get("fact_check_article"),
        "fact_check_evidence": supplied_evidence.get("fact_check_evidence"),
        "transcript": supplied_evidence.get("transcript"),
        "visual_analysis": supplied_evidence.get("visual_analysis"),
        "title": current_video.get("title"),
        "description": current_video.get("description"),
    }
    event_grounding_text = json.dumps(event_evidence, ensure_ascii=False)
    return grounding_text, event_grounding_text


def generate_text_result(model, tokenizer, device, grounding_text):
    """Ask Qwen for one result dictionary and preserve parse errors."""
    import torch

    messages = [
        {"role": "system", "content": "Extract conservative, evidence-grounded JSON."},
        {"role": "user", "content": TEXT_PROMPT + grounding_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=6000).to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=400,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )

    try:
        return {
            "status": "ok",
            "analysis": parse_json_output(generated),
            "model": MODELS["text"],
        }
    except (ValueError, json.JSONDecodeError) as error:
        return {
            "status": "parse_error",
            "raw_output": generated,
            "error": str(error),
            "model": MODELS["text"],
        }


def run(records, output, model_cache, force=False, offline=False):
    """Generate or reuse Qwen output, validate it, and update sidecars."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache = output / "cache" / "text"
    cache.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        MODELS["text"], cache_dir=str(model_cache), local_files_only=offline
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODELS["text"],
        cache_dir=str(model_cache),
        torch_dtype=dtype,
        local_files_only=offline,
    ).to(device)
    model.eval()

    for number, record in enumerate(records, 1):
        cached = cache / (record["claim_id"] + ".json")
        sidecar_file = sidecar_path(output, record["claim_id"])
        sidecar = read_json(sidecar_file)
        grounding_text, event_grounding_text = build_text_grounding(record, sidecar)

        if cached.exists() and not force:
            result = read_json(cached)
        else:
            result = generate_text_result(model, tokenizer, device, grounding_text)
            write_json(cached, result)

        reset_text_analysis(sidecar)
        if result.get("status") == "ok":
            apply_text_analysis(sidecar, result["analysis"], grounding_text, event_grounding_text)
        mark_automated(sidecar, "text", result)
        write_json(sidecar_file, sidecar)
        if number % 10 == 0:
            print("text", number, "/", len(records), flush=True)
    release_models(model, tokenizer)
