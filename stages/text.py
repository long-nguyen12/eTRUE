"""Extract text candidates and apply conservative pillar validation."""

import json

from build_etrue import normalize_date
from pillars.date import DATE_GRANULARITIES, DATE_MISMATCHES, apply_date_rules, date_is_grounded
from pillars.location import LOCATION_MISMATCHES, apply_location_rules, clean_visual_location_candidates
from pillars.motivation import CONTEXT_CATEGORIES, MOTIVATION_MISMATCHES, apply_motivation_rules
from pillars.provenance import (
    PROVENANCE_STATUSES,
    apply_provenance_rules,
    previous_context_is_useful,
)
from pillars.source import OFFICIAL_SOURCES, SOURCE_TYPES, apply_source_rules, source_name_is_grounded
from stages import MODELS
from stages.text_prompts import COMMON_RULES, TEXT_PROMPT_JOBS
from utils.evidence import add_field_evidence
from utils.files import read_json, trim, write_json
from utils.model_output import enum_value, parse_json_output, release_models
from utils.records import sidecar_path
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

PILLAR_EVIDENCE_TYPES = {
    "provenance": {
        "archive_record",
        "dataset_transcript",
        "fact_check_article",
        "fact_check_evidence",
        "local_visual_match",
        "provenance_image_candidate",
        "provenance_search_candidate",
    },
    "source": {
        "dataset_video_metadata",
        "fact_check_article",
        "fact_check_evidence",
        "platform_metadata",
        "provenance_image_candidate",
        "provenance_search_candidate",
    },
    "date": {
        "archive_record",
        "claim_text",
        "dataset_transcript",
        "dataset_video_metadata",
        "fact_check_article",
        "fact_check_evidence",
        "platform_metadata",
        "provenance_image_candidate",
        "provenance_search_candidate",
    },
    "location": {
        "claim_text",
        "dataset_transcript",
        "fact_check_article",
        "fact_check_evidence",
        "geocoder_result",
        "keyframe_analysis",
        "location_text_candidate",
        "provenance_image_candidate",
        "provenance_search_candidate",
    },
    "motivation": {
        "claim_text",
        "dataset_transcript",
        "dataset_video_metadata",
        "fact_check_article",
        "fact_check_evidence",
        "platform_metadata",
        "provenance_image_candidate",
        "provenance_search_candidate",
    },
}

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

def bounded_value(value, length=1000):
    if isinstance(value, str):
        return trim(value, length)
    if isinstance(value, list):
        return value[:10]
    if isinstance(value, dict):
        return {key: bounded_value(item, length) for key, item in value.items()}
    return value


def bounded_web_retrieval(value):
    if not isinstance(value, dict):
        return None
    metadata = value.get("platform_metadata") or {}
    fields = (
        "uploader",
        "uploader_url",
        "channel",
        "channel_url",
        "upload_date",
        "timestamp",
        "title",
        "description",
        "webpage_url",
        "extractor",
    )
    return {
        "url": value.get("url"),
        "retrieved_at": value.get("retrieved_at"),
        "platform_metadata": {
            key: trim(metadata[key], 1000) if key == "description" else metadata[key]
            for key in fields
            if metadata.get(key) is not None
        },
        "wayback": value.get("wayback"),
        "errors": (value.get("errors") or [])[:5],
    }


def text_input(record, extra):
    data = record["data"]
    video = data.get("video_information") or {}
    transcript = video.get("video_transcript")
    verification = extra.get("verification") or {}
    candidate_facts = {}
    for pillar, values in verification.items():
        for field, value in values.items():
            if field in TEXT_OUTPUT_FIELDS or "mismatch" in field or value is None or value == []:
                continue
            if field != "near_duplicate_matches":
                candidate_facts[pillar + "." + field] = bounded_value(value)
    vision = (extra.get("automation") or {}).get("vision") or {}
    if vision.get("status") != "ok":
        vision = None
    search = (extra.get("automation") or {}).get("search") or {}
    image_search = search.get("image_search") or {}
    reverse_image_candidates = None
    if image_search:
        pages = []
        fields = (
            "url",
            "canonical_url",
            "title",
            "description",
            "context_excerpt",
            "author",
            "site_name",
            "published_at",
            "archive_first_seen",
            "match_types",
            "matched_frames",
            "match_count",
        )
        for page in (image_search.get("pages") or [])[:5]:
            pages.append(
                {
                    key: trim(page[key], 400)
                    if key in {"description", "context_excerpt"}
                    else page[key]
                    for key in fields
                    if page.get(key) is not None
                }
            )
        reverse_image_candidates = {
            "status": "unverified_candidates",
            "web_entities": (image_search.get("web_entities") or [])[:10],
            "best_guess_labels": (image_search.get("best_guess_labels") or [])[:10],
            "pages": pages,
        }
    web_search_candidates = None
    if search.get("results"):
        fields = (
            "url",
            "canonical_url",
            "title",
            "description",
            "context_excerpt",
            "author",
            "site_name",
            "published_at",
            "archive_first_seen",
            "query",
            "query_type",
            "rank",
        )
        web_search_candidates = {
            "status": "unverified_candidates",
            "provider": search.get("search_provider"),
            "results": [
                {
                    key: trim(item[key], 400)
                    if key in {"description", "context_excerpt"}
                    else item[key]
                    for key in fields
                    if item.get(key) is not None
                }
                for item in search.get("results", [])[:5]
            ],
        }
    return {
        "claim": data.get("claim"),
        "fact_check_publisher": {"name": "Snopes", "url": data.get("url")},
        "fact_check_article": trim(data.get("content"), 5000),
        "current_video": {
            "url": video.get("video_url"),
            "platform": video.get("platform"),
            "upload_date": normalize_date(video.get("video_date")),
            "title": video.get("video_headline"),
            "description": trim(video.get("video_description"), 2000),
        },
        "transcript": trim(transcript, 3500),
        "candidate_facts": candidate_facts,
        "visual_analysis": vision,
        "web_retrieval": bounded_web_retrieval(
            (extra.get("automation") or {}).get("web")
        ),
        "web_search_candidates": web_search_candidates,
        "reverse_image_candidates": reverse_image_candidates,
        "fact_check_evidence": [
            {
                "id": evidence.get("id"),
                "type": evidence.get("type"),
                "source": evidence.get("source"),
                "observation": trim(evidence.get("observation"), 1200),
                "references": (evidence.get("references") or [])[:5],
            }
            for evidence in extra.get("evidence", [])
            if evidence.get("type") == "fact_check_evidence"
        ][:3],
    }


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
    """Return only evidence relevant to the field being assigned."""
    field = "verification." + pillar + "." + TEXT_FIELD_MAP[key][1]
    existing = list((extra.get("field_evidence") or {}).get(field) or [])
    allowed = PILLAR_EVIDENCE_TYPES[pillar]
    candidates = [
        evidence
        for evidence in extra.get("evidence", [])
        if evidence.get("id") and evidence.get("type") in allowed
    ]

    # Categorical/Boolean fields are conclusions over the pillar evidence, while
    # free-text fields should point to evidence that actually contains the value.
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
    evidence_ids=None,
):
    """Copy model fields that pass their basic type and evidence checks."""
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
        if (
            key in {"source_type", "source_is_uploader"}
            and protected_source in OFFICIAL_SOURCES.values()
        ):
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
    """Copy model output, then validate each verification pillar in turn."""
    verification = extra["verification"]
    protected_status = (
        verification["provenance"].get("provenance_status")
        == "earlier version found"
    )
    protected_context = protected_status and bool(
        verification["provenance"].get("previous_context_summary")
    )
    source_name = verification["source"].get("original_source_name")
    official_source = source_name if source_name in OFFICIAL_SOURCES.values() else None
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
    """Remove prior text-derived values while preserving stronger evidence."""
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
    official_source = OFFICIAL_SOURCES.get(platform)
    protected_search_source = field_has_evidence_type(
        extra,
        "verification.source.original_source_name",
        {"provenance_image_candidate", "provenance_search_candidate"},
    )
    if official_source:
        verification["source"]["original_source_name"] = official_source
        verification["source"]["source_type"] = "news outlet"
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


def build_text_grounding(record, extra):
    """Return the full prompt evidence and the stricter event-only evidence."""
    supplied_evidence = text_input(record, extra)
    grounding_text = json.dumps(supplied_evidence, ensure_ascii=False)
    current_video = supplied_evidence.get("current_video") or {}
    event_evidence = {
        "claim": supplied_evidence.get("claim"),
        "fact_check_article": supplied_evidence.get("fact_check_article"),
        "fact_check_evidence": supplied_evidence.get("fact_check_evidence"),
        "transcript": supplied_evidence.get("transcript"),
        "visual_analysis": supplied_evidence.get("visual_analysis"),
        "title": current_video.get("title"),
        "description": current_video.get("description"),
        "reverse_image_candidates": supplied_evidence.get("reverse_image_candidates"),
        "retrieved_event_candidates": [
            item
            for item in (
                (supplied_evidence.get("web_search_candidates") or {}).get("results")
                or []
            )
            if item.get("query_type") in {"transcript", "caption"}
        ],
    }
    event_grounding_text = json.dumps(event_evidence, ensure_ascii=False)
    return grounding_text, event_grounding_text


def generate_prompt_result(model, tokenizer, device, job, evidence):
    """Run one focused extraction prompt and preserve its parse error."""
    import torch

    messages = [
        {"role": "system", "content": COMMON_RULES},
        {
            "role": "user",
            "content": job["instructions"] + "\n\nEVIDENCE:\n" + evidence,
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=job["max_new_tokens"],
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )
    print("text", job["name"], "model output:", generated, flush=True)

    try:
        analysis = parse_json_output(generated)
        if isinstance(analysis, list) and len(analysis) == 1:
            analysis = analysis[0]
        if not isinstance(analysis, dict):
            raise ValueError("Text model output must be a JSON object")
        return {
            "status": "ok",
            "analysis": {field: analysis.get(field) for field in job["fields"]},
            "model": MODELS["text"],
        }
    except (ValueError, json.JSONDecodeError) as error:
        return {
            "status": "parse_error",
            "raw_output": generated,
            "error": str(error),
            "model": MODELS["text"],
        }


def generate_text_result(model, tokenizer, device, grounding_text, event_grounding_text):
    """Merge independent provenance/source, date/location, and motivation passes."""
    evidence = {"full": grounding_text, "event": event_grounding_text}
    analysis = {}
    parts = {}
    for job in TEXT_PROMPT_JOBS:
        job_evidence = evidence[job["evidence"]]
        if job["name"] == "motivation":
            job_evidence += "\n\nPROVISIONAL PROVENANCE/SOURCE:\n" + json.dumps(
                {
                    field: analysis[field]
                    for field in TEXT_PROMPT_JOBS[0]["fields"]
                    if field in analysis
                },
                ensure_ascii=False,
            )
        result = generate_prompt_result(
            model,
            tokenizer,
            device,
            job,
            job_evidence,
        )
        parts[job["name"]] = result
        if result["status"] == "ok":
            analysis.update(result["analysis"])

    success_count = sum(part["status"] == "ok" for part in parts.values())
    if success_count == len(parts):
        status = "ok"
    else:
        status = "partial" if analysis else "parse_error"
    return {
        "status": status,
        "analysis": analysis,
        "parts": parts,
        "model": MODELS["text"],
    }


def run(records, output, model_cache, offline=False):
    """Generate Qwen output, validate it, and update sidecars."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
        sidecar_file = sidecar_path(output, record["claim_id"])
        extra = read_json(sidecar_file)
        grounding_text, event_grounding_text = build_text_grounding(record, extra)
        result = generate_text_result(
            model,
            tokenizer,
            device,
            grounding_text,
            event_grounding_text,
        )

        reset_text_analysis(extra)
        if result["analysis"]:
            apply_text_analysis(extra, result["analysis"], grounding_text, event_grounding_text)
        extra.setdefault("automation", {})["text"] = result
        write_json(sidecar_file, extra)
        if number % 10 == 0:
            print("text", number, "/", len(records), flush=True)
    release_models(model, tokenizer)
