"""Extract and verify locations mentioned by independent evidence sources."""

import re

from pillars.location import location_candidate_is_plausible
from stages import MODELS
from utils.evidence import add_evidence, add_field_evidence, mark_automated, remove_evidence_type
from utils.files import read_json, trim, write_json
from utils.model_output import release_models
from utils.records import sidecar_path


LOCATION_FIELDS = (
    "claimed_location",
    "candidate_locations",
    "verified_location",
    "verified_coordinates",
    "location_mismatch_type",
)


def build_sources(record, extra):
    """Return category, text, URL, and any already-grounded visual places."""
    data = record["data"]
    motivation = extra["verification"]["motivation"]
    video = extra["normalized_video_information"]
    source_file = extra.get("source_file")
    sources = []

    def add(category, text, source, known=()):
        text = str(text or "").strip()
        if text or known:
            sources.append(
                {"category": category, "text": text, "source": str(source or ""), "known": known}
            )

    add("claim", data.get("claim"), source_file)
    add("platform", motivation.get("original_caption"), video.get("video_url"))
    add("platform", motivation.get("original_description"), video.get("video_url"))
    add("transcript", video.get("video_transcript"), source_file)
    add("fact_check", trim(data.get("content"), 12000), data.get("url"))
    for evidence in extra.get("evidence") or []:
        if evidence.get("type") == "fact_check_evidence":
            add("fact_check", evidence.get("observation"), evidence.get("source"))

    vision = (extra.get("automation") or {}).get("vision") or {}
    if vision.get("status") == "ok":
        visual_text = "\n".join(
            str(item)
            for field in ("ocr_text", "landmarks", "signs_and_logos")
            for item in vision.get(field) or []
        )
        add(
            "vision",
            visual_text,
            "cache/vision/" + record["claim_id"] + ".json",
            vision.get("candidate_locations") or [],
        )
    return sources


def clean_location(value):
    text = str(value or "").replace(" ##", "").replace("##", "")
    return re.sub(r"\s+", " ", text).strip(" \t\r\n.,;:!?()[]{}\"'")


def extract_candidates(sources, recognizer):
    """Keep one supporting excerpt per evidence category and place."""
    found = {}
    for source in sources:
        mentions = [(place, 1.0, source["text"]) for place in source["known"]]
        text = source["text"]
        for start in range(0, len(text), 750):
            chunk = text[start : start + 800]
            for entity in recognizer(chunk):
                entity_type = entity.get("entity_group") or entity.get("entity") or ""
                score = float(entity.get("score") or 0)
                if str(entity_type).endswith("LOC") and score >= 0.80:
                    mentions.append((entity.get("word"), score, chunk))

        for value, score, excerpt in mentions:
            name = clean_location(value)
            if not location_candidate_is_plausible(name):
                continue
            candidate = found.setdefault(
                name.casefold(), {"name": name, "sources": [], "evidence": []}
            )
            if source["category"] in candidate["sources"]:
                continue
            candidate["sources"].append(source["category"])
            candidate["evidence"].append(
                {
                    "category": source["category"],
                    "source": source["source"],
                    "text": trim(excerpt, 240),
                    "score": round(score, 3),
                }
            )

    candidates = list(found.values())
    return sorted(candidates, key=lambda item: (-len(item["sources"]), item["name"].lower()))


def choose_claimed_location(candidates):
    names = [candidate["name"] for candidate in candidates if "claim" in candidate["sources"]]
    return names[0] if len(names) == 1 else None


def choose_verified_location(candidates):
    """Require a unique place supported by at least two non-claim categories."""
    eligible = []
    for candidate in candidates:
        support = len(set(candidate["sources"]) - {"claim"})
        if support >= 2:
            eligible.append((support, candidate["name"]))
    if not eligible:
        return None
    best = max(support for support, _ in eligible)
    winners = [name for support, name in eligible if support == best]
    return winners[0] if len(winners) == 1 else None


def apply_candidates(extra, result):
    location = extra["verification"]["location"]
    links = extra.setdefault("field_evidence", {})
    for field in LOCATION_FIELDS:
        links.pop("verification.location." + field, None)
    remove_evidence_type(extra, "location_text_candidate")

    candidates = result.get("candidates") or []
    evidence_ids = {}
    for candidate in candidates:
        evidence_ids[candidate["name"]] = add_evidence(
            extra,
            "location_text_candidate",
            candidate["evidence"][0]["source"] if candidate["evidence"] else "",
            "Supported by: " + ", ".join(candidate["sources"]),
            evidence_categories=candidate["sources"],
            mentions=candidate["evidence"],
            model=result.get("model"),
        )

    claimed = choose_claimed_location(candidates)
    verified = choose_verified_location(candidates)
    location.update(
        {
            "claimed_location": claimed,
            "candidate_locations": [candidate["name"] for candidate in candidates],
            "verified_location": verified,
            "verified_coordinates": None,
            "location_mismatch_type": (
                "same" if claimed and verified and claimed.casefold() == verified.casefold() else None
            ),
        }
    )

    if evidence_ids:
        add_field_evidence(
            extra, "verification.location.candidate_locations", list(evidence_ids.values())
        )
    for field, value in (("claimed_location", claimed), ("verified_location", verified)):
        if value:
            add_field_evidence(
                extra, "verification.location." + field, [evidence_ids[value]]
            )
    if location["location_mismatch_type"] == "same":
        add_field_evidence(
            extra, "verification.location.location_mismatch_type", [evidence_ids[claimed]]
        )


def load_recognizer(model_cache, offline):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    options = {"cache_dir": str(model_cache), "local_files_only": offline}
    tokenizer = AutoTokenizer.from_pretrained(MODELS["location"], **options)
    model = AutoModelForTokenClassification.from_pretrained(MODELS["location"], **options)
    # recognizer = pipeline(
    #     "token-classification",
    #     model=model,
    #     tokenizer=tokenizer,
    #     aggregation_strategy="simple",
    #     device=0 if torch.cuda.is_available() else -1,
    # )
    recognizer = pipeline("ner", model=model, tokenizer=tokenizer)

    return recognizer, model, tokenizer


def run(records, output, model_cache, force=False, offline=False):
    cache = output / "cache" / "location"
    cache.mkdir(parents=True, exist_ok=True)
    recognizer = model = tokenizer = None

    for number, record in enumerate(records, 1):
        cached = cache / (record["claim_id"] + ".json")
        sidecar_file = sidecar_path(output, record["claim_id"])
        extra = read_json(sidecar_file)
        if cached.exists() and not force:
            result = read_json(cached)
            for candidate in result.get("candidates") or []:
                candidate.setdefault("evidence", candidate.get("mentions") or [])
        else:
            if recognizer is None:
                recognizer, model, tokenizer = load_recognizer(model_cache, offline)
            result = {
                "status": "ok",
                "model": MODELS["location"],
                "candidates": extract_candidates(build_sources(record, extra), recognizer),
            }
            write_json(cached, result)

        apply_candidates(extra, result)
        mark_automated(extra, "location", result)
        write_json(sidecar_file, extra)
        if number % 10 == 0:
            print("location", number, "/", len(records), flush=True)

    if model is not None:
        release_models(recognizer, model, tokenizer)
