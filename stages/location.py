"""Extract and verify locations mentioned by independent evidence sources."""

import re
from urllib.parse import urlsplit

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

LOCATION_CACHE_VERSION = 3

EVENT_TERMS = re.compile(
    r"\b(?:video|footage|clip|photo(?:graph)?|image|scene|incident|event|rally|protest|"
    r"attack|explosion|fire|crash|demonstration|meeting|speech|ceremony|storm|flood|"
    r"earthquake|filmed|recorded|shot|captured|taken|happened|occurred|depicted|shows?)\b",
    re.IGNORECASE,
)


def build_sources(record, extra):
    """Return category, text, URL, and any already-grounded visual places."""
    data = record["data"]
    motivation = extra["verification"]["motivation"]
    video = extra["normalized_video_information"]
    source_file = extra.get("source_file")
    sources = []

    def add(category, text, source, known=(), support_group=None):
        text = str(text or "").strip()
        if text or known:
            sources.append(
                {
                    "category": category,
                    "text": text,
                    "source": str(source or ""),
                    "known": known,
                    "support_group": support_group or category,
                }
            )

    add("claim", data.get("claim"), source_file, support_group="claim")
    add(
        "platform",
        motivation.get("original_caption"),
        video.get("video_url"),
        support_group="current_video",
    )
    add(
        "platform",
        motivation.get("original_description"),
        video.get("video_url"),
        support_group="current_video",
    )
    add(
        "transcript",
        video.get("video_transcript"),
        source_file,
        support_group="current_video",
    )
    add(
        "fact_check",
        trim(data.get("content"), 12000),
        data.get("url"),
        support_group="fact_check",
    )
    for evidence in extra.get("evidence") or []:
        if evidence.get("type") == "fact_check_evidence":
            add(
                "fact_check",
                evidence.get("observation"),
                evidence.get("source"),
                support_group="fact_check",
            )

    search = (extra.get("automation") or {}).get("search") or {}
    retrieved_pages = list(search.get("results") or [])
    retrieved_pages += list((search.get("image_search") or {}).get("pages") or [])
    known_urls = [data.get("url"), video.get("video_url")]
    known_urls += [
        evidence.get("source")
        for evidence in extra.get("evidence") or []
        if evidence.get("type") == "fact_check_evidence"
    ]
    seen_pages = {
        str(url).rstrip("/").casefold() for url in known_urls if url
    }
    for page in retrieved_pages:
        page_url = page.get("canonical_url") or page.get("final_url") or page.get("url")
        page_key = str(page_url or "").rstrip("/").casefold()
        if not page_url or page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        page_text = " ".join(
            str(page.get(field) or "")
            for field in ("title", "description", "context_excerpt")
        )
        try:
            hostname = urlsplit(page_url).hostname or page_url
        except ValueError:
            hostname = page_url
        add(
            "retrieved_page",
            trim(page_text, 2000),
            page_url,
            support_group="retrieved:" + hostname.casefold(),
        )

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
            support_group="vision",
        )
    return sources


def clean_location(value):
    text = str(value or "").replace(" ##", "").replace("##", "")
    return re.sub(r"\s+", " ", text).strip(" \t\r\n.,;:!?()[]{}\"'")


def mention_excerpt(text, start, end, radius=180):
    """Return a compact sentence-like window around a recognized mention."""
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    for separator in ".!?\n":
        boundary = text.rfind(separator, left, start)
        if boundary >= 0:
            left = max(left, boundary + 1)
    following = [text.find(separator, end, right) for separator in ".!?\n"]
    following = [position for position in following if position >= 0]
    if following:
        right = min(following) + 1
    return trim(text[left:right].strip(), 360)


def event_location_is_asserted(text, name, start=None, end=None):
    """Conservatively distinguish an event location from a topical place mention."""
    if start is None or end is None:
        match = re.search(
            r"(?<!\w)" + re.escape(name) + r"(?!\w)",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return False
        start, end = match.span()

    excerpt = mention_excerpt(text, start, end)
    place = re.escape(name)
    spatial = re.search(
        r"\b(?:in|at|near|outside|inside|within|around)\s+(?:the\s+)?"
        + place
        + r"(?!\w)",
        excerpt,
        flags=re.IGNORECASE,
    )
    admin_spatial = re.search(
        r"\b(?:in|at|near|outside|inside|within|around)\s+(?:the\s+)?"
        r"[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,3}\s*,\s*"
        + place
        + r"(?!\w)",
        excerpt,
    )
    direct = re.search(
        r"\b(?:filmed|recorded|shot|captured|taken|happened|occurred|located|live|"
        r"reporting)\b.{0,50}\b(?:in|at|near|outside|inside|within|around|from)\s+"
        r"(?:the\s+)?"
        + place
        + r"(?!\w)",
        excerpt,
        flags=re.IGNORECASE,
    )
    deictic = re.search(
        r"\b(?:here|live|reporting)\s+(?:in|at|from)\s+(?:the\s+)?"
        + place
        + r"(?!\w)",
        excerpt,
        flags=re.IGNORECASE,
    )
    spatial_match = spatial or admin_spatial
    event_before_place = bool(
        spatial_match
        and EVENT_TERMS.search(excerpt[max(0, spatial_match.start() - 100) : spatial_match.start()])
    )
    place_led_event = re.search(
        r"(?:^|[.!?]\s+)(?:in|at|near)\s+(?:the\s+)?"
        + place
        + r"\s*,?\s+(?:(?:an?|the)\s+)?(?:attack|explosion|fire|crash|rally|"
        r"protest|demonstration|meeting|speech|ceremony|storm|flood|earthquake)\b",
        excerpt,
        flags=re.IGNORECASE,
    )
    return bool(
        direct
        or deictic
        or event_before_place
        or place_led_event
    )


def _store_mention(candidate, mention):
    """Keep the strongest excerpt for each genuinely independent support group."""
    group = mention["support_group"]
    existing = next(
        (item for item in candidate["evidence"] if item.get("support_group") == group),
        None,
    )
    if existing is None:
        candidate["evidence"].append(mention)
        return
    if mention["event_location"] and not existing.get("event_location"):
        existing.update(mention)
    elif mention["event_location"] == existing.get("event_location") and mention["score"] > existing["score"]:
        existing.update(mention)


def extract_candidates(sources, recognizer):
    """Keep one supporting excerpt per evidence category and place."""
    found = {}
    for source in sources:
        mentions = [(place, 1.0, source["text"], True) for place in source["known"]]
        text = source["text"]
        for start in range(0, len(text), 750):
            chunk = text[start : start + 800]
            for entity in recognizer(chunk):
                entity_type = entity.get("entity_group") or entity.get("entity") or ""
                score = float(entity.get("score") or 0)
                if str(entity_type).endswith("LOC") and score >= 0.80:
                    entity_start = entity.get("start")
                    entity_end = entity.get("end")
                    value = entity.get("word")
                    if (
                        isinstance(entity_start, int)
                        and isinstance(entity_end, int)
                        and 0 <= entity_start < entity_end <= len(chunk)
                    ):
                        value = chunk[entity_start:entity_end]
                    name = clean_location(value)
                    if not name:
                        continue
                    if not (
                        isinstance(entity_start, int)
                        and isinstance(entity_end, int)
                        and (entity_start == 0 or not chunk[entity_start - 1].isalnum())
                        and (entity_end == len(chunk) or not chunk[entity_end].isalnum())
                    ):
                        exact = re.search(
                            r"(?<!\w)" + re.escape(name) + r"(?!\w)",
                            chunk,
                            flags=re.IGNORECASE,
                        )
                        if not exact:
                            continue
                        entity_start, entity_end = exact.span()
                    excerpt = mention_excerpt(chunk, entity_start, entity_end)
                    mentions.append(
                        (
                            name,
                            score,
                            excerpt,
                            event_location_is_asserted(
                                chunk,
                                name,
                                entity_start,
                                entity_end,
                            ),
                        )
                    )

        for value, score, excerpt, event_location in mentions:
            name = clean_location(value)
            if not location_candidate_is_plausible(name):
                continue
            candidate = found.setdefault(
                name.casefold(), {"name": name, "sources": [], "evidence": []}
            )
            _store_mention(
                candidate,
                {
                    "category": source["category"],
                    "support_group": source["support_group"],
                    "source": source["source"],
                    "text": trim(excerpt, 360),
                    "score": round(score, 3),
                    "event_location": bool(event_location),
                },
            )

    candidates = list(found.values())
    for candidate in candidates:
        candidate["sources"] = list(
            dict.fromkeys(item["category"] for item in candidate["evidence"])
        )
        candidate["support_groups"] = list(
            dict.fromkeys(item["support_group"] for item in candidate["evidence"])
        )
        event_evidence = [item for item in candidate["evidence"] if item["event_location"]]
        candidate["event_sources"] = list(
            dict.fromkeys(item["category"] for item in event_evidence)
        )
        candidate["event_support_groups"] = list(
            dict.fromkeys(item["support_group"] for item in event_evidence)
        )
    return sorted(
        candidates,
        key=lambda item: (
            -len(item["event_support_groups"]),
            -len(item["support_groups"]),
            item["name"].lower(),
        ),
    )


def choose_claimed_location(candidates):
    names = [
        candidate["name"]
        for candidate in candidates
        if "claim" in candidate.get("event_sources", candidate.get("sources", []))
    ]
    return names[0] if len(names) == 1 else None


def choose_verified_location(candidates):
    """Require a unique place supported by at least two non-claim categories."""
    eligible = []
    for candidate in candidates:
        groups = candidate.get("event_support_groups")
        if groups is None:
            groups = candidate.get("event_sources", candidate.get("sources", []))
        support = len(set(groups) - {"claim"})
        if support >= 2:
            eligible.append((support, candidate["name"]))
    if not eligible:
        return None
    best = max(support for support, _ in eligible)
    winners = [name for support, name in eligible if support == best]
    return winners[0] if len(winners) == 1 else None


def _stronger_existing_value(extra, field, value, stage_evidence_ids):
    """Return whether a current value has evidence not produced by this NER stage."""
    if not value or not location_candidate_is_plausible(value):
        return False
    path = "verification.location." + field
    linked = set((extra.get("field_evidence") or {}).get(path) or [])
    if linked - stage_evidence_ids:
        return True
    if (extra.get("review") or {}).get("status") not in {None, "not_started", "automated"}:
        return True
    return False


def _candidate_evidence_id(evidence_ids, value):
    key = clean_location(value).casefold()
    return next(
        (evidence_id for name, evidence_id in evidence_ids.items() if name.casefold() == key),
        None,
    )


def apply_candidates(extra, result):
    location = extra["verification"]["location"]
    old_result = (extra.get("automation") or {}).get("location") or {}
    stage_evidence_ids = {
        evidence["id"]
        for evidence in extra.get("evidence") or []
        if evidence.get("type") == "location_text_candidate"
    }
    prior_claimed = location.get("claimed_location")
    prior_verified = location.get("verified_location")
    prior_coordinates = location.get("verified_coordinates")
    prior_mismatch = location.get("location_mismatch_type")
    preserve_claimed = _stronger_existing_value(
        extra, "claimed_location", prior_claimed, stage_evidence_ids
    )
    preserve_verified = _stronger_existing_value(
        extra, "verified_location", prior_verified, stage_evidence_ids
    )

    old_generated = {
        clean_location(candidate.get("name")).casefold()
        for candidate in old_result.get("candidates") or []
        if candidate.get("name")
    }
    upstream_candidates = []
    for value in location.get("candidate_locations") or []:
        if isinstance(value, dict):
            value = value.get("name") or value.get("location") or value.get("value")
        name = clean_location(value)
        if (
            location_candidate_is_plausible(name)
            and name.casefold() not in old_generated
            and name.casefold() not in {item.casefold() for item in upstream_candidates}
        ):
            upstream_candidates.append(name)

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
            event_evidence_categories=candidate.get("event_sources") or [],
            mentions=candidate["evidence"],
            model=result.get("model"),
        )

    extracted_names = [candidate["name"] for candidate in candidates]
    merged_candidates = list(extracted_names)
    for name in upstream_candidates:
        if name.casefold() not in {item.casefold() for item in merged_candidates}:
            merged_candidates.append(name)

    selected_claimed = choose_claimed_location(candidates)
    selected_verified = choose_verified_location(candidates)
    claimed = prior_claimed if preserve_claimed else selected_claimed
    verified = prior_verified if preserve_verified else selected_verified
    for name in (claimed, verified):
        if name and name.casefold() not in {item.casefold() for item in merged_candidates}:
            merged_candidates.append(name)

    inputs_unchanged = (
        clean_location(prior_claimed).casefold() == clean_location(claimed).casefold()
        and clean_location(prior_verified).casefold() == clean_location(verified).casefold()
    )
    mismatch = prior_mismatch if inputs_unchanged else None
    if (
        claimed
        and verified
        and clean_location(claimed).casefold() == clean_location(verified).casefold()
    ):
        mismatch = "same"
    location.update(
        {
            "claimed_location": claimed,
            "candidate_locations": merged_candidates,
            "verified_location": verified,
            "verified_coordinates": prior_coordinates if inputs_unchanged else None,
            "location_mismatch_type": mismatch,
        }
    )

    if evidence_ids:
        add_field_evidence(
            extra, "verification.location.candidate_locations", list(evidence_ids.values())
        )
    for field, value, selected in (
        ("claimed_location", claimed, selected_claimed if not preserve_claimed else None),
        ("verified_location", verified, selected_verified if not preserve_verified else None),
    ):
        evidence_id = _candidate_evidence_id(evidence_ids, selected)
        if value and evidence_id:
            add_field_evidence(
                extra, "verification.location." + field, [evidence_id]
            )
    if mismatch == "same" and not inputs_unchanged:
        matching_ids = [
            evidence_id
            for value in (selected_claimed, selected_verified)
            if (evidence_id := _candidate_evidence_id(evidence_ids, value))
        ]
        if matching_ids:
            add_field_evidence(
                extra,
                "verification.location.location_mismatch_type",
                list(dict.fromkeys(matching_ids)),
            )


def load_recognizer(model_cache, offline):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    options = {"cache_dir": str(model_cache), "local_files_only": offline}
    tokenizer = AutoTokenizer.from_pretrained(MODELS["location"], **options)
    model = AutoModelForTokenClassification.from_pretrained(MODELS["location"], **options)
    recognizer = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1,
    )

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
        else:
            result = None
        if not result or result.get("cache_version") != LOCATION_CACHE_VERSION:
            if recognizer is None:
                recognizer, model, tokenizer = load_recognizer(model_cache, offline)
            result = {
                "status": "ok",
                "cache_version": LOCATION_CACHE_VERSION,
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
