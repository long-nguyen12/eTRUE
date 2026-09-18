"""Extract location candidates from text and classify event-place assertions."""

import re

from pillars.location import location_candidate_is_plausible
from utils.files import trim


EVENT_TERMS = re.compile(
    r"\b(?:video|footage|clip|photo(?:graph)?|image|scene|incident|event|rally|protest|"
    r"attack|explosion|fire|crash|demonstration|meeting|speech|ceremony|storm|flood|"
    r"earthquake|filmed|recorded|shot|captured|taken|happened|occurred|depicted|shows?)\b",
    re.IGNORECASE,
)


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
        and EVENT_TERMS.search(
            excerpt[max(0, spatial_match.start() - 100) : spatial_match.start()]
        )
    )
    place_led_event = re.search(
        r"(?:^|[.!?]\s+)(?:in|at|near)\s+(?:the\s+)?"
        + place
        + r"\s*,?\s+(?:(?:an?|the)\s+)?(?:attack|explosion|fire|crash|rally|"
        r"protest|demonstration|meeting|speech|ceremony|storm|flood|earthquake)\b",
        excerpt,
        flags=re.IGNORECASE,
    )
    return bool(direct or deictic or event_before_place or place_led_event)


def _store_mention(candidate, mention):
    """Keep the strongest excerpt for each genuinely independent support group."""
    group = mention["support_group"]
    existing = next(
        (
            item
            for item in candidate["evidence"]
            if item.get("support_group") == group
        ),
        None,
    )
    if existing is None:
        candidate["evidence"].append(mention)
        return
    if mention["event_location"] and not existing.get("event_location"):
        existing.update(mention)
    elif (
        mention["event_location"] == existing.get("event_location")
        and mention["score"] > existing["score"]
    ):
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
                        and (
                            entity_end == len(chunk)
                            or not chunk[entity_end].isalnum()
                        )
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
        event_evidence = [
            item for item in candidate["evidence"] if item["event_location"]
        ]
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
