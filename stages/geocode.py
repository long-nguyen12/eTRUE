"""Geocode verified locations and compare them with claimed locations."""

import hashlib
import re
import time
import unicodedata
from urllib.parse import quote

from utils.evidence import add_evidence, add_field_evidence, mark_automated, remove_evidence_type
from utils.files import now, read_json, write_json
from utils.records import sidecar_path


GEOCODE_CACHE_VERSION = 2


def normalized(value):
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[\W_]+", " ", text, flags=re.UNICODE).strip()


def address_value(row, keys):
    address = (row or {}).get("address") or {}
    return next((normalized(address[key]) for key in keys if address.get(key)), None)


ADMIN_ADDRESS_KEYS = (
    "amenity",
    "building",
    "road",
    "neighbourhood",
    "suburb",
    "city_district",
    "city",
    "town",
    "village",
    "municipality",
    "county",
    "state_district",
    "state",
    "region",
    "province",
    "country",
    "country_code",
)


def _row_names(row):
    address = (row or {}).get("address") or {}
    values = [row.get("name"), *((address.get(key) for key in ADMIN_ADDRESS_KEYS))]
    display_parts = str(row.get("display_name") or "").split(",")
    values.extend(display_parts)
    return {normalized(value) for value in values if normalized(value)}


def _candidate_score(query, row, context):
    """Rank a Nominatim row using exact query and administrative-context matches."""
    query_name = normalized(query)
    names = _row_names(row)
    display_name = normalized(row.get("display_name"))
    score = float(row.get("importance") or 0)

    if query_name in names:
        score += 8.0
    elif query_name and query_name in display_name:
        score += 3.0

    query_parts = [normalized(part) for part in str(query).split(",") if normalized(part)]
    score += 2.0 * sum(part in names for part in query_parts)

    context_names = {
        normalized(value)
        for value in context or []
        if normalized(value) and normalized(value) != query_name
    }
    score += 3.0 * sum(value in names for value in context_names)

    place_type = normalized(row.get("addresstype") or row.get("type"))
    if place_type in {
        "country",
        "state",
        "province",
        "region",
        "county",
        "city",
        "town",
        "village",
        "municipality",
        "suburb",
        "neighbourhood",
    }:
        score += 0.5
    return score


def select_geocode_candidate(query, rows, context=()):
    """Select the best result without assuming Nominatim's first row is correct."""
    candidates = [row for row in rows or [] if isinstance(row, dict)]
    if not candidates:
        return None, {"strategy": "no_candidates", "candidate_count": 0}
    scored = [
        (_candidate_score(query, row, context), -index, index, row)
        for index, row in enumerate(candidates)
    ]
    score, _, index, selected = max(scored, key=lambda item: (item[0], item[1]))
    return selected, {
        "strategy": "query_and_admin_context",
        "candidate_count": len(candidates),
        "selected_index": index,
        "score": round(score, 4),
    }


def classify_location_mismatch(claimed, verified, claimed_row=None, verified_row=None):
    """Compare names, OSM identity, country, region, then city."""
    if not claimed or not verified:
        return None
    same_osm_place = (
        claimed_row
        and verified_row
        and claimed_row.get("osm_type") == verified_row.get("osm_type")
        and claimed_row.get("osm_id") == verified_row.get("osm_id")
    )
    if normalized(claimed) == normalized(verified) or same_osm_place:
        return "same"
    if not claimed_row or not verified_row:
        return None

    levels = (
        ("different country", ("country_code", "country")),
        ("different region", ("state", "region", "province", "state_district")),
        ("different city", ("city", "town", "village", "municipality", "hamlet")),
    )
    for mismatch, keys in levels:
        claimed_value = address_value(claimed_row, keys)
        verified_value = address_value(verified_row, keys)
        if claimed_value and verified_value and claimed_value != verified_value:
            return mismatch
    return "unknown"


def geocode(query, cache, session, force, context=()):
    key = hashlib.sha256(query.casefold().encode("utf-8")).hexdigest()[:16]
    cached = cache / (key + ".json")
    if cached.exists() and not force:
        saved = read_json(cached)
        if saved.get("cache_version") == GEOCODE_CACHE_VERSION:
            rows = saved.get("candidates")
            if isinstance(rows, list):
                row, selection = select_geocode_candidate(query, rows, context)
                return {**saved, "result": row, "selection": selection}
            return saved

    try:
        response = session.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "jsonv2", "limit": 5, "addressdetails": 1},
            timeout=20,
        )
        response.raise_for_status()
        rows = response.json()
        row, selection = select_geocode_candidate(query, rows, context)
        result = {
            "cache_version": GEOCODE_CACHE_VERSION,
            "query": query,
            "retrieved_at": now(),
            "candidates": rows,
            "result": row,
            "selection": selection,
        }
    except Exception as error:
        result = {
            "cache_version": GEOCODE_CACHE_VERSION,
            "query": query,
            "retrieved_at": now(),
            "result": None,
            "error": str(error),
        }
    write_json(cached, result)
    if not result.get("error"):
        time.sleep(1.1)
    return result


def run(records, output, force=False):
    import requests

    cache = output / "cache" / "geocode"
    cache.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "eTRUE-research-dataset/1.0"

    for number, record in enumerate(records, 1):
        sidecar_file = sidecar_path(output, record["claim_id"])
        extra = read_json(sidecar_file)
        location = extra["verification"]["location"]
        claimed = location.get("claimed_location")
        verified = location.get("verified_location")

        if not verified:
            location["verified_coordinates"] = None
            location["location_mismatch_type"] = None
            remove_evidence_type(extra, "geocoder_result")
            mark_automated(
                extra,
                "geocode",
                {"status": "skipped", "reason": "no verified location"},
            )
            write_json(sidecar_file, extra)
            continue

        location_analysis = (extra.get("automation") or {}).get("location") or {}
        contextual_candidates = [
            candidate.get("name")
            for candidate in location_analysis.get("candidates") or []
            if candidate.get("name")
            and candidate.get("event_support_groups", candidate.get("event_sources"))
        ]
        if not contextual_candidates:
            contextual_candidates = [
                value
                for value in location.get("candidate_locations") or []
                if isinstance(value, str)
            ]
        verified_context = [
            value
            for value in contextual_candidates
            if not claimed
            or normalized(claimed) == normalized(verified)
            or normalized(value) != normalized(claimed)
        ]
        results = {
            "verified": geocode(verified, cache, session, force, verified_context)
        }
        if claimed:
            results["claimed"] = (
                results["verified"]
                if normalized(claimed) == normalized(verified)
                else geocode(
                    claimed,
                    cache,
                    session,
                    force,
                    [
                        value
                        for value in contextual_candidates
                        if normalized(value) != normalized(verified)
                    ],
                )
            )

        verified_row = results["verified"].get("result")
        if not verified_row:
            mark_automated(
                extra,
                "geocode",
                {
                    "status": "error",
                    "reason": "no geocoder result for verified location",
                    **results,
                },
            )
            write_json(sidecar_file, extra)
            continue

        remove_evidence_type(extra, "geocoder_result")

        evidence_ids = {}
        queries = {"verified": verified, "claimed": claimed}
        for role, result in results.items():
            row = result.get("result")
            if not row:
                continue
            fields = (
                ("verification.location.verified_coordinates",) if role == "verified" else ()
            )
            evidence_ids[role] = add_evidence(
                extra,
                "geocoder_result",
                "https://nominatim.openstreetmap.org/ui/search.html?q=" + quote(queries[role]),
                row.get("display_name"),
                fields=fields,
                retrieved_at=result.get("retrieved_at"),
                role=role,
                address=row.get("address"),
                selection=result.get("selection"),
            )

        claimed_row = (results.get("claimed") or {}).get("result")
        if verified_row.get("lat") is not None and verified_row.get("lon") is not None:
            location["verified_coordinates"] = {
                "latitude": float(verified_row["lat"]),
                "longitude": float(verified_row["lon"]),
            }
        else:
            location["verified_coordinates"] = None
        mismatch = classify_location_mismatch(
            claimed, verified, claimed_row, verified_row
        )
        location["location_mismatch_type"] = mismatch
        if mismatch:
            add_field_evidence(
                extra,
                "verification.location.location_mismatch_type",
                list(dict.fromkeys(evidence_ids.values())),
            )

        mark_automated(extra, "geocode", {"status": "ok", **results})
        write_json(sidecar_file, extra)
        if number % 10 == 0:
            print("geocode", number, "/", len(records), flush=True)
