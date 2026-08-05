"""Geocode verified locations and compare them with claimed locations."""

import hashlib
import re
import time
from urllib.parse import quote

from utils.evidence import add_evidence, add_field_evidence, mark_automated, remove_evidence_type
from utils.files import now, read_json, write_json
from utils.records import sidecar_path


def normalized(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def address_value(row, keys):
    address = (row or {}).get("address") or {}
    return next((normalized(address[key]) for key in keys if address.get(key)), None)


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


def geocode(query, cache, session, force):
    key = hashlib.sha256(query.casefold().encode("utf-8")).hexdigest()[:16]
    cached = cache / (key + ".json")
    if cached.exists() and not force:
        saved = read_json(cached)
        row = saved.get("result")
        if not row or row.get("address"):
            return saved

    try:
        response = session.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "jsonv2", "limit": 1, "addressdetails": 1},
            timeout=20,
        )
        response.raise_for_status()
        rows = response.json()
        result = {"query": query, "retrieved_at": now(), "result": rows[0] if rows else None}
    except Exception as error:
        result = {"query": query, "retrieved_at": now(), "result": None, "error": str(error)}
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
        location["verified_coordinates"] = None
        location["location_mismatch_type"] = None

        remove_evidence_type(extra, "geocoder_result")
        links = extra.setdefault("field_evidence", {})
        links.pop("verification.location.verified_coordinates", None)
        links.pop("verification.location.location_mismatch_type", None)

        if not verified:
            mark_automated(extra, "geocode", {"status": "skipped", "reason": "no verified location"})
            write_json(sidecar_file, extra)
            continue

        results = {"verified": geocode(verified, cache, session, force)}
        if claimed:
            results["claimed"] = (
                results["verified"]
                if normalized(claimed) == normalized(verified)
                else geocode(claimed, cache, session, force)
            )

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
            )

        verified_row = results["verified"].get("result")
        claimed_row = (results.get("claimed") or {}).get("result")
        if verified_row:
            location["verified_coordinates"] = {
                "latitude": float(verified_row["lat"]),
                "longitude": float(verified_row["lon"]),
            }
        location["location_mismatch_type"] = classify_location_mismatch(
            claimed, verified, claimed_row, verified_row
        )
        if location["location_mismatch_type"]:
            add_field_evidence(
                extra,
                "verification.location.location_mismatch_type",
                list(dict.fromkeys(evidence_ids.values())),
            )

        mark_automated(extra, "geocode", {"status": "ok", **results})
        write_json(sidecar_file, extra)
        if number % 10 == 0:
            print("geocode", number, "/", len(records), flush=True)
