"""Canonical public verification schema."""

from copy import deepcopy


VERIFICATION_DEFAULTS = {
    "provenance": {
        "provenance_status": None,
        "earliest_known_url": None,
        "earliest_known_date": None,
        "near_duplicate_matches": [],
        "previous_context_summary": None,
        "provenance_mismatch": None,
    },
    "source": {
        "uploader_name": None,
        "uploader_profile": None,
        "original_source_name": None,
        "source_type": None,
        "source_is_uploader": None,
        "source_mismatch": None,
    },
    "date": {
        "claimed_date": None,
        "video_upload_date": None,
        "earliest_online_date": None,
        "estimated_date": None,
        "capture_date_granularity": None,
        "date_mismatch_type": None,
    },
    "location": {
        "claimed_location": None,
        "candidate_locations": [],
        "visual_location_clues": [],
        "verified_location": None,
        "verified_coordinates": None,
        "location_mismatch_type": None,
    },
    "motivation": {
        "claimed_framing": None,
        "original_caption": None,
        "original_description": None,
        "original_context_category": None,
        "motivation_mismatch_type": None,
    },
}


def empty_verification():
    return deepcopy(VERIFICATION_DEFAULTS)


def project_verification(verification):
    """Return only fields intended for the public result file."""
    verification = verification or {}
    projected = {}
    for pillar, defaults in VERIFICATION_DEFAULTS.items():
        values = verification.get(pillar) or {}
        projected[pillar] = {
            field: deepcopy(values.get(field) or [])
            if isinstance(default, list)
            else deepcopy(values.get(field))
            for field, default in defaults.items()
        }
    return projected
