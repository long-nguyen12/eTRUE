"""Create the small JSON file intended for dataset readers."""

from utils.files import write_json


def readable_result(extra):
    verification = extra["verification"]
    provenance = verification["provenance"]
    source = verification["source"]
    date = verification["date"]
    location = verification["location"]
    motivation = verification["motivation"]
    video = extra["normalized_video_information"]

    claim_components = extra.get("claim_components") or []
    claim = claim_components[0].get("text") if claim_components else None

    return {
        "claim_id": extra["claim_id"],
        "claim": claim,
        "video_information": {
            "platform": video.get("platform"),
            "video_url": video.get("video_url"),
            "video_date": video.get("video_date"),
            "video_transcript": video.get("video_transcript"),
        },
        "verification": {
            "provenance": {
                "provenance_status": provenance.get("provenance_status"),
                "earliest_known_url": provenance.get("earliest_known_url"),
                "earliest_known_date": provenance.get("earliest_known_date"),
                "near_duplicate_matches": provenance.get("near_duplicate_matches") or [],
                "previous_context_summary": provenance.get("previous_context_summary"),
                "provenance_mismatch": provenance.get("provenance_mismatch"),
            },
            "source": {
                "uploader_name": source.get("uploader_name"),
                "uploader_profile": source.get("uploader_profile"),
                "original_source_name": source.get("original_source_name"),
                "source_type": source.get("source_type"),
                "source_is_uploader": source.get("source_is_uploader"),
                "source_mismatch": source.get("source_mismatch"),
            },
            "date": {
                "claimed_date": date.get("claimed_date"),
                "video_upload_date": date.get("video_upload_date"),
                "earliest_online_date": date.get("earliest_online_date"),
                "estimated_date": date.get("estimated_date"),
                "capture_date_granularity": date.get("capture_date_granularity"),
                "date_mismatch_type": date.get("date_mismatch_type"),
            },
            "location": {
                "claimed_location": location.get("claimed_location"),
                "candidate_locations": location.get("candidate_locations") or [],
                "visual_location_clues": location.get("visual_location_clues") or [],
                "verified_location": location.get("verified_location"),
                "verified_coordinates": location.get("verified_coordinates"),
                "location_mismatch_type": location.get("location_mismatch_type"),
            },
            "motivation": {
                "claimed_framing": motivation.get("claimed_framing"),
                "original_caption": motivation.get("original_caption"),
                "original_description": motivation.get("original_description"),
                "original_context_category": motivation.get("original_context_category"),
                "motivation_mismatch_type": motivation.get("motivation_mismatch_type"),
            },
        }
    }


def write_readable_result(extra, output):
    path = output / "results" / (extra["claim_id"] + ".json")
    write_json(path, readable_result(extra))
