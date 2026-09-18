"""Create the small JSON file intended for dataset readers."""

from utils.files import write_json
from utils.schema import project_verification


def readable_result(extra):
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
        "verification": project_verification(extra["verification"]),
    }


def write_readable_result(extra, output):
    path = output / "results" / (extra["claim_id"] + ".json")
    write_json(path, readable_result(extra))
