"""Field definitions used by text extraction and validation."""

from pillars.date import DATE_GRANULARITIES, DATE_MISMATCHES
from pillars.location import LOCATION_MISMATCHES
from pillars.motivation import CONTEXT_CATEGORIES, MOTIVATION_MISMATCHES
from pillars.provenance import PROVENANCE_STATUSES
from pillars.source import SOURCE_TYPES


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
