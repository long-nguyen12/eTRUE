"""Import claims, metadata, articles, and existing TRUE evidence."""

from build_etrue import normalize_date
from pillars.source import OFFICIAL_SOURCES
from utils.evidence import add_evidence, add_field_evidence, mark_automated, remove_evidence_type
from utils.files import read_json, trim, write_json
from utils.records import sidecar_path


def import_existing_evidence(extra, data):
    evidences = data.get("evidences") or {}
    count = int(evidences.get("num_of_evidence") or 0)
    for number in range(1, count + 1):
        item = evidences.get("evidence" + str(number))
        if not isinstance(item, list) or not item:
            continue
        text = trim(item[0], 2000)
        references = item[1] if len(item) > 1 and isinstance(item[1], list) else []
        add_evidence(
            extra,
            "fact_check_evidence",
            references[0] if references else data.get("url"),
            text,
            references=references,
        )


def run(records, source, output, force=False):
    """Copy claims, metadata, articles, evidence, and known publishers."""
    for number, record in enumerate(records, 1):
        path = sidecar_path(output, record["claim_id"])
        extra = read_json(path)
        data = record["data"]
        video = data.get("video_information") or {}
        verification = extra["verification"]

        claim_evidence = add_evidence(
            extra,
            "claim_text",
            record["path"].relative_to(source).as_posix(),
            data.get("claim"),
            fields=("verification.motivation.claimed_framing",),
        )
        metadata_evidence = add_evidence(
            extra,
            "dataset_video_metadata",
            record["path"].relative_to(source).as_posix(),
            "Video URL, platform, upload date, title, and description imported from TRUE.",
            fields=(
                "verification.date.video_upload_date",
                "verification.provenance.earliest_known_url",
                "verification.provenance.earliest_known_date",
                "verification.motivation.original_caption",
                "verification.motivation.original_description",
            ),
        )
        article_evidence = add_evidence(
            extra,
            "fact_check_article",
            data.get("url"),
            trim(data.get("content"), 2000),
        )
        import_existing_evidence(extra, data)

        transcript = video.get("video_transcript")
        has_transcript = bool(str(transcript or "").strip())
        normalized = extra["normalized_video_information"]
        normalized["video_transcript"] = transcript if has_transcript else None
        normalized["transcript_status"] = "available" if has_transcript else "missing"
        normalized.pop("generated_transcript", None)
        extra.get("automation", {}).pop("asr", None)
        remove_evidence_type(extra, "asr_transcript")
        if has_transcript:
            add_evidence(
                extra,
                "dataset_transcript",
                record["path"].relative_to(source).as_posix(),
                trim(transcript, 2000),
            )

        if force or verification["motivation"]["claimed_framing"] is None:
            verification["motivation"]["claimed_framing"] = data.get("claim")
        upload_date = normalize_date(video.get("video_date"))
        if force or verification["date"]["video_upload_date"] is None:
            verification["date"]["video_upload_date"] = upload_date
        if force or verification["provenance"]["earliest_known_url"] is None:
            verification["provenance"]["earliest_known_url"] = video.get("video_url")
            verification["provenance"]["earliest_known_date"] = upload_date
            verification["provenance"]["provenance_status"] = "unknown"
            add_field_evidence(
                extra,
                "verification.provenance.provenance_status",
                [metadata_evidence],
            )
        if force or verification["motivation"]["original_caption"] is None:
            verification["motivation"]["original_caption"] = video.get("video_headline")
            verification["motivation"]["original_description"] = video.get("video_description")

        platform = extra["normalized_video_information"].get("platform")
        official_source = OFFICIAL_SOURCES.get(platform)
        if official_source:
            verification["source"]["uploader_name"] = official_source
            verification["source"]["original_source_name"] = official_source
            verification["source"]["source_type"] = "news outlet"
            verification["source"]["source_is_uploader"] = True
            for field in (
                "verification.source.uploader_name",
                "verification.source.original_source_name",
                "verification.source.source_type",
                "verification.source.source_is_uploader",
            ):
                add_field_evidence(extra, field, [metadata_evidence])

        mark_automated(
            extra,
            "bootstrap",
            {
                "claim_evidence": claim_evidence,
                "metadata_evidence": metadata_evidence,
                "article_evidence": article_evidence,
            },
        )
        write_json(path, extra)
        if number % 100 == 0:
            print("bootstrap", number, "/", len(records), flush=True)
