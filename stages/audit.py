
import json
from collections import Counter

from stages import MODELS
from utils.files import now, read_json, write_json
from utils.media_metadata import CORE_METADATA_FIELDS
from utils.records import sidecar_path


METADATA_FIELD_PATHS = {
    "uploader_name": "verification.source.uploader_name",
    "uploader_profile": "verification.source.uploader_profile",
    "upload_date": "verification.date.video_upload_date",
    "title": "verification.motivation.original_caption",
    "description": "verification.motivation.original_description",
}


def verification_fields(extra):
    for pillar, values in extra.get("verification", {}).items():
        for field, value in values.items():
            yield "verification." + pillar + "." + field, value


def is_filled(value):
    return value is not None and value != [] and value != ""


def run(records, output, report_path=None):
    coverage = Counter()
    unsupported = []
    dangling_links = []
    stage_issues = []
    evidence_types = Counter()
    stale_metadata_links = []
    for record in records:
        extra = read_json(sidecar_path(output, record["claim_id"]))
        links = extra.get("field_evidence", {})
        evidence_ids = {
            evidence.get("id")
            for evidence in extra.get("evidence", [])
            if evidence.get("id")
        }
        for field, value in verification_fields(extra):
            if is_filled(value):
                coverage[field] += 1
                if not evidence_ids.intersection(links.get(field) or []):
                    unsupported.append({"claim_id": record["claim_id"], "field": field})
        for field, linked_ids in links.items():
            missing = [evidence_id for evidence_id in linked_ids if evidence_id not in evidence_ids]
            if missing:
                dangling_links.append(
                    {
                        "claim_id": record["claim_id"],
                        "field": field,
                        "evidence_ids": missing,
                    }
                )
        for stage, result in (extra.get("automation") or {}).items():
            if not isinstance(result, dict):
                continue
            status = result.get("status")
            errors = result.get("errors") or []
            if status in {"error", "parse_error", "low_quality"} or errors:
                stage_issues.append(
                    {
                        "claim_id": record["claim_id"],
                        "stage": stage,
                        "status": status,
                        "errors": errors[:3] if isinstance(errors, list) else [str(errors)],
                    }
                )
        web = (extra.get("automation") or {}).get("web") or {}
        web_metadata = web.get("metadata") or {}
        platform_evidence = {
            evidence.get("id")
            for evidence in extra.get("evidence", [])
            if evidence.get("type") == "platform_metadata"
        }
        for field in CORE_METADATA_FIELDS:
            path = METADATA_FIELD_PATHS[field]
            if not web_metadata.get(field) and platform_evidence.intersection(links.get(path) or []):
                stale_metadata_links.append(
                    {"claim_id": record["claim_id"], "field": path}
                )
        evidence_types.update(evidence.get("type") for evidence in extra.get("evidence", []))
    report = {
        "samples": len(records),
        "field_coverage": dict(sorted(coverage.items())),
        "unsupported_non_null_fields": len(unsupported),
        "unsupported_examples": unsupported[:50],
        "dangling_field_evidence": len(dangling_links),
        "dangling_examples": dangling_links[:50],
        "stage_issues": len(stage_issues),
        "stage_issue_examples": stage_issues[:50],
        "stale_metadata_evidence_links": len(stale_metadata_links),
        "stale_metadata_examples": stale_metadata_links[:50],
        "evidence_types": dict(sorted(evidence_types.items())),
        "models": MODELS,
        "generated_at": now(),
    }
    write_json(report_path or output / "enrichment_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report
