"""Audit field coverage and evidence links after enrichment."""

import json
from collections import Counter

from stages import MODELS
from utils.files import now, read_json, write_json
from utils.records import sidecar_path
from utils.results import write_readable_result


def verification_fields(extra):
    for pillar, values in extra.get("verification", {}).items():
        for field, value in values.items():
            yield "verification." + pillar + "." + field, value


def is_filled(value):
    return value is not None and value != [] and value != ""


def run(records, output):
    coverage = Counter()
    unsupported = []
    dangling_links = []
    stage_issues = []
    review_status = Counter()
    evidence_types = Counter()
    for record in records:
        extra = read_json(sidecar_path(output, record["claim_id"]))
        write_readable_result(extra, output)
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
        review_status[(extra.get("review") or {}).get("status") or "missing"] += 1
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
        "review_status": dict(sorted(review_status.items())),
        "evidence_types": dict(sorted(evidence_types.items())),
        "models": MODELS,
        "generated_at": now(),
    }
    write_json(output / "enrichment_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report
