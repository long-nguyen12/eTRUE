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
    review_status = Counter()
    evidence_types = Counter()
    for record in records:
        extra = read_json(sidecar_path(output, record["claim_id"]))
        write_readable_result(extra, output)
        links = extra.get("field_evidence", {})
        for field, value in verification_fields(extra):
            if is_filled(value):
                coverage[field] += 1
                if not links.get(field):
                    unsupported.append({"claim_id": record["claim_id"], "field": field})
        review_status[(extra.get("review") or {}).get("status") or "missing"] += 1
        evidence_types.update(evidence.get("type") for evidence in extra.get("evidence", []))
    report = {
        "samples": len(records),
        "field_coverage": dict(sorted(coverage.items())),
        "unsupported_non_null_fields": len(unsupported),
        "unsupported_examples": unsupported[:50],
        "review_status": dict(sorted(review_status.items())),
        "evidence_types": dict(sorted(evidence_types.items())),
        "models": MODELS,
        "generated_at": now(),
    }
    # write_json(output / "enrichment_report.json", report)
    # print(json.dumps(report, indent=2), flush=True)
