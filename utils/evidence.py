"""Evidence creation, linking, and automated-review bookkeeping."""

import hashlib
import json

from utils.files import now


def add_field_evidence(extra, field, evidence_ids):
    links = extra.setdefault("field_evidence", {}).setdefault(field, [])
    for evidence_id in evidence_ids:
        if evidence_id not in links:
            links.append(evidence_id)


def add_evidence(extra, evidence_type, source, observation, fields=(), **details):
    identity = json.dumps(
        [evidence_type, str(source), str(observation), details.get("model")],
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    evidence_id = evidence_type.replace("_", "-") + "-" + digest
    evidence = {
        "id": evidence_id,
        "type": evidence_type,
        "source": str(source),
        "observation": observation,
    }
    evidence.update({key: value for key, value in details.items() if value is not None})

    existing = {item["id"] for item in extra.setdefault("evidence", [])}
    if evidence_id not in existing:
        extra["evidence"].append(evidence)
    for field in fields:
        add_field_evidence(extra, field, [evidence_id])
    return evidence_id


def remove_evidence_type(extra, evidence_type):
    removed = {
        evidence["id"]
        for evidence in extra.get("evidence", [])
        if evidence.get("type") == evidence_type
    }
    extra["evidence"] = [
        evidence for evidence in extra.get("evidence", []) if evidence.get("id") not in removed
    ]
    for field, evidence_ids in list(extra.get("field_evidence", {}).items()):
        remaining = [evidence_id for evidence_id in evidence_ids if evidence_id not in removed]
        if remaining:
            extra["field_evidence"][field] = remaining
        else:
            del extra["field_evidence"][field]


def mark_automated(extra, stage, value):
    extra.setdefault("automation", {})[stage] = value
    review = extra.setdefault("review", {})
    if review.get("status") in {None, "not_started", "automated"}:
        review["status"] = "automated"
    review["last_automated_at"] = now()
