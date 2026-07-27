"""Evidence creation, linking, and automated-review bookkeeping."""

import hashlib
import json

from utils.files import now


def add_field_evidence(sidecar, field, evidence_ids):
    links = sidecar.setdefault("field_evidence", {}).setdefault(field, [])
    for evidence_id in evidence_ids:
        if evidence_id not in links:
            links.append(evidence_id)


def add_evidence(sidecar, evidence_type, source, observation, fields=(), **details):
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

    existing = {item["id"] for item in sidecar.setdefault("evidence", [])}
    if evidence_id not in existing:
        sidecar["evidence"].append(evidence)
    for field in fields:
        add_field_evidence(sidecar, field, [evidence_id])
    return evidence_id


def remove_evidence_type(sidecar, evidence_type):
    removed = {
        evidence["id"]
        for evidence in sidecar.get("evidence", [])
        if evidence.get("type") == evidence_type
    }
    sidecar["evidence"] = [
        evidence for evidence in sidecar.get("evidence", []) if evidence.get("id") not in removed
    ]
    for field, evidence_ids in list(sidecar.get("field_evidence", {}).items()):
        remaining = [evidence_id for evidence_id in evidence_ids if evidence_id not in removed]
        if remaining:
            sidecar["field_evidence"][field] = remaining
        else:
            del sidecar["field_evidence"][field]


def mark_automated(sidecar, stage, value):
    sidecar.setdefault("automation", {})[stage] = value
    review = sidecar.setdefault("review", {})
    if review.get("status") in {None, "not_started", "automated"}:
        review["status"] = "automated"
    review["last_automated_at"] = now()
