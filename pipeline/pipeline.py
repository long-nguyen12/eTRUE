"""Request-scoped orchestration, independent of enrichment and annotation."""

import json
import re
import time
from pathlib import Path

from utils.files import now
from utils.json_ld import public_http_url

from .assessment import analyze_component, predict_verdict
from .evidence import EvidenceStore
from .media import prepare_media
from .retrieval import (
    collect_evidence,
    inspect_page,
    is_factcheck_text,
    is_factcheck_url,
)
from .runtime import QwenRuntime

INPUT_FIELDS = {
    "claim",
    "video_path",
    "keyframes",
    "post_url",
    "metadata",
    "transcript",
}
METADATA_FIELDS = {
    "title",
    "description",
    "platform",
    "uploader_name",
    "uploader_profile",
    "upload_date",
}
DIMENSIONS = ("subject", "action", "time", "location", "attribution", "context")
NORMALIZE_PROMPT = """Decompose this claim into independent atomic propositions.
Preserve negation, uncertainty, quoted attribution, conjunctions, and every important assertion.
Never add facts from outside the claim. Keep the language of the original claim.
Return {"normalized_claim":text,"claim_components":[{"text":text,"subject":text_or_null,
"action":text_or_null,"time":text_or_null,"location":text_or_null,
"attribution":text_or_null,"context":text_or_null}]}.
Copy the original claim into normalized_claim, changing only whitespace or punctuation.
Each component text must quote a contiguous clause of the original claim; never move a
negation between clauses. Subject may be inherited in the subject field to make the
proposition independent without rewriting the quoted text. Dimension values must be literal
phrases from the original claim; action must occur in the quoted component text.
Missing dimensions are null, not inferred. Do not resolve
relative dates without an explicit reference date. Return one JSON object, no hidden thought.
"""


def check_artifact_path(path):
    path = Path(path).resolve()
    repo = Path(__file__).resolve().parent.parent
    for protected in (repo / "data" / "eTRUE", repo / "data" / "TRUE_Dataset"):
        if path.is_relative_to(protected.resolve()):
            raise ValueError("Pipeline artifacts must be outside the existing datasets")
    return path


def validate_payload(payload, base_dir=None):
    clean = {**payload, "claim": payload["claim"].strip()}
    metadata = payload.get("metadata", {})
    if metadata is None:
        metadata = {}
    clean["metadata"] = dict(metadata)
    transcript = payload.get("transcript")
    url = payload.get("post_url")
    frames = payload.get("keyframes", [])
    if frames is None:
        frames = []
    base = Path(base_dir or Path.cwd())

    def media_path(value):
        path = Path(value)
        path = (base / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file():
            raise ValueError("Media file does not exist: " + str(path))
        return str(path)

    clean["keyframes"] = [media_path(p) for p in frames]
    if payload.get("video_path") is not None:
        clean["video_path"] = media_path(payload["video_path"])
    return clean


def _normalize_claim(runtime, claim):
    raw = runtime.generate(NORMALIZE_PROMPT, {"claim": claim}, max_new_tokens=2048)
    normalized = raw.get("normalized_claim")
    components = raw.get("claim_components")
    words = set(re.findall(r"\w+", claim.casefold()))
    original_tokens = re.findall(r"\w+", claim.casefold())
    output = []
    for index, component in enumerate(components, 1):
        text = component["text"].strip()
        tokens = re.findall(r"\w+", text.casefold())
        row = {"id": "claim_" + str(index), "text": text}
        for dimension in DIMENSIONS:
            value = component.get(dimension)
            row[dimension] = value
        if row["action"] and row["action"].casefold() not in text.casefold():
            raise ValueError("Action must be grounded in the quoted component clause")
        output.append(row)
    # A dropped negation/qualification must never silently turn a claim affirmative.
    qualifiers = {
        "not",
        "never",
        "no",
        "cannot",
        "can't",
        "didn't",
        "wasn't",
        "isn't",
        "may",
        "might",
        "allegedly",
        "supposedly",
        "possibly",
    }
    for token in qualifiers:
        pattern = r"(?<!\w)" + re.escape(token) + r"(?!\w)"
        if re.search(pattern, claim, re.I) and (
            not re.search(pattern, normalized, re.I)
            or not any(re.search(pattern, c["text"], re.I) for c in output)
        ):
            raise ValueError("Claim normalization dropped negation or uncertainty")
    component_words = set(
        re.findall(r"\w+", " ".join(c["text"] for c in output).casefold())
    )
    if (
        words
        - component_words
        - {
            "and",
            "the",
            "a",
            "an",
            "that",
            "this",
            "it",
            "its",
            "is",
            "was",
            "were",
            "are",
        }
    ):
        raise ValueError("Claim normalization omitted part of the original assertion")
    return normalized.strip(), output


def _relations(results):
    relations, trace = [], []
    for result in results:
        for pillar, assessment in {
            "core": result["core_assessment"],
            **result["pillar_assessments"],
        }.items():
            for key, stance in (
                ("supporting_citations", "supports"),
                ("refuting_citations", "refutes"),
            ):
                for citation in assessment[key]:
                    relations.append(
                        {
                            "component_id": result["component_id"],
                            "pillar": pillar,
                            "stance": stance,
                            **citation,
                        }
                    )
            trace.append(
                {
                    "component_id": result["component_id"],
                    "pillar": pillar,
                    "agent": assessment.get("agent"),
                    "status": assessment["status"],
                    "conclusion": assessment["explanation"],
                    "evidence_ids": list(
                        dict.fromkeys(
                            c["evidence_id"]
                            for key in ("supporting_citations", "refuting_citations")
                            for c in assessment[key]
                        )
                    ),
                }
            )
    return relations, trace


def verify_claim(payload: dict, *, work_dir: Path, model_cache: Path) -> dict:
    """Verify one request. Invalid input raises ValueError; run failures are JSON errors."""
    payload = validate_payload(payload)
    work_dir = check_artifact_path(work_dir)
    started = time.perf_counter()
    result = {
        "claim": payload["claim"],
        "normalized_claim": None,
        "claim_components": [],
        "component_results": [],
        "verdict": None,
        "explanation": "",
        "open_questions": [],
        "citations": [],
        "evidence": [],
        "evidence_relations": [],
        "trace": [],
        "status": "error",
        "errors": [],
        "started_at": now(),
        "metrics": {},
    }
    errors = result["errors"]
    store = EvidenceStore()
    runtime = None
    stage = "model"
    try:
        runtime = QwenRuntime(Path(model_cache))
        runtime.load_text()
        stage = "normalize"
        normalized, components = _normalize_claim(runtime, payload["claim"])
        result.update(normalized_claim=normalized, claim_components=components)
        runtime.unload()
        stage = "media"
        media_payload = dict(payload)
        source_page = None
        if payload.get("post_url"):
            import requests

            with requests.Session() as session:
                session.headers["User-Agent"] = "eTRUE-independent-verification/1.0"
                try:
                    source_page = inspect_page(session, payload["post_url"])
                except Exception as error:
                    errors.append(
                        {"stage": "media.source", "message": str(error)[:500]}
                    )
            if source_page is None:
                errors.append(
                    {
                        "stage": "media.source",
                        "message": "Post page unavailable or excluded; URL media not used",
                    }
                )
                media_payload.pop("post_url", None)
        if (
            media_payload.get("keyframes")
            or media_payload.get("video_path")
            or media_payload.get("post_url")
        ):
            media = prepare_media(media_payload, work_dir)
        else:
            media = {
                "frames": [],
                "video_path": None,
                "frame_timestamps": [],
                "errors": [],
            }
        errors.extend(media["errors"])
        if source_page:
            media["source_page"] = source_page
        try:
            media["analysis"] = runtime.analyze_frames(media["frames"])
            if media["frames"]:
                observation = json.dumps(
                    {
                        k: v
                        for k, v in media["analysis"].items()
                        if k not in {"model", "frames", "status"}
                    },
                    ensure_ascii=False,
                )
                if is_factcheck_text(observation):
                    media["analysis"] = {}
                    media["frames"] = []
                    errors.append(
                        {
                            "stage": "vision",
                            "message": "Fact-check content in media analysis excluded",
                        }
                    )
                else:
                    store.add(
                        "keyframe_analysis",
                        str(media["frames"][0]),
                        observation,
                        method="qwen_vision",
                        verified=False,
                        frames=[str(p) for p in media["frames"]],
                        frame_timestamps=media["frame_timestamps"],
                    )
        except Exception as error:
            errors.append({"stage": "vision", "message": str(error)[:500]})
            media["analysis"] = {}
        for kind, observation in (
            ("supplied_metadata", payload["metadata"]),
            ("supplied_transcript", payload.get("transcript")),
        ):
            if observation:
                store.add(
                    kind,
                    "input:" + kind,
                    json.dumps(observation, ensure_ascii=False),
                    method="user_input",
                    verified=False,
                )
        stage = "retrieval"
        collect_evidence(payload, components, media, store, errors)
        stage = "model"
        runtime.load_text()
        stage = "pillars"
        for component in components:
            reference = {
                "post_url": payload.get("post_url"),
                "frames": [str(p) for p in media["frames"]],
                "video_path": media["video_path"],
            }
            result["component_results"].append(
                analyze_component(component, store, runtime, errors, reference)
            )
        stage = "verdict"
        result.update(
            predict_verdict(components, result["component_results"], store, runtime)
        )
        result["status"] = "partial" if errors else "ok"
        result["media"] = {
            "frames": [str(p) for p in media["frames"]],
            "video_path": media["video_path"],
            "frame_timestamps": media["frame_timestamps"],
        }
    except Exception as error:
        errors.append({"stage": stage, "message": str(error)[:500]})
    finally:
        result["evidence"] = store.items
        result["evidence_relations"], result["trace"] = _relations(
            result["component_results"]
        )
        result["finished_at"] = now()
        if runtime is not None:
            try:
                result["metrics"] = runtime.metrics()
                result["metrics"]["omitted_evidence_ids"] = list(
                    runtime.omitted_evidence_ids
                )
            except Exception as error:
                errors.append({"stage": "metrics", "message": str(error)[:500]})
                result["status"] = "error"
            try:
                runtime.unload()
            except Exception as error:
                errors.append({"stage": "cleanup", "message": str(error)[:500]})
                result["status"] = "error"
        result["metrics"]["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return result
