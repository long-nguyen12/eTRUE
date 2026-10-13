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
NORMALIZE_PROMPT = """Decompose the input claim into independent atomic propositions.

CONSTRAINTS:
1. Never add outside facts. Keep the exact language of the original claim.
2. Copy the original claim into "normalized_claim", altering only whitespace or punctuation.
3. Each component "text" must be an exact, contiguous substring of the "normalized_claim".
4. Never move a negation between clauses.
5. Missing dimensions must be strictly null. Do not infer dates, locations, or actions.

HANDLING MULTIPLE SUBJECTS:
If a contiguous clause contains multiple subjects (e.g., "A and B did X"), create a separate JSON object for each subject. Keep the "text" field identical for both objects, but specify the individual subject in the "subject" field to make the propositions independent.

OUTPUT FORMAT:
You must output ONLY a raw JSON object.
Do NOT wrap the output in markdown code blocks (do not use ```json).
Do NOT include any conversational text, explanations, or hidden thoughts.

REQUIRED JSON SKELETON:
{
  "normalized_claim": "",
  "claim_components": [
    {
      "text": "",
      "subject": null,
      "action": null,
      "time": null,
      "location": null,
      "attribution": null,
      "context": null
    }
  ]
}
Instructions for Skeleton: Replace the empty strings with extracted phrases. Leave as null if a dimension is not explicitly present in the text. Duplicate the component object in the array as many times as necessary to capture all propositions.

INPUT CLAIM:
[INSERT CLAIM HERE]
"""


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
        return str(path)

    clean["transcript"] = transcript.strip() if transcript else None
    if url is not None:
        clean["post_url"] = public_http_url(url)
    clean["keyframes"] = [media_path(p) for p in frames]
    if payload.get("video_path") is not None:
        clean["video_path"] = media_path(payload["video_path"])
    return clean


def _normalize_claim(runtime, claim):
    raw = runtime.generate(NORMALIZE_PROMPT, {"claim": claim}, max_new_tokens=2048)
    normalized = raw.get("normalized_claim")
    components = raw.get("claim_components")
    words = set(re.findall(r"\w+", claim.casefold()))
    output = []
    for index, component in enumerate(components, 1):
        text = component["text"].strip()
        row = {"id": "claim_" + str(index), "text": text}
        for dimension in DIMENSIONS:
            value = component.get(dimension)
            row[dimension] = value

        output.append(row)
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
    payload = validate_payload(payload)
    work_dir = Path(work_dir).resolve()
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
