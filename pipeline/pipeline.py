import json
from pathlib import Path

from utils.files import now

from .assessment import analyze_component, predict_verdict
from .evidence import EvidenceStore
from .media import prepare_media
from .retrieval import collect_evidence
from .runtime import QwenRuntime

DIMENSIONS = ("subject", "action", "time", "location", "attribution", "context")
NORMALIZE_PROMPT = """Read the "claim" field of the supplied JSON data and extract its assertions.
Return exactly one JSON object with the following keys and no additional keys:
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

Copy the entire input claim into normalized_claim, changing whitespace only.
Use one or more components. Each text must be an exact contiguous passage from
normalized_claim. Retain every assertion and its original language, negation,
modality, quantifiers, uncertainty, attribution, and temporal/causal/conditional
relationships. Never add outside facts, paraphrase, or resolve an ambiguous claim.
Split only clauses that can stand independently without changing meaning.
Keep collective actions together: "A and B met" is one component.
Keep a linked assertion together if splitting loses its relationship: "The FBI
was on its way when a fire destroyed the servers" is one complete component.
Keep that event relationship in the full text and copied action phrase, rather
than assigning it to context or inventing a calendar time.

Every component must contain all seven keys shown above. Dimension values must
be exact copied phrases from the claim, or null when not explicitly stated:
- subject: the entity or entities performing or experiencing the action.
- action: the asserted action or state; preserve its negation and qualifications.
- time: an explicit calendar date, clock time, or stated period; never infer
  capture time. An event-only phrase such as "when a fire broke out" stays in
  text; leave time null unless a date, clock time, or period is also stated.
- location: an explicitly asserted physical site of the event; do not treat
  objects such as "servers and hard drives" as geographic location clues.
- attribution: an explicitly named speaker or source to whom the claim is ascribed.
- context: explicitly asserted framing or stated purpose, not merely a temporal
  or causal link between events. Leave null when that framing is absent.
Do not invent values to fill missing dimensions. Empty strings are invalid.
End immediately after the JSON object; no Markdown or explanatory text.
"""


def validate_payload(payload, base_dir=None):
    clean = {**payload, "claim": payload["claim"].strip()}
    metadata = payload.get("metadata", {})
    if metadata is None:
        metadata = {}
    clean["metadata"] = dict(metadata)
    transcript = payload.get("transcript")
    frames = payload.get("keyframes", [])
    if frames is None:
        frames = []
    base = Path(base_dir or Path.cwd())

    def media_path(value):
        path = Path(value)
        path = (base / path).resolve() if not path.is_absolute() else path.resolve()
        return str(path)

    clean["transcript"] = transcript.strip() if transcript else None
    clean["keyframes"] = [media_path(p) for p in frames]
    if payload.get("video_path") is not None:
        clean["video_path"] = media_path(payload["video_path"])
    return clean


def _normalize_claim(runtime, claim):
    raw = runtime.generate(NORMALIZE_PROMPT, {"claim": claim}, max_new_tokens=2048)
    if not isinstance(raw, dict) or set(raw) != {"normalized_claim", "claim_components"}:
        raise ValueError("Claim normalization must contain normalized_claim and claim_components")
    normalized = raw.get("normalized_claim")
    components = raw.get("claim_components")
    expected = " ".join(claim.split())
    if not isinstance(normalized, str) or " ".join(normalized.split()) != expected:
        raise ValueError("normalized_claim must preserve the claim exactly apart from whitespace")
    normalized = expected
    if not isinstance(components, list) or not components:
        raise ValueError("claim_components must be a nonempty list")
    output = []
    for index, component in enumerate(components, 1):
        if not isinstance(component, dict) or set(component) != {"text", *DIMENSIONS}:
            raise ValueError("Each claim component must contain text and all six dimensions")
        text = component["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Claim component text must be a nonempty string")
        text = " ".join(text.split())
        if text not in normalized:
            raise ValueError("Claim component text must be an exact passage from normalized_claim")
        row = {"id": "claim_" + str(index), "text": text}
        for dimension in DIMENSIONS:
            value = component[dimension]
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("Claim dimension " + dimension + " must be a nonempty string or null")
                value = " ".join(value.split())
                if value not in normalized:
                    raise ValueError("Claim dimension " + dimension + " must be copied from the claim")
            row[dimension] = value
        output.append(row)
    return normalized, output


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
                            "stance": (
                                "unresolved" if assessment["status"] == "unknown" else stance
                            ),
                            **citation,
                        }
                    )
            trace.append(
                {
                    "component_id": result["component_id"],
                    "pillar": pillar,
                    "agent": assessment.get("agent"),
                    **(
                        {"reviewed_by": assessment["reviewed_by"]}
                        if "reviewed_by" in assessment else {}
                    ),
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


def verify_claim(
    payload: dict,
    *,
    work_dir: Path,
    model_cache: Path,
    runtime: QwenRuntime | None = None,
) -> dict:
    payload = validate_payload(payload)
    work_dir = Path(work_dir).resolve()
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
    }
    errors = result["errors"]
    store = EvidenceStore()
    owns_runtime = runtime is None
    stage = "model"
    try:
        if runtime is None:
            runtime = QwenRuntime(Path(model_cache))
        runtime.last_evidence_ids = []
        runtime.load_text()
        stage = "normalize"
        normalized, components = _normalize_claim(runtime, payload["claim"])
        result.update(normalized_claim=normalized, claim_components=components)
        stage = "media"
        if (
            payload.get("keyframes")
            or payload.get("video_path")
            or payload.get("post_url")
        ):
            media = prepare_media(payload, work_dir)
        else:
            media = {
                "frames": [],
                "video_path": None,
                "frame_timestamps": [],
                "errors": [],
            }
        errors.extend(media["errors"])
        result["media"] = {
            "frames": [str(p) for p in media["frames"]],
            "video_path": media["video_path"],
            "frame_timestamps": media["frame_timestamps"],
        }
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
        reference = {
            "post_url": payload.get("post_url"),
            "frames": [str(p) for p in media["frames"]],
            "video_path": media["video_path"],
        }
        for component in components:
            result["component_results"].append(
                analyze_component(component, store, runtime, errors, reference)
            )
        stage = "verdict"
        result.update(
            predict_verdict(components, result["component_results"], store, runtime)
        )
        result["status"] = "partial" if errors else "ok"
    except Exception as error:
        errors.append({"stage": stage, "message": str(error)[:500]})
    finally:
        if owns_runtime and runtime is not None:
            try:
                runtime.unload()
            except Exception as error:
                errors.append({"stage": "cleanup", "message": str(error)[:500]})
                result["status"] = "error"
        result["evidence"] = store.items
        result["evidence_relations"], result["trace"] = _relations(
            result["component_results"]
        )
        result["finished_at"] = now()
    return result
