import json

from stages import MODELS
from stages.text_grounding import build_text_grounding
from stages.text_prompts import COMMON_RULES, TEXT_PROMPT_JOBS
from stages.text_validation import apply_text_analysis, reset_text_analysis
from utils.files import read_json, write_json
from utils.model_output import parse_json_output, release_models
from utils.records import sidecar_path


def generate_prompt_result(model, tokenizer, device, job, evidence):
    import torch

    messages = [
        {"role": "system", "content": COMMON_RULES},
        {
            "role": "user",
            "content": job["instructions"] + "\n\nEVIDENCE:\n" + evidence,
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=8192
    ).to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=job["max_new_tokens"],
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )
    print("text", job["name"], "model output:", generated, flush=True)

    try:
        analysis = parse_json_output(generated)
        if isinstance(analysis, list) and len(analysis) == 1:
            analysis = analysis[0]
        if not isinstance(analysis, dict):
            raise ValueError("Text model output must be a JSON object")
        return {
            "status": "ok",
            "analysis": {field: analysis.get(field) for field in job["fields"]},
            "model": MODELS["text"],
        }
    except (ValueError, json.JSONDecodeError) as error:
        return {
            "status": "parse_error",
            "raw_output": generated,
            "error": str(error),
            "model": MODELS["text"],
        }


def generate_text_result(
    model, tokenizer, device, grounding_text, event_grounding_text
):
    evidence = {"full": grounding_text, "event": event_grounding_text}
    analysis = {}
    parts = {}
    for job in TEXT_PROMPT_JOBS:
        job_evidence = evidence[job["evidence"]]
        if job["name"] == "motivation":
            job_evidence += "\n\nPROVISIONAL PROVENANCE/SOURCE:\n" + json.dumps(
                {
                    field: analysis[field]
                    for field in TEXT_PROMPT_JOBS[0]["fields"]
                    if field in analysis
                },
                ensure_ascii=False,
            )
        result = generate_prompt_result(
            model,
            tokenizer,
            device,
            job,
            job_evidence,
        )
        parts[job["name"]] = result
        if result["status"] == "ok":
            analysis.update(result["analysis"])

    success_count = sum(part["status"] == "ok" for part in parts.values())
    if success_count == len(parts):
        status = "ok"
    else:
        status = "partial" if analysis else "parse_error"
    return {
        "status": status,
        "analysis": analysis,
        "parts": parts,
        "model": MODELS["text"],
    }


def run(records, output, model_cache, offline=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        MODELS["text"], cache_dir=str(model_cache), local_files_only=offline
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODELS["text"],
        cache_dir=str(model_cache),
        torch_dtype=dtype,
        local_files_only=offline,
    ).to(device)
    model.eval()

    for number, record in enumerate(records, 1):
        sidecar_file = sidecar_path(output, record["claim_id"])
        extra = read_json(sidecar_file)
        grounding_text, event_grounding_text = build_text_grounding(record, extra)
        result = generate_text_result(
            model,
            tokenizer,
            device,
            grounding_text,
            event_grounding_text,
        )

        reset_text_analysis(extra)
        if result["analysis"]:
            apply_text_analysis(
                extra, result["analysis"], grounding_text, event_grounding_text
            )
        extra.setdefault("automation", {})["text"] = result
        write_json(sidecar_file, extra)
        if number % 10 == 0:
            print("text", number, "/", len(records), flush=True)
    release_models(model, tokenizer)
