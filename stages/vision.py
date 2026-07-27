"""Analyze extracted keyframes for visual verification clues."""

import json

from pillars.location import location_candidate_is_plausible
from stages import MODELS
from utils.evidence import add_evidence, add_field_evidence, mark_automated
from utils.files import read_json, trim, write_json
from utils.model_output import parse_json_output, release_models
from utils.records import frame_paths, sidecar_path


VISION_PROMPT = """Analyze all keyframes together. Return exactly one compact JSON object:
{
  "scene_summary": "short factual description",
  "ocr_text": ["unique visible text"],
  "landmarks": ["recognizable landmark"],
  "signs_and_logos": ["unique sign or logo"],
  "languages": ["visible language"],
  "terrain_and_weather": ["visible environmental clue"],
  "candidate_locations": ["place explicitly supported by visible text or landmark"]
}
Use at most three unique items per list. Use [] when absent. Never repeat an item.
Do not guess a location from appearance alone. End immediately after the JSON object."""


def normalize_vision_analysis(value):
    aliases = {
        "scenesummary": "scene_summary",
        "ocrtext": "ocr_text",
        "landmarks": "landmarks",
        "signsandlogos": "signs_and_logos",
        "signsandlogs": "signs_and_logos",
        "languages": "languages",
        "terrainandweather": "terrain_and_weather",
        "terrainandwether": "terrain_and_weather",
        "candidatelocations": "candidate_locations",
    }
    normalized = {
        "scene_summary": "",
        "ocr_text": [],
        "landmarks": [],
        "signs_and_logos": [],
        "languages": [],
        "terrain_and_weather": [],
        "candidate_locations": [],
    }
    if isinstance(value, (list, tuple)):
        flattened = []

        def collect(item):
            if isinstance(item, (list, tuple)):
                for child in item:
                    collect(child)
            elif isinstance(item, str) and item.strip():
                flattened.append(item.strip())

        collect(value)
        normalized["scene_summary"] = trim("; ".join(flattened), 500)
        return normalized

    if not isinstance(value, dict):
        return normalized

    for key, item in value.items():
        canonical = aliases.get("".join(character for character in key.lower() if character.isalnum()))
        if not canonical:
            continue
        if canonical == "scene_summary":
            normalized[canonical] = trim(item, 500)
        elif isinstance(item, list):
            normalized[canonical] = item
        elif item:
            normalized[canonical] = [item]

    for key in ("ocr_text", "landmarks", "signs_and_logos", "languages", "terrain_and_weather"):
        unique = []
        for item in normalized[key]:
            text = trim(item, 300)
            placeholder = text.lower() in {
                "unique visible text",
                "recognizable landmark",
                "unique sign or logo",
                "visible language",
                "visible environmental clue",
                "visible weather",
                "visible terrain and weather",
            }
            if text and not placeholder and text not in unique:
                unique.append(text)
        normalized[key] = unique[:3]

    if normalized["scene_summary"].lower() in {"short factual description", "short, factual description"}:
        normalized["scene_summary"] = ""
    if "no additional information provided" in normalized["scene_summary"].lower():
        normalized["scene_summary"] = ""

    support = " ".join(
        [normalized["scene_summary"]]
        + normalized["ocr_text"]
        + normalized["landmarks"]
        + normalized["signs_and_logos"]
    ).lower()
    supported_locations = []
    for item in normalized["candidate_locations"]:
        candidate = item.get("name") if isinstance(item, dict) else item
        candidate = trim(candidate, 200)
        if (
            candidate
            and location_candidate_is_plausible(candidate)
            and candidate.lower() in support
            and candidate not in supported_locations
        ):
            supported_locations.append(candidate)
    normalized["candidate_locations"] = supported_locations[:3]
    return normalized


def evenly_spaced(items, maximum):
    if len(items) <= maximum:
        return items
    indexes = [round(index * (len(items) - 1) / (maximum - 1)) for index in range(maximum)]
    return [items[index] for index in indexes]


def run(records, source, output, model_cache, force=False, offline=False):
    """Extract conservative scene, OCR, landmark, and location clues."""
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoModelForCausalLM, Qwen3VLForConditionalGeneration

    cache = output / "cache" / "vision"
    cache.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading models on", device)
    dtype = torch.float16 if device != "cpu" else torch.float32,
    # processor = AutoProcessor.from_pretrained(
    #     MODELS["vision"], cache_dir=str(model_cache), local_files_only=offline
    # )
    # if hasattr(processor, "image_processor"):
    #     processor.image_processor.size = {"longest_edge": 1024}
    # model = AutoModelForImageTextToText.from_pretrained(
    #     MODELS["vision"],
    #     cache_dir=str(model_cache),
    #     torch_dtype=dtype,
    #     local_files_only=offline,
    # ).to(device)
    processor = AutoProcessor.from_pretrained(MODELS["vision"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODELS["vision"], dtype="auto", device_map="auto"
    ).to(device)
    model.eval()
    print("Loaded models")

    for number, record in enumerate(records, 1):
        cached = cache / (record["claim_id"] + ".json")
        if cached.exists() and not force:
            analysis = read_json(cached)
            if analysis.get("status") == "parse_error" and analysis.get("raw_output"):
                try:
                    repaired = normalize_vision_analysis(parse_json_output(analysis["raw_output"]))
                    repaired["status"] = "ok" if any(
                        repaired.get(key)
                        for key in (
                            "scene_summary",
                            "ocr_text",
                            "landmarks",
                            "signs_and_logos",
                            "terrain_and_weather",
                        )
                    ) else "low_quality"
                    repaired["frames"] = analysis.get("frames")
                    repaired["model"] = MODELS["vision"]
                    analysis = repaired
                    write_json(cached, analysis)
                except (ValueError, SyntaxError, json.JSONDecodeError):
                    pass
        else:
            frames = evenly_spaced(frame_paths(source, record), 4)
            if not frames:
                analysis = {"status": "no_keyframes"}
                write_json(cached, analysis)
            else:
                images = []
                for frame in frames:
                    with Image.open(frame) as image:
                        images.append(image.convert("RGB").copy())
                content = [{"type": "image"} for _ in images]
                content.append({"type": "text", "text": VISION_PROMPT})
                messages = [{"role": "user", "content": content}]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
    return_tensors="pt"
                )
                inputs = inputs.to(device)
                # inputs = processor(text=prompt, images=images, return_tensors="pt")
                # inputs = {
                #     key: value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device)
                #     for key, value in inputs.items()
                # }
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=256,
                        repetition_penalty=1.15,
                        no_repeat_ngram_size=6,
                    )
                generated = processor.decode(
                    output_ids[0][inputs["input_ids"].shape[-1] :],
                    skip_special_tokens=True,
                )
                try:
                    analysis = normalize_vision_analysis(parse_json_output(generated))
                    analysis["status"] = "ok"
                except (ValueError, SyntaxError, json.JSONDecodeError) as error:
                    analysis = {"status": "parse_error", "raw_output": generated, "error": str(error)}
                analysis["frames"] = [frame.relative_to(source).as_posix() for frame in frames]
                analysis["model"] = MODELS["vision"]
                write_json(cached, analysis)

        if analysis.get("status") == "ok":
            metadata = {
                "status": "ok",
                "frames": analysis.get("frames") or [],
                "model": analysis.get("model") or MODELS["vision"],
            }
            analysis = normalize_vision_analysis(analysis)
            analysis.update(metadata)
            write_json(cached, analysis)

        sidecar_file = sidecar_path(output, record["claim_id"])
        sidecar = read_json(sidecar_file)
        if analysis.get("status") == "ok":
            clues = []
            for category in ("ocr_text", "landmarks", "signs_and_logos", "languages", "terrain_and_weather"):
                for value in analysis.get(category) or []:
                    clues.append({"type": category, "value": value})
            location = sidecar["verification"]["location"]
            location["visual_location_clues"] = clues
            for candidate in analysis.get("candidate_locations") or []:
                if candidate not in location["candidate_locations"]:
                    location["candidate_locations"].append(candidate)
            evidence_id = add_evidence(
                sidecar,
                "keyframe_analysis",
                "cache/vision/" + record["claim_id"] + ".json",
                analysis.get("scene_summary"),
                fields=(
                    "verification.location.visual_location_clues",
                    "verification.location.candidate_locations",
                ),
                model=MODELS["vision"],
                frames=analysis.get("frames"),
            )
            add_field_evidence(sidecar, "verification.location.visual_location_clues", [evidence_id])
        mark_automated(sidecar, "vision", analysis)
        write_json(sidecar_file, sidecar)
        if number % 10 == 0:
            print("vision", number, "/", len(records), flush=True)
    release_models(model, processor)
