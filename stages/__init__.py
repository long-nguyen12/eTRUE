"""Enrichment pipeline stages and model choices."""

MODELS = {
    "clip": "openai/clip-vit-base-patch32",
    # "vision": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    "vision": "Qwen/Qwen3-VL-2B-Instruct",
    "text": "Qwen/Qwen2.5-1.5B-Instruct",
    "location": "dslim/bert-base-NER",
}

PIPELINE_STAGES = (
    "bootstrap",
    "clip",
    "match",
    "vision",
    "web",
    "search",
    "text",
    "location",
    "geocode",
    "audit",
)
