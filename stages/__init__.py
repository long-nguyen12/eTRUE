"""Enrichment pipeline stages and model choices."""

MODELS = {
    "clip": "openai/clip-vit-base-patch32",
    "vision": "Qwen/Qwen3-VL-4B-Instruct",
    "text": "Qwen/Qwen3-4B",
    "location": "dslim/bert-base-NER",
}

PIPELINE_STAGES = (
    "bootstrap",
    "match",
    "vision",
    "web",
    "search",
    "text",
    "location",
    "geocode",
    "audit",
    "export",
)
