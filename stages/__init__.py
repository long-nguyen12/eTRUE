
MODELS = {
    "clip": "openai/clip-vit-base-patch32",
    "transcript": "small",
    "vision": "Qwen/Qwen3.5-4B",
    "text": "Qwen/Qwen3.5-4B",
    "location": "dslim/distilbert-NER",
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
