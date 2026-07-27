"""Parsing and cleanup helpers shared by local model stages."""

import ast
import gc
import json


def parse_json_output(text):
    text = str(text or "").strip()
    if "```" in text:
        parts = text.split("```")
        text = max(parts, key=lambda part: part.count("{") + part.count("["))
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:].lstrip()

    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        raise ValueError("Model output did not contain structured data")
    start = min(starts)
    closing = "}" if text[start] == "{" else "]"
    end = text.rfind(closing)
    if end < start:
        raise ValueError("Model output did not close its structured data")

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return ast.literal_eval(candidate)


def enum_value(value, allowed):
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized if normalized in allowed else None


def release_models(*objects):
    for value in objects:
        del value
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
