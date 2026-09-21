"""Small file and value helpers used by enrichment stages."""

import json
from datetime import datetime, timezone


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_safe(value):
    if isinstance(value, str):
        return "".join(
            "\ufffd" if 0xD800 <= ord(character) <= 0xDFFF else character
            for character in value
        )
    if isinstance(value, dict):
        return {_json_safe(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def trim(value, length):
    text = str(value or "").strip()
    if len(text) <= length:
        return text
    return text[:length] + "..."
