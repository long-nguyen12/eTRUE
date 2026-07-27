"""Small file and value helpers used by enrichment stages."""

import json
from datetime import datetime, timezone


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def trim(value, length):
    text = str(value or "").strip()
    if len(text) <= length:
        return text
    return text[:length] + "..."
