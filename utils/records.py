"""Load source records and locate their generated artifacts."""

from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit

from utils.files import read_json


PLATFORM_ALIASES = {
    "redd": "reddit",
    "tiktokcdn": "tiktok",
    "twitter": "x",
    "youtu": "youtube",
}


def normalize_platform(value):
    platform = str(value or "unknown").strip().lower()
    return PLATFORM_ALIASES.get(platform, platform)


def normalize_date(value):
    if value is None or str(value).strip() == "":
        return None

    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]

    for date_format in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, date_format).date().isoformat()
        except ValueError:
            pass
    return None


def canonical_video_key(url, claim_id):
    """Return a stable key for exact URL-equivalent videos."""
    if not url:
        return "claim:" + claim_id

    parsed = urlsplit(str(url).strip())
    host = parsed.netloc.lower().split("@")[-1]
    if host.startswith("www."):
        host = host[4:]

    path = parsed.path.rstrip("/") or "/"
    query = parse_qsl(parsed.query, keep_blank_values=True)

    youtube_id = None
    if host == "youtu.be" and path != "/":
        youtube_id = path.strip("/").split("/")[0]
    elif host.endswith("youtube.com"):
        parts = [part for part in path.split("/") if part]
        if len(parts) >= 2 and parts[0] in {"embed", "shorts", "live"}:
            youtube_id = parts[1]
        else:
            youtube_id = dict(query).get("v")
    if youtube_id:
        return "youtube:" + youtube_id

    if host in {"twitter.com", "x.com"}:
        host = "x.com"

    if path == "/" and not query:
        return "claim:" + claim_id

    query_text = urlencode(sorted(query))
    return host + path + (("?" + query_text) if query_text else "")


def load_records(source):
    records = []
    seen_ids = set()
    for legacy_split in ("train_val", "test"):
        annotation_dir = source / legacy_split
        if not annotation_dir.is_dir():
            raise FileNotFoundError("Missing annotation directory: " + str(annotation_dir))

        for path in sorted(annotation_dir.glob("*.json")):
            claim_id = path.stem
            if claim_id in seen_ids:
                raise ValueError("Duplicate claim ID: " + claim_id)
            seen_ids.add(claim_id)
            records.append(
                {
                    "claim_id": claim_id,
                    "legacy_split": legacy_split,
                    "path": path,
                    "data": read_json(path),
                }
            )
    return records


def selected_records(source, ids_value, limit):
    records = load_records(source)
    if ids_value != "all":
        ids_path = Path(ids_value)
        wanted = {
            line.strip()
            for line in ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        records = [record for record in records if record["claim_id"] in wanted]
    if limit:
        records = records[:limit]
    return records


def sidecar_path(output, claim_id):
    return output / "annotations" / (claim_id + ".json")


def frame_paths(source, record):
    directory = source / (record["legacy_split"] + "_output") / record["claim_id"]
    return sorted(directory.glob("*.jpeg"), key=lambda path: int(path.stem))
