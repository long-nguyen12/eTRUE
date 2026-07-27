"""Select source records and locate their generated artifacts."""

from pathlib import Path

from build_etrue import load_records


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


def video_path(source, record):
    return source / (record["legacy_split"] + "_video") / (record["claim_id"] + ".mp4")
