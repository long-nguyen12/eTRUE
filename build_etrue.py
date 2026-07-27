"""Build simple eTRUE sidecar annotations from the original TRUE dataset."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit


PLATFORM_ALIASES = {
    "redd": "reddit",
    "tiktokcdn": "tiktok",
    "twitter": "x",
    "youtu": "youtube",
}


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


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


def choose_split(group_key, seed):
    digest = hashlib.sha256((str(seed) + ":" + group_key).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


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


def empty_verification():
    return {
        "provenance": {
            "provenance_status": None,
            "earliest_known_url": None,
            "earliest_known_date": None,
            "near_duplicate_matches": [],
            "previous_context_summary": None,
            "provenance_mismatch": None,
        },
        "source": {
            "uploader_name": None,
            "uploader_profile": None,
            "original_source_name": None,
            "source_type": None,
            "source_is_uploader": None,
            "source_mismatch": None,
        },
        "date": {
            "claimed_date": None,
            "video_upload_date": None,
            "earliest_online_date": None,
            "estimated_date": None,
            "capture_date_granularity": None,
            "date_mismatch_type": None,
        },
        "location": {
            "claimed_location": None,
            "candidate_locations": [],
            "visual_location_clues": [],
            "verified_location": None,
            "verified_coordinates": None,
            "location_mismatch_type": None,
        },
        "motivation": {
            "claimed_framing": None,
            "original_caption": None,
            "original_description": None,
            "original_context_category": None,
            "motivation_mismatch_type": None,
        },
    }


def make_sidecar(record, source):
    data = record["data"]
    video = data.get("video_information") or {}
    transcript = video.get("video_transcript")
    has_transcript = bool(str(transcript or "").strip())

    return {
        "claim_id": record["claim_id"],
        "source_file": record["path"].relative_to(source).as_posix(),
        "legacy_split": record["legacy_split"],
        "normalized_video_information": {
            "platform": normalize_platform(video.get("platform")),
            "video_date": normalize_date(video.get("video_date")),
            "video_url": video.get("video_url"),
            "video_transcript": transcript if has_transcript else None,
            "transcript_status": "available" if has_transcript else "missing",
        },
        "claim_components": [
            {
                "id": "claim_1",
                "text": data.get("claim"),
            }
        ],
        "verification": empty_verification(),
        "evidence": [],
        "field_evidence": {},
        "review": {
            "status": "not_started",
            "reviewer": None,
            "reviewed_at": None,
            "notes": None,
        },
    }


def count_existing_evidence(data):
    evidences = data.get("evidences") or {}
    count = int(evidences.get("num_of_evidence") or 0)
    referenced = 0
    for number in range(1, count + 1):
        evidence = evidences.get("evidence" + str(number))
        if isinstance(evidence, list) and len(evidence) > 1 and evidence[1]:
            referenced += 1
    return count, referenced


def make_report(records, split_by_id, group_by_id):
    ratings = Counter()
    platforms = Counter()
    dates_missing = 0
    dates_invalid = 0
    transcripts_missing = 0
    samples_without_evidence = 0
    evidence_items = 0
    referenced_evidence_items = 0
    empty_relationships = 0
    id_lengths = Counter()
    split_ratings = defaultdict(Counter)
    groups = defaultdict(list)

    for record in records:
        claim_id = record["claim_id"]
        data = record["data"]
        video = data.get("video_information") or {}
        rating = str(data.get("rating") or "unknown")
        platform = normalize_platform(video.get("platform"))
        raw_date = video.get("video_date")
        evidence_count, referenced_count = count_existing_evidence(data)

        ratings[rating] += 1
        platforms[platform] += 1
        id_lengths[str(len(claim_id))] += 1
        split_ratings[split_by_id[claim_id]][rating] += 1
        groups[group_by_id[claim_id]].append(claim_id)
        evidence_items += evidence_count
        referenced_evidence_items += referenced_count

        if raw_date is None or str(raw_date).strip() == "":
            dates_missing += 1
        elif normalize_date(raw_date) is None:
            dates_invalid += 1
        if not str(video.get("video_transcript") or "").strip():
            transcripts_missing += 1
        if evidence_count == 0:
            samples_without_evidence += 1
        if not data.get("relationship_with_evidence"):
            empty_relationships += 1

    duplicate_groups = [ids for ids in groups.values() if len(ids) > 1]
    split_counts = Counter(split_by_id.values())

    return {
        "samples": len(records),
        "ratings": dict(sorted(ratings.items())),
        "normalized_platforms": dict(sorted(platforms.items())),
        "id_length_distribution": dict(sorted(id_lengths.items(), key=lambda item: int(item[0]))),
        "missing_transcripts": transcripts_missing,
        "missing_video_dates": dates_missing,
        "invalid_video_dates": dates_invalid,
        "samples_without_evidence": samples_without_evidence,
        "evidence_items": evidence_items,
        "referenced_evidence_items": referenced_evidence_items,
        "empty_evidence_relationships": empty_relationships,
        "new_split_counts": dict(sorted(split_counts.items())),
        "new_split_ratings": {
            split: dict(sorted(counts.items())) for split, counts in sorted(split_ratings.items())
        },
        "url_duplicate_groups": len(duplicate_groups),
        "samples_in_url_duplicate_groups": sum(len(ids) for ids in duplicate_groups),
        "largest_url_duplicate_group": max((len(ids) for ids in duplicate_groups), default=1),
        "split_grouping": "canonical video URL",
    }


def write_splits(output, split_by_id):
    split_dir = output / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        ids = sorted(claim_id for claim_id, name in split_by_id.items() if name == split)
        (split_dir / (split + ".txt")).write_text("\n".join(ids) + "\n", encoding="utf-8")


def choose_pilot(records, split_by_id, seed, per_rating):
    candidates = defaultdict(lambda: defaultdict(list))
    for record in records:
        claim_id = record["claim_id"]
        if split_by_id[claim_id] != "train":
            continue
        data = record["data"]
        rating = str(data.get("rating") or "unknown")
        video = data.get("video_information") or {}
        platform = normalize_platform(video.get("platform"))
        candidates[rating][platform].append(claim_id)

    selected = []
    for rating in sorted(candidates):
        by_platform = candidates[rating]
        platforms = sorted(by_platform)
        for ids in by_platform.values():
            ids.sort(key=lambda claim_id: hashlib.sha256(
                (str(seed) + ":pilot:" + claim_id).encode("utf-8")
            ).hexdigest())

        rating_selection = []
        while len(rating_selection) < per_rating:
            added = False
            for platform in platforms:
                if by_platform[platform] and len(rating_selection) < per_rating:
                    rating_selection.append(by_platform[platform].pop(0))
                    added = True
            if not added:
                break
        selected.extend(rating_selection)
    return sorted(selected)


def check_output(records, output, split_by_id, group_by_id):
    expected_ids = {record["claim_id"] for record in records}
    annotation_dir = output / "annotations"
    actual_ids = {path.stem for path in annotation_dir.glob("*.json")}
    errors = []

    if actual_ids != expected_ids:
        errors.append("Generated annotation IDs do not match source IDs")

    listed_ids = []
    for split in ("train", "validation", "test"):
        path = output / "splits" / (split + ".txt")
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        listed_ids.extend(ids)
        if any(split_by_id.get(claim_id) != split for claim_id in ids):
            errors.append("Incorrect ID in " + path.name)
    if len(listed_ids) != len(set(listed_ids)) or set(listed_ids) != expected_ids:
        errors.append("Split files must contain every ID exactly once")

    group_splits = defaultdict(set)
    for claim_id, group in group_by_id.items():
        group_splits[group].add(split_by_id[claim_id])
    if any(len(splits) != 1 for splits in group_splits.values()):
        errors.append("A canonical video group crosses splits")

    if errors:
        raise ValueError("; ".join(errors))


def build(source, output, seed=2026, pilot_per_rating=15, reset_annotations=False):
    records = load_records(source)
    annotation_dir = output / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    group_by_id = {}
    split_by_id = {}
    for record in records:
        video = record["data"].get("video_information") or {}
        group = canonical_video_key(video.get("video_url"), record["claim_id"])
        group_by_id[record["claim_id"]] = group
        split_by_id[record["claim_id"]] = choose_split(group, seed)
        annotation_path = annotation_dir / (record["claim_id"] + ".json")
        if reset_annotations or not annotation_path.exists():
            write_json(annotation_path, make_sidecar(record, source))

    write_splits(output, split_by_id)
    pilot_ids = choose_pilot(records, split_by_id, seed, pilot_per_rating)
    (output / "pilot.txt").write_text("\n".join(pilot_ids) + "\n", encoding="utf-8")
    report = make_report(records, split_by_id, group_by_id)
    report["split_seed"] = seed
    report["pilot_samples"] = len(pilot_ids)
    report["pilot_per_rating"] = pilot_per_rating
    pilot_set = set(pilot_ids)
    pilot_records = [record for record in records if record["claim_id"] in pilot_set]
    report["pilot_ratings"] = dict(sorted(Counter(
        str(record["data"].get("rating") or "unknown") for record in pilot_records
    ).items()))
    report["pilot_platforms"] = dict(sorted(Counter(
        normalize_platform((record["data"].get("video_information") or {}).get("platform"))
        for record in pilot_records
    ).items()))
    write_json(output / "build_report.json", report)
    check_output(records, output, split_by_id, group_by_id)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/TRUE_Dataset"))
    parser.add_argument("--output", type=Path, default=Path("data/eTRUE"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pilot-per-rating", type=int, default=15)
    parser.add_argument("--reset-annotations", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    report = build(
        args.source,
        args.output,
        args.seed,
        args.pilot_per_rating,
        args.reset_annotations,
    )
    counts = report["new_split_counts"]
    print("Built {0} sidecars: {1} train, {2} validation, {3} test".format(
        report["samples"], counts.get("train", 0), counts.get("validation", 0), counts.get("test", 0)
    ))
    print("Audit report: " + str(args.output / "build_report.json"))


if __name__ == "__main__":
    main()
