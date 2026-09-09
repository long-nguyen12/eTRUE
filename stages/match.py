"""Confirm local near-duplicate videos with CLIP and perceptual hashes."""

from build_etrue import normalize_date
from stages import MODELS
from utils.evidence import add_evidence, add_field_evidence, mark_automated, remove_evidence_type
from utils.files import read_json, trim, write_json
from utils.records import frame_paths, sidecar_path


def image_dhash(path):
    from PIL import Image

    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | (pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return value


def hash_distance(first, second):
    return bin(first ^ second).count("1")


def run(records, source, output, threshold=0.95):
    """Keep CLIP candidates only when perceptual hashes also agree."""
    import numpy as np

    clip_cache = output / "cache" / "clip"
    available = []
    frame_embeddings = {}
    frame_names = {}
    frame_hashes = {}
    record_by_id = {record["claim_id"]: record for record in records}
    for record in records:
        path = clip_cache / (record["claim_id"] + ".npz")
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=False)
        embeddings = data["embeddings"].astype("float32")
        if not len(embeddings):
            continue
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or (norms <= np.finfo("float32").eps).any():
            continue
        embeddings = embeddings / norms
        paths = frame_paths(source, record)
        names = [str(value) for value in data["frames"].tolist()]
        if len(paths) != len(embeddings) or len(names) != len(embeddings):
            continue
        claim_id = record["claim_id"]
        available.append(claim_id)
        frame_embeddings[claim_id] = embeddings
        frame_names[claim_id] = names
        frame_hashes[claim_id] = [image_dhash(path) for path in paths]

    if len(available) < 2:
        print("match skipped: fewer than two CLIP records", flush=True)
        return

    means = np.stack([frame_embeddings[claim_id].mean(axis=0) for claim_id in available])
    mean_norms = np.linalg.norm(means, axis=1, keepdims=True)
    means /= np.maximum(mean_norms, np.finfo("float32").eps)
    similarities = means @ means.T
    np.fill_diagonal(similarities, -1)

    for number, claim_id in enumerate(available, 1):
        candidate_count = min(20, len(available) - 1)
        candidate_indexes = np.argpartition(-similarities[number - 1], candidate_count)[:candidate_count]
        matches = []
        current_frames = frame_embeddings[claim_id]
        for candidate_index in candidate_indexes:
            other_id = available[int(candidate_index)]
            pair_scores = current_frames @ frame_embeddings[other_id].T
            maximum = float(pair_scores.max())
            current_coverage = float((pair_scores.max(axis=1) >= 0.90).mean())
            other_coverage = float((pair_scores.max(axis=0) >= 0.90).mean())
            coverage = min(current_coverage, other_coverage)
            if maximum < threshold or coverage < 0.25:
                continue
            hash_distances = [
                hash_distance(first, second)
                for first in frame_hashes[claim_id]
                for second in frame_hashes[other_id]
            ]
            best_hash_distance = min(hash_distances)
            current_hash_coverage = sum(
                min(hash_distance(first, second) for second in frame_hashes[other_id]) <= 12
                for first in frame_hashes[claim_id]
            ) / len(frame_hashes[claim_id])
            other_hash_coverage = sum(
                min(hash_distance(first, second) for first in frame_hashes[claim_id]) <= 12
                for second in frame_hashes[other_id]
            ) / len(frame_hashes[other_id])
            hash_coverage = min(current_hash_coverage, other_hash_coverage)
            if best_hash_distance > 12 or hash_coverage < 0.25:
                continue
            other = record_by_id[other_id]
            other_video = other["data"].get("video_information") or {}
            matching_frames = []
            for current_index, scores in enumerate(pair_scores):
                other_index = int(scores.argmax())
                score = float(scores[other_index])
                if score >= 0.90:
                    matching_frames.append(
                        {
                            "current": frame_names[claim_id][current_index],
                            "candidate": frame_names[other_id][other_index],
                            "similarity": round(min(score, 1.0), 4),
                        }
                    )
            matches.append(
                {
                    "claim_id": other_id,
                    "url": other_video.get("video_url"),
                    "date": normalize_date(other_video.get("video_date")),
                    "similarity": round(min(maximum, 1.0), 4),
                    "frame_coverage": round(coverage, 4),
                    "perceptual_distance": best_hash_distance,
                    "perceptual_coverage": round(hash_coverage, 4),
                    "matching_frames": matching_frames,
                }
            )
        matches.sort(key=lambda item: (-item["similarity"], -item["frame_coverage"]))
        matches = matches[:5]

        sidecar_file = sidecar_path(output, claim_id)
        extra = read_json(sidecar_file)
        remove_evidence_type(extra, "local_visual_match")
        provenance = extra["verification"]["provenance"]
        current_video = record_by_id[claim_id]["data"].get("video_information") or {}
        provenance["provenance_status"] = "unknown"
        provenance["earliest_known_url"] = current_video.get("video_url")
        provenance["earliest_known_date"] = normalize_date(current_video.get("video_date"))
        provenance["previous_context_summary"] = None
        provenance["near_duplicate_matches"] = matches
        evidence_ids = []
        for match in matches:
            evidence_ids.append(
                add_evidence(
                    extra,
                    "local_visual_match",
                    "cache/clip/" + claim_id + ".npz",
                    "Keyframes match local claim {0} (similarity {1}, coverage {2}).".format(
                        match["claim_id"], match["similarity"], match["frame_coverage"]
                    ),
                    model=MODELS["clip"],
                    details=match,
                )
            )
        if evidence_ids:
            add_field_evidence(
                extra,
                "verification.provenance.near_duplicate_matches",
                evidence_ids,
            )

        current_date = extra["verification"]["date"].get("video_upload_date")
        dated = [
            {
                "claim_id": claim_id,
                "date": current_date,
                "url": extra["normalized_video_information"].get("video_url"),
            }
        ] + [match for match in matches if match.get("date")]
        dated = [item for item in dated if item.get("date")]
        if dated:
            earliest = min(dated, key=lambda item: item["date"])
            provenance["earliest_known_url"] = earliest.get("url")
            provenance["earliest_known_date"] = earliest.get("date")
            if (
                earliest["claim_id"] != claim_id
                and current_date
                and earliest["date"] < current_date
            ):
                provenance["provenance_status"] = "earlier version found"
                date_fields = extra["verification"]["date"]
                online_dates = [
                    value
                    for value in (
                        date_fields.get("earliest_online_date"),
                        earliest.get("date"),
                    )
                    if value
                ]
                if online_dates:
                    date_fields["earliest_online_date"] = min(online_dates)
                earlier = record_by_id[earliest["claim_id"]]["data"].get("video_information") or {}
                context = " - ".join(
                    value
                    for value in [
                        earlier.get("video_headline"),
                        trim(earlier.get("video_description"), 500),
                    ]
                    if value
                )
                provenance["previous_context_summary"] = context or None
                for field in (
                    "verification.provenance.provenance_status",
                    "verification.provenance.earliest_known_url",
                    "verification.provenance.earliest_known_date",
                    "verification.provenance.previous_context_summary",
                    "verification.date.earliest_online_date",
                ):
                    add_field_evidence(extra, field, evidence_ids)
        mark_automated(extra, "local_visual_matching", {"match_count": len(matches)})
        write_json(sidecar_file, extra)
        if number % 100 == 0:
            print("match", number, "/", len(available), flush=True)
