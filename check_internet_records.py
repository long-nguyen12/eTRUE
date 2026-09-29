"""Check public video availability and metadata coverage for TRUE records."""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

from utils.files import now, write_json
from utils.media_metadata import (
    CORE_METADATA_FIELDS,
    cookie_file_from_env,
    extract_media,
)
from utils.records import canonical_video_key, load_records, normalize_platform
from utils.utils import normalize_social_url


REASON_LABELS = {
    "missing_url": "Missing video URL",
    "invalid_url": "Invalid URL",
    "no_video_format": "No video stream exposed by yt-dlp",
    "probe_failed": "yt-dlp could not read the URL",
    "all_equivalent_urls_failed": "All equivalent URLs failed",
    "probe_inconclusive": "Video availability is inconclusive",
    "all_equivalent_urls_inconclusive": "All equivalent URLs are inconclusive",
}


def probe_video(url, session=None, cookie_file=None):
    import requests

    owns_session = session is None
    session = session or requests.Session()
    try:
        extraction = extract_media(url, session, cookie_file)
    finally:
        if owns_session:
            session.close()
    return {
        **extraction["video"],
        "metadata_status": extraction["metadata_status"],
        "metadata": extraction["metadata"],
        "metadata_field_sources": extraction["field_sources"],
        "missing_metadata_fields": extraction["missing_fields"],
        "metadata_errors": extraction["errors"],
    }


def _metadata_count(result):
    metadata = result.get("metadata") or {}
    return sum(bool(metadata.get(field)) for field in CORE_METADATA_FIELDS)


def check_group(records, session=None, cookie_file=None):
    urls = list(
        dict.fromkeys(
            str(
                (record["data"].get("video_information") or {}).get("video_url")
                or ""
            ).strip()
            for record in records
        )
    )
    checks = []
    for url in urls:
        result = probe_video(url, session, cookie_file)
        checks.append({"url": url or None, **result})
        if result["decision"] == "keep" and result["metadata_status"] == "complete":
            break

    best = max(checks, key=_metadata_count)
    kept = next((check for check in checks if check["decision"] == "keep"), None)
    if kept:
        decision = "keep"
        reason = kept["reason"]
        successful_url = kept["url"]
    elif any(check["decision"] == "unknown" for check in checks):
        decision = "unknown"
        reason = checks[0]["reason"] if len(checks) == 1 else "all_equivalent_urls_inconclusive"
        successful_url = None
    else:
        decision = "discard"
        reason = checks[0]["reason"] if len(checks) == 1 else "all_equivalent_urls_failed"
        successful_url = None

    return {
        "decision": decision,
        "reason": reason,
        "successful_url": successful_url,
        "metadata_status": best["metadata_status"],
        "metadata": best["metadata"],
        "metadata_field_sources": best["metadata_field_sources"],
        "missing_metadata_fields": best["missing_metadata_fields"],
        "metadata_url": best["url"],
        "checks": checks,
    }


def breakdown(rows, field):
    values = defaultdict(Counter)
    for row in rows:
        values[row[field]]["total"] += 1
        values[row[field]][row["decision"]] += 1
    return {
        name: {
            "total": counts["total"],
            "kept": counts["keep"],
            "discarded": counts["discard"],
            "unknown": counts["unknown"],
        }
        for name, counts in sorted(values.items())
    }


def metadata_breakdown(rows, field):
    values = defaultdict(Counter)
    for row in rows:
        counts = values[row[field]]
        counts["total"] += 1
        counts[row["metadata_status"]] += 1
        for metadata_field in CORE_METADATA_FIELDS:
            if (row.get("metadata") or {}).get(metadata_field):
                counts[metadata_field] += 1
    return {
        name: {
            "total": counts["total"],
            "complete": counts["complete"],
            "partial": counts["partial"],
            "missing": counts["missing"],
            "field_coverage": {
                metadata_field: counts[metadata_field]
                for metadata_field in CORE_METADATA_FIELDS
            },
        }
        for name, counts in sorted(values.items())
    }


def analyze(source, session=None, cookie_file=None):
    import requests

    records = load_records(source)
    groups = defaultdict(list)
    for record in records:
        video = record["data"].get("video_information") or {}
        video["video_url"] = normalize_social_url(video.get("video_url"))["url"]
        key = canonical_video_key(video.get("video_url"), record["claim_id"])
        groups[key].append(record)

    owns_session = session is None
    session = session or requests.Session()
    session.headers["User-Agent"] = "eTRUE-research-dataset/1.0"
    cookie_file = cookie_file if cookie_file is not None else cookie_file_from_env()
    group_results = {}
    try:
        for number, (key, group) in enumerate(sorted(groups.items()), 1):
            claim_ids = sorted(record["claim_id"] for record in group)
            print(
                f"Checking record {number}/{len(groups)}: {', '.join(claim_ids)}",
                flush=True,
            )
            result = check_group(group, session, cookie_file)
            group_results[key] = {"claim_ids": claim_ids, **result}
            for check in result["checks"]:
                if check["error"]:
                    print(
                        f"  {', '.join(claim_ids)} ({check['url'] or 'missing URL'}): "
                        f"{check['error']}",
                        flush=True,
                    )
    finally:
        if owns_session:
            session.close()

    rows = []
    for record in records:
        video = record["data"].get("video_information") or {}
        key = canonical_video_key(video.get("video_url"), record["claim_id"])
        result = group_results[key]
        rows.append(
            {
                "claim_id": record["claim_id"],
                "split": record["legacy_split"],
                "platform": normalize_platform(video.get("platform")),
                "video_url": video.get("video_url"),
                "canonical_key": key,
                "decision": result["decision"],
                "reason": result["reason"],
                "metadata_status": result["metadata_status"],
                "metadata": result["metadata"],
                "metadata_field_sources": result["metadata_field_sources"],
                "missing_metadata_fields": result["missing_metadata_fields"],
            }
        )
    rows.sort(key=lambda row: row["claim_id"])

    kept_ids = [row["claim_id"] for row in rows if row["decision"] == "keep"]
    discarded_ids = [row["claim_id"] for row in rows if row["decision"] == "discard"]
    unknown_ids = [row["claim_id"] for row in rows if row["decision"] == "unknown"]
    all_ids = {record["claim_id"] for record in records}
    assert len(rows) == len(kept_ids) + len(discarded_ids) + len(unknown_ids)
    assert set(kept_ids).isdisjoint(discarded_ids)
    assert set(kept_ids).isdisjoint(unknown_ids)
    assert set(discarded_ids).isdisjoint(unknown_ids)
    assert set(kept_ids).union(discarded_ids, unknown_ids) == all_ids

    total = len(rows)
    status_counts = Counter(row["metadata_status"] for row in rows)
    summary = {
        "total_records": total,
        "kept_records": len(kept_ids),
        "discarded_records": len(discarded_ids),
        "unknown_records": len(unknown_ids),
        "kept_percentage": round(100 * len(kept_ids) / total, 2) if total else 0,
        "discarded_percentage": round(100 * len(discarded_ids) / total, 2) if total else 0,
        "unknown_percentage": round(100 * len(unknown_ids) / total, 2) if total else 0,
    }
    metadata_summary = {
        "complete": status_counts["complete"],
        "partial": status_counts["partial"],
        "missing": status_counts["missing"],
        "field_coverage": {
            field: sum(bool((row.get("metadata") or {}).get(field)) for row in rows)
            for field in CORE_METADATA_FIELDS
        },
    }
    return (
        {
            "generated_at": now(),
            "summary": summary,
            "metadata_summary": metadata_summary,
            "discard_reasons": dict(
                sorted(Counter(row["reason"] for row in rows if row["decision"] == "discard").items())
            ),
            "unknown_reasons": dict(
                sorted(Counter(row["reason"] for row in rows if row["decision"] == "unknown").items())
            ),
            "by_platform": breakdown(rows, "platform"),
            "by_split": breakdown(rows, "split"),
            "metadata_by_platform": metadata_breakdown(rows, "platform"),
            "groups": group_results,
            "records": rows,
        },
        kept_ids,
        discarded_ids,
        unknown_ids,
    )


def markdown_cell(value):
    return str(value or "").replace("\n", " ").replace("|", "\\|")


def render_report(result):
    summary = result["summary"]
    metadata = result["metadata_summary"]
    lines = [
        "# Internet video and metadata extraction report",
        "",
        f"- Total records: **{summary['total_records']}**",
        f"- Video available: **{summary['kept_records']}** ({summary['kept_percentage']:.2f}%)",
        f"- Video unavailable: **{summary['discarded_records']}** ({summary['discarded_percentage']:.2f}%)",
        f"- Video inconclusive: **{summary['unknown_records']}** ({summary['unknown_percentage']:.2f}%)",
        f"- Metadata complete: **{metadata['complete']}**",
        f"- Metadata partial: **{metadata['partial']}**",
        f"- Metadata missing: **{metadata['missing']}**",
        "",
        "## Metadata field coverage",
        "",
        "| Field | Records |",
        "| --- | ---: |",
    ]
    for field, count in metadata["field_coverage"].items():
        lines.append(f"| {markdown_cell(field)} | {count} |")

    for title, reasons in (
        ("Unavailable reasons", result["discard_reasons"]),
        ("Inconclusive reasons", result["unknown_reasons"]),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Reason | Records |",
                "| --- | ---: |",
            ]
        )
        if reasons:
            for reason, count in reasons.items():
                lines.append(f"| {REASON_LABELS.get(reason, reason)} (`{reason}`) | {count} |")
        else:
            lines.append("| None | 0 |")

    lines.extend(
        [
            "",
            "## Metadata coverage by platform",
            "",
            "| Platform | Total | Complete | Partial | Missing | Title | Description | Uploader | Profile | Date |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for platform, counts in result["metadata_by_platform"].items():
        fields = counts["field_coverage"]
        lines.append(
            f"| {markdown_cell(platform)} | {counts['total']} | {counts['complete']} | "
            f"{counts['partial']} | {counts['missing']} | {fields['title']} | "
            f"{fields['description']} | {fields['uploader_name']} | "
            f"{fields['uploader_profile']} | {fields['upload_date']} |"
        )

    for title, values in (("Platform", result["by_platform"]), ("Split", result["by_split"])):
        lines.extend(
            [
                "",
                f"## Video status by {title.lower()}",
                "",
                f"| {title} | Total | Available | Unavailable | Inconclusive |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, counts in values.items():
            lines.append(
                f"| {markdown_cell(name)} | {counts['total']} | {counts['kept']} | "
                f"{counts['discarded']} | {counts['unknown']} |"
            )

    for decision, title in (("discard", "Unavailable records"), ("unknown", "Inconclusive records")):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Claim ID | Split | Platform | URL | Reason | Metadata |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in result["records"]:
            if row["decision"] == decision:
                lines.append(
                    f"| {markdown_cell(row['claim_id'])} | {markdown_cell(row['split'])} | "
                    f"{markdown_cell(row['platform'])} | {markdown_cell(row['video_url'])} | "
                    f"{REASON_LABELS.get(row['reason'], row['reason'])} | {row['metadata_status']} |"
                )
    return "\n".join(lines) + "\n"


def write_outputs(output, result, kept_ids, discarded_ids, unknown_ids):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "internet_extraction.json", result)
    (output / "kept_ids.txt").write_text(
        "\n".join(kept_ids) + ("\n" if kept_ids else ""), encoding="utf-8"
    )
    (output / "discarded_ids.txt").write_text(
        "\n".join(discarded_ids) + ("\n" if discarded_ids else ""), encoding="utf-8"
    )
    (output / "unknown_ids.txt").write_text(
        "\n".join(unknown_ids) + ("\n" if unknown_ids else ""), encoding="utf-8"
    )
    (output / "internet_extraction_report.md").write_text(
        render_report(result), encoding="utf-8"
    )


def main():
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/TRUE_Dataset"))
    parser.add_argument("--output", type=Path, default=Path("data/eTRUE_filtered"))
    args = parser.parse_args()
    load_dotenv()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    result, kept_ids, discarded_ids, unknown_ids = analyze(args.source)
    write_outputs(args.output, result, kept_ids, discarded_ids, unknown_ids)
    summary = result["summary"]
    metadata = result["metadata_summary"]
    print(f"Checked: {summary['total_records']} records")
    print(f"Video available: {summary['kept_records']} ({summary['kept_percentage']:.2f}%)")
    print(f"Video unavailable: {summary['discarded_records']} ({summary['discarded_percentage']:.2f}%)")
    print(f"Video inconclusive: {summary['unknown_records']} ({summary['unknown_percentage']:.2f}%)")
    print(
        "Metadata: "
        f"{metadata['complete']} complete, {metadata['partial']} partial, "
        f"{metadata['missing']} missing"
    )


if __name__ == "__main__":
    main()
