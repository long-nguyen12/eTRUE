"""Generate the dataset analysis used in the eTRUE dataset report."""

import argparse
from collections import Counter
from pathlib import Path

from utils.files import read_json
from utils.schema import VERIFICATION_DEFAULTS


NEWS_PLATFORMS = {"nytimes", "foxnews", "nbcnews", "cnbc", "today"}
PLATFORM_LABELS = {
    "x": "X (Twitter)",
    "youtube": "YouTube",
    "tiktok": "TikTok",
    "facebook": "Facebook",
    "instagram": "Instagram",
}


def is_missing(value):
    return value is None or isinstance(value, (str, list, dict)) and not value


def percentage(count, total):
    return 100 * count / total if total else 0.0


def field_value(record, pillar, field):
    verification = record.get("verification") or {}
    return (verification.get(pillar) or {}).get(field)


def platform_group(platform):
    platform = str(platform or "").strip().lower()
    if platform in PLATFORM_LABELS:
        return PLATFORM_LABELS[platform]
    if platform in NEWS_PLATFORMS:
        return "News sites (NYT, Fox, NBC, CNBC, Today)"
    return "Others"


def comparison_note(records, pillar, field, video_field):
    matches = sum(
        field_value(record, pillar, field)
        == (record.get("video_information") or {}).get(video_field)
        and (record.get("video_information") or {}).get(video_field) is not None
        for record in records
    )
    return (
        f"Matches {video_field} in {matches}/{len(records)} records "
        f"({percentage(matches, len(records)):.1f}%)"
    )


def unknown_note(records, pillar, field):
    unknown = sum(
        isinstance(value := field_value(record, pillar, field), str)
        and value.strip().lower() == "unknown"
        for record in records
    )
    if not unknown:
        return ""
    return f'"unknown" in {unknown}/{len(records)} records ({percentage(unknown, len(records)):.1f}%)'


def analyze(records):
    total = len(records)
    fields = {}
    pillar_fill_rate = {}

    for pillar, defaults in VERIFICATION_DEFAULTS.items():
        fields[pillar] = {}
        filled_in_pillar = 0
        for field in defaults:
            filled = sum(
                not is_missing(field_value(record, pillar, field))
                for record in records
            )
            fields[pillar][field] = {
                "filled": filled,
                "missing": total - filled,
                "coverage": percentage(filled, total),
            }
            filled_in_pillar += filled
        pillar_fill_rate[pillar] = percentage(
            filled_in_pillar, total * len(defaults)
        )

    source_fields = tuple(VERIFICATION_DEFAULTS["source"])
    source_missing = {
        "all_fields": sum(
            all(is_missing(field_value(record, "source", field)) for field in source_fields)
            for record in records
        ),
        "uploader_name": sum(
            is_missing(field_value(record, "source", "uploader_name"))
            for record in records
        ),
        "uploader_profile": sum(
            is_missing(field_value(record, "source", "uploader_profile"))
            for record in records
        ),
        "uploader_name_and_profile": sum(
            is_missing(field_value(record, "source", "uploader_name"))
            and is_missing(field_value(record, "source", "uploader_profile"))
            for record in records
        ),
    }

    platforms = Counter(
        platform_group((record.get("video_information") or {}).get("platform"))
        for record in records
    )
    return {
        "total": total,
        "source_missing": source_missing,
        "fields": fields,
        "pillar_fill_rate": pillar_fill_rate,
        "platforms": platforms,
    }


def field_notes(records):
    return {
        ("provenance", "provenance_status"): unknown_note(
            records, "provenance", "provenance_status"
        ),
        ("provenance", "earliest_known_url"): comparison_note(
            records, "provenance", "earliest_known_url", "video_url"
        ),
        ("provenance", "earliest_known_date"): comparison_note(
            records, "provenance", "earliest_known_date", "video_date"
        ),
        ("source", "source_type"): unknown_note(records, "source", "source_type"),
        ("date", "video_upload_date"): comparison_note(
            records, "date", "video_upload_date", "video_date"
        ),
        ("motivation", "original_context_category"): unknown_note(
            records, "motivation", "original_context_category"
        ),
    }


def ratio(count, total):
    return f"{count}/{total} ({percentage(count, total):.2f}%)"


def render_report(stats, records, results_directory):
    total = stats["total"]
    source = stats["source_missing"]
    notes = field_notes(records)
    lines = [
        "# eTRUE Dataset Analysis",
        "",
        f"Results directory: `{results_directory.as_posix()}`",
        "",
        "## Dataset detail",
        "",
        f"- Total result files: {total}",
        f"- Records with all source fields missing: {ratio(source['all_fields'], total)}",
        f"- Records with missing uploader name: {ratio(source['uploader_name'], total)}",
        f"- Records with missing uploader profile: {ratio(source['uploader_profile'], total)}",
        "- Missing uploader name/profile overlap: "
        f"{ratio(source['uploader_name_and_profile'], total)}",
        "",
        "A field is filled when it is not null or empty. Values such as `unknown` count as filled.",
        "",
        "## Pillar fill rates",
        "",
        "| Pillar | Average fill rate |",
        "| --- | ---: |",
    ]
    for pillar, fill_rate in stats["pillar_fill_rate"].items():
        lines.append(f"| {pillar.title()} | {fill_rate:.1f}% |")

    for pillar, fields in stats["fields"].items():
        lines.extend(
            [
                "",
                f"### {pillar.title()}",
                "",
                "| Field | Filled | Coverage | Note |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for field, values in fields.items():
            lines.append(
                f"| `{field}` | {values['filled']}/{total} | "
                f"{values['coverage']:.1f}% | {notes.get((pillar, field), '')} |"
            )

    lines.extend(
        [
            "",
            "## Source platform",
            "",
            "| Platform | Records | Percent |",
            "| --- | ---: | ---: |",
        ]
    )
    for platform, count in stats["platforms"].most_common():
        lines.append(f"| {platform} | {count} | {percentage(count, total):.1f}% |")

    lines.extend(
        [
            "",
            "## Analysis limitations",
            "",
            "Collection errors and tool/model usage are not stored in the public result JSON files, "
            "so they cannot be measured from this directory. Those sections require annotation "
            "sidecars or execution logs.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_directory",
        nargs="?",
        type=Path,
        default=Path("data/eTRUE/results"),
    )
    parser.add_argument("--output", type=Path, help="Write the Markdown report here.")
    args = parser.parse_args()

    paths = sorted(args.results_directory.glob("*.json"))
    if not paths:
        raise SystemExit(f"No JSON result files found in {args.results_directory}")

    records = [read_json(path) for path in paths]
    report = render_report(analyze(records), records, args.results_directory)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
