"""Report missing source fields in eTRUE result files."""

import argparse
import json
from pathlib import Path


SOURCE_FIELDS = (
    "uploader_name",
    "uploader_profile",
    "original_source_name",
    "source_type",
    "source_is_uploader",
    "source_mismatch",
)


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, (str, list, dict)):
        return len(value) == 0
    return False


def missing_source_stats(results_directory):
    files = sorted(results_directory.glob("*.json"))
    if not files:
        raise ValueError(f"No JSON result files found in {results_directory}")

    missing_by_field = {field: 0 for field in SOURCE_FIELDS}
    records_with_any_missing = 0
    records_with_all_missing = 0
    records_missing_uploader_name_and_profile = 0

    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)

        verification = result.get("verification") or {}
        source = verification.get("source") or {}
        if not isinstance(source, dict):
            source = {}

        missing_fields = [
            field for field in SOURCE_FIELDS if is_missing(source.get(field))
        ]

        for field in missing_fields:
            missing_by_field[field] += 1

        if missing_fields:
            records_with_any_missing += 1
        if len(missing_fields) == len(SOURCE_FIELDS):
            records_with_all_missing += 1
        if "uploader_name" in missing_fields and "uploader_profile" in missing_fields:
            records_missing_uploader_name_and_profile += 1

    return {
        "total_records": len(files),
        "missing_by_field": missing_by_field,
        "records_with_any_missing": records_with_any_missing,
        "records_with_all_missing": records_with_all_missing,
        "records_missing_uploader_name_and_profile": (
            records_missing_uploader_name_and_profile
        ),
    }


def percentage(count, total):
    return 100 * count / total


def print_stats(stats):
    total = stats["total_records"]
    missing_by_field = stats["missing_by_field"]

    print(f"Total result files: {total}")
    print()
    print(f"{'Source field':<24} {'Missing':>8} {'Missing %':>10}")
    print("-" * 44)

    for field in SOURCE_FIELDS:
        missing = missing_by_field[field]
        print(f"{field:<24} {missing:>8} {percentage(missing, total):>9.2f}%")

    missing_values = sum(missing_by_field.values())
    total_values = total * len(SOURCE_FIELDS)

    print()
    print(
        "All source values: "
        f"{missing_values}/{total_values} missing "
        f"({percentage(missing_values, total_values):.2f}%)"
    )
    print(
        "Records with any source field missing: "
        f"{stats['records_with_any_missing']}/{total} "
        f"({percentage(stats['records_with_any_missing'], total):.2f}%)"
    )
    print(
        "Records with all source fields missing: "
        f"{stats['records_with_all_missing']}/{total} "
        f"({percentage(stats['records_with_all_missing'], total):.2f}%)"
    )
    overlap = stats["records_missing_uploader_name_and_profile"]
    print(
        "Missing uploader_name and uploader_profile overlap: "
        f"{overlap}/{total} ({percentage(overlap, total):.2f}%)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute missing-source percentages for eTRUE results."
    )
    parser.add_argument(
        "results_directory",
        nargs="?",
        type=Path,
        default=Path("data/eTRUE/results"),
        help="Folder containing eTRUE result JSON files.",
    )
    args = parser.parse_args()

    try:
        stats = missing_source_stats(args.results_directory)
    except ValueError as error:
        raise SystemExit(error) from error

    print_stats(stats)


if __name__ == "__main__":
    main()
