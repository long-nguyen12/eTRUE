"""Run verification directly on the TRUE_Dataset test directory."""

import argparse
from pathlib import Path

from .batch import run_true_test


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/TRUE_Dataset/test"),
        help="Test directory or TRUE_Dataset root (default: data/TRUE_Dataset/test)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/verification/true_test"),
        help="Batch directory (default: outputs/verification/true_test)",
    )
    parser.add_argument(
        "--limit", type=int, help="Maximum test records (positive integer)"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write clean test requests without GPU inference or retrieval",
    )
    parser.add_argument("--model-cache", type=Path, default=Path("data/eTRUE/models"))
    args = parser.parse_args(argv)
    try:
        report = run_true_test(
            args.dataset,
            output_dir=args.output_dir,
            model_cache=args.model_cache,
            limit=args.limit,
            prepare_only=args.prepare_only,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 1 if report["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
