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
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/verification/true_test"),
    )
    parser.add_argument(
        "--limit", type=int, help="Maximum test records (positive integer)"
    )
    parser.add_argument("--model-cache", type=Path, default=Path("data/eTRUE/models"))
    args = parser.parse_args(argv)
    try:
        run_true_test(
            args.dataset,
            output_dir=args.output_dir,
            model_cache=args.model_cache,
            limit=args.limit,
        )
    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
