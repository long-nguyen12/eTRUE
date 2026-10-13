"""Verify a claim/media JSON or the TRUE_Dataset test split."""

import argparse
from pathlib import Path

from utils.files import read_json, write_json

from .batch import run_true_test
from .pipeline import check_artifact_path, validate_payload, verify_claim


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="Claim/media JSON")
    inputs.add_argument("--dataset", type=Path, help="TRUE_Dataset root containing test/ and test_output/")
    parser.add_argument("--output", type=Path, help="Single-request result JSON outside existing datasets")
    parser.add_argument("--output-dir", type=Path, help="Batch directory (default: outputs/verification/true_test)")
    parser.add_argument("--limit", type=int, help="Maximum test records (positive integer)")
    parser.add_argument("--prepare-only", action="store_true", help="Write clean test requests without GPU inference or retrieval")
    parser.add_argument("--model-cache", type=Path, default=Path("data/eTRUE/models"))
    args = parser.parse_args(argv)
    if args.dataset:
        if args.output:
            parser.error("Use --output-dir for --dataset")
        try:
            report = run_true_test(args.dataset, output_dir=args.output_dir or Path("outputs/verification/true_test"),
                                   model_cache=args.model_cache, limit=args.limit, prepare_only=args.prepare_only)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        return 1 if report["status"] == "error" else 0
    try:
        output = check_artifact_path(args.output)
        if output == args.input.resolve():
            raise ValueError("Output must not overwrite the input request")
        work_dir = check_artifact_path(output.parent / (output.stem + "_assets"))
    except ValueError as error:
        parser.error(str(error))
    try:
        payload = validate_payload(read_json(args.input), args.input.resolve().parent)
        result = verify_claim(payload, work_dir=work_dir, model_cache=args.model_cache)
    except (OSError, ValueError) as error:
        result = {"status": "error", "verdict": None, "errors": [{"stage": "input", "message": str(error)}]}
    write_json(output, result)
    print("status:", result["status"], "verdict:", result.get("verdict"), "output:", output)
    return 1 if result["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
