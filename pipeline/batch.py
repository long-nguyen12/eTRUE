import time
from collections import Counter
from pathlib import Path

from utils.files import now, write_json

from .dataset import list_test_ids, load_test_request, resolve_dataset_root
from .pipeline import check_artifact_path, verify_claim


def run_true_test(
    dataset_root: Path,
    *,
    output_dir: Path,
    model_cache: Path,
    limit=None,
    prepare_only=False
) -> dict:
    dataset_root = resolve_dataset_root(dataset_root)
    output_dir = check_artifact_path(output_dir)
    identifiers = list_test_ids(dataset_root, limit)
    if not identifiers:
        raise ValueError("No JSON records found in the test directory")
    started = time.perf_counter()
    report = {
        "dataset": str(dataset_root),
        "split": "test",
        "selected_records": len(identifiers),
        "processed_records": 0,
        "mode": "prepare" if prepare_only else "verify",
        "started_at": now(),
        "status": "running",
        "records": [],
        "status_counts": {},
        "verdict_counts": {},
        "stopped_early": False,
    }
    status_counts, verdict_counts = Counter(), Counter()
    for index, identifier in enumerate(identifiers, 1):
        request_file = output_dir / "requests" / (identifier + ".json")
        result_file = output_dir / "results" / (identifier + ".json")
        request_written = False
        try:
            payload = load_test_request(dataset_root, identifier)
            write_json(request_file, payload)
            request_written = True
            if prepare_only:
                result = {"status": "prepared", "verdict": None, "errors": []}
            else:
                result = verify_claim(
                    payload,
                    work_dir=output_dir / "assets" / identifier,
                    model_cache=Path(model_cache),
                )
        except (OSError, ValueError) as error:
            result = {
                "status": "error",
                "verdict": None,
                "errors": [{"stage": "input", "message": str(error)}],
            }
        if not prepare_only or result["status"] == "error":
            result["claim_id"] = identifier
            write_json(result_file, result)
        status_counts[result["status"]] += 1
        if result.get("verdict"):
            verdict_counts[result["verdict"]] += 1
        report["records"].append(
            {
                "claim_id": identifier,
                "status": result["status"],
                "verdict": result.get("verdict"),
                "request_file": str(request_file) if request_written else None,
                "result_file": (
                    str(result_file)
                    if not prepare_only or result["status"] == "error"
                    else None
                ),
                "errors": result.get("errors", []),
            }
        )
        report.update(
            processed_records=len(report["records"]),
            status_counts=dict(status_counts),
            verdict_counts=dict(verdict_counts),
        )
        # A failed CUDA/model setup will fail every request; leave remaining IDs unprocessed.
        if result["status"] == "error" and any(
            e.get("stage") == "model" for e in result.get("errors", [])
        ):
            report["stopped_early"] = True
        write_json(output_dir / "batch_report.json", report)
        print(
            "[{0}/{1}] {2}: {3}, verdict={4}".format(
                index,
                len(identifiers),
                identifier,
                result["status"],
                result.get("verdict"),
            ),
            flush=True,
        )
        if report["stopped_early"]:
            break
    error_count = status_counts.get("error", 0)
    report["status"] = (
        "error"
        if report["stopped_early"] or error_count == report["processed_records"]
        else "partial" if error_count or status_counts.get("partial") else "ok"
    )
    report.update(
        finished_at=now(), elapsed_seconds=round(time.perf_counter() - started, 3)
    )
    write_json(output_dir / "batch_report.json", report)
    return report
