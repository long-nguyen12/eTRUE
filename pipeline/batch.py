import time
from collections import Counter
from pathlib import Path

from utils.files import now, write_json

from .dataset import list_test_ids, load_record
from .pipeline import verify_claim
from .runtime import QwenRuntime


def run_true_test(
    dataset_root: Path,
    *,
    output_dir: Path,
    model_cache: Path,
    limit=None,
) -> dict:
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    items = list_test_ids(dataset_root, limit)
    started = time.perf_counter()
    report = {
        "dataset": str(dataset_root),
        "processed_records": 0,
        "started_at": now(),
        "records": [],
    }
    try:
        runtime = QwenRuntime(Path(model_cache))
    except Exception as error:
        print(f"Failed to initialize QwenRuntime: {error}", flush=True)
    for index, record in enumerate(items, 1):
        result_file = output_dir / "results" / (record + ".json")
        try:
            payload = load_record(dataset_root, record)
            result = verify_claim(
                payload,
                work_dir=output_dir / "assets" / record,
                model_cache=Path(model_cache),
                runtime=runtime,
            )
        except (OSError, ValueError) as error:
            print(f"Error processing record {record}: {error}", flush=True)

        result["claim_id"] = record
        write_json(result_file, result)
        report["records"].append(
            {
                "claim_id": record,
                "status": result["status"],
                "verdict": result.get("verdict"),
                "result_file": str(result_file),
                "errors": result.get("errors", []),
            }
        )
        report.update(
            processed_records=len(report["records"]),
        )
        write_json(output_dir / "batch_report.json", report)
        print(
            "[{0}/{1}] {2}: {3}, verdict={4}".format(
                index,
                len(items),
                record,
                result["status"],
                result.get("verdict"),
            ),
            flush=True,
        )
    if runtime is not None:
        runtime.unload()
    report.update(
        finished_at=now(), elapsed_seconds=round(time.perf_counter() - started, 3)
    )
    write_json(output_dir / "batch_report.json", report)
    return report
