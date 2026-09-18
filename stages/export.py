"""Export compact reader-facing results from enrichment sidecars."""

from utils.files import read_json
from utils.records import sidecar_path
from utils.results import write_readable_result


def run(records, output):
    for record in records:
        extra = read_json(sidecar_path(output, record["claim_id"]))
        write_readable_result(extra, output)

    print("exported", len(records), "readable results", flush=True)
