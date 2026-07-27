"""Run the eTRUE enrichment pipeline.

Each pipeline stage lives in its matching file under ``stages/``. Pillar rules
live in ``pillars/`` and reusable helpers live in ``utils/``. Records remain
ordinary Python dictionaries; there is no schema framework or hidden model.
"""

import argparse
import os
from pathlib import Path

from stages import PIPELINE_STAGES
from stages.audit import run as run_audit
from stages.bootstrap import run as run_bootstrap
from stages.clip import run as run_clip
from stages.geocode import run as run_geocode
from stages.location import run as run_location
from stages.match import run as run_matches
from stages.search import run as run_search
from stages.text import run as run_text
from stages.vision import run as run_vision
from stages.web import run as run_web
from utils.records import selected_records


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=PIPELINE_STAGES + ("all",))
    parser.add_argument("--source", type=Path, default=Path("data/TRUE_Dataset"))
    parser.add_argument("--output", type=Path, default=Path("data/eTRUE"))
    parser.add_argument("--ids", default="data/eTRUE/pilot.txt", help="ID file or 'all'")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of records to process")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Load Hugging Face models from cache only")
    parser.add_argument("--clip-threshold", type=float, default=0.95)
    return parser.parse_args()


def run_stage(stage, args, records, model_cache):
    """Call one pipeline stage with the arguments it needs."""
    print("starting", stage, "for", len(records), "records", flush=True)
    if stage == "bootstrap":
        run_bootstrap(records, args.source, args.output, args.force)
    elif stage == "clip":
        run_clip(records, args.source, args.output, model_cache, args.force, args.offline)
    elif stage == "match":
        run_matches(records, args.source, args.output, args.clip_threshold)
    elif stage == "vision":
        run_vision(records, args.source, args.output, model_cache, args.force, args.offline)
    elif stage == "web":
        run_web(records, args.output, args.force)
    elif stage == "search":
        run_search(records, args.source, args.output, args.force)
    elif stage == "text":
        run_text(records, args.output, model_cache, args.force, args.offline)
    elif stage == "location":
        run_location(records, args.output, model_cache, args.force, args.offline)
    elif stage == "geocode":
        run_geocode(records, args.output, args.force)
    elif stage == "audit":
        run_audit(records, args.output)


def main():
    args = parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    records = selected_records(args.source, args.ids, args.limit)
    if not records:
        raise SystemExit("No records selected")

    model_cache = args.output / "models"
    model_cache.mkdir(parents=True, exist_ok=True)
    stage_names = PIPELINE_STAGES if args.stage == "all" else (args.stage,)
    for stage in stage_names:
        run_stage(stage, args, records, model_cache)


if __name__ == "__main__":
    main()
