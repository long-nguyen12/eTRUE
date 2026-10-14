Independent eTRUE verification
==============================

This module accepts a claim/media request or the TRUE_Dataset test split, retrieves independent evidence,
runs five pillar agents with the existing pillar rules, and returns JSON
decisions with quoted evidence. It does not import the website, run enrichment, or write dataset
annotations/results. There is no audio analysis or ASR; a supplied transcript
is optional evidence, not independent corroboration.

Server setup
------------

Run from the repository root with Python 3.10 or newer. Install the existing
requirements into the server environment and install FFmpeg/ffprobe on PATH::

    python -m pip install -r requirements.txt
    python -c "import torch; print(torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
    ffmpeg -version
    ffprobe -version

Inference uses Qwen/Qwen3.5-4B, FP16, entirely on cuda:0. There is no CPU fallback
or offload. Text and vision run sequentially using one multimodal model, retained
across claims in a batch and released when the batch ends. The default cache is
data/eTRUE/models; use --model-cache to select a server cache directory.
Missing weights are downloaded through Transformers. The module does not load
Whisper, even if the repository requirements installed faster-whisper.

Set BRAVE_SEARCH_API_KEY and GOOGLE_CLOUD_VISION_API_KEY in the environment or
the repository .env. Without a Brave key, text search uses DDGS. Without a
Google Vision key, reverse-image retrieval is recorded as skipped and the run
is partial. Configured provider failures are reported rather than silently
switching providers. Social search uses web queries scoped to platform sites.

CLI
---

Run the pipeline directly on the existing test folder; no request JSON needs
to be entered or created manually::

    python -m pipeline
    python -m pipeline --limit 5
    python -m pipeline --model-cache /srv/models/etrue

The default dataset folder is data/TRUE_Dataset/test and the default output
directory is outputs/verification/true_test. The first command processes the
full test split; --limit 5 selects the first five records for a smoke run.
Use --dataset to select another test folder or a TRUE_Dataset root.

Python API
----------

Single-request verification remains available through verify_claim. Accepted
top-level fields are claim, video_path, keyframes, post_url, metadata, and
transcript. A non-empty claim and at least one media input are required.
Unknown fields, including rating, content and verification, are rejected.
Metadata accepts title, description, platform, uploader_name, uploader_profile,
and upload_date, with string or null values.

Paths in Python API requests are relative to the process working directory::

    from pathlib import Path
    from pipeline import verify_claim

    result = verify_claim(
        {"claim": "This video records a march.", "video_path": "/srv/input/clip.mp4"},
        work_dir=Path("/srv/output/request_assets"),
        model_cache=Path("/srv/models/etrue"),
    )

Supplied keyframes take priority over local video, then video downloaded from
post_url. At most four frames are analyzed and three used for reverse-image
retrieval. Downloaded video and extracted frames persist under work_dir;
supplied media files are only read. Output/work directories inside the existing
eTRUE or TRUE_Dataset trees are rejected. Invalid input raises ValueError.
Runtime failures return status=error and verdict=null.

verify_claim accepts an optional runtime=QwenRuntime(...) for repeated calls.
Without it, the function creates and releases its own runtime. With a supplied
runtime, the caller owns it and must call runtime.unload() when finished.

TRUE_Dataset test split
----------------------

Run the existing test split on the server without creating requests manually::

    python -m pipeline --dataset data/TRUE_Dataset/test --limit 5
    python -m pipeline --dataset data/TRUE_Dataset/test

The first command is a five-record smoke run; omit --limit to process the full
test split. --dataset also accepts the dataset root, data/TRUE_Dataset.
The reader preserves test_set.txt order when present, otherwise
uses sorted test/*.json files. It reads only claim and original video fields,
never the article URL/content, rating, rationales, evidence or annotations.
Existing test_output/<ID> frames take priority. Records without frames use
test_video/<ID>.mp4 or a video inside test_video/<ID>, then the video URL.
Input videos are assumed to come from non-fact-checking sources. Their original
metadata and supplied transcripts are preserved as unverified input, without ASR.
Fact-check source filtering applies to retrieved web evidence.

Outputs are requests/<ID>.json, results/<ID>.json, assets/<ID>/ and
batch_report.json under --output-dir. The report is updated after every record.
Every processed record has a result file, including ok, partial and error results.
Record errors do not stop other records; a CUDA/model setup failure stops the
batch immediately and leaves remaining records unprocessed. Run summaries
report processed counts and verdict counts, not accuracy or gold labels.
Output directories inside the source dataset are rejected.

To inspect/prepare inputs without loading Qwen or contacting search services::

    python -m pipeline --prepare-only

Preparation writes clean requests and a report, without predicted results.
Run the regular command on the server to perform actual verification.

The Python batch API is::

    from pipeline import run_true_test

    report = run_true_test(
        Path("data/TRUE_Dataset/test"),
        output_dir=Path("outputs/verification/true_test"),
        model_cache=Path("data/eTRUE/models"),
        limit=5,
    )

Results and evidence
--------------------

Provenance, Source, Date, Location and Motivation each have a separate agent
prompt. They assess each claim component independently using a shared frozen
evidence store; no pillar agent receives another agent's conclusions. The
Provenance agent also assesses the core event. Existing pillar rules and quote
validators check each agent's output before synthesis.

The agents are logical roles within the local pipeline, without an agent
framework or extra models. They and the final aggregator reuse one loaded
Qwen text model and run sequentially on the GPU. The aggregator receives the
validated five-agent results and their evidence to produce the component and
overall verdicts and explanations.

Every claim component triggers all five pillar-agent calls, including pillars
that return not_applicable. After all components, verdict_agent combines their
validated results in a separate call. A malformed pillar response is recorded
as an error without skipping the remaining pillar agents.

The aggregator receives compact findings and their accepted quotes, reviews every
supported/refuted assessment, and may retain it or downgrade it to unknown. It cannot
promote an unknown assessment or reverse a finding. Python recomputes the verdicts
and attaches the accepted citations without asking the model to copy them again.
Invalid aggregation output gets one correction attempt with validation feedback.
Component results are updated only after the entire response passes validation.

Results contain the original/normalized claim, claim_components,
component_results, overall verdict, explanation, open_questions, citations,
evidence, evidence_relations, trace, errors, media references and model_calls.
Each component keeps the existing verification field structure plus a core
assessment and pillar assessments. Applicable assessments are supported,
refuted or unknown; unasserted dimensions are not_applicable. Component text
quotes an unchanged source clause; inherited subjects live in structured fields.
Reviewed assessments identify verdict_agent in reviewed_by. Citations retained on
unknown assessments have unresolved relations, rather than decisive supports/refutes.

model_calls records the agent, component, prompt/output token counts, output limit,
whether generation hit that limit, and elapsed time. Failed model calls also retain
the error and raw response when available, including JSON and validation failures.
Diagnostics are reset per claim even when the loaded model is reused across a batch.

Every explanation field, including overall, component and pillar explanations,
is a plain prose string: no Markdown, HTML, JSON, URLs or evidence IDs.
Verdicts, citations, evidence and trace remain separate structured JSON fields.

True requires the core event and all applicable pillars to be supported.
False requires core-event refutation. Misleading requires a supported core
event and a refuted context pillar. Unverified represents missing or conflicting
evidence. Overall decisions preserve all component decisions; they do not vote
across pillars. Invalid verdict/citation output is a technical error.

Evidence's verified flag means its document was acquired, not that its claim
is true. Snippets, reverse-image matches, user metadata/transcripts, and model
observations are candidates. Decisive assertions need acquired relevant pages
and exact quotes; geocoding only normalizes an independently grounded place.
Both supporting and refuting evidence survive unresolved conflicts.

Known fact-check domains, redirects/canonical URLs, ClaimReview markup and
visible fact-check/rating signals are excluded before evidence reaches prompts.
Screening is heuristic: unidentified mirrors can still escape detection.
Quote validation checks provenance and extraction constraints; semantic
relevance/entailment still depends on Qwen and warrants review of the evidence.

Text prompts reserve generated tokens within an 8192-token budget. Whole
evidence items are selected without truncating the claim. metrics records
which evidence was sent/omitted per prompt; an oversized non-evidence input
is an explicit error. Relative dates without reference context and capture
dates less precise than the claim cannot receive a supported date assessment.

A6000 smoke test
----------------

On the A6000 server, run the CLI on representative actual media and review the
result JSON. Acceptance requires valid cited decisions, no out-of-memory
failure, media under the assets directory, and unchanged dataset files. Check
metrics.elapsed_seconds and the explicit PyTorch peak allocated/reserved
VRAM fields. These are process allocator measurements, not total GPU usage
from other processes. The smoke test is not an accuracy benchmark.

