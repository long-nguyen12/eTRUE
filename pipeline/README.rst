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
Runtime failures return status=error and verdict=NEI; the error status indicates
that verification did not complete.

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

Outputs are results/<ID>.json, assets/<ID>/ and
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

Normalization preserves the original claim apart from whitespace. Component text
and dimensions copy source phrases; collective actions and linked assertions stay
together when splitting would change their meaning. Empty or malformed components
are technical errors rather than silently omitted claims.

Each pillar returns a complete JSON object with fields, field_citations and
assessments, all keyed by its own pillar name. Provenance also returns a separate
core_assessment. Each agent receives a response_template containing all defined
fields and dotted field_citations keys. Nested field citations are flattened when
their field names are explicit; ambiguous group citations are discarded. Missing
field_citations defaults to {}, and missing core_assessment remains unknown without
discarding the other provenance findings. Malformed citation entries are dropped
without inventing quotes; other invalid response structures are rejected.
Missing observations remain null or [], while unsupported findings become unknown.

The agents are logical roles within the local pipeline, without an agent
framework or extra models. They and the final aggregator reuse one loaded
Qwen text model and run sequentially on the GPU. The aggregator receives the
validated five-agent results and their evidence to produce the component and
overall verdicts and explanations.

Every claim component triggers all five pillar-agent calls, including pillars
that return not_applicable. After all components, verdict_agent combines their
validated results in a separate call. A malformed pillar response is recorded
as an error without skipping the remaining pillar agents.

The aggregator receives compact findings, accepted quotes and available evidence,
with keyframe observations prioritized within the evidence budget. It reviews every
applicable assessment in one call without rerunning pillar agents or retrieval.
Its video_context consolidates supplied metadata, a transcript excerpt, vision
observations and the first platform metadata record, keeping the sources separate.
Metadata values, transcript excerpts and unparsed platform excerpts are capped at
500 characters; full observations remain in the evidence store. Source references
allow citation checks against exactly the context retained in the prompt.
pipeline_context summarizes evidence counts and technical failures. Pillar errors
carry component_id; each packet's partial_findings identifies only that component's
failed unknown assessments. Provenance failure also explains the unknown core event.
Failure reasons survive the pillar rules, while unasserted dimensions remain
not_applicable. These context packets are model inputs, not new result fields or logs.
Every assessment review has status, explanation, supporting_citations and
refuting_citations. Citations contain exactly evidence_id and quote. Known findings
may be retained or downgraded, never reversed. Unknown findings may
be resolved using video observations and existing evidence, with exact supporting
or refuting quotes; missing evidence and unresolved conflicts remain unknown.
New citations are accepted only from evidence actually shown to the aggregator.
Python validates ratings against grounded findings and attaches accepted citations. Reviewed decisions
do not rewrite the pillar's extracted verification fields.
Independent reports can establish the asserted event and its documented date
without visual confirmation in keyframes. Event truth and video provenance are
assessed separately; uncertain provenance does not invalidate an event report.
The response template contains the actual component IDs and required reviews.
Missing, duplicate or extra IDs/reviews and status reversals are technical errors.
Missing or conflicting decisive citations downgrade an assessment to unknown;
Unsupported verdicts become NEI, with explanations updated to reflect missing evidence.
Invalid JSON and other structural validation errors propagate without masking the
original failure.
Component results are updated only after the entire response passes validation.

Results contain the original/normalized claim, claim_components,
component_results, overall verdict, explanation, open_questions, citations,
evidence, evidence_relations, trace, errors and media references.
Each component keeps the existing verification field structure plus a core
assessment and pillar assessments. Applicable assessments are supported,
refuted or unknown; unasserted dimensions are not_applicable. Component text
quotes an unchanged source clause; inherited subjects live in structured fields.
Reviewed assessments identify verdict_agent in reviewed_by. Citations retained on
unknown assessments have unresolved relations, rather than decisive supports/refutes.

Every explanation field, including overall, component and pillar explanations,
is a plain prose string: no Markdown, HTML, JSON, URLs or evidence IDs.
Verdicts, citations, evidence and trace remain separate structured JSON fields.

Component and overall verdicts use exactly TRUE, FALSE and NEI. The dataset label
composition is defined in pipeline.assessment.LABEL_COMPOSITION:

* TRUE: True, Mostly True, Correct Attribution.
* FALSE: False, Miscaptioned, Mixture, Fake, Mostly False.
* NEI: insufficient or conflicting evidence.

The verdict agent judges the substance and importance of assertions, not counts of
pillars/components. Established minor errors may still yield TRUE as Mostly True;
material false assertions, including Mixture, yield FALSE. Uncertainty alone is not
FALSE or a proven mixture. TRUE needs a supported core assertion and no unresolved
assessments; FALSE needs a grounded refutation. The agent judges whether established
errors are minor or material. Decisive quotes are always checked.
Completed verifications returning NEI include open_questions. The status field
independently describes execution success or failure. Original per-record dataset
ratings never enter model prompts.
If the model supplies no questions, the NEI fallback distinguishes technical agent
failures from available video context lacking independent corroboration, then asks
which evidence would verify the unresolved claim. A technical failure is never
treated as contradictory evidence and does not trigger retries.

Evidence's verified flag means its document was acquired, not that its claim
is true. Snippets, reverse-image matches and user metadata/transcripts are
context or candidates. Pillar facts require acquired relevant pages and exact
quotes. The verdict agent may also cite keyframe observations for directly visible
facts, without inferring capture dates, original sources or unobserved context.
Geocoding only normalizes an independently grounded place.
Both supporting and refuting evidence survive unresolved conflicts.

Uploader observations may come from fetched platform metadata. Original creator,
media provenance, capture/event date, verified location and original context need
retrieved-page evidence. Upload dates use platform metadata; earliest online dates
use archive records. Only text in evidence.observation is quotable; a published_at
value stored outside that observation cannot be used as an invented quote.

Known fact-check domains, redirects/canonical URLs, ClaimReview markup and
visible fact-check/rating signals are excluded before evidence reaches prompts.
Screening is heuristic: unidentified mirrors can still escape detection.
Quote validation checks provenance and extraction constraints; semantic
relevance/entailment still depends on Qwen and warrants review of the evidence.

Text prompts reserve generated tokens within an 8192-token budget. Whole
evidence items are selected without truncating the claim. An oversized non-evidence
input is an explicit error. Relative dates without reference context and capture
dates less precise than the claim cannot receive a supported date assessment.
Normalization and pillar calls reserve 2048 output tokens. Vision reserves 512 and
returns seven observation fields with short literal OCR/clues. Aggregation reserves
2048 to 4096 tokens according to the number of components and required reviews,
within the same 8192-token total context. Required aggregation quotes are not removed
to make the packet fit; an oversized packet fails explicitly.

A6000 smoke test
----------------

On the A6000 server, run the CLI on representative actual media and review the
result JSON. Acceptance requires valid cited decisions, no out-of-memory
failure, media under the assets directory, and unchanged dataset files.
Review the evidence and explanations manually. The smoke test is not an
accuracy benchmark.

