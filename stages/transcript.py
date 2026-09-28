"""Generate a textual transcript from a record's remote video audio."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

from stages import MODELS
from utils.evidence import add_evidence
from utils.files import now, read_json, trim, write_json
from utils.records import sidecar_path
from utils.utils import normalize_social_url


MAX_AUDIO_BYTES = 512 * 1024 * 1024
MAX_TRANSCRIPT_CHARACTERS = 200000


def _video_url(record):
    url = str(
        (record["data"].get("video_information") or {}).get("video_url") or ""
    ).strip()
    return normalize_social_url(url)["url"]


def _set_record_transcript(record, transcript):
    video = record["data"].setdefault("video_information", {})
    video["video_transcript"] = transcript


def _download_audio(url, directory):
    """Download the best available audio stream into a temporary directory."""
    import yt_dlp

    directory = Path(directory)
    template = directory / "audio.%(ext)s"
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(template),
        "noplaylist": True,
        "playlist_items": "1",
        "quiet": True,
        "no_warnings": True,
        "overwrites": True,
        "socket_timeout": 30,
        "retries": 2,
        "fragment_retries": 2,
        "max_filesize": MAX_AUDIO_BYTES,
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
            if info and info.get("entries"):
                info = next((entry for entry in info["entries"] if entry), info)
            prepared = Path(downloader.prepare_filename(info)) if info else None
    except Exception as error:
        raise RuntimeError(
            f"Could not download audio from the video URL: {error}"
        ) from error

    candidates = []
    for item in (info or {}).get("requested_downloads") or []:
        if item.get("filepath"):
            candidates.append(Path(item["filepath"]))
    if prepared:
        candidates.append(prepared)
    candidates.extend(sorted(directory.glob("audio.*")))
    audio = next(
        (
            path
            for path in candidates
            if path.is_file() and path.suffix not in {".part", ".ytdl"}
        ),
        None,
    )
    if audio is None:
        raise RuntimeError("The video URL did not provide a downloadable audio stream.")
    return audio


def _transcribe_audio(audio_path, model_cache):
    """Run multilingual Faster Whisper and return text plus generation metadata."""
    from faster_whisper import WhisperModel

    model_name = (
        os.environ.get("ETRUE_WHISPER_MODEL", "").strip() or MODELS["transcript"]
    )
    device = os.environ.get("ETRUE_WHISPER_DEVICE", "cpu").strip().lower()
    if device not in {"cpu", "cuda"}:
        raise RuntimeError("ETRUE_WHISPER_DEVICE must be either cpu or cuda.")
    compute_type = "float16" if device == "cuda" else "int8"
    cache = Path(model_cache) / "whisper"
    cache.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=str(cache),
    )
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,
    )
    texts = []
    for segment in segments:
        text = str(segment.text or "").strip()
        if text:
            texts.append(text)
    return {
        "text": " ".join(texts),
        "model": model_name,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "segment_count": len(texts),
    }


def run(records, output, model_cache):
    for number, record in enumerate(records, 1):
        sidecar_file = sidecar_path(output, record["claim_id"])
        sidecar = read_json(sidecar_file)
        video = sidecar["normalized_video_information"]
        existing_transcript = str(video.get("video_transcript") or "").strip()
        if existing_transcript:
            _set_record_transcript(record, existing_transcript)
            continue

        url = _video_url(record)
        if not url:
            raise RuntimeError("The record does not contain a video URL to transcribe.")
        with TemporaryDirectory(prefix=f"etrue-{record['claim_id']}-") as temporary:
            audio_path = _download_audio(url, temporary)
            transcription = _transcribe_audio(audio_path, model_cache)

        transcript = str(transcription.get("text") or "").strip()
        if not transcript:
            raise RuntimeError("Whisper did not detect speech in the video audio.")
        if len(transcript) > MAX_TRANSCRIPT_CHARACTERS:
            raise RuntimeError(
                "The generated transcript exceeds the supported 200,000 characters."
            )

        video["video_transcript"] = transcript
        video["transcript_status"] = "generated"
        evidence_id = add_evidence(
            sidecar,
            "asr_transcript",
            url,
            trim(transcript, 2000),
            fields=(
                "normalized_video_information.video_transcript",
                "normalized_video_information.transcript_status",
            ),
            model=transcription.get("model"),
            language=transcription.get("language"),
            language_probability=transcription.get("language_probability"),
        )
        sidecar.setdefault("automation", {})["asr"] = {
            "status": "ok",
            "generated_at": now(),
            "model": transcription.get("model"),
            "language": transcription.get("language"),
            "language_probability": transcription.get("language_probability"),
            "duration": transcription.get("duration"),
            "segment_count": transcription.get("segment_count"),
            "evidence_id": evidence_id,
        }
        _set_record_transcript(record, transcript)
        write_json(sidecar_file, sidecar)
        print("transcript", number, "/", len(records), flush=True)
