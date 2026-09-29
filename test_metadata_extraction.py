import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from check_internet_records import check_group, metadata_breakdown
from stages.web import apply_metadata, restore_true_metadata
from utils.media_metadata import cookie_file_from_env, extract_media
from utils.results import readable_result
from utils.schema import empty_verification, project_verification


class FakeYoutubeDL:
    info = None
    error = None
    options = None

    def __init__(self, options):
        type(self).options = options

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def extract_info(self, _url, download=False):
        if type(self).error:
            raise type(self).error
        return type(self).info


class MetadataExtractorTests(unittest.TestCase):
    def setUp(self):
        FakeYoutubeDL.error = None
        FakeYoutubeDL.options = None
        self.yt_dlp = types.SimpleNamespace(YoutubeDL=FakeYoutubeDL)
        self.session = object()

    def extract(self, info, page=None, error=None, cookie_file=None):
        FakeYoutubeDL.info = info
        FakeYoutubeDL.error = error
        page_effect = page if not isinstance(page, Exception) else page
        with patch.dict(sys.modules, {"yt_dlp": self.yt_dlp}), patch(
            "utils.media_metadata.fetch_page_metadata",
            side_effect=page_effect if isinstance(page_effect, Exception) else None,
            return_value=page_effect if not isinstance(page_effect, Exception) else None,
        ):
            return extract_media(
                "https://example.com/video/1", self.session, cookie_file
            )

    def test_complete_yt_dlp_metadata_skips_html(self):
        result = self.extract(
            {
                "title": "YT title",
                "description": "YT description",
                "uploader": "Uploader",
                "uploader_url": "https://example.com/uploader",
                "upload_date": "20240102",
                "formats": [{"url": "https://cdn.example.com/video.mp4", "vcodec": "h264"}],
            }
        )
        self.assertEqual(result["metadata_status"], "complete")
        self.assertEqual(result["video"]["decision"], "keep")
        self.assertEqual(set(result["field_sources"].values()), {"yt_dlp"})

    def test_html_fills_only_missing_fields(self):
        result = self.extract(
            {
                "title": "YT title",
                "formats": [{"url": "https://cdn.example.com/video.mp4", "vcodec": "h264"}],
            },
            {
                "title": "HTML title",
                "description": "HTML description",
                "uploader_name": "HTML author",
                "uploader_profile": "https://example.com/author",
                "upload_date": "2024-01-03",
            },
        )
        self.assertEqual(result["metadata_status"], "complete")
        self.assertEqual(result["metadata"]["title"], "YT title")
        self.assertEqual(result["field_sources"]["title"], "yt_dlp")
        self.assertEqual(result["field_sources"]["description"], "html")

    def test_html_metadata_does_not_prove_video_availability(self):
        result = self.extract(
            None,
            {
                "title": "Page title",
                "description": "Page description",
                "uploader_name": "Publisher",
                "uploader_profile": "https://example.com/publisher",
                "upload_date": "2024-01-03",
            },
            error=RuntimeError("yt-dlp failed"),
        )
        self.assertEqual(result["metadata_status"], "complete")
        self.assertEqual(result["video"]["decision"], "unknown")
        self.assertEqual(result["errors"][0]["provider"], "yt_dlp")

    def test_total_failure_is_missing(self):
        result = self.extract(
            None,
            RuntimeError("page failed"),
            error=RuntimeError("yt-dlp failed"),
        )
        self.assertEqual(result["metadata_status"], "missing")
        self.assertEqual(len(result["errors"]), 2)

    def test_playlist_entry_overrides_parent_and_parent_fills_gaps(self):
        result = self.extract(
            {
                "description": "Parent description",
                "uploader": "Parent uploader",
                "uploader_url": "https://example.com/uploader",
                "upload_date": "20240203",
                "entries": [
                    {
                        "title": "Entry title",
                        "formats": [
                            {"url": "https://cdn.example.com/video.mp4", "vcodec": "h264"}
                        ],
                    }
                ],
            }
        )
        self.assertEqual(result["metadata"]["title"], "Entry title")
        self.assertEqual(result["metadata"]["description"], "Parent description")
        self.assertEqual(result["metadata_status"], "complete")

    def test_timestamp_is_converted_to_utc_date_and_cookie_is_forwarded(self):
        result = self.extract(
            {
                "title": "Title",
                "description": "Description",
                "uploader": "Uploader",
                "uploader_url": "https://example.com/uploader",
                "timestamp": 1704153600,
                "formats": [],
            },
            cookie_file="cookies.txt",
        )
        self.assertEqual(result["metadata"]["upload_date"], "2024-01-02")
        self.assertEqual(FakeYoutubeDL.options["cookiefile"], "cookies.txt")

    def test_cookie_environment_is_optional_and_validated(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(cookie_file_from_env())
        cookie = Path(__file__).resolve()
        with patch.dict(os.environ, {"YTDLP_COOKIE_FILE": str(cookie)}, clear=True):
            self.assertEqual(cookie_file_from_env(), str(cookie))
        with patch.dict(os.environ, {"YTDLP_COOKIE_FILE": "missing.txt"}, clear=True):
            with self.assertRaises(FileNotFoundError):
                cookie_file_from_env()


class WebStageTests(unittest.TestCase):
    def extra(self):
        return {
            "verification": empty_verification(),
            "evidence": [
                {
                    "id": "platform-metadata-old",
                    "type": "platform_metadata",
                    "source": "https://old.example.com",
                    "observation": "Old metadata",
                }
            ],
            "field_evidence": {
                "verification.source.uploader_name": ["platform-metadata-old"]
            },
        }

    def record(self):
        return {
            "path": Path("train_val/1.json"),
            "data": {
                "video_information": {
                    "platform": "foxnews",
                    "video_date": "20200102",
                    "video_headline": "TRUE title",
                    "video_description": "TRUE description",
                }
            },
        }

    def test_failed_rerun_restores_true_values_and_removes_stale_evidence(self):
        extra = self.extra()
        extra["verification"]["source"]["uploader_name"] = "Stale uploader"
        restore_true_metadata(self.record(), extra)
        extraction = {
            "metadata_status": "missing",
            "metadata": {},
            "field_sources": {},
            "missing_fields": [
                "title",
                "description",
                "uploader_name",
                "uploader_profile",
                "upload_date",
            ],
            "errors": [{"provider": "yt_dlp", "message": "failed"}],
        }
        apply_metadata(extra, extraction, "2026-01-01T00:00:00+00:00", "train_val/1.json")
        self.assertEqual(extra["verification"]["source"]["uploader_name"], "Fox News")
        self.assertIsNone(extra["verification"]["source"]["uploader_profile"])
        self.assertEqual(extra["verification"]["date"]["video_upload_date"], "2020-01-02")
        self.assertEqual(extra["verification"]["motivation"]["original_caption"], "TRUE title")
        self.assertEqual(extra["verification"]["source"]["metadata_status"], "missing")
        self.assertNotIn(
            "verification.source.uploader_name", extra["field_evidence"]
        )

    def test_partial_extraction_updates_only_extracted_fields(self):
        extra = self.extra()
        restore_true_metadata(self.record(), extra)
        extraction = {
            "metadata_status": "partial",
            "metadata": {"title": "Current title", "uploader_name": "Current uploader"},
            "field_sources": {"title": "yt_dlp", "uploader_name": "html"},
            "missing_fields": ["description", "uploader_profile", "upload_date"],
            "errors": [{"provider": "wayback", "message": "unavailable"}],
        }
        apply_metadata(extra, extraction, "2026-01-01T00:00:00+00:00", "https://example.com/1")
        self.assertEqual(extra["verification"]["source"]["uploader_name"], "Current uploader")
        self.assertEqual(extra["verification"]["motivation"]["original_caption"], "Current title")
        self.assertEqual(extra["verification"]["motivation"]["original_description"], "TRUE description")
        self.assertEqual(extra["verification"]["source"]["metadata_status"], "partial")


class CheckerAndSchemaTests(unittest.TestCase):
    def test_equivalent_urls_keep_video_and_choose_richest_metadata(self):
        records = [
            {"data": {"video_information": {"video_url": "https://example.com/1"}}},
            {"data": {"video_information": {"video_url": "https://example.com/2"}}},
        ]
        partial = {
            "decision": "keep",
            "reason": "video_format_found",
            "error": None,
            "metadata_status": "partial",
            "metadata": {"title": "Title"},
            "metadata_field_sources": {"title": "yt_dlp"},
            "missing_metadata_fields": ["description"],
            "metadata_errors": [],
        }
        complete = {
            "decision": "unknown",
            "reason": "no_video_format",
            "error": None,
            "metadata_status": "complete",
            "metadata": {field: field for field in (
                "title", "description", "uploader_name", "uploader_profile", "upload_date"
            )},
            "metadata_field_sources": {},
            "missing_metadata_fields": [],
            "metadata_errors": [],
        }
        with patch("check_internet_records.probe_video", side_effect=[partial, complete]) as probe:
            result = check_group(records)
        self.assertEqual(probe.call_count, 2)
        self.assertEqual(result["decision"], "keep")
        self.assertEqual(result["metadata_status"], "complete")
        self.assertEqual(result["metadata_url"], "https://example.com/2")

    def test_metadata_breakdown_counts_fields(self):
        rows = [
            {
                "platform": "youtube",
                "metadata_status": "partial",
                "metadata": {"title": "Title"},
            },
            {
                "platform": "youtube",
                "metadata_status": "missing",
                "metadata": {},
            },
        ]
        result = metadata_breakdown(rows, "platform")["youtube"]
        self.assertEqual(result["partial"], 1)
        self.assertEqual(result["missing"], 1)
        self.assertEqual(result["field_coverage"]["title"], 1)

    def test_public_schema_contains_metadata_status_without_nested_metadata(self):
        extra = {
            "claim_id": "1",
            "claim_components": [{"id": "claim_1", "text": "Claim"}],
            "normalized_video_information": {
                "platform": "youtube",
                "video_url": "https://example.com/1",
                "video_date": "2024-01-01",
                "video_transcript": None,
            },
            "verification": project_verification(
                {"source": {"metadata_status": "complete"}}
            ),
        }
        result = readable_result(extra)
        self.assertEqual(result["verification"]["source"]["metadata_status"], "complete")
        self.assertNotIn("metadata", result["video_information"])

    def test_metadata_status_is_a_readonly_website_enum(self):
        from website.annotation.schema import CORE_RECORD_FIELDS, ENUM_FIELDS

        path = "verification.source.metadata_status"
        self.assertEqual(ENUM_FIELDS[path], {"complete", "partial", "missing"})
        self.assertIn(path, CORE_RECORD_FIELDS)


if __name__ == "__main__":
    unittest.main()
