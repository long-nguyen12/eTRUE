import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from build_etrue import (
    build,
    canonical_video_key,
    empty_verification,
    normalize_date,
    normalize_platform,
)
from pillars.date import (
    apply_date_rules,
    compare_date_values,
    date_is_grounded,
    first_year,
    infer_date_granularity,
    normalize_date_value,
)
from pillars.location import extract_location_contrast, location_candidate_is_plausible
from pillars.motivation import apply_motivation_rules
from pillars.provenance import apply_search_provenance, previous_context_is_useful
from pillars.source import (
    apply_source_rules,
    normalize_source_name,
    source_entities_match,
    source_name_is_grounded,
)
from stages import MODELS, PIPELINE_STAGES
from stages.audit import run as run_audit
from stages.clip import cached_embeddings_are_normalized
from stages.geocode import classify_location_mismatch, select_geocode_candidate
from stages.location import (
    apply_candidates,
    build_sources,
    choose_claimed_location,
    choose_verified_location,
    event_location_is_asserted,
    extract_candidates,
)
from stages.match import hash_distance, image_dhash
from stages.search import (
    build_queries,
    candidate_page_metadata,
    ddgs_results,
    enrich_image_pages,
    google_image_matches,
    public_http_url,
    search_record,
    select_keyframes,
)
from stages.text import field_evidence_ids, text_input
from stages.vision import normalize_vision_analysis, run as run_vision
from utils.model_output import parse_json_output
from utils.records import frame_paths
from utils.results import readable_result
from utils.text import containment_overlap, word_overlap


def sample(claim_id, url, rating="True", transcript="words"):
    return {
        "claim": "Claim " + claim_id,
        "rating": rating,
        "video_information": {
            "video_id": claim_id,
            "video_date": 20240102.0,
            "platform": "youtu",
            "video_transcript": transcript,
            "video_url": url,
        },
        "evidences": {"num_of_evidence": 0},
        "relationship_with_evidence": [],
    }


class BuildEtrueTests(unittest.TestCase):
    def test_creates_readable_result_without_internal_details(self):
        sidecar = {
            "claim_id": "123",
            "claim_components": [{"id": "claim_1", "text": "Example claim"}],
            "normalized_video_information": {
                "platform": "x",
                "video_url": "https://example.com/video",
                "video_date": "2024-01-02",
                "video_transcript": "Original transcript",
            },
            "verification": {
                "provenance": {
                    "provenance_status": "unknown",
                    "earliest_known_url": "https://example.com/video",
                },
                "source": {"uploader_name": "Uploader", "original_source_name": None},
                "date": {"claimed_date": None, "estimated_date": None, "date_mismatch_type": None},
                "location": {
                    "claimed_location": "London",
                    "verified_location": "London",
                    "location_mismatch_type": "same",
                },
                "motivation": {"original_caption": "Original caption", "motivation_mismatch_type": None},
            },
            "evidence": [{"id": "evidence-1"}],
            "field_evidence": {"verification.location.verified_location": ["evidence-1"]},
            "automation": {"text": {"status": "ok"}},
            "review": {"status": "automated"},
        }
        original = json.loads(json.dumps(sidecar))

        result = readable_result(sidecar)

        self.assertEqual(sidecar, original)
        self.assertEqual(result["claim"], "Example claim")
        self.assertEqual(result["video_information"]["video_transcript"], "Original transcript")
        self.assertEqual(result["verification"]["location"]["location_mismatch_type"], "same")
        self.assertEqual(result["review_status"], "automated")
        self.assertNotIn("evidence", result)
        self.assertNotIn("field_evidence", result)
        self.assertNotIn("automation", result)
        self.assertEqual(
            set(result),
            {"claim_id", "claim", "video_information", "verification", "review_status"},
        )
        self.assertEqual(
            set(result["video_information"]),
            {"platform", "video_url", "video_date", "video_transcript"},
        )
        self.assertEqual(
            set(result["verification"]),
            {"provenance", "source", "date", "location", "motivation"},
        )

        self.assertEqual(
            set(result["verification"]["provenance"]),
            {
                "provenance_status",
                "earliest_known_url",
                "earliest_known_date",
                "near_duplicate_matches",
                "previous_context_summary",
                "provenance_mismatch",
            },
        )
        self.assertEqual(
            set(result["verification"]["source"]),
            {
                "uploader_name",
                "uploader_profile",
                "original_source_name",
                "source_type",
                "source_is_uploader",
                "source_mismatch",
            },
        )
        self.assertEqual(
            set(result["verification"]["date"]),
            {
                "claimed_date",
                "video_upload_date",
                "earliest_online_date",
                "estimated_date",
                "capture_date_granularity",
                "date_mismatch_type",
            },
        )
        self.assertEqual(
            set(result["verification"]["location"]),
            {
                "claimed_location",
                "candidate_locations",
                "visual_location_clues",
                "verified_location",
                "verified_coordinates",
                "location_mismatch_type",
            },
        )
        self.assertEqual(
            set(result["verification"]["motivation"]),
            {
                "claimed_framing",
                "original_caption",
                "original_description",
                "original_context_category",
                "motivation_mismatch_type",
            },
        )

    def test_finds_extracted_keyframes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            frame_directory = source / "train_val_output" / "123"
            frame_directory.mkdir(parents=True)
            for name in ("10.jpeg", "2.jpeg", "1.jpeg"):
                (frame_directory / name).touch()

            record = {"claim_id": "123", "legacy_split": "train_val"}
            names = [path.name for path in frame_paths(source, record)]
            self.assertEqual(names, ["1.jpeg", "2.jpeg", "10.jpeg"])

    def test_parses_fenced_model_json(self):
        self.assertEqual(parse_json_output('```json\n{"value": 1}\n```'), {"value": 1})
        self.assertEqual(parse_json_output("{'value': 2}"), {"value": 2})
        self.assertEqual(parse_json_output('[["building"]]'), [["building"]])

    def test_repairs_control_characters_in_model_json(self):
        malformed = '{"claimed_framing": "first line\nsecond line"}'
        self.assertEqual(
            parse_json_output(malformed),
            {"claimed_framing": "first line\nsecond line"},
        )

    def test_reports_incomplete_model_json_as_parse_error(self):
        malformed = '{"claimed_framing": "unfinished}'
        with self.assertRaises(ValueError):
            parse_json_output(malformed)

    def test_normalizes_vision_output_and_rejects_unsupported_places(self):
        tuple_output = normalize_vision_analysis(("crowd", "street"))
        self.assertEqual(tuple_output["scene_summary"], "crowd; street")

        value = normalize_vision_analysis(
            {
                "scene_summary": "Live in New York City",
                "ocr_Text": "LIVE NEW YORK CITY",
                "candidateLocations": [
                    {"name": "New York City"},
                    {"name": "Washington DC"},
                ],
            }
        )
        self.assertEqual(value["ocr_text"], ["LIVE NEW YORK CITY"])
        self.assertEqual(value["candidate_locations"], ["New York City"])

    def test_rejects_scene_descriptions_as_locations(self):
        self.assertTrue(location_candidate_is_plausible("Rock Hill, South Carolina"))
        self.assertTrue(location_candidate_is_plausible("new york city"))
        self.assertFalse(location_candidate_is_plausible("wooden surface"))
        self.assertFalse(location_candidate_is_plausible("Room"))
        self.assertFalse(location_candidate_is_plausible("Field"))
        self.assertFalse(location_candidate_is_plausible("D"))
        self.assertFalse(location_candidate_is_plausible("Harris on board"))
        self.assertFalse(location_candidate_is_plausible("A person wearing glasses and frowning"))

    def test_extracts_directional_location_contrasts(self):
        self.assertEqual(
            extract_location_contrast("The incident happened in Brazil, not the United States."),
            ("United States", "Brazil"),
        )
        self.assertEqual(
            extract_location_contrast("It was not in France, but in Belgium."),
            ("France", "Belgium"),
        )

    def test_word_overlap(self):
        self.assertEqual(word_overlap("same words", "same words"), 1.0)
        self.assertLess(word_overlap("sexual assault allegation", "presidential campaign speech"), 0.25)

    def test_rejects_ungrounded_sources_and_claim_restatements(self):
        self.assertTrue(source_name_is_grounded("Associated Press", "Footage from Associated Press"))
        self.assertFalse(source_name_is_grounded("Reddit", "Posted to Reddit"))
        self.assertEqual(containment_overlap("A cat sat on the mat", "The cat sat on a mat"), 1.0)
        self.assertFalse(
            previous_context_is_useful(
                "A video shows an explosives test of a newly developed football helmet.",
                "Video of an explosives test of the newly developed football helmet.",
                "The article discusses the explosives test video.",
            )
        )
        self.assertFalse(
            previous_context_is_useful(
                "Alexandria Ocasio-Cortez won a Democratic primary in New York.",
                "Alexandria Ocasio-Cortez said billions and trillions are similar.",
                "Alexandria Ocasio-Cortez won a Democratic primary in New York.",
            )
        )
        self.assertTrue(
            previous_context_is_useful(
                "The footage depicts a 1948 daredevil show in Birmingham, Alabama.",
                "Video of a football helmet safety test.",
                "The footage depicts Helen Howe at a 1948 daredevil show in Birmingham, Alabama.",
            )
        )

    def test_links_inferred_source_type_to_source_evidence(self):
        sidecar = {
            "verification": {
                "source": {
                    "uploader_name": "Mark Kennedy",
                    "original_source_name": "Mark Kennedy",
                    "source_type": None,
                    "source_is_uploader": None,
                    "source_mismatch": None,
                }
            },
            "field_evidence": {
                "verification.source.original_source_name": ["source-1"],
                "verification.source.uploader_name": ["uploader-1"],
            },
        }

        apply_source_rules(sidecar, "Mark Kennedy uploaded the video.", None, [])

        self.assertEqual(sidecar["verification"]["source"]["source_type"], "unknown")
        self.assertEqual(
            sidecar["field_evidence"]["verification.source.source_type"],
            ["source-1", "uploader-1"],
        )

    def test_extracts_first_year(self):
        self.assertEqual(first_year("between 2017 and 2018"), 2017)
        self.assertIsNone(first_year("unknown"))

    def test_dates_require_event_evidence(self):
        self.assertTrue(date_is_grounded("2018-03-14", "The event occurred on March 14, 2018."))
        self.assertFalse(date_is_grounded("2007-07-29", "The stunt occurred in November 1925."))
        self.assertEqual(infer_date_granularity("March 2018"), "month")
        self.assertEqual(infer_date_granularity("28 June 2018"), "day")

    def test_pipeline_does_not_use_asr(self):
        self.assertNotIn("asr", PIPELINE_STAGES)
        self.assertNotIn("asr", MODELS)

    def test_builds_optional_provenance_search_queries(self):
        record = sample(
            "1",
            "https://example.com/video",
            transcript="This unusually specific sentence came directly from the original transcript.",
        )
        record["claim"] = "A sufficiently distinctive claim about this specific video"
        sidecar = {
            "normalized_video_information": {
                "video_transcript": record["video_information"]["video_transcript"]
            },
            "verification": {"motivation": {"original_caption": "A distinctive video title"}},
        }

        queries = build_queries({"data": record}, sidecar)

        self.assertEqual(len(queries), 3)
        self.assertTrue(all(query.startswith('"') and query.endswith('"') for query in queries))

    def test_normalizes_ddgs_search_results(self):
        class DDGS:
            def text(self, query, max_results):
                self.request = (query, max_results)
                return [
                    {
                        "href": "https://example.com/result",
                        "title": "Earlier source",
                        "body": "Description of the earlier source",
                    }
                ]

        client = DDGS()
        options = {}

        def create_ddgs(**kwargs):
            options.update(kwargs)
            return client

        module = SimpleNamespace(DDGS=create_ddgs)
        with patch.dict("sys.modules", {"ddgs": module}):
            results = ddgs_results('"distinctive query"')

        self.assertEqual(options, {"timeout": 20})
        self.assertEqual(client.request, ('"distinctive query"', 5))
        self.assertEqual(
            results,
            [
                {
                    "url": "https://example.com/result",
                    "title": "Earlier source",
                    "description": "Description of the earlier source",
                    "rank": 1,
                    "query": '"distinctive query"',
                }
            ],
        )

    def test_uses_ddgs_when_search_api_keys_are_missing(self):
        record = {
            "claim_id": "1",
            "legacy_split": "train_val",
            "data": sample(
                "1",
                "https://example.com/video",
                transcript="This unusually specific sentence came from the video.",
            ),
        }
        sidecar = {
            "normalized_video_information": {
                "video_transcript": record["data"]["video_information"]["video_transcript"]
            },
            "verification": {
                "motivation": {"original_caption": None},
                "source": {},
            },
            "evidence": [],
        }

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "stages.search.ddgs_results",
                return_value=[
                    {
                        "url": "https://example.com/result",
                        "title": "Exact transcript result",
                        "description": "Earlier copy",
                        "rank": 1,
                        "query": "quoted query",
                    }
                ],
            ) as search:
                result = search_record(
                    record,
                    sidecar,
                    Path(directory),
                    session=None,
                    brave_key=None,
                    vision_key=None,
                )
            with patch("stages.search.ddgs_results", return_value=[]) as vision_fallback:
                search_record(
                    record,
                    sidecar,
                    Path(directory),
                    session=None,
                    brave_key=None,
                    vision_key="vision-key",
                )

        self.assertGreater(search.call_count, 0)
        self.assertGreater(vision_fallback.call_count, 0)
        self.assertEqual(result["search_provider"], "DDGS")
        self.assertEqual(result["results"][0]["query_type"], "transcript")
        self.assertEqual(
            result["image_search_status"],
            "skipped: GOOGLE_CLOUD_VISION_API_KEY is not set",
        )

    def test_location_requires_two_independent_non_claim_sources(self):
        candidates = [
            {"name": "London", "sources": ["claim", "fact_check", "vision"]},
            {"name": "Paris", "sources": ["claim", "transcript"]},
        ]

        self.assertIsNone(choose_claimed_location(candidates))
        self.assertEqual(choose_verified_location(candidates), "London")
        self.assertIsNone(
            choose_verified_location([{"name": "Paris", "sources": ["claim", "transcript"]}])
        )

    def test_reads_reverse_image_candidates_without_storing_image_bytes(self):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "responses": [
                        {
                            "webDetection": {
                                "webEntities": [
                                    {
                                        "entityId": "event",
                                        "description": "Example Event",
                                        "score": 0.8,
                                    }
                                ],
                                "pagesWithMatchingImages": [
                                    {
                                        "url": "https://example.com/page",
                                        "pageTitle": "<b>Earlier</b> page",
                                        "fullMatchingImages": [
                                            {
                                                "url": "https://example.com/image.jpg",
                                                "score": 0.9,
                                            }
                                        ],
                                    }
                                ],
                                "fullMatchingImages": [
                                    {
                                        "url": "https://example.com/image.jpg",
                                        "score": 0.9,
                                    }
                                ],
                                "bestGuessLabels": [
                                    {"label": "example event", "languageCode": "en"}
                                ],
                            }
                        },
                        {
                            "webDetection": {
                                "webEntities": [
                                    {
                                        "entityId": "event",
                                        "description": "Example Event",
                                        "score": 0.9,
                                    }
                                ],
                                "pagesWithMatchingImages": [
                                    {
                                        "url": "https://example.com/page",
                                        "pageTitle": "Earlier page",
                                        "partialMatchingImages": [
                                            {"url": "https://example.com/crop.jpg"}
                                        ],
                                    }
                                ],
                                "partialMatchingImages": [
                                    {"url": "https://example.com/crop.jpg"}
                                ],
                                "visuallySimilarImages": [
                                    {"url": "https://example.com/similar.jpg"}
                                ],
                            }
                        },
                    ]
                }

        class Session:
            def post(self, *args, **kwargs):
                self.request = kwargs
                return Response()

        with tempfile.TemporaryDirectory() as directory:
            frames = [
                Path(directory) / "first.jpeg",
                Path(directory) / "last.jpeg",
            ]
            for frame in frames:
                frame.write_bytes(b"image bytes")
            session = Session()
            result = google_image_matches(session, "secret", frames)

        self.assertEqual(len(session.request["json"]["requests"]), 2)
        self.assertEqual(result["pages"][0]["url"], "https://example.com/page")
        self.assertEqual(result["pages"][0]["title"], "Earlier page")
        self.assertEqual(result["pages"][0]["match_types"], ["full", "partial"])
        self.assertEqual(
            result["pages"][0]["matched_frames"],
            ["first.jpeg", "last.jpeg"],
        )
        self.assertEqual(
            result["pages"][0]["image_urls"],
            [
                "https://example.com/image.jpg",
                "https://example.com/crop.jpg",
            ],
        )
        self.assertEqual(result["web_entities"][0]["score"], 0.9)
        self.assertEqual(result["best_guess_labels"][0]["language_code"], "en")
        self.assertEqual(
            result["visually_similar_images"][0]["url"],
            "https://example.com/similar.jpg",
        )
        self.assertNotIn("image bytes", json.dumps(result))

    def test_selects_evenly_spaced_reverse_image_frames(self):
        frames = [Path(str(index) + ".jpeg") for index in range(7)]
        self.assertEqual(
            [frame.name for frame in select_keyframes(frames)],
            ["0.jpeg", "3.jpeg", "6.jpeg"],
        )

    def test_extracts_bounded_candidate_page_metadata(self):
        html = b"""
        <html>
          <head>
            <meta property="og:title" content="Original report">
            <meta property="og:description" content="Earlier event context">
            <meta property="og:site_name" content="Example News">
            <link rel="canonical" href="/canonical">
            <script type="application/ld+json">
              {"datePublished": "2018-02-10T12:00:00Z",
               "author": {"name": "Reporter Name"}}
            </script>
          </head>
          <body><article>The complete original event context.</article></body>
        </html>
        """

        class Response:
            url = "https://example.com/report"
            headers = {"Content-Type": "text/html; charset=utf-8"}
            encoding = "utf-8"

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                yield html

            def close(self):
                self.closed = True

        class Session:
            def get(self, *args, **kwargs):
                self.request = (args, kwargs)
                return Response()

        session = Session()
        metadata = candidate_page_metadata(session, "https://example.com/report")

        self.assertEqual(metadata["canonical_url"], "https://example.com/canonical")
        self.assertEqual(metadata["title"], "Original report")
        self.assertEqual(metadata["author"], "Reporter Name")
        self.assertEqual(metadata["published_at"], "2018-02-10")
        self.assertIn("original event context", metadata["context_excerpt"])
        self.assertTrue(session.request[1]["stream"])
        self.assertFalse(public_http_url("http://127.0.0.1/private"))
        self.assertFalse(public_http_url("file:///tmp/private"))

    def test_enriches_reverse_image_pages_with_archive_dates(self):
        image_search = {"pages": [{"url": "https://example.com/page"}], "errors": []}
        metadata = {
            "canonical_url": "https://example.com/canonical",
            "title": "Original page",
        }
        archive = {
            "date": "2018-02-11",
            "snapshot_url": "https://web.archive.org/snapshot",
        }
        with patch("stages.search.candidate_page_metadata", return_value=metadata):
            with patch("stages.search.wayback_earliest", return_value=archive) as wayback:
                result = enrich_image_pages(session=None, image_search=image_search)

        self.assertEqual(result["pages"][0]["archive_first_seen"], "2018-02-11")
        wayback.assert_called_once_with("https://example.com/canonical", None)

    def test_supplies_enriched_reverse_image_candidates_to_text_stage(self):
        record = {"data": sample("1", "https://example.com/video")}
        extra = {
            "verification": {},
            "evidence": [],
            "automation": {
                "search": {
                    "status": "ok",
                    "image_search": {
                        "web_entities": [{"description": "Example Event", "score": 0.9}],
                        "best_guess_labels": [{"label": "example event"}],
                        "pages": [
                            {
                                "url": "https://example.com/original",
                                "title": "Original report",
                                "context_excerpt": "Earlier context",
                                "match_types": ["full"],
                                "matched_frames": ["1.jpeg", "5.jpeg"],
                                "match_count": 2,
                            }
                        ],
                    },
                }
            },
        }

        candidates = text_input(record, extra)["reverse_image_candidates"]

        self.assertEqual(candidates["status"], "unverified_candidates")
        self.assertEqual(candidates["pages"][0]["match_count"], 2)
        self.assertNotIn("visually_similar_images", candidates)

    def test_compares_geocoded_location_levels(self):
        london = {
            "osm_type": "relation",
            "osm_id": 1,
            "address": {"city": "London", "state": "England", "country_code": "gb"},
        }
        manchester = {
            "osm_type": "relation",
            "osm_id": 2,
            "address": {"city": "Manchester", "state": "England", "country_code": "gb"},
        }
        paris = {
            "osm_type": "relation",
            "osm_id": 3,
            "address": {"city": "Paris", "state": "Ile-de-France", "country_code": "fr"},
        }

        self.assertEqual(classify_location_mismatch("London", "London", london, london), "same")
        self.assertEqual(
            classify_location_mismatch("London", "Manchester", london, manchester),
            "different city",
        )
        self.assertEqual(
            classify_location_mismatch("London", "Paris", london, paris),
            "different country",
        )

    def test_normalizes_date_ranges_and_compares_full_dates(self):
        self.assertEqual(
            normalize_date_value("March 5-7, 2024"),
            "2024-03-05 to 2024-03-07",
        )
        self.assertEqual(compare_date_values("2024-03", "2024-03-16"), 0)
        self.assertEqual(compare_date_values("2024-04", "2024-03"), 1)
        self.assertTrue(
            date_is_grounded(
                "March 5-7, 2024",
                "The event ran from March 5-7, 2024.",
            )
        )

        verification = {
            "date": {
                "claimed_date": "March 2024",
                "estimated_date": "April 2024",
                "video_upload_date": "2024-05-01",
                "capture_date_granularity": None,
                "date_mismatch_type": "older video",
            }
        }
        apply_date_rules(verification)
        self.assertEqual(verification["date"]["date_mismatch_type"], "newer video")
        self.assertEqual(verification["date"]["capture_date_granularity"], "month")

    def test_matches_source_aliases_without_conflating_source_mismatch(self):
        self.assertEqual(normalize_source_name("https://x.com/nytimes"), "new york times")
        self.assertTrue(source_entities_match("The New York Times", "@nytimes"))

        sidecar = {
            "verification": {
                "source": {
                    "uploader_name": "C-SPAN",
                    "original_source_name": "The New York Times",
                    "source_type": "news outlet",
                    "source_is_uploader": True,
                    "source_mismatch": True,
                }
            },
            "field_evidence": {},
        }
        apply_source_rules(sidecar, "The New York Times published the report.", None, [])
        self.assertFalse(sidecar["verification"]["source"]["source_is_uploader"])
        self.assertTrue(sidecar["verification"]["source"]["source_mismatch"])

    def test_preserves_valid_motivation_categories(self):
        verification = {
            "source": {"source_type": "entertainment source"},
            "date": {"claimed_date": "2024", "estimated_date": "2024"},
            "motivation": {
                "claimed_framing": "This is a real news event",
                "original_caption": "A staged film scene",
                "original_description": None,
                "original_context_category": "entertainment",
                "motivation_mismatch_type": "entertainment as news",
            },
        }
        apply_motivation_rules(verification, None)
        self.assertEqual(
            verification["motivation"]["original_context_category"],
            "entertainment",
        )
        self.assertEqual(
            verification["motivation"]["motivation_mismatch_type"],
            "entertainment as news",
        )

    def test_aggregates_full_location_mentions_and_rejects_topical_places(self):
        text = "The footage was filmed in Charlottesville during the rally."
        start = text.index("Charlottesville")

        def recognizer(_):
            return [
                {
                    "entity_group": "LOC",
                    "score": 0.99,
                    "word": "Charlottes ##ville",
                    "start": start,
                    "end": start + len("Charlottesville"),
                }
            ]

        candidates = extract_candidates(
            [
                {
                    "category": "fact_check",
                    "support_group": "fact_check",
                    "text": text,
                    "source": "https://example.com/check",
                    "known": (),
                }
            ],
            recognizer,
        )
        self.assertEqual(candidates[0]["name"], "Charlottesville")
        self.assertEqual(candidates[0]["event_sources"], ["fact_check"])
        self.assertTrue(event_location_is_asserted(text, "Charlottesville"))
        self.assertFalse(
            event_location_is_asserted(
                "Officials in London discussed aid for the incident.",
                "London",
            )
        )

    def test_preserves_stronger_upstream_verified_location(self):
        extra = {
            "verification": {
                "location": {
                    "claimed_location": None,
                    "candidate_locations": ["Charlottesville"],
                    "verified_location": "Charlottesville",
                    "verified_coordinates": {"latitude": 1.0, "longitude": 2.0},
                    "location_mismatch_type": None,
                }
            },
            "evidence": [
                {
                    "id": "fact-1",
                    "type": "fact_check_evidence",
                    "source": "https://example.com/check",
                    "observation": "The footage was filmed in Charlottesville.",
                }
            ],
            "field_evidence": {
                "verification.location.verified_location": ["fact-1"]
            },
            "automation": {
                "location": {"candidates": [{"name": "Charlotte"}]}
            },
            "review": {"status": "automated"},
        }
        result = {
            "model": "test-model",
            "candidates": [
                {
                    "name": "Charlotte",
                    "sources": ["fact_check", "transcript"],
                    "event_sources": ["fact_check", "transcript"],
                    "event_support_groups": ["fact_check", "current_video"],
                    "evidence": [
                        {
                            "category": "fact_check",
                            "support_group": "fact_check",
                            "source": "https://example.com/check",
                            "text": "A topical Charlotte mention.",
                            "score": 0.99,
                            "event_location": True,
                        }
                    ],
                }
            ],
        }
        apply_candidates(extra, result)
        self.assertEqual(
            extra["verification"]["location"]["verified_location"],
            "Charlottesville",
        )
        self.assertEqual(
            extra["verification"]["location"]["verified_coordinates"],
            {"latitude": 1.0, "longitude": 2.0},
        )

    def test_ranks_multiple_geocoder_candidates_with_context(self):
        rows = [
            {
                "display_name": "Springfield, Illinois, United States",
                "importance": 0.8,
                "address": {"city": "Springfield", "state": "Illinois"},
            },
            {
                "display_name": "Springfield, Massachusetts, United States",
                "importance": 0.5,
                "address": {"city": "Springfield", "state": "Massachusetts"},
            },
        ]
        selected, details = select_geocode_candidate(
            "Springfield", rows, context=["Massachusetts"]
        )
        self.assertEqual(selected["address"]["state"], "Massachusetts")
        self.assertEqual(details["selected_index"], 1)

    def test_reverse_image_candidates_update_earliest_provenance(self):
        extra = {
            "normalized_video_information": {"video_url": "https://current.example/video"},
            "verification": {
                "provenance": {
                    "provenance_status": "unknown",
                    "earliest_known_url": "https://current.example/video",
                    "earliest_known_date": "2024-01-02",
                    "near_duplicate_matches": [],
                    "previous_context_summary": None,
                    "provenance_mismatch": None,
                },
                "date": {
                    "video_upload_date": "2024-01-02",
                    "earliest_online_date": None,
                },
                "source": {"original_source_name": None},
                "motivation": {
                    "original_caption": "Current caption",
                    "original_description": None,
                },
            },
            "field_evidence": {},
        }
        page = {
            "url": "https://old.example/page",
            "canonical_url": "https://old.example/original",
            "title": "Earlier report",
            "description": "The footage depicts an earlier event.",
            "author": "Example News",
            "published_at": "2018-02-10",
            "archive_first_seen": "2018-02-11",
            "match_types": ["full"],
            "matched_frames": ["1.jpeg"],
            "match_count": 1,
        }
        transcript_result = {
            "url": "https://archive.example/transcript",
            "canonical_url": "https://archive.example/original",
            "title": "Original transcript page",
            "description": (
                "The footage was originally recorded with this distinctive sentence "
                "from the original recording."
            ),
            "query": '"distinctive sentence from the original recording"',
            "query_type": "transcript",
            "published_at": "2017-01-03",
        }
        apply_search_provenance(
            extra,
            {"pages": [page]},
            {
                "https://old.example/page": "image-1",
                "https://archive.example/transcript": "search-1",
            },
            [transcript_result],
        )
        provenance = extra["verification"]["provenance"]
        self.assertEqual(provenance["provenance_status"], "earlier version found")
        self.assertEqual(
            provenance["earliest_known_url"], "https://archive.example/original"
        )
        self.assertEqual(provenance["earliest_known_date"], "2017-01-03")
        self.assertIn("Original transcript page", provenance["previous_context_summary"])
        self.assertEqual(
            extra["verification"]["date"]["earliest_online_date"],
            "2017-01-03",
        )
        self.assertEqual(
            {match["source"] for match in provenance["near_duplicate_matches"]},
            {"reverse_image", "transcript_search"},
        )

    def test_archive_date_before_upload_marks_an_earlier_version(self):
        extra = {
            "normalized_video_information": {
                "video_url": "https://www.youtube.com/embed/example"
            },
            "verification": {
                "provenance": {
                    "provenance_status": "unknown",
                    "earliest_known_url": "https://youtube.com/watch?v=example",
                    "earliest_known_date": "2018-01-01",
                    "near_duplicate_matches": [],
                    "previous_context_summary": None,
                },
                "date": {
                    "video_upload_date": "2024-01-01",
                    "earliest_online_date": "2018-01-01",
                },
                "source": {},
                "motivation": {},
            },
            "field_evidence": {
                "verification.provenance.earliest_known_date": ["archive-1"]
            },
        }
        apply_search_provenance(extra, {"pages": []})
        self.assertEqual(
            extra["verification"]["provenance"]["provenance_status"],
            "earlier version found",
        )

    def test_supplies_text_search_candidates_and_specific_field_evidence(self):
        record = {"data": sample("1", "https://example.com/video")}
        extra = {
            "verification": {},
            "evidence": [
                {
                    "id": "source-1",
                    "type": "provenance_search_candidate",
                    "source": "https://example.com/ap",
                    "observation": "Associated Press published the footage.",
                },
                {
                    "id": "claim-1",
                    "type": "claim_text",
                    "source": "record.json",
                    "observation": "An unrelated claim.",
                },
            ],
            "field_evidence": {},
            "automation": {
                "search": {
                    "status": "ok",
                    "search_provider": "DDGS",
                    "results": [
                        {
                            "url": "https://example.com/ap",
                            "title": "Earlier source",
                            "description": "Associated Press published the footage.",
                            "rank": 1,
                        }
                    ],
                }
            },
        }
        supplied = text_input(record, extra)
        self.assertEqual(
            supplied["web_search_candidates"]["results"][0]["title"],
            "Earlier source",
        )
        self.assertEqual(
            field_evidence_ids(
                extra, "original_source_name", "source", "Associated Press"
            ),
            ["source-1"],
        )

    def test_location_stage_reads_retrieved_pages(self):
        record = {"claim_id": "1", "data": sample("1", "https://example.com/video")}
        extra = {
            "source_file": "train_val/1.json",
            "normalized_video_information": {
                "video_url": "https://example.com/video",
                "video_transcript": "",
            },
            "verification": {
                "motivation": {
                    "original_caption": None,
                    "original_description": None,
                }
            },
            "evidence": [],
            "automation": {
                "search": {
                    "results": [
                        {
                            "url": "https://news.example/report",
                            "title": "Footage filmed in Brussels",
                            "context_excerpt": "The event occurred in Brussels.",
                        }
                    ]
                }
            },
        }
        sources = build_sources(record, extra)
        retrieved = [item for item in sources if item["category"] == "retrieved_page"]
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0]["support_group"], "retrieved:news.example")
        self.assertIn("Brussels", retrieved[0]["text"])

    def test_audit_reports_unsupported_fields_dangling_links_and_stage_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            verification = empty_verification()
            verification["motivation"]["claimed_framing"] = "Example claim"
            extra = {
                "claim_id": "1",
                "claim_components": [{"id": "claim_1", "text": "Example claim"}],
                "normalized_video_information": {
                    "platform": "youtube",
                    "video_url": "https://example.com/video",
                    "video_date": "2024-01-01",
                    "video_transcript": None,
                },
                "verification": verification,
                "evidence": [],
                "field_evidence": {
                    "verification.motivation.claimed_framing": ["missing-1"]
                },
                "automation": {
                    "vision": {"status": "parse_error", "error": "invalid JSON"}
                },
                "review": {"status": "automated"},
            }
            annotation = output / "annotations" / "1.json"
            annotation.parent.mkdir(parents=True)
            annotation.write_text(json.dumps(extra), encoding="utf-8")

            report = run_audit([{"claim_id": "1"}], output)

            self.assertEqual(report["unsupported_non_null_fields"], 1)
            self.assertEqual(report["dangling_field_evidence"], 1)
            self.assertEqual(report["stage_issues"], 1)
            self.assertTrue((output / "enrichment_report.json").exists())

    def test_rejects_unnormalized_or_stale_clip_caches(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "clip.npz"
            np.savez_compressed(
                cache,
                embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
                frames=np.array(["1.jpeg", "2.jpeg"]),
            )
            self.assertTrue(
                cached_embeddings_are_normalized(cache, ["1.jpeg", "2.jpeg"])
            )
            self.assertFalse(cached_embeddings_are_normalized(cache, ["1.jpeg"]))

            np.savez_compressed(
                cache,
                embeddings=np.array([[2.0, 0.0]], dtype="float32"),
                frames=np.array(["1.jpeg"]),
            )
            self.assertFalse(cached_embeddings_are_normalized(cache, ["1.jpeg"]))

    def test_vision_stage_passes_real_images_to_the_processor(self):
        from PIL import Image

        class FakeInputs(dict):
            def to(self, device):
                self.device = device
                return self

        class FakeInputIds:
            shape = (1, 2)

        class FakeProcessor:
            instance = None

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.instance = cls()
                return cls.instance

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return FakeInputs(input_ids=FakeInputIds())

            def decode(self, values, skip_special_tokens=True):
                return json.dumps(
                    {
                        "scene_summary": "A blue test frame.",
                        "ocr_text": [],
                        "landmarks": [],
                        "languages": [],
                        "visual_clues": [],
                        "candidate_locations": [],
                    }
                )

        class FakeModel:
            device = "cpu"

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def to(self, device):
                self.device = device
                return self

            def eval(self):
                return self

            def generate(self, **kwargs):
                return [[0, 1, 2]]

        class InferenceMode:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: False,
                empty_cache=lambda: None,
            ),
            float16="float16",
            float32="float32",
            inference_mode=InferenceMode,
        )
        fake_transformers = SimpleNamespace(
            AutoProcessor=FakeProcessor,
            Qwen3VLForConditionalGeneration=FakeModel,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            frames = source / "train_val_output" / "1"
            frames.mkdir(parents=True)
            Image.new("RGB", (16, 16), "blue").save(frames / "1.jpeg")

            verification = empty_verification()
            sidecar = {
                "claim_id": "1",
                "normalized_video_information": {
                    "platform": "youtube",
                    "video_url": "https://example.com/video",
                    "video_date": "2024-01-01",
                    "video_transcript": None,
                },
                "verification": verification,
                "evidence": [],
                "field_evidence": {},
                "automation": {},
                "review": {"status": "not_started"},
            }
            annotation = output / "annotations" / "1.json"
            annotation.parent.mkdir(parents=True)
            annotation.write_text(json.dumps(sidecar), encoding="utf-8")
            record = {
                "claim_id": "1",
                "legacy_split": "train_val",
                "data": sample("1", "https://example.com/video"),
            }

            with patch.dict(
                "sys.modules",
                {"torch": fake_torch, "transformers": fake_transformers},
            ):
                run_vision(
                    [record],
                    source,
                    output,
                    output / "models",
                    force=True,
                    offline=True,
                )

            image_item = FakeProcessor.instance.messages[0]["content"][0]
            self.assertEqual(image_item["type"], "image")
            self.assertIsInstance(image_item["image"], Image.Image)
            cached = json.loads(
                (output / "cache" / "vision" / "1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(cached["scene_summary"], "A blue test frame.")

    def test_perceptual_hash_is_stable_for_same_image(self):
        with tempfile.TemporaryDirectory() as directory:
            from PIL import Image

            path = Path(directory) / "image.png"
            Image.new("RGB", (32, 32), "white").save(path)
            self.assertEqual(image_dhash(path), image_dhash(path))
        self.assertEqual(hash_distance(0b1010, 0b0011), 2)

    def test_normalization(self):
        self.assertEqual(normalize_date(20170904.0), "2017-09-04")
        self.assertEqual(normalize_date("2024-02-29"), "2024-02-29")
        self.assertIsNone(normalize_date("20240230"))
        self.assertEqual(normalize_platform("Twitter"), "x")
        self.assertEqual(normalize_platform("youtu"), "youtube")

    def test_youtube_urls_share_a_group(self):
        embed = canonical_video_key("https://www.youtube.com/embed/abc123", "1")
        watch = canonical_video_key("https://youtube.com/watch?v=abc123", "2")
        self.assertEqual(embed, watch)

    def test_builds_sidecars_and_keeps_duplicate_urls_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            (source / "train_val").mkdir(parents=True)
            (source / "test").mkdir()

            first = sample("1", "https://youtube.com/embed/shared", transcript="  words  ")
            second = sample("2", "https://youtu.be/shared", rating="False", transcript="")
            (source / "train_val" / "1.json").write_text(json.dumps(first), encoding="utf-8")
            (source / "test" / "2.json").write_text(json.dumps(second), encoding="utf-8")

            report = build(source, output, seed=7)

            self.assertEqual(report["samples"], 2)
            self.assertEqual(report["missing_transcripts"], 1)
            self.assertEqual(report["url_duplicate_groups"], 1)
            sidecar = json.loads((output / "annotations" / "1.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["claim_id"], "1")
            self.assertEqual(sidecar["normalized_video_information"]["platform"], "youtube")
            self.assertEqual(sidecar["normalized_video_information"]["video_transcript"], "  words  ")
            self.assertIsNone(sidecar["verification"]["provenance"]["earliest_known_url"])
            second_sidecar = json.loads((output / "annotations" / "2.json").read_text(encoding="utf-8"))
            self.assertIsNone(second_sidecar["normalized_video_information"]["video_transcript"])

            membership = {}
            for split in ("train", "validation", "test"):
                for claim_id in (output / "splits" / (split + ".txt")).read_text().splitlines():
                    membership[claim_id] = split
            self.assertEqual(membership["1"], membership["2"])
            expected_pilot_size = 2 if membership["1"] == "train" else 0
            self.assertEqual(report["pilot_samples"], expected_pilot_size)


if __name__ == "__main__":
    unittest.main()
