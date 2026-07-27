# Pillar 1: Provenance

**Goal.** Determine whether the video is original, reused, reposted, manipulated, cropped, or presented outside its original context.

## Questions

- **Q1.** Was the same or near-duplicate video, keyframe, audio segment, or transcript phrase available online before the claimed context?
- **Q2.** What is the earliest known online version of the video or visually matching keyframe sequence?
- **Q3.** Is the video an original upload, repost, edited excerpt, cropped version, compilation, or screen recording?
- **Q4.** Did earlier versions of the video have a different title, caption, description, or event context?
- **Q5.** Does the provenance evidence contradict the claim's asserted event, date, location, source, or framing?

## Required Fields and Tools

| **Required field** | **Retrieval tool(s)** | **Tool input and output** |
| --- | --- | --- |
| `provenance_status` | Rule-based aggregator over visual-search, transcript-search, and archive results | Input: duplicate evidence and dates. Output: original, earlier version found, repost, edited excerpt, compilation, screen recording, or unknown. |
| `earliest_known_url` | Reverse image search (RIS) | Input: thumbnail or sampled keyframes. Output: earliest matching page or platform URL. |
| `earliest_known_date` | `Wayback Machine CDX API`, page-metadata scraper, platform-metadata scraper | Input: candidate duplicate URL. Output: first archived date, publication date, or upload date. |
| `near_duplicate_matches` | `CLIPChunk` keyframes, CLIP similarity | Input: sampled frames or video thumbnails. Output: matching frames, duplicate URLs, and similarity scores. |
| `previous_context_summary` | `Trafilatura`, `BeautifulSoup`, LLM summarizer | Input: earliest matching pages and captions. Output: concise summary of the original context. |
| `provenance_mismatch` | NLI model | Input: claimed context and previous context summary. Output: Boolean mismatch label with rationale. |

## Workflow

1. **Input.** Video file, video URL, thumbnail, title, description, transcript, claim text, and metadata.
2. **Keyframe extraction.** Extract representative keyframes from the video and select distinctive transcript phrases likely to be searchable.
3. **Reverse visual search.** Apply thumbnails/keyframes to RIS tools to retrieve visually similar images, matching frames, near-duplicate posts, and possible earlier uses.
4. **Transcript and text search.** Search exact transcript phrases, video title, caption, and distinctive named entities to retrieve earlier copies that may not be found through visual search.
5. **Archive and date checking.** Query archive services and page metadata for each candidate duplicate URL to identify the earliest known online version.
6. **Duplicate clustering.** Group visually or textually similar results into duplicate clusters and rank them by publication date, source reliability, and similarity score.
7. **Context comparison.** Compare the earliest known context with the claim context to determine whether the claim reuses the video for a different event, date, location, source, or framing.
8. **Output.** Fill data fields.

# Pillar 2: Source

**Goal.** Identify who originally created, captured, published, or broadcast the video, while distinguishing the original source from the current uploader.

## Questions

- **Q1.** Who uploaded the video or the version used in the fact-check article?
- **Q2.** Is the uploader also the original creator, publisher, broadcaster, or eyewitness source?
- **Q3.** Who is the earliest or most authoritative source associated with the video?
- **Q4.** What type of source is it: eyewitness, news outlet, news agency, official account, political actor, activist group, entertainment source, satire source, or unknown?
- **Q5.** Does the source evidence support or contradict the claim's attribution of the video?

## Required Fields and Tools

| **Required field** | **Retrieval tool(s)** | **Tool input and output** |
| --- | --- | --- |
| `uploader_name` | `yt-dlp`, metadata, `InVID` analysis | Input: `video_url`. Output: uploader, channel, account, or page name. |
| `uploader_profile` | `yt-dlp`, metadata | Input: `video_url`. Output: public channel or profile URL. |
| `original_source_name` | Scraper, metadata, source extractor | Input: `earliest_known_url` and duplicate pages. Output: original creator, outlet, agency, or account. |
| `source_type` | LLM classifier | Input: source name, domain, and profile description. Output: source category. |
| `source_is_uploader` | Rule-based entity matcher | Input: current uploader and original source. Output: Boolean value. |
| `source_mismatch` | NLI model | Input: claim attribution and original-source fields. Output: Boolean mismatch label with rationale. |

## Workflow

1. **Input.** Use the video URL, uploader metadata, earliest known URL, duplicate pages, captions, and retrieved provenance evidence.
2. **Current-uploader extraction.** Retrieve the uploader name, channel name, account ID, and public profile URL from the platform page or metadata-extraction tool.
3. **Earliest-source inspection.** Inspect the earliest known video page and duplicate pages to identify whether another creator, agency, outlet, eyewitness, or official account is credited.
4. **Source normalization.** Normalize source names across pages by resolving aliases, channel names, outlet names, and account handles.
5. **Uploader–source comparison.** Compare the current uploader with the earliest or most authoritative source to decide whether the uploader is also the original source.
6. **Claim-attribution comparison.** Compare the claim's attribution with the recovered original source and decide whether there is a source mismatch.
7. **Output.** Fill data fields.

# Pillar 3: Date

**Goal.** Establish when the video was captured, when the depicted event occurred, and whether that date matches the claim. The upload date should not be treated as the capture date by default.

## Questions

- **Q1.** What date or time period does the claim explicitly or implicitly assert?
- **Q2.** What is the upload or publication date of the video version?
- **Q3.** What is the earliest known online date of the same or near-duplicate video?
- **Q4.** What is the most plausible capture date or event date supported by evidence?
- **Q5.** Does the estimated capture or event date match the claimed date?

## Required Fields and Tools

| **Required field** | **Retrieval tool(s)** | **Tool input and output** |
| --- | --- | --- |
| `claimed_date` | NER, LLM extractor | Input: `claim`, headline, description, transcript, and article content. Output: normalized claimed date or date range. |
| `video_upload_date` | `video_date`, `yt-dlp`, metadata | Input: JSON and video URL. Output: platform upload or publication date. |
| `earliest_online_date` | `Wayback Machine CDX API`, TinEye date sorting, metadata | Input: duplicate URL or earliest candidate URL. Output: earliest archived, indexed, or published date. |
| `estimated_date` | Event search, official statements, metadata, transcript | Input: event keywords, place, actors, source, and transcript phrases. Output: date or date range. |
| `capture_date_granularity` | Date normalizer | Input: estimated date. Output: day, month, year, range, or unknown. |
| `date_mismatch_type` | Rule-based comparator | Input: claimed date and estimated capture or event date. Output: same, older video, newer video, wrong event date, or unknown. |

## Workflow

1. **Claim-date extraction.** Extract explicit and implicit temporal expressions from the claim, title, description, transcript, and article text.
2. **Upload-date extraction.** Retrieve the platform upload date from the dataset metadata, platform page, or video-metadata extraction tool.
3. **Earliest-online-date retrieval.** Use provenance results, archive records, duplicate pages, and metadata to identify the earliest known online date of the same or near-duplicate video.
4. **Event-date retrieval.** Search event keywords, actor names, location names, transcript phrases, and source information to retrieve external evidence about when the depicted event occurred.
5. **Capture-date estimation.** Combine upload date, earliest online date, event date, transcript clues, and external evidence to estimate the most plausible capture or event date.
6. **Date normalization.** Normalize the estimated date into day-level, month-level, year-level, range-level, or unknown granularity.
7. **Temporal mismatch detection.** Compare the claimed date with the estimated capture or event date and assign a mismatch type.
8. **Output.** Fill data fields.

# Pillar 4: Location

**Goal.** Determine where the video was captured or where the depicted event occurred, then compare that verified location with the claimed location.

## Questions

- **Q1.** What location does the claim explicitly or implicitly assert?
- **Q2.** What candidate locations are mentioned in the transcript, title, description, captions, or retrieved pages?
- **Q3.** What visual clues are visible in keyframes, such as landmarks, road signs, shop names, terrain, language, license plates, uniforms, or weather?
- **Q4.** What is the most specific verified location supported by map, visual, textual, or external evidence?
- **Q5.** Does the verified location match the claimed location at the venue, city, region, or country level?

## Required Fields and Tools

| **Required field** | **Retrieval tool(s)** | **Tool input and output** |
| --- | --- | --- |
| `claimed_location` | Location NER, LLM extractor, geocoder | Input: claim, title, description, transcript, and article text. Output: claimed place name and normalized location. |
| `candidate_locations` | NER, Wikidata, GeoNames, Nominatim, retrieved-page parser | Input: text from the claim, transcript, and evidence pages. Output: list of candidate places. |
| `visual_location_clues` | OCR, object detection, logo detection, landmark recognition, manual keyframe review, or MLLMs | Input: keyframes. Output: signs, buildings, landmarks, terrain, road markings, license plates, weather clues, and related visual evidence. |
| `verified_location` | Nominatim, GeoNames, Google Geocoding API, OpenStreetMap, Google Maps, Google Earth, Mapillary | Input: candidate places and visual clues. Output: verified place name. |
| `verified_coordinates` | Nominatim, GeoNames, Google Geocoding API | Input: verified location. Output: latitude and longitude. |
| `location_mismatch_type` | Spatial comparator and administrative-hierarchy matcher | Input: claimed and verified locations. Output: same, different city, different region, different country, or unknown. |

## Workflow

1. **Claim-location extraction.** Extract claimed locations from the claim, title, description, transcript, and article text.
2. **Candidate-location generation.** Extract locations mentioned in retrieved pages, captions, source pages, transcript segments, and provenance evidence.
3. **Visual-clue extraction.** Inspect keyframes using OCR, object detection, logo detection, landmark recognition, MLLMs, or manual review to identify signs, buildings, landmarks, terrain, road markings, license plates, uniforms, language, and weather cues.
4. **Geocoding.** Use geocoding tools to normalize candidate locations into structured place names, administrative levels, and coordinates.
5. **Map-based verification.** Cross-check candidate locations against maps, satellite imagery, street-level imagery, road layouts, landmarks, terrain, and visible textual clues.
6. **Location selection.** Select the most specific verified location supported by the strongest combination of textual, visual, map, and external evidence.
7. **Spatial mismatch detection.** Compare the claimed and verified locations using administrative hierarchy and geographic distance.
8. **Output.** Fill data fields.

# Pillar 5: Motivation

**Goal.** Infer the original purpose or framing of the video and determine whether the current claim reframes that purpose.

## Questions

- **Q1.** What purpose or framing does the claim assign to the video?
- **Q2.** What was the likely original purpose of the video according to the earliest caption, description, source, or event context?
- **Q3.** Was the video originally eyewitness documentation, news reporting, official communication, campaign material, activism, entertainment, satire, advertisement, archive footage, or something else?
- **Q4.** Has the video been reframed by a new caption, claim, edited excerpt, or misleading context?
- **Q5.** Does the motivation or framing mismatch help explain the verification label?

## Required Fields and Tools

| **Required field** | **Retrieval tool(s)** | **Tool input and output** |
| --- | --- | --- |
| `claimed_framing` | LLM extractor | Input: claim, title, description, transcript, and article text. Output: what the claim implies the video shows or proves. |
| `original_caption` | `yt-dlp`, metadata, scraper | Input: earliest known URL. Output: original caption or title. |
| `original_description` | `yt-dlp`, metadata, scraper, `Trafilatura` | Input: earliest known page or video page. Output: original description or surrounding page text. |
| `original_context_category` | LLM classifier | Input: original caption, description, source type, and event context. Output: news report, eyewitness, official record, campaign, activism, entertainment, satire, advertisement, archive, or unknown. |
| `motivation_mismatch_type` | LLM-as-a-judge | Input: claimed framing and original context category. Output: same, satire as real, entertainment as news, old news as current, political reframing, or unknown. |

## Workflow

1. **Claim-framing extraction.** Extract what the claim implies the video shows, proves, documents, exposes, or represents.
2. **Original-caption retrieval.** Retrieve the original caption, title, and description from the earliest known video page or most authoritative source page.
3. **Source-context analysis.** Use the source type, source profile, page context, and event context to infer why the video was originally produced or published.
4. **Context-category classification.** Classify the original context as news reporting, eyewitness documentation, official communication, campaign material, activism, entertainment, satire, advertisement, archive footage, or unknown.
5. **Reframing detection.** Compare the claimed framing with the original caption, description, source type, and context category to determine whether the video has been reframed.
6. **Motivation-mismatch classification.**
7. **Prediction.** Determine whether the motivation mismatch helps explain the final decision.
8. **Output.** Fill data fields.
