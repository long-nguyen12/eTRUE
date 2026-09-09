"""Claimed, upload, and estimated-date validation rules."""

import calendar
import re
from datetime import date, datetime
from typing import NamedTuple


DATE_MISMATCHES = {"same", "older video", "newer video", "wrong event date", "unknown"}
DATE_GRANULARITIES = {"day", "month", "year", "range", "unknown"}


class _ParsedDate(NamedTuple):
    start: date
    end: date
    granularity: str
    normalized: str
    endpoints: tuple


def first_year(value):
    match = re.search(r"(?:19|20)\d{2}", str(value or ""))
    return int(match.group(0)) if match else None


def _clean_date_text(value):
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", text, flags=re.I).strip(" .")


def _single_date(value, default_year=None, default_month=None):
    text = _clean_date_text(value)
    if not text:
        return None

    if re.match(r"^\d{4}-\d{2}-\d{2}[T ]", text):
        text = text[:10]

    formats = (
        ("%Y-%m-%d", "day"),
        ("%Y/%m/%d", "day"),
        ("%B %d, %Y", "day"),
        ("%b %d, %Y", "day"),
        ("%B %d %Y", "day"),
        ("%b %d %Y", "day"),
        ("%d %B %Y", "day"),
        ("%d %b %Y", "day"),
        ("%Y-%m", "month"),
        ("%Y/%m", "month"),
        ("%B %Y", "month"),
        ("%b %Y", "month"),
        ("%Y", "year"),
    )
    parsed = None
    granularity = None
    for date_format, candidate_granularity in formats:
        try:
            parsed = datetime.strptime(text, date_format)
            granularity = candidate_granularity
            break
        except ValueError:
            pass

    if parsed is None and default_year:
        for date_format, candidate_granularity in (
            ("%B %d", "day"),
            ("%b %d", "day"),
            ("%B", "month"),
            ("%b", "month"),
        ):
            try:
                partial = datetime.strptime(text, date_format)
                parsed = partial.replace(year=default_year)
                granularity = candidate_granularity
                break
            except ValueError:
                pass

    if parsed is None and default_year and default_month and re.fullmatch(r"\d{1,2}", text):
        try:
            parsed = datetime(default_year, default_month, int(text))
            granularity = "day"
        except ValueError:
            pass

    if parsed is None:
        return None

    if granularity == "day":
        start = end = parsed.date()
        normalized = start.isoformat()
    elif granularity == "month":
        start = date(parsed.year, parsed.month, 1)
        end = date(parsed.year, parsed.month, calendar.monthrange(parsed.year, parsed.month)[1])
        normalized = f"{parsed.year:04d}-{parsed.month:02d}"
    else:
        start = date(parsed.year, 1, 1)
        end = date(parsed.year, 12, 31)
        normalized = f"{parsed.year:04d}"

    parts = (parsed.year, parsed.month, parsed.day, granularity)
    return _ParsedDate(start, end, granularity, normalized, (parts,))


def _range_text_parts(value):
    text = _clean_date_text(value)
    if not text:
        return None

    same_month_days = re.fullmatch(
        r"([A-Za-z]+)\s+(\d{1,2})\s*[-–—]\s*(\d{1,2}),?\s+((?:19|20)\d{2})",
        text,
        re.I,
    )
    if same_month_days:
        month, first_day, last_day, year = same_month_days.groups()
        return f"{month} {first_day}, {year}", f"{month} {last_day}, {year}"

    month_range = re.fullmatch(
        r"([A-Za-z]+)\s*[-–—]\s*([A-Za-z]+)\s+((?:19|20)\d{2})",
        text,
        re.I,
    )
    if month_range:
        first_month, last_month, year = month_range.groups()
        return f"{first_month} {year}", f"{last_month} {year}"

    year_range = re.fullmatch(r"((?:19|20)\d{2})\s*[-–—]\s*((?:19|20)\d{2})", text)
    if year_range:
        return year_range.group(1), year_range.group(2)

    patterns = (
        r"^between\s+(.+?)\s+and\s+(.+)$",
        r"^from\s+(.+?)\s+(?:to|through|until)\s+(.+)$",
        r"^(.+?)\s+(?:to|through|until)\s+(.+)$",
        r"^(.+?)\s+[–—]\s+(.+)$",
        r"^(.+?)\s+-\s+(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, text, re.I)
        if match:
            return match.group(1), match.group(2)
    return None


def _parse_date_value(value):
    range_parts = _range_text_parts(value)
    if not range_parts:
        return _single_date(value)

    left_text, right_text = range_parts
    right = _single_date(right_text)
    left = _single_date(left_text, default_year=right.start.year if right else None)
    if left and not right:
        right = _single_date(
            right_text,
            default_year=left.start.year,
            default_month=left.start.month,
        )
    if right and not left:
        left = _single_date(
            left_text,
            default_year=right.start.year,
            default_month=right.start.month,
        )
    if not left or not right or left.start > right.end:
        return None

    return _ParsedDate(
        left.start,
        right.end,
        "range",
        left.normalized + " to " + right.normalized,
        left.endpoints + right.endpoints,
    )


def normalize_date_value(value):
    """Normalize a supported date or date range while retaining unsupported text."""
    parsed = _parse_date_value(value)
    return parsed.normalized if parsed else value


def parse_date_parts(value):
    parsed = _single_date(value)
    return parsed.endpoints[0] if parsed else None


def infer_date_granularity(value):
    parsed = _parse_date_value(value)
    return parsed.granularity if parsed else "unknown"


def _parts_are_grounded(parts, evidence, shared_year=False):
    year, month, day, granularity = parts
    if granularity == "year":
        return bool(re.search(r"\b" + str(year) + r"\b", evidence))

    month_date = datetime(year, month, 1)
    month_names = (month_date.strftime("%B").lower(), month_date.strftime("%b").lower())
    if granularity == "month":
        if shared_year:
            return bool(re.search(rf"\b{year}[-/]0?{month}\b", evidence)) or (
                str(year) in evidence
                and any(
                    re.search(r"\b" + re.escape(name) + r"\b", evidence)
                    for name in month_names
                )
            )
        return any(
            re.search(r"\b" + re.escape(name) + r"\b.{0,15}\b" + str(year) + r"\b", evidence)
            for name in month_names
        ) or bool(re.search(rf"\b{year}[-/]0?{month}\b", evidence))

    if shared_year:
        return bool(re.search(rf"\b{year}[-/]0?{month}[-/]0?{day}\b", evidence)) or (
            str(year) in evidence
            and any(
                re.search(
                    r"\b" + re.escape(name) + r"\s+0?" + str(day) + r"(?:st|nd|rd|th)?\b",
                    evidence,
                )
                or re.search(
                    r"\b0?" + str(day) + r"(?:st|nd|rd|th)?\s+" + re.escape(name) + r"\b",
                    evidence,
                )
                for name in month_names
            )
        )

    patterns = [rf"\b{year}[-/]0?{month}[-/]0?{day}\b"]
    for name in month_names:
        patterns.extend(
            [
                r"\b" + re.escape(name) + r"\s+0?" + str(day) + r"(?:st|nd|rd|th)?(?:,)?\s+" + str(year) + r"\b",
                r"\b0?" + str(day) + r"(?:st|nd|rd|th)?\s+" + re.escape(name) + r"\s+" + str(year) + r"\b",
            ]
        )
    return any(re.search(pattern, evidence) for pattern in patterns)


def date_is_grounded(value, grounding_text):
    parsed = _parse_date_value(value)
    if not parsed:
        return False
    evidence = str(grounding_text or "").lower()
    raw_value = re.sub(r"\s+", " ", str(value or "").strip().lower())
    normalized_evidence = re.sub(r"\s+", " ", evidence)
    if raw_value and raw_value in normalized_evidence:
        return True
    shared_year = parsed.granularity == "range"
    return all(
        _parts_are_grounded(parts, evidence, shared_year=shared_year)
        for parts in parsed.endpoints
    )


def compare_date_values(left, right):
    """Return -1, 0, or 1 when left is before, overlaps, or follows right."""
    left_date = _parse_date_value(left)
    right_date = _parse_date_value(right)
    if not left_date or not right_date:
        return None
    if left_date.end < right_date.start:
        return -1
    if left_date.start > right_date.end:
        return 1
    return 0


def apply_date_rules(verification):
    date_fields = verification["date"]
    claimed_date = date_fields.get("claimed_date")
    estimated_date = date_fields.get("estimated_date")
    upload_date = date_fields.get("video_upload_date")

    claimed_parsed = _parse_date_value(claimed_date)
    estimated_parsed = _parse_date_value(estimated_date)
    upload_parsed = _parse_date_value(upload_date)
    if claimed_parsed:
        date_fields["claimed_date"] = claimed_parsed.normalized
    if estimated_parsed:
        date_fields["estimated_date"] = estimated_parsed.normalized

    if not estimated_date:
        date_fields["capture_date_granularity"] = None
        date_fields["date_mismatch_type"] = None
        return

    if estimated_parsed and upload_parsed and estimated_parsed.start > upload_parsed.end:
        date_fields["estimated_date"] = None
        date_fields["capture_date_granularity"] = None
        date_fields["date_mismatch_type"] = None
        return

    date_fields["capture_date_granularity"] = (
        estimated_parsed.granularity if estimated_parsed else "unknown"
    )
    if not claimed_date:
        date_fields["date_mismatch_type"] = None
        return

    mismatch = date_fields.get("date_mismatch_type")
    if mismatch not in DATE_MISMATCHES:
        mismatch = None

    comparison = compare_date_values(estimated_date, claimed_date)
    if comparison == 0:
        date_fields["date_mismatch_type"] = "same"
    elif comparison == -1:
        date_fields["date_mismatch_type"] = (
            "wrong event date" if mismatch == "wrong event date" else "older video"
        )
    elif comparison == 1:
        date_fields["date_mismatch_type"] = (
            "wrong event date" if mismatch == "wrong event date" else "newer video"
        )
    elif str(claimed_date).strip().casefold() == str(estimated_date).strip().casefold():
        date_fields["date_mismatch_type"] = "same"
    elif mismatch == "same":
        date_fields["date_mismatch_type"] = None
    else:
        date_fields["date_mismatch_type"] = mismatch
