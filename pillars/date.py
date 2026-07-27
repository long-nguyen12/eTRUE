"""Claimed, upload, and estimated-date validation rules."""

import re
from datetime import datetime


DATE_MISMATCHES = {"same", "older video", "newer video", "wrong event date", "unknown"}
DATE_GRANULARITIES = {"day", "month", "year", "range", "unknown"}


def first_year(value):
    match = re.search(r"(?:19|20)\d{2}", str(value or ""))
    return int(match.group(0)) if match else None


def parse_date_parts(value):
    text = str(value or "").strip()
    formats = (
        ("%Y-%m-%d", "day"),
        ("%B %d, %Y", "day"),
        ("%b %d, %Y", "day"),
        ("%d %B %Y", "day"),
        ("%d %b %Y", "day"),
        ("%Y-%m", "month"),
        ("%B %Y", "month"),
        ("%b %Y", "month"),
        ("%Y", "year"),
    )
    for date_format, granularity in formats:
        try:
            parsed = datetime.strptime(text, date_format)
            return parsed.year, parsed.month, parsed.day, granularity
        except ValueError:
            pass
    return None


def infer_date_granularity(value):
    if re.search(r"\b(between|from)\b.+\b(and|to|through|until)\b", str(value or ""), re.I):
        return "range"
    parts = parse_date_parts(value)
    return parts[3] if parts else "unknown"


def date_is_grounded(value, grounding_text):
    parts = parse_date_parts(value)
    if not parts:
        return False
    year, month, day, granularity = parts
    evidence = str(grounding_text or "").lower()
    if granularity == "year":
        return str(year) in evidence

    month_date = datetime(year, month, 1)
    month_names = (
        month_date.strftime("%B").lower(),
        month_date.strftime("%b").lower(),
    )
    if granularity == "month":
        return any(
            re.search(r"\b" + name + r"\b.{0,15}\b" + str(year) + r"\b", evidence)
            for name in month_names
        )

    patterns = [r"\b" + str(year) + r"[-/]0?" + str(month) + r"[-/]0?" + str(day) + r"\b"]
    for name in month_names:
        patterns.extend(
            [
                r"\b" + name + r"\s+0?" + str(day) + r"(?:st|nd|rd|th)?(?:,)?\s+" + str(year) + r"\b",
                r"\b0?" + str(day) + r"(?:st|nd|rd|th)?\s+" + name + r"\s+" + str(year) + r"\b",
            ]
        )
    return any(re.search(pattern, evidence) for pattern in patterns)


def apply_date_rules(verification):
    date_fields = verification["date"]
    claimed_date = date_fields.get("claimed_date")
    estimated_date = date_fields.get("estimated_date")
    upload_date = date_fields.get("video_upload_date")

    if not claimed_date or not estimated_date:
        date_fields["date_mismatch_type"] = None
    if not estimated_date:
        date_fields["capture_date_granularity"] = None
        return

    estimated_year = first_year(estimated_date)
    upload_year = first_year(upload_date)
    if estimated_year and upload_year and estimated_year > upload_year:
        date_fields["estimated_date"] = None
        date_fields["capture_date_granularity"] = None
        date_fields["date_mismatch_type"] = None
        return

    date_fields["capture_date_granularity"] = infer_date_granularity(estimated_date)
    if not claimed_date:
        return

    mismatch = date_fields.get("date_mismatch_type")
    if claimed_date == estimated_date and mismatch != "same":
        date_fields["date_mismatch_type"] = None
    if claimed_date != estimated_date and mismatch == "same":
        date_fields["date_mismatch_type"] = None

    claimed_year = first_year(claimed_date)
    if not claimed_year or not estimated_year:
        return
    if estimated_year < claimed_year and mismatch == "newer video":
        date_fields["date_mismatch_type"] = None
    if estimated_year > claimed_year and mismatch == "older video":
        date_fields["date_mismatch_type"] = None
