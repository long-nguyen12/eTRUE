"""Check which TRUE records still expose a public video stream online."""

import argparse
import html
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

from utils.files import now, write_json
from utils.records import canonical_video_key, load_records, normalize_platform
from utils.utils import normalize_social_url

REASON_LABELS = {
    "missing_url": "Thiếu URL video",
    "invalid_url": "URL không hợp lệ",
    "no_video_format": "Không tìm thấy luồng video",
    "probe_failed": "yt-dlp không thể đọc URL",
    "cnbc_page_unavailable": "Không thể đọc trang CNBC",
    "all_equivalent_urls_failed": "Tất cả URL tương đương đều thất bại",
    "probe_inconclusive": "yt-dlp chưa xác định được trạng thái video",
    "all_equivalent_urls_inconclusive": "Tất cả URL tương đương chưa xác định được",
}


def is_cnbc_url(url):
    hostname = (urlparse(url).hostname or "").lower()
    return hostname == "cnbc.com" or hostname.endswith(".cnbc.com")


def probe_cnbc_page(url, video_result):
    """Keep a CNBC record when its article metadata is still public."""
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
            "Range": "bytes=0-999999",
        },
    )
    try:
        print(f"  Probing CNBC page: {url}", flush=True)
        with build_opener(ProxyHandler({})).open(request, timeout=30) as response:
            if not is_cnbc_url(response.geturl()):
                raise ValueError("CNBC URL redirected to another domain")
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"
            page = response.read(1_000_000).decode(charset, errors="replace")
        if content_type != "text/html":
            raise ValueError("CNBC response is not HTML")
        match = re.search(r"<title[^>]*>(.*?)</title>", page, re.I | re.S)
        title = (
            re.sub(r"\s+", " ", html.unescape(match.group(1))).strip() if match else ""
        )
        if not title or re.search(r"access denied|captcha|just a moment", title, re.I):
            raise ValueError("CNBC page does not contain usable metadata")
        return {
            "decision": "keep",
            "reason": "cnbc_page_metadata_found",
            "error": None,
            "page_title": title[:300],
            "video_error": video_result["error"] or video_result["reason"],
        }
    except Exception as error:
        page_error = re.sub(r"\s+", " ", str(error)).strip()[:500]
        video_error = video_result["error"] or video_result["reason"]
        return {
            "decision": "discard",
            "reason": "cnbc_page_unavailable",
            "error": f"yt-dlp: {video_error}; CNBC page: {page_error}",
        }


VIDEO_EXTENSIONS = {"mp4", "webm", "mov", "mkv", "flv", "avi", "m4v", "ts"}


def _format_has_video(item):
    """Return True when one yt-dlp format/result has positive video evidence."""
    if not isinstance(item, dict) or not item.get("url"):
        return False

    vcodec = item.get("vcodec")
    if isinstance(vcodec, str) and vcodec.lower() == "none":
        return False
    if vcodec:
        return True

    if item.get("height") or item.get("width") or item.get("fps"):
        return True

    ext = str(item.get("ext") or "").lower()
    if ext in VIDEO_EXTENSIONS:
        return True

    path = urlparse(str(item.get("url") or "")).path.lower()
    return any(path.endswith(f".{ext}") for ext in VIDEO_EXTENSIONS)


def has_video_stream(info):
    """Recursively detect positive video evidence in a yt-dlp info result."""
    if not isinstance(info, dict):
        return False

    # Playlist / multi-video results: inspect every entry, not only the first one.
    for entry in info.get("entries") or []:
        if has_video_stream(entry):
            return True

    # Standard yt-dlp formats.
    for item in info.get("formats") or []:
        if _format_has_video(item):
            return True

    # Some processed results expose selected/requested formats separately.
    for item in info.get("requested_formats") or []:
        if _format_has_video(item):
            return True

    # Extractors are allowed to return a direct media URL instead of `formats`.
    if _format_has_video(info):
        return True

    return False


def probe_video(url):
    """Probe one URL. Unknown yt-dlp results are not treated as dead content."""
    url = normalize_social_url(url)["url"]
    if not url:
        return {"decision": "discard", "reason": "missing_url", "error": None}

    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return {"decision": "discard", "reason": "invalid_url", "error": None}
    except ValueError:
        return {"decision": "discard", "reason": "invalid_url", "error": None}

    import yt_dlp

    options = {
        "skip_download": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "no_color": True,
        "proxy": "",
        "socket_timeout": 30,
        # A transient request should not immediately become a discarded record.
        "retries": 2,
        "fragment_retries": 1,
        "extractor_retries": 2,
        # Let us inspect the metadata even when yt-dlp cannot build formats.
        "ignore_no_formats_error": True,
    }

    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=False)

        if has_video_stream(info):
            return {"decision": "keep", "reason": "video_format_found", "error": None}

        # This means only that yt-dlp did not expose a video stream.
        # It does NOT prove that the webpage/post contains no video.
        result = {
            "decision": "unknown",
            "reason": "no_video_format",
            "error": None,
        }
    except Exception as error:
        message = re.sub(r"\s+", " ", str(error)).strip()[:500]
        result = {
            "decision": "unknown",
            "reason": "probe_failed",
            "error": message,
        }

    # Preserve the existing CNBC fallback policy. Note that page metadata alone
    # proves the article is alive, not that a video stream is available.
    return probe_cnbc_page(url, result) if is_cnbc_url(url) else result


def check_group(records):
    urls = list(
        dict.fromkeys(
            str(
                (record["data"].get("video_information") or {}).get("video_url") or ""
            ).strip()
            for record in records
        )
    )
    checks = []
    for url in urls:
        result = probe_video(url)
        checks.append({"url": url or None, **result})
        if result["decision"] == "keep":
            return {
                "decision": "keep",
                "reason": result["reason"],
                "successful_url": url,
                "checks": checks,
            }

    # Missing/invalid URLs are confirmed discard conditions.
    # yt-dlp failures or missing formats remain unknown.
    if any(check["decision"] == "unknown" for check in checks):
        reason = (
            checks[0]["reason"]
            if len(checks) == 1
            else "all_equivalent_urls_inconclusive"
        )
        return {
            "decision": "unknown",
            "reason": reason,
            "successful_url": None,
            "checks": checks,
        }

    reason = checks[0]["reason"] if len(checks) == 1 else "all_equivalent_urls_failed"
    return {
        "decision": "discard",
        "reason": reason,
        "successful_url": None,
        "checks": checks,
    }


def breakdown(rows, field):
    values = defaultdict(Counter)
    for row in rows:
        values[row[field]]["total"] += 1
        values[row[field]][row["decision"]] += 1
    return {
        name: {
            "total": counts["total"],
            "kept": counts["keep"],
            "discarded": counts["discard"],
            "unknown": counts["unknown"],
        }
        for name, counts in sorted(values.items())
    }


def analyze(source):
    records = load_records(source)
    groups = defaultdict(list)
    for record in records:
        video = record["data"].get("video_information") or {}
        video["video_url"] = normalize_social_url(video.get("video_url"))["url"]
        key = canonical_video_key(video.get("video_url"), record["claim_id"])
        groups[key].append(record)

    group_results = {}
    for number, (key, group) in enumerate(sorted(groups.items()), 1):
        claim_ids = sorted(record["claim_id"] for record in group)
        print(
            f"Checking record {number}/{len(groups)}: {', '.join(claim_ids)}",
            flush=True,
        )
        result = check_group(group)
        group_results[key] = {
            "claim_ids": claim_ids,
            **result,
        }
        for check in result["checks"]:
            if check["decision"] == "discard":
                detail = REASON_LABELS.get(check["reason"], check["reason"])
                if check["error"]:
                    detail += ": " + check["error"]
                print(
                    f"  Error for {', '.join(claim_ids)} "
                    f"({check['url'] or 'missing URL'}): {detail}",
                    flush=True,
                )
            elif check["reason"] == "cnbc_page_metadata_found":
                print(
                    f"  Kept {', '.join(claim_ids)} from CNBC page metadata: "
                    f"{check['page_title']}",
                    flush=True,
                )

    rows = []
    for record in records:
        video = record["data"].get("video_information") or {}
        key = canonical_video_key(video.get("video_url"), record["claim_id"])
        result = group_results[key]
        rows.append(
            {
                "claim_id": record["claim_id"],
                "split": record["legacy_split"],
                "platform": normalize_platform(video.get("platform")),
                "video_url": video.get("video_url"),
                "canonical_key": key,
                "decision": result["decision"],
                "reason": result["reason"],
            }
        )
    rows.sort(key=lambda row: row["claim_id"])

    kept_ids = [row["claim_id"] for row in rows if row["decision"] == "keep"]
    discarded_ids = [row["claim_id"] for row in rows if row["decision"] == "discard"]
    unknown_ids = [row["claim_id"] for row in rows if row["decision"] == "unknown"]
    all_ids = {record["claim_id"] for record in records}
    assert len(rows) == len(kept_ids) + len(discarded_ids) + len(unknown_ids)
    assert set(kept_ids).isdisjoint(discarded_ids)
    assert set(kept_ids).isdisjoint(unknown_ids)
    assert set(discarded_ids).isdisjoint(unknown_ids)
    assert set(kept_ids).union(discarded_ids, unknown_ids) == all_ids

    total = len(rows)
    summary = {
        "total_records": total,
        "kept_records": len(kept_ids),
        "discarded_records": len(discarded_ids),
        "unknown_records": len(unknown_ids),
        "kept_percentage": round(100 * len(kept_ids) / total, 2) if total else 0,
        "discarded_percentage": (
            round(100 * len(discarded_ids) / total, 2) if total else 0
        ),
        "unknown_percentage": (
            round(100 * len(unknown_ids) / total, 2) if total else 0
        ),
    }
    return (
        {
            "generated_at": now(),
            "summary": summary,
            "discard_reasons": dict(
                sorted(
                    Counter(
                        row["reason"] for row in rows if row["decision"] == "discard"
                    ).items()
                )
            ),
            "unknown_reasons": dict(
                sorted(
                    Counter(
                        row["reason"] for row in rows if row["decision"] == "unknown"
                    ).items()
                )
            ),
            "by_platform": breakdown(rows, "platform"),
            "by_split": breakdown(rows, "split"),
            "groups": group_results,
            "records": rows,
        },
        kept_ids,
        discarded_ids,
        unknown_ids,
    )


def markdown_cell(value):
    return str(value or "").replace("\n", " ").replace("|", "\\|")


def render_report(result):
    summary = result["summary"]
    lines = [
        "# Báo cáo khả năng trích xuất video từ Internet",
        "",
        f"- Tổng số bản ghi: **{summary['total_records']}**",
        f"- Giữ lại: **{summary['kept_records']}** ({summary['kept_percentage']:.2f}%)",
        f"- Đã loại: **{summary['discarded_records']}** ({summary['discarded_percentage']:.2f}%)",
        f"- Chưa xác định: **{summary['unknown_records']}** ({summary['unknown_percentage']:.2f}%)",
        "",
        "## Lý do loại",
        "",
        "| Lý do | Số bản ghi |",
        "| --- | ---: |",
    ]
    for reason, count in result["discard_reasons"].items():
        lines.append(f"| {REASON_LABELS.get(reason, reason)} (`{reason}`) | {count} |")
    if not result["discard_reasons"]:
        lines.append("| Không có | 0 |")

    for title, values in (
        ("Nền tảng", result["by_platform"]),
        ("Split", result["by_split"]),
    ):
        lines.extend(
            [
                "",
                f"## Theo {title.lower()}",
                "",
                f"| {title} | Tổng | Giữ lại | Đã loại | Chưa xác định |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, counts in values.items():
            lines.append(
                f"| {markdown_cell(name)} | {counts['total']} | "
                f"{counts['kept']} | {counts['discarded']} | {counts['unknown']} |"
            )

    lines.extend(
        [
            "",
            "## Danh sách bản ghi bị loại",
            "",
            "| Claim ID | Split | Nền tảng | URL | Lý do |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in result["records"]:
        if row["decision"] == "discard":
            lines.append(
                f"| {markdown_cell(row['claim_id'])} | {markdown_cell(row['split'])} | "
                f"{markdown_cell(row['platform'])} | {markdown_cell(row['video_url'])} | "
                f"{REASON_LABELS.get(row['reason'], row['reason'])} |"
            )
    lines.extend(
        [
            "",
            "## Danh sách bản ghi chưa xác định",
            "",
            "| Claim ID | Split | Nền tảng | URL | Lý do |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in result["records"]:
        if row["decision"] == "unknown":
            lines.append(
                f"| {markdown_cell(row['claim_id'])} | {markdown_cell(row['split'])} | "
                f"{markdown_cell(row['platform'])} | {markdown_cell(row['video_url'])} | "
                f"{REASON_LABELS.get(row['reason'], row['reason'])} |"
            )

    return "\n".join(lines) + "\n"


def write_outputs(output, result, kept_ids, discarded_ids, unknown_ids):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "internet_extraction.json", result)
    (output / "kept_ids.txt").write_text(
        "\n".join(kept_ids) + ("\n" if kept_ids else ""),
        encoding="utf-8",
    )
    (output / "discarded_ids.txt").write_text(
        "\n".join(discarded_ids) + ("\n" if discarded_ids else ""),
        encoding="utf-8",
    )
    (output / "unknown_ids.txt").write_text(
        "\n".join(unknown_ids) + ("\n" if unknown_ids else ""),
        encoding="utf-8",
    )
    (output / "internet_extraction_report.md").write_text(
        render_report(result), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/TRUE_Dataset"))
    parser.add_argument("--output", type=Path, default=Path("data/eTRUE_filtered"))
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    result, kept_ids, discarded_ids, unknown_ids = analyze(args.source)
    write_outputs(args.output, result, kept_ids, discarded_ids, unknown_ids)
    summary = result["summary"]
    print(f"Đã kiểm tra: {summary['total_records']} bản ghi")
    print(
        f"Giữ lại: {summary['kept_records']} bản ghi ({summary['kept_percentage']:.2f}%)"
    )
    print(
        f"Đã loại: {summary['discarded_records']} bản ghi "
        f"({summary['discarded_percentage']:.2f}%)"
    )
    print(
        f"Chưa xác định: {summary['unknown_records']} bản ghi "
        f"({summary['unknown_percentage']:.2f}%)"
    )


if __name__ == "__main__":
    main()
