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


def probe_video(url):
    """Return whether one public URL exposes a video-bearing format."""
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
        "retries": 0,
        "fragment_retries": 0,
        "extractor_retries": 0,
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=False)
        if info and info.get("entries"):
            info = next((entry for entry in info["entries"] if entry), None)
        formats = (info or {}).get("formats") or []
        if any(str(item.get("vcodec") or "none").lower() != "none" for item in formats):
            return {"decision": "keep", "reason": "video_format_found", "error": None}
        result = {"decision": "discard", "reason": "no_video_format", "error": None}
    except Exception as error:
        message = re.sub(r"\s+", " ", str(error)).strip()[:500]
        result = {"decision": "discard", "reason": "probe_failed", "error": message}

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
                "reason": "video_format_found",
                "successful_url": url,
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
    all_ids = {record["claim_id"] for record in records}
    assert len(rows) == len(kept_ids) + len(discarded_ids)
    assert not set(kept_ids).intersection(discarded_ids)
    assert set(kept_ids).union(discarded_ids) == all_ids

    total = len(rows)
    summary = {
        "total_records": total,
        "kept_records": len(kept_ids),
        "discarded_records": len(discarded_ids),
        "kept_percentage": round(100 * len(kept_ids) / total, 2) if total else 0,
        "discarded_percentage": (
            round(100 * len(discarded_ids) / total, 2) if total else 0
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
            "by_platform": breakdown(rows, "platform"),
            "by_split": breakdown(rows, "split"),
            "groups": group_results,
            "records": rows,
        },
        kept_ids,
        discarded_ids,
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
                f"| {title} | Tổng | Giữ lại | Đã loại |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for name, counts in values.items():
            lines.append(
                f"| {markdown_cell(name)} | {counts['total']} | "
                f"{counts['kept']} | {counts['discarded']} |"
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
    return "\n".join(lines) + "\n"


def write_outputs(output, result, kept_ids, discarded_ids):
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

    result, kept_ids, discarded_ids = analyze(args.source)
    write_outputs(args.output, result, kept_ids, discarded_ids)
    summary = result["summary"]
    print(f"Đã kiểm tra: {summary['total_records']} bản ghi")
    print(
        f"Giữ lại: {summary['kept_records']} bản ghi ({summary['kept_percentage']:.2f}%)"
    )
    print(
        f"Đã loại: {summary['discarded_records']} bản ghi "
        f"({summary['discarded_percentage']:.2f}%)"
    )


if __name__ == "__main__":
    main()
