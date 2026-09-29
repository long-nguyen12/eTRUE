import re
from urllib.parse import urlparse, parse_qs, unquote

import requests


def normalize_social_url(url: str, resolve_redirects: bool = True) -> dict:
    original_url = url

    url = str(url or "").strip()
    url = url.strip("'\"")
    url = url.replace(r"\&", "&")
    url = url.replace("&amp;", "&")
    parsed = urlparse(url)


    host = parsed.netloc.lower()
    host_no_www = host.removeprefix("www.")

    result = {
        "original_url": original_url,
        "url": url,
        "platform": None,
        "status": "ok",
        "changed": url != original_url,
    }

    if host_no_www in {
        "facebook.com",
        "m.facebook.com",
        "web.facebook.com",
        "fb.watch",
    }:
        result["platform"] = "facebook"

        if "/plugins/video.php" in parsed.path:
            params = parse_qs(parsed.query)

            href = params.get("href")

            if href:
                decoded = unquote(href[0])

                nested = normalize_social_url(
                    decoded,
                    resolve_redirects=resolve_redirects,
                )

                nested["original_url"] = original_url
                nested["changed"] = True

                return nested

        params = parse_qs(parsed.query)

        if parsed.path.rstrip("/") == "/watch" and params.get("v"):
            video_id = params["v"][0]

            result["url"] = f"https://www.facebook.com/watch/?v={video_id}"
            result["status"] = "normalized"
            result["changed"] = result["url"] != original_url

            return result

        if "/videos/" in parsed.path:
            numbers = re.findall(r"/(\d+)(?:/|$)", parsed.path)

            if numbers:
                video_id = numbers[-1]

                result["url"] = f"https://www.facebook.com/watch/?v={video_id}"
                result["status"] = "normalized"
                result["changed"] = True

                return result

        match = re.search(r"/reel/(\d+)", parsed.path)

        if match:
            video_id = match.group(1)

            result["url"] = f"https://www.facebook.com/reel/{video_id}/"
            result["status"] = "normalized"
            result["changed"] = result["url"] != original_url

            return result

        is_redirect = (
            host_no_www == "fb.watch"
            or "/share/v/" in parsed.path
            or "/share/r/" in parsed.path
        )

        if is_redirect:
            if resolve_redirects:
                resolved = _resolve_redirect(url)

                if resolved and resolved != url:
                    nested = normalize_social_url(
                        resolved,
                        resolve_redirects=False,
                    )

                    nested["original_url"] = original_url
                    nested["changed"] = True

                    return nested

            result["status"] = "needs_redirect"
            return result

        clean = f"https://www.facebook.com{parsed.path}"

        result["url"] = clean
        result["changed"] = clean != original_url

        if result["changed"]:
            result["status"] = "normalized"

        return result

    if host_no_www in {
        "twitter.com",
        "x.com",
        "mobile.twitter.com",
        "mobile.x.com",
    }:
        result["platform"] = "x"

        match = re.search(r"/status(?:es)?/(\d+)", parsed.path)

        if match:
            status_id = match.group(1)

            result["url"] = f"https://x.com/i/web/status/{status_id}"
            result["status"] = "normalized"
            result["changed"] = result["url"] != original_url

            return result

        return result

    if host_no_www == "t.co":
        result["platform"] = "x"

        if resolve_redirects:
            resolved = _resolve_redirect(url)

            if resolved and resolved != url:
                nested = normalize_social_url(
                    resolved,
                    resolve_redirects=False,
                )

                nested["original_url"] = original_url
                nested["changed"] = True

                return nested

        result["status"] = "needs_redirect"
        return result

    if host_no_www in {
        "youtube.com",
        "m.youtube.com",
        "youtu.be",
    }:
        result["platform"] = "youtube"

        if host_no_www == "youtu.be":
            video_id = parsed.path.strip("/").split("/")[0]

            if video_id:
                result["url"] = f"https://www.youtube.com/watch?v={video_id}"
                result["status"] = "normalized"
                result["changed"] = True

            return result

        if parsed.path == "/results":
            result["status"] = "search_page"
            return result

        match = re.match(r"/embed/([^/?]+)", parsed.path)

        if match:
            video_id = match.group(1)

            result["url"] = f"https://www.youtube.com/watch?v={video_id}"
            result["status"] = "normalized"
            result["changed"] = True

            return result

        match = re.match(r"/shorts/([^/?]+)", parsed.path)

        if match:
            video_id = match.group(1)

            result["url"] = f"https://www.youtube.com/watch?v={video_id}"
            result["status"] = "normalized"
            result["changed"] = True

            return result

        if parsed.path == "/watch":
            params = parse_qs(parsed.query)

            if params.get("v"):
                video_id = params["v"][0]

                result["url"] = f"https://www.youtube.com/watch?v={video_id}"
                result["status"] = "normalized"
                result["changed"] = result["url"] != original_url

                return result

        return result

    if host_no_www.endswith("tiktok.com") or "tiktokcdn.com" in host_no_www:
        result["platform"] = "tiktok"

        if "tiktokcdn.com" in host_no_www:
            result["status"] = "ephemeral_media_url"
            return result

        match = re.search(
            r"/@[^/]+/video/(\d+)",
            parsed.path,
        )

        if match:
            video_id = match.group(1)

            username_match = re.search(
                r"/@([^/]+)/video/",
                parsed.path,
            )

            if username_match:
                username = username_match.group(1)

                result["url"] = (
                    f"https://www.tiktok.com/" f"@{username}/video/{video_id}"
                )

            result["status"] = "normalized"
            result["changed"] = result["url"] != original_url

            return result

        match = re.match(
            r"/embed/(\d+)",
            parsed.path,
        )

        if match:
            video_id = match.group(1)

            result["url"] = f"https://www.tiktok.com/embed/{video_id}"
            result["status"] = "normalized"
            result["changed"] = result["url"] != original_url

            return result

        return result

    if host_no_www in {
        "instagram.com",
        "m.instagram.com",
    }:
        result["platform"] = "instagram"

        match = re.search(
            r"/(p|reel|tv)/([^/?#]+)",
            parsed.path,
        )

        if match:
            content_type = match.group(1)
            shortcode = match.group(2)

            result["url"] = f"https://www.instagram.com/" f"{content_type}/{shortcode}/"
            result["status"] = "normalized"
            result["changed"] = result["url"] != original_url

            return result

        return result

    if host_no_www.endswith("reddit.com") or host_no_www == "redd.it":
        result["platform"] = "reddit"

        if host_no_www in {
            "packaged-media.redd.it",
            "v.redd.it",
        }:
            result["status"] = "ephemeral_media_url"
            return result

        result["url"] = f"https://www.reddit.com{parsed.path}"
        result["status"] = "normalized"
        result["changed"] = result["url"] != original_url

        return result

    result["status"] = "unsupported"

    return result


def _resolve_redirect(url: str) -> str | None:

    try:
        response = requests.get(
            url,
            allow_redirects=True,
            timeout=10,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/130.0 Safari/537.36"
                )
            },
        )

        return response.url

    except requests.RequestException:
        return None
