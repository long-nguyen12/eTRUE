from utils.json_ld import fetch_page_metadata, public_http_url


def candidate_page_metadata(session, url):
    return fetch_page_metadata(session, url)
