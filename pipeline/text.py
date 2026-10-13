"""Plain prose for explanations; citations remain in their separate fields."""

import json
import re

from bs4 import BeautifulSoup


def plain_explanation(value: str) -> str:
    """Remove presentation markup and references without stringifying objects."""
    if not isinstance(value, str):
        raise ValueError("Explanation must be a plain text string")
    text = value.strip().replace("\r\n", "\n").replace("\r", "\n")
    if re.search(
        r"(?m)^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(?:\|[ \t]*:?-{3,}:?[ \t]*)+\|?[ \t]*$",
        text,
    ):
        raise ValueError("Explanation must be prose, not a Markdown table")
    for block in re.finditer(r"(?P<fence>`{3,}|~{3,})(.*?)(?P=fence)", text, re.DOTALL):
        body = re.sub(r"^[A-Za-z0-9_+-]+[ \t]*\n", "", block.group(2).lstrip()).strip()
        body = re.sub(r"^json[ \t]*(?=[{\[])", "", body, flags=re.IGNORECASE)
        if body.startswith(("{", "[")):
            raise ValueError("Explanation cannot contain a structured code block")
    text = re.sub(
        r"(?m)^[ \t]*(?:`{3,}|~{3,})(?:[A-Za-z0-9_+-]+)?[ \t]*$", "", text
    ).strip()
    text = re.sub(r"!?\[([^\]\n]*)\]\((?:[^\n()]+|\([^\n()]*\))*\)", r"\1", text)
    if re.search(r"</?[A-Za-z][^>]*>", text):
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup.find_all(["script", "style"]):
            tag.decompose()
        for tag in soup.find_all("br"):
            tag.replace_with("\n")
        for tag in soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"]):
            tag.insert_before("\n")
            tag.insert_after("\n")
        text = soup.get_text()

    # Preserve sentence punctuation after URLs while removing the URLs themselves.
    text = re.sub(
        r"\b(?:https?://|www\.)[^\s<>\[\]]+",
        lambda match: match.group(0)[len(match.group(0).rstrip(".,;:!?)]}")):],
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bev-[\w]+(?:[.-][\w]+)*\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*\d{1,3}(?:\s*[,;-]\s*\d{1,3})*\s*\]", "", text)
    text = re.sub(r"\[[\s,;]*\]|\([\s,;]*\)", "", text)
    text = re.sub(r"(?m)^[ \t]*>+[ \t]?", "", text)
    text = re.sub(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+", "", text)
    text = re.sub(r"(?m)[ \t]+#+[ \t]*$", "", text)
    text = re.sub(r"(?m)^[ \t]*(?:[-+*\u2022]|\d{1,3}[.)])[ \t]+", "", text)
    text = re.sub(r"(?m)^[ \t]*(?:-{3,}|_{3,}|\*{3,})[ \t]*$", "", text)
    text = re.sub(r"(\*{1,3}|~~)(?=\S)(.*?\S)\1", r"\2", text)
    text = re.sub(r"(?<!\w)(_{1,3})(?=\S)(.*?\S)\1(?!\w)", r"\2", text)
    text = re.sub(r"`+", "", text)
    text = re.sub(r"\[[\s,;]*\]|\([\s,;]*\)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    text = re.sub(r"[,;:]+([.!?])", r"\1", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text or not any(character.isalnum() for character in text):
        raise ValueError("Explanation cannot be empty")
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    if isinstance(parsed, (dict, list)) or text.startswith("{") or (
        text.startswith("[") and text.endswith("]")
    ):
        raise ValueError("Explanation cannot contain a serialized object or list")
    return text
