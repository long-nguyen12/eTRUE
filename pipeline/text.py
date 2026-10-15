import json
import re

from bs4 import BeautifulSoup


def plain_explanation(value: str) -> str:
    text = value.strip().replace("\r\n", "\n").replace("\r", "\n")
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
        for tag in soup.find_all(
            ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"]
        ):
            tag.insert_before("\n")
            tag.insert_after("\n")
        text = soup.get_text()

    # Preserve sentence punctuation after URLs while removing the URLs themselves.
    text = re.sub(
        r"\b(?:https?://|www\.)[^\s<>\[\]]+",
        lambda match: match.group(0)[len(match.group(0).rstrip(".,;:!?)]}")) :],
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
    return text
