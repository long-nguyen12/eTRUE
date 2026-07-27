"""Simple text comparison helpers."""

import re


def word_overlap(first, second):
    first_words = set(re.findall(r"[a-z0-9]+", str(first or "").lower()))
    second_words = set(re.findall(r"[a-z0-9]+", str(second or "").lower()))
    if not first_words or not second_words:
        return 0.0
    return len(first_words & second_words) / len(first_words | second_words)


def containment_overlap(first, second):
    first_words = set(re.findall(r"[a-z0-9]+", str(first or "").lower()))
    second_words = set(re.findall(r"[a-z0-9]+", str(second or "").lower()))
    if not first_words or not second_words:
        return 0.0
    return len(first_words & second_words) / min(len(first_words), len(second_words))


def text_value_is_grounded(value, grounding_text):
    phrase = re.sub(r"\s+", " ", str(value or "").strip().lower())
    evidence = re.sub(r"\s+", " ", str(grounding_text or "").lower())
    return bool(phrase and phrase in evidence)
