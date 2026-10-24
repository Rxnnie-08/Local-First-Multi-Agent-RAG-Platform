"""Small text helpers used by the sample project."""

import re


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def truncate_words(text: str, max_words: int) -> str:
    """Keep at most max_words words, adding an ellipsis when text was cut."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


class TokenCounter:
    """Approximates token counts as characters divided by four."""

    def count(self, text: str) -> int:
        return max(1, len(text) // 4)
