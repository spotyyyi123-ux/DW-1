"""Conservative template alignment with typed, observable variable slots."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

PATTERNS = [
    ("LINK", re.compile(r"https?://[^\s<>]+", re.I)),
    (
        "AMOUNT",
        re.compile(r"(?:[$€₽]\s*\d[\d.,]*|\d[\d.,]*\s*(?:руб\.?|RUB|USD|EUR|₽|долларов))", re.I),
    ),
    ("NAME", re.compile(r"(?<=(?:Dear ))[^,\n]+")),
    ("NAME", re.compile(r"(?<=Здравствуйте, )[^!\n]+")),
    ("BRAND", re.compile(r"(?<=бренда )[\w-]+|(?<=brand )[\w-]+", re.I)),
]


def typed(text: str) -> tuple[str, dict[str, list[str]]]:
    slots: dict[str, list[str]] = {}
    for kind, pattern in PATTERNS:

        def replace(match, kind=kind):
            slots.setdefault(kind, []).append(match.group())
            return "{" + kind + "}"

        text = pattern.sub(replace, text)
    return text, slots


def skeleton(texts: list[str]) -> dict:
    if not texts:
        return {"text": "", "slots": {}, "method": "typed-alignment-1"}
    normalized = []
    slots: dict[str, list[str]] = {}
    for text in texts:
        value, found = typed(text)
        normalized.append(value.split())
        for key, values in found.items():
            slots.setdefault(key, []).extend(values)
    common = normalized[0]
    for words in normalized[1:]:
        aligned = []
        for tag, a, b, _, _ in SequenceMatcher(None, common, words, autojunk=False).get_opcodes():
            aligned.extend(common[a:b] if tag == "equal" else ["{VARIABLE}"])
        common = aligned
    output = re.sub(r"(?:\{VARIABLE\}\s*)+", "{VARIABLE} ", " ".join(common)).strip()
    return {
        "text": output,
        "slots": {k: sorted(set(v))[:20] for k, v in slots.items()},
        "method": "typed-alignment-1",
        "member_count": len(texts),
    }
