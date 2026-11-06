"""Shared, dependency-free document format used locally and in the external exporter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def document_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_documents(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not rows or any(
        not isinstance(r.get("id"), str)
        or not r["id"]
        or not isinstance(r.get("text"), str)
        or not r["text"].strip()
        for r in rows
    ):
        raise ValueError("Expected non-empty JSONL records with string id and text")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate document IDs")
    return rows
