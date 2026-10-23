"""Evidence JSON v1 — единственный контракт обмена между экстракторами, фузией и API.

Полная схема: docs/SPEC.md §4.1. Версионирование:
- добавление поля       -> минорная версия (1.1, 1.2, ...);
- удаление/переименование -> мажорная (2.0).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any

SCHEMA_VERSION = "1.0"

PROVENANCE_LABELS = ("human", "ai_generated", "ai_polished", "template", "unknown")
INTENT_LABELS = ("phishing", "spam", "legitimate", "unknown")

# Элементы письма, по которым атрибутируется происхождение (SPEC §1, §3)
ELEMENTS = ("subject", "body_text", "cta", "html", "images", "qr", "links", "headers", "template")


class EvidenceError(ValueError):
    """Некорректный Evidence JSON."""


@dataclass
class ProvenanceSignal:
    """Атрибуция происхождения элемента (SPEC §2)."""

    label: str = "unknown"
    confidence: float = 0.0
    evidence: list[dict] = field(default_factory=list)
    model_version: str = "unset"

    def validate(self, where: str) -> None:
        if self.label not in PROVENANCE_LABELS:
            raise EvidenceError(f"{where}.label={self.label!r} не в {PROVENANCE_LABELS}")
        if not 0.0 <= self.confidence <= 1.0:
            raise EvidenceError(f"{where}.confidence={self.confidence} вне [0,1]")


@dataclass
class IntentSignal:
    """Атрибуция намерения (SPEC §2)."""

    label: str = "unknown"
    confidence: float = 0.0
    evidence: list[dict] = field(default_factory=list)
    model_version: str = "unset"

    def validate(self, where: str) -> None:
        if self.label not in INTENT_LABELS:
            raise EvidenceError(f"{where}.label={self.label!r} не в {INTENT_LABELS}")
        if not 0.0 <= self.confidence <= 1.0:
            raise EvidenceError(f"{where}.confidence={self.confidence} вне [0,1]")


@dataclass
class Evidence:
    schema_version: str = SCHEMA_VERSION
    message_id: str = ""
    ingest: dict = field(default_factory=dict)
    parts: dict = field(default_factory=dict)
    signals: dict = field(default_factory=dict)
    fusion: dict = field(default_factory=dict)
    campaign: dict = field(default_factory=dict)
    explanations: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> Evidence:
        if not isinstance(d, dict):
            raise EvidenceError("Evidence должен быть dict")
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise EvidenceError(f"неизвестные поля: {sorted(unknown)}")
        return cls(**{k: d[k] for k in known if k in d})

    def validate(self) -> None:
        """Валидация контракта: версия схемы, метки и уверенности всех сигналов."""
        if self.schema_version != SCHEMA_VERSION:
            raise EvidenceError(
                f"schema_version={self.schema_version!r}, ожидается {SCHEMA_VERSION!r}"
            )
        for elem, blob in self.signals.items():
            blobs: list[Any] = blob if isinstance(blob, list) else [blob]
            for b in blobs:
                if not isinstance(b, dict):
                    raise EvidenceError(f"signals.{elem}: ожидается dict или список dict")
                for kind, sig in b.items():
                    if sig is None or (isinstance(sig, dict) and not sig):
                        continue
                    if not isinstance(sig, dict):
                        raise EvidenceError(f"signals.{elem}.{kind}: ожидается dict")
                    if kind == "ai_provenance":
                        ProvenanceSignal(**sig).validate(f"signals.{elem}.{kind}")
                    elif kind == "intent":
                        IntentSignal(**sig).validate(f"signals.{elem}.{kind}")
                    else:
                        raise EvidenceError(f"signals.{elem}.{kind}: неизвестный вид сигнала")
        for axis in ("ai_provenance", "intent"):
            blob = self.fusion.get(axis)
            if isinstance(blob, dict) and blob:
                label = blob.get("label", "unknown")
                allowed = PROVENANCE_LABELS if axis == "ai_provenance" else INTENT_LABELS
                if label not in allowed:
                    raise EvidenceError(f"fusion.{axis}.label={label!r} не в {allowed}")
