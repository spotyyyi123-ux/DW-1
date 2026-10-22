"""Экстракторы сигналов. Единый интерфейс (SPEC §4.2):

    extract(parts) -> секция signals для своего элемента

Правила: не ронять пайплайн — при ошибке метка "unknown" + запись в лог;
у каждого экстрактора есть model_version; кэш по content-hash части (M1+).
"""
from __future__ import annotations

import logging

from dw1.evidence import IntentSignal, ProvenanceSignal

log = logging.getLogger("dw1.extractors")

STUB_VERSION = "stub-0.1"

_REGISTRY: list[type[BaseExtractor]] = []


def register(cls: type[BaseExtractor]) -> type[BaseExtractor]:
    """Регистрирует класс экстрактора (декоратор)."""
    _REGISTRY.append(cls)
    return cls


class BaseExtractor:
    """Базовый экстрактор. Наследник задаёт name (= ключ элемента в Evidence)
    и version. M0-поведение: оба вида сигналов = unknown."""

    name: str = ""
    version: str = STUB_VERSION

    def __init__(self) -> None:
        if not self.name:
            raise ValueError("экстрактор без имени: задайте class-атрибут name")

    def _unknown(self, kind: str) -> dict:
        sig = ProvenanceSignal() if kind == "ai_provenance" else IntentSignal()
        sig.model_version = self.version
        sig.evidence.append(
            {"signal": "stub", "value": f"{self.name}: не реализован (M0)", "weight": 0.0}
        )
        return {
            "label": sig.label,
            "confidence": sig.confidence,
            "evidence": sig.evidence,
            "model_version": self.version,
        }

    def extract(self, parts: dict) -> dict:
        """M0-заглушка. Реализация — по SPEC §6 (см. docstring модуля)."""
        return {
            "ai_provenance": self._unknown("ai_provenance"),
            "intent": self._unknown("intent"),
        }


def get_extractors() -> list[BaseExtractor]:
    """Возвращает по одному инстансу каждого зарегистрированного экстрактора."""
    from dw1.extractors import (  # noqa: F401  (регистрация происходит при импорте)
        body_text,
        cta,
        headers,
        html,
        images,
        links,
        qr,
        subject,
        template,
    )

    out: list[BaseExtractor] = [cls() for cls in _REGISTRY]
    out.sort(key=lambda e: e.name)
    return out
