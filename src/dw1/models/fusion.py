# -*- coding: utf-8 -*-
"""Фузия (SPEC §6 W4): две оси, калибровка, доверительные интервалы, отказ от ответа.

M0: честная заглушка — сигналов нет -> rejection=true. Реализация в M5.
"""
from __future__ import annotations

FUSION_STUB = "fusion-stub-0.1"


def _axis(label: str) -> dict:
    return {
        "label": label,
        "confidence": 0.0,
        "calibrated": False,
        "ci95": [],
        "model_version": FUSION_STUB,
    }


def fuse(signals: dict) -> dict:
    """Собирает две оси из позлементных сигналов (SPEC §3).

    Правила (M5): раздельные модели для provenance и intent; isotonic-калибровка;
    bootstrap-CI; rejection при нехватке сигналов.
    """
    return {
        "ai_provenance": _axis("unknown"),
        "intent": _axis("unknown"),
        "rejection": True,  # M0: ни одного реализованного экстрактора
    }
