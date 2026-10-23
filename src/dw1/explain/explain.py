"""Объяснения (SPEC §6 W5): атрибуция, подсветка, контр-свидетельства, faithfulness.

M0: заглушка. Реализация в M6; faithfulness (comprehensiveness/sufficiency) обязателен
(SPEC §3, §8) — объяснение без измерения достоверности не существует.
"""
from __future__ import annotations


def explain(signals: dict, fusion: dict, campaign: dict) -> dict:
    return {
        "top_factors": [],
        "highlights": [],
        "counter_evidence": [],
        "faithfulness": {"comprehensiveness": None, "sufficiency": None},
    }
