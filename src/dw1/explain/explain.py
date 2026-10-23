"""Объяснения (SPEC §6 W5): атрибуция, подсветка, контр-свидетельства, faithfulness.

M0/M1: контр-свидетельства и усилители берутся из фузии (fusion v1);
полный explainability-слой (SHAP/IG, faithfulness) — M6.
"""
from __future__ import annotations


def explain(signals: dict, fusion: dict, campaign: dict) -> dict:
    fusion = fusion if isinstance(fusion, dict) else {}
    counter = list(fusion.get("counter_evidence", []) or [])
    threats = list(fusion.get("threat_signals", []) or [])
    return {
        "top_factors": [{"signal": name, "direction": "+"} for name in threats],
        "highlights": [],
        "counter_evidence": counter,
        "faithfulness": {"comprehensiveness": None, "sufficiency": None},
    }
