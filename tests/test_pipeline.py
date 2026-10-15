# -*- coding: utf-8 -*-
from pathlib import Path

from dw1.evidence import ELEMENTS
from dw1.pipeline import analyze

FIXTURE = Path(__file__).parent / "fixtures" / "sample.eml"


def test_analyze_produces_valid_evidence():
    ev = analyze(FIXTURE)
    assert ev.schema_version == "1.0"
    assert set(ev.signals) == set(ELEMENTS)  # 9 экстракторов


def test_m0_honest_rejection():
    # M0: все экстракторы — заглушки, поэтому фузия обязана отказаться от ответа
    ev = analyze(FIXTURE)
    assert ev.fusion["rejection"] is True
    assert ev.fusion["ai_provenance"]["label"] == "unknown"
    assert ev.fusion["intent"]["label"] == "unknown"


def test_all_stub_signals_are_unknown():
    ev = analyze(FIXTURE)
    for name, blob in ev.signals.items():
        assert blob["ai_provenance"]["label"] == "unknown", name
        assert blob["intent"]["label"] == "unknown", name
        assert blob["ai_provenance"]["model_version"].startswith("stub-"), name
