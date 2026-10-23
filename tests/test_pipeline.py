from pathlib import Path

from dw1.evidence import ELEMENTS
from dw1.pipeline import analyze

FIXTURE = Path(__file__).parent / "fixtures" / "sample.eml"


def test_analyze_produces_valid_evidence():
    ev = analyze(FIXTURE)
    assert ev.schema_version == "1.0"
    assert set(ev.signals) == set(ELEMENTS)  # 9 экстракторов


def test_fusion_produces_two_axes():
    # Fusion v1: две оси заполняются из сигналов, уверенность — эвристическая (не калибрована)
    ev = analyze(FIXTURE)
    assert ev.fusion["heuristic"] is True
    assert ev.fusion["rejection"] is False
    for axis in ("ai_provenance", "intent"):
        blob = ev.fusion[axis]
        assert blob["calibrated"] is False, axis
        assert blob["label"] in (
            "human",
            "ai_generated",
            "ai_polished",
            "template",
            "unknown",
            "phishing",
            "spam",
            "legitimate",
        ), axis
    assert isinstance(ev.explanations["counter_evidence"], list)


def test_remaining_stub_signals_are_unknown():
    # Ещё не реализованные экстракторы остаются честными заглушками
    ev = analyze(FIXTURE)
    for name in ("images", "qr", "template"):
        blob = ev.signals[name]
        assert blob["ai_provenance"]["label"] == "unknown", name
        assert blob["intent"]["label"] == "unknown", name
        assert blob["ai_provenance"]["model_version"].startswith("stub-"), name


def test_text_extractors_produce_evidence():
    # Текстовые экстракторы (Задача 2): body_text, subject, cta производят валидные сигналы
    ev = analyze(FIXTURE)
    for name in ("body_text", "subject", "cta"):
        blob = ev.signals[name]
        assert "ai_provenance" in blob, name
        assert "intent" in blob, name
        assert not blob["ai_provenance"]["model_version"].startswith("stub-"), name
        assert len(blob["ai_provenance"]["evidence"]) > 0, name
