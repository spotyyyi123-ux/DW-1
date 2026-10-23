from pathlib import Path

from dw1.evidence import ELEMENTS
from dw1.pipeline import analyze

FIXTURE = Path(__file__).parent / "fixtures" / "sample.eml"


def test_analyze_produces_valid_evidence():
    ev = analyze(FIXTURE)
    assert ev.schema_version == "1.0"
    assert set(ev.signals) == set(ELEMENTS)  # 9 экстракторов


def test_m0_honest_rejection():
    # До вехи M5 (фузия) система корректно возвращает rejection=True
    ev = analyze(FIXTURE)
    assert ev.fusion["rejection"] is True
    assert ev.fusion["ai_provenance"]["label"] == "unknown"
    assert ev.fusion["intent"]["label"] == "unknown"


def test_remaining_stub_signals_are_unknown():
    # Проверка того, что ещё не реализованные экстракторы остаются честными заглушками
    ev = analyze(FIXTURE)
    stubs = {"headers", "html", "images", "links", "qr", "template"}
    for name in stubs:
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
