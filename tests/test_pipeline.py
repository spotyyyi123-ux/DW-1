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


def test_template_measurements_are_unknown():
    ev = analyze(FIXTURE)
    for name in ("template",):
        blob = ev.signals[name]
        assert blob["ai_provenance"]["label"] == "unknown", name
        assert blob["intent"]["label"] == "unknown", name
        assert blob["ai_provenance"]["model_version"] == "campaign-template-1.0", name


def test_media_measurements_do_not_claim_authorship():
    ev = analyze(FIXTURE)
    for name in ("images", "qr"):
        for axis in ("ai_provenance", "intent"):
            assert ev.signals[name][axis]["label"] == "unknown"
            assert not ev.signals[name][axis]["model_version"].startswith("stub-")


def test_text_extractors_produce_evidence():
    # Текстовые экстракторы (Задача 2): body_text, subject, cta производят валидные сигналы
    ev = analyze(FIXTURE)
    for name in ("body_text", "subject", "cta"):
        blob = ev.signals[name]
        assert "ai_provenance" in blob, name
        assert "intent" in blob, name
        assert not blob["ai_provenance"]["model_version"].startswith("stub-"), name
        assert len(blob["ai_provenance"]["evidence"]) > 0, name
