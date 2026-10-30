from dw1.models.fusion import fuse

UNKNOWN = {"label": "unknown", "confidence": 0.0, "evidence": [], "model_version": "test"}


def test_llm_signal_used_as_provenance_component():
    # Калиброванная вероятность Fast-DetectGPT имеет приоритет над TF-IDF-моделью
    llm_evidence = [
        {
            "signal": "fast_detectgpt",
            "weight": 1.0,
            "value": {
                "criterion": 1.0,
                "p_ai": 0.82,
                "language": "en",
                "calibrated": True,
            },
        }
    ]
    signals = {
        "body_text": {
            "ai_provenance": {
                "label": "ai_generated",
                "confidence": 0.82,
                "evidence": llm_evidence,
                "model_version": "test-fixture",
            },
            "intent": dict(UNKNOWN),
        }
    }
    fused = fuse(signals)
    assert fused["ai_provenance"]["components"]["ai"] == 0.82
    assert fused["ai_provenance"]["components"]["model"] == "fast-detectgpt"


def test_fast_detectgpt_not_double_calibrated(tmp_path, monkeypatch):
    import json

    import dw1.models.fusion as fusion

    calibration = {
        "axes": {
            "ai_provenance": {"calibrator_x": [0.0, 1.0], "calibrator_y": [0.1, 0.9]},
            "intent": {"calibrator_x": [0.0, 1.0], "calibrator_y": [0.2, 0.8]},
        }
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")
    monkeypatch.setenv("DW1_FUSION_CALIBRATION", str(path))
    fusion._CALIBRATION = None
    fusion._CALIBRATION_MTIME = 0.0

    llm_evidence = [
        {
            "signal": "fast_detectgpt",
            "weight": 1.0,
            "value": {"criterion": 1.0, "p_ai": 0.82, "language": "en", "calibrated": True},
        }
    ]
    signals = {
        "body_text": {
            "ai_provenance": {
                "label": "ai_generated",
                "confidence": 0.82,
                "evidence": llm_evidence,
                "model_version": "test-fixture",
            },
            "intent": dict(UNKNOWN),
        }
    }
    fused = fusion.fuse(signals)
    # вероятность LLM уже калибрована по языку — isotonic фузии не применяется
    assert "calibrated_confidence" not in fused["ai_provenance"]
    # в компоненту оси уходит сама калиброванная LLM-вероятность
    assert fused["ai_provenance"]["components"]["ai"] == 0.82
    assert fused["ai_provenance"]["components"]["model"] == "fast-detectgpt"


def signal(label: str, conf: float, evidence=None) -> dict:
    return {
        "ai_provenance": {
            "label": label,
            "confidence": conf,
            "evidence": evidence or [],
            "model_version": "t",
        },
        "intent": {
            "label": label,
            "confidence": conf,
            "evidence": evidence or [],
            "model_version": "t",
        },
    }


def test_all_unknown_rejects():
    signals = {
        e: {"ai_provenance": dict(UNKNOWN), "intent": dict(UNKNOWN)}
        for e in ("body_text", "subject", "cta", "html", "headers", "links")
    }
    fused = fuse(signals)
    assert fused["rejection"] is True
    assert fused["ai_provenance"]["label"] == "unknown"
    assert fused["intent"]["label"] == "unknown"


def test_provenance_votes_and_intent_boost():
    signals = {
        "body_text": {
            "ai_provenance": {
                "label": "ai_generated",
                "confidence": 0.8,
                "evidence": [],
                "model_version": "t",
            },
            "intent": {
                "label": "phishing",
                "confidence": 0.7,
                "evidence": [],
                "model_version": "t",
            },
        },
        "subject": {
            "ai_provenance": {
                "label": "ai_polished",
                "confidence": 0.6,
                "evidence": [],
                "model_version": "t",
            },
            "intent": {
                "label": "phishing",
                "confidence": 0.7,
                "evidence": [],
                "model_version": "t",
            },
        },
        "headers": {
            "ai_provenance": dict(UNKNOWN),
            "intent": {
                "label": "unknown",
                "confidence": 0.0,
                "evidence": [{"signal": "spf_fail", "value": "fail"}],
                "model_version": "t",
            },
        },
        "links": {
            "ai_provenance": dict(UNKNOWN),
            "intent": {
                "label": "unknown",
                "confidence": 0.0,
                "evidence": [{"signal": "link_0_punycode", "value": True}],
                "model_version": "t",
            },
        },
    }
    fused = fuse(signals)
    assert fused["rejection"] is False
    # provenance: body_text (0.5*0.8) > subject (0.2*0.6)
    assert fused["ai_provenance"]["label"] == "ai_generated"
    # intent: phishing (0.07+0.07) + усилители (2*0.05)
    assert fused["intent"]["label"] == "phishing"
    assert fused["intent"]["confidence"] >= 0.2
    assert "spf_fail" in fused["threat_signals"]


def test_calibration_applied_when_file_present(tmp_path, monkeypatch):
    import json

    import dw1.models.fusion as fusion

    calibration = {
        "axes": {
            "ai_provenance": {"calibrator_x": [0.0, 1.0], "calibrator_y": [0.1, 0.9]},
            "intent": {"calibrator_x": [0.0, 1.0], "calibrator_y": [0.2, 0.8]},
        }
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")
    monkeypatch.setenv("DW1_FUSION_CALIBRATION", str(path))
    fusion._CALIBRATION = None
    fusion._CALIBRATION_MTIME = 0.0

    signals = {
        "body_text": {
            "ai_provenance": {
                "label": "ai_generated",
                "confidence": 1.0,
                "evidence": [],
                "model_version": "t",
            },
            "intent": {
                "label": "phishing",
                "confidence": 1.0,
                "evidence": [],
                "model_version": "t",
            },
        }
    }
    fused = fusion.fuse(signals)
    assert fused["calibrated"] is True
    assert fused["ai_provenance"]["calibrated"] is True
    # калибровка кладётся отдельным полем, эвристическая уверенность не подменяется
    assert fused["ai_provenance"]["calibrated_confidence"] == 0.9
    assert fused["intent"]["calibrated_confidence"] == 0.8


def test_suppressors_lower_intent_confidence():
    signals = {
        "body_text": {
            "ai_provenance": {
                "label": "ai_generated",
                "confidence": 0.7,
                "evidence": [],
                "model_version": "t",
            },
            "intent": {
                "label": "spam",
                "confidence": 0.6,
                "evidence": [],
                "model_version": "t",
            },
        },
        "html": {
            "ai_provenance": dict(UNKNOWN),
            "intent": {
                "label": "unknown",
                "confidence": 0.0,
                "evidence": [
                    {"signal": "html_mailing_list_unsubscribe", "value": True}
                ],
                "model_version": "t",
            },
        },
    }
    fused = fuse(signals)
    assert fused["intent"]["label"] == "spam"
    # базовая уверенность 0.1*0.6 = 0.06, после гасителя — меньше
    assert fused["intent"]["confidence"] < 0.06
    assert any("гаситель" in item for item in fused["counter_evidence"])
