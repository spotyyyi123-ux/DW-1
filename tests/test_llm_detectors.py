"""Проверки калибровки на искусственных фикстурах, без скачивания LLM."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from dw1.eval.llm_calibration import calibrate, expected_calibration_error
from dw1.extractors.body_text import BodyTextExtractor
from dw1.models.llm_detectors import LLMScoreStore, text_hash, validate_scores


def fixture_text(language, split, label, index):
    return f"This is a local test fixture with enough words. {language} {split} {label} {index}"


@pytest.fixture
def scores():
    rows = []
    rng = np.random.default_rng(42)
    for language, dataset in [("en", "hc3-en"), ("ru", "coat")]:
        for split in ["train", "validation", "test"]:
            for label in [0, 1]:
                for index in range(20):
                    key = f"{language}-{split}-{label}-{index}"
                    rows.append(
                        dict(
                            message_id=key,
                            group_id=key,
                            dataset=dataset,
                            language=language,
                            label=label,
                            split=split,
                            text_sha256=text_hash(fixture_text(language, split, label, index)),
                            criterion=float(label * 2 - 1 + rng.normal(0, 0.3)),
                            model_version="test-fixture-not-real-llm",
                            n_tokens=64,
                            truncated=False,
                        )
                    )
    return pd.DataFrame(rows)


def artifacts(tmp_path, scores):
    config, _, _ = calibrate(scores)
    path = tmp_path / "scores.parquet"
    scores.to_parquet(path, index=False)
    config["score_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(config), encoding="utf-8")
    return path, calibration


def test_ece_definition_and_bin_endpoints():
    assert expected_calibration_error([0, 1], [0.0, 1.0]) == 0
    assert expected_calibration_error([0, 0], [0.5, 0.5]) == 0.5
    assert expected_calibration_error([0, 1], [0.5, 0.5]) == 0
    with pytest.raises(ValueError):
        expected_calibration_error([0], [float("nan")])


def test_test_labels_do_not_fit_calibrator(scores):
    first, _, _ = calibrate(scores)
    altered = scores.copy()
    selected = altered.split == "test"
    altered.loc[selected, "label"] = 1 - altered.loc[selected, "label"]
    second, _, _ = calibrate(altered)
    assert first == second


def test_group_leakage_rejected(scores):
    bad = scores.copy()
    bad.loc[bad.index[40], "group_id"] = bad.iloc[0].group_id
    with pytest.raises(ValueError, match="Родственные"):
        validate_scores(bad)


def test_score_export_must_not_contain_text(scores):
    bad = scores.assign(text="private content is not allowed in score exports")
    with pytest.raises(ValueError, match="исходные тексты"):
        validate_scores(bad)


def test_missing_class_is_not_silently_calibrated(scores):
    bad = scores[
        ~((scores.language == "ru") & (scores.split == "validation") & (scores.label == 1))
    ]
    with pytest.raises(ValueError, match="Недостаточно"):
        calibrate(bad)


def test_cached_signal_and_unknown_text(tmp_path, scores):
    path, calibration = artifacts(tmp_path, scores)
    store = LLMScoreStore(path, calibration)
    signal = store.lookup(fixture_text("en", "test", 1, 0))
    assert signal["label"] == "ai_generated"
    assert signal["evidence"][0]["value"]["calibrated"] is True
    assert "test-fixture" in signal["model_version"]
    assert store.lookup("Unseen email") is None


def test_wrong_calibration_file_rejected(tmp_path, scores):
    path, calibration = artifacts(tmp_path, scores)
    cfg = json.loads(calibration.read_text())
    cfg["score_sha256"] = "0" * 64
    calibration.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="другому файлу"):
        LLMScoreStore(path, calibration)


def test_body_text_keeps_intent_when_llm_signal_exists(tmp_path, scores, monkeypatch):
    path, calibration = artifacts(tmp_path, scores)
    monkeypatch.setenv("DW1_LLM_SCORES", str(path))
    monkeypatch.setenv("DW1_LLM_CALIBRATION", str(calibration))
    text = fixture_text("en", "test", 1, 0)
    result = BodyTextExtractor().extract({"body_text": {"text": text}})
    assert "test-fixture" in result["ai_provenance"]["model_version"]
    assert result["intent"]["label"] == "unknown"
    assert result["ai_provenance"]["evidence"][0]["signal"] == "fast_detectgpt"
