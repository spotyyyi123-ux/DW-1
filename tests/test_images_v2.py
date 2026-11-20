from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dw1.extractors.images import ImagesExtractor
from dw1.models.image_detector import VERSION, ImageDetector, image_signal, sha256
from scripts.images_v2 import ATTACKS, distort, evaluate
from scripts.validate_sprint5 import audit_ru


def test_measurements_without_weights_do_not_claim_ai(monkeypatch):
    monkeypatch.delenv("DW1_IMAGE_MODEL", raising=False)
    monkeypatch.delenv("DW1_IMAGE_ENCODER", raising=False)
    status = image_signal(b"not used when model absent")
    result = ImagesExtractor().extract({"images": [{"phash": "ffff", "ai_detector": status}]})
    assert result["ai_provenance"]["label"] == "unknown"
    assert result["intent"]["label"] == "unknown"


def test_detector_only_affects_image_origin_axis():
    result = ImagesExtractor().extract(
        {
            "images": [
                {
                    "ai_detector": dict(
                        status="scored",
                        p_ai=0.9,
                        label="ai_generated",
                        confidence=0.9,
                        model_version="fixture",
                    )
                }
            ]
        }
    )
    assert result["ai_provenance"]["label"] == "ai_generated"
    assert result["intent"]["label"] == "unknown"
    assert result["intent"]["confidence"] == 0


def test_partial_coverage_cannot_claim_all_images_human():
    result = ImagesExtractor().extract(
        {
            "images": [
                dict(
                    ai_detector=dict(
                        status="scored",
                        p_ai=0.1,
                        label="human",
                        confidence=0.9,
                        model_version="fixture",
                    )
                ),
                {},
            ]
        }
    )
    assert result["ai_provenance"]["label"] == "unknown"


def test_artifact_inference_replays_saved_calibration_without_torch(tmp_path, monkeypatch):
    class Encoder:
        def __init__(self, *args):
            pass

        def features(self, image):
            return np.ones(592)

    monkeypatch.setattr("dw1.models.image_detector.ImageEncoder", Encoder)
    model = tmp_path / "model.npz"
    np.savez(
        model,
        mean=np.zeros(592),
        scale=np.ones(592),
        coef=np.zeros(592),
        intercept=np.array([2.0]),
        platt_coef=np.array([0.5]),
        platt_intercept=np.array([-1.0]),
    )
    model.with_suffix(".json").write_text(
        json.dumps(
            dict(feature_version=VERSION, model_sha256=sha256(model), encoder_sha256="fixture")
        )
    )
    detector = ImageDetector(model, tmp_path / "encoder")
    assert detector.predict(Image.new("RGB", (32, 32)))["p_ai"] == 0.5
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="mismatch"):
        ImageDetector(model, tmp_path / "encoder")


def test_attacks_have_real_compression_and_size_changes():
    rgb = Image.fromarray(np.random.default_rng(42).integers(0, 255, (32, 32, 3), dtype=np.uint8))
    assert distort(rgb, "resize25").size == (8, 8)
    assert distort(rgb, "crop80").size == (26, 26)
    assert not np.array_equal(np.asarray(rgb), np.asarray(distort(rgb, "jpeg30")))


def test_eval_rejects_partial_grid_and_tracks_degradation(tmp_path):
    rows = []
    for i, label in enumerate([0, 1, 0, 1]):
        key = hashlib.sha256(str(i).encode()).hexdigest()
        for attack in ATTACKS:
            probability = 0.9 if label else 0.1
            if attack == "jpeg30":
                probability = 1 - probability
            rows.append(
                dict(
                    image_id=str(i),
                    original_sha256=key,
                    pixel_sha256=key,
                    split="test",
                    label=label,
                    attack=attack,
                    p_ai=probability,
                    model_version="test-fixture-only",
                )
            )
    predictions = tmp_path / "predictions.parquet"
    pd.DataFrame(rows).to_parquet(predictions, index=False)
    result = evaluate(predictions, tmp_path / "out")
    assert result["metrics"][0]["auc"] == 1
    compressed = next(r for r in result["metrics"] if r["attack"] == "jpeg30")
    assert compressed["auc"] == 0 and compressed["f1_delta"] == -1
    pd.DataFrame(rows[:-1]).to_parquet(predictions, index=False)
    with pytest.raises(ValueError, match="population"):
        evaluate(predictions, tmp_path / "bad")


def test_ru_aggregate_is_not_presented_as_recomputed():
    baseline = dict(ece=0.1, f1=0.5, roc_auc=0.6)
    source = dict(
        metrics={
            k: dict(n_test=10, before=baseline, after=baseline)
            for k in ("legacy_increasing", "train_direction")
        },
        score_sha256="fixture",
        n_input=30,
        n_scored=30,
    )
    assert audit_ru(source, "fixture")["status"] == "reported_not_recomputed"
    source["metrics"]["train_direction"]["after"] = dict(baseline, roc_auc=2)
    with pytest.raises(ValueError, match="Invalid"):
        audit_ru(source, "fixture")


def test_invalid_runtime_model_fails_to_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("DW1_IMAGE_MODEL", str(tmp_path / "missing.npz"))
    monkeypatch.setenv("DW1_IMAGE_ENCODER", str(tmp_path / "missing.onnx"))
    content = io.BytesIO()
    Image.new("RGB", (32, 32)).save(content, format="PNG")
    assert image_signal(content.getvalue())["label"] == "unknown"
