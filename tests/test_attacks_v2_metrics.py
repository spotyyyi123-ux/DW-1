from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from dw1.eval.attacks_v2 import (
    CONDITIONS,
    RECEIPT_HASHES,
    SCORING_SCOPE,
    evaluate_predictions,
    main,
    threshold_at_fpr,
    validate_predictions,
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def predictions() -> pd.DataFrame:
    rows = []
    # Each split has two AI phishing, two human spam, and two human legitimate.
    for split in ("validation", "test"):
        for number in range(6):
            message_id = f"{split}-{number}"
            ai, malicious = int(number < 2), int(number < 4)
            for attack, strength in CONDITIONS:
                changed = bool(ai and attack != "clean")
                text = f"{message_id}/{attack}/{strength}" if changed else message_id
                p_prov = 0.9 - 0.7 * strength if ai else (0.4 if malicious else 0.1)
                p_intent = (0.9 - 0.7 * strength if ai else 0.9) if malicious else 0.4
                rows.append({
                    "variant_id": f"{message_id}:{attack}:{strength}",
                    "message_id": message_id, "attack": attack, "strength": strength,
                    "text_sha256": digest(text), "source_sha256": digest(message_id + ".eml"),
                    "y_provenance": ai, "y_intent": malicious, "split": split,
                    "changed": changed,
                    "p_tfidf_provenance": p_prov, "p_fast_detectgpt": p_prov,
                    "p_ensemble_provenance": p_prov, "p_intent": p_intent,
                    "tfidf_model_version": "test-tfidf", "fd_model_version": "test-fd",
                    "intent_model_version": "test-intent",
                })
    return pd.DataFrame(rows)


def test_fixed_clean_thresholds_expose_degradation(predictions, tmp_path):
    path = tmp_path / "predictions.parquet"
    predictions.to_parquet(path, index=False)
    result = evaluate_predictions(path, tmp_path / "out", expected_messages=12)
    assert result["status"] == "validation_run_not_600"
    assert result["population"] == {"total": 12, "validation": 6, "test": 6, "ai_test": 2}
    assert result["predictions_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["thresholds"]["tfidf"] > 0.4
    records = result["conditions"]
    for signal in ("tfidf", "fast_detectgpt", "ensemble"):
        clean = next(r for r in records if r["signal"] == signal and r["attack"] == "clean")
        attacked = next(r for r in records if r["signal"] == signal
                        and r["attack"] == "llm_paraphrase" and r["strength"] == 1)
        assert clean["tpr"] == 1
        assert attacked["tpr"] == 0  # Retuning on attacked scores would hide this loss.
        assert attacked["degradation_pct"] == 100
        assert attacked["threshold"] == clean["threshold"]
    intent = next(r for r in records if r["signal"] == "intent"
                  and r["attack"] == "llm_paraphrase" and r["strength"] == 1)
    assert intent["tpr"] == 0.5  # Unchanged human spam inflates aggregate recall.
    assert intent["ai_subset_tpr"] == 0
    assert intent["n_changed"] == 2
    output = tmp_path / "out"
    assert json.loads((output / "metrics.json").read_text()) == result
    assert "NaN" not in (output / "metrics.json").read_text()
    curves = json.loads((output / "degradation_curves.json").read_text())
    assert len(curves) == 32
    assert all(len(curve["points"]) == 5 for curve in curves)
    assert (output / "curves.svg").read_text().startswith('<svg xmlns=')
    assert "критерий 600 писем не выполнен" in (output / "report.md").read_text(encoding="utf-8")


def test_threshold_ties_and_small_negative_samples():
    y = np.array([0] * 100 + [1] * 3)
    p = np.array([0.8, 0.8] + [0.1] * 98 + [0.9, 0.8, 0.7])
    threshold = threshold_at_fpr(y, p)
    assert threshold > 0.8
    assert ((p[y == 0] >= threshold).mean()) == 0
    p[1] = 0.7
    threshold = threshold_at_fpr(y, p)
    assert 0.7 < threshold <= 0.8
    assert ((p[y == 0] >= threshold).mean()) == 0.01
    assert threshold_at_fpr(np.array([0, 1]), np.array([1., 1.])) > 1
    with pytest.raises(ValueError, match="both classes"):
        threshold_at_fpr(np.array([1]), np.array([0.5]))


@pytest.mark.parametrize("kind,match", [
    ("missing", "Incomplete population"),
    ("duplicate", "Duplicate variant_id"),
    ("metadata", "metadata changed"),
    ("ensemble", "Ensemble"),
    ("probability", "finite"),
    ("text", "source text"),
    ("hash", "Invalid text_sha256"),
    ("human", "human messages"),
    ("cross_split", "Cross-split"),
    ("unchanged_llm", "actually change"),
    ("same_text_different_score", "Identical texts"),
    ("changed", "changed flag"),
    ("version", "single nonempty"),
])
def test_invalid_predictions_are_rejected(predictions, kind, match):
    frame = predictions.copy()
    # This is an AI attack row; clean rows remain intact unless a test says otherwise.
    index = frame.index[(frame.message_id == "test-0")
                        & (frame.attack == "llm_paraphrase") & (frame.strength == 1)][0]
    if kind == "missing":
        frame = frame.drop(index)
    elif kind == "duplicate":
        frame = pd.concat([frame, frame.iloc[[index]]], ignore_index=True)
    elif kind == "metadata":
        frame.loc[index, "y_intent"] = 0
    elif kind == "ensemble":
        frame.loc[index, "p_ensemble_provenance"] = 0.99
    elif kind == "probability":
        frame.loc[index, "p_intent"] = float("nan")
        match = "incomplete"
    elif kind == "text":
        frame["text"] = "private email"
    elif kind == "hash":
        frame.loc[index, "text_sha256"] = "bad"
    elif kind == "human":
        index = frame.index[(frame.message_id == "test-5")
                            & (frame.attack == "paraphrase") & (frame.strength == 1)][0]
        frame.loc[index, "text_sha256"] = digest("changed human")
        frame.loc[index, "changed"] = True
    elif kind == "cross_split":
        test_hash = frame.loc[index, "text_sha256"]
        frame.loc[frame.index[1], "text_sha256"] = test_hash
    elif kind == "unchanged_llm":
        clean = frame[(frame.message_id == "test-0") & (frame.attack == "clean")].iloc[0]
        frame.loc[index, "text_sha256"] = clean.text_sha256
        frame.loc[index, "changed"] = False
    elif kind == "same_text_different_score":
        index = frame.index[(frame.message_id == "test-5")
                            & (frame.attack == "paraphrase") & (frame.strength == 1)][0]
        frame.loc[index, "p_intent"] = 0.5
    elif kind == "changed":
        frame.loc[index, "changed"] = False
    elif kind == "version":
        frame.loc[index, "fd_model_version"] = "different-fd"
    with pytest.raises(ValueError, match=match):
        validate_predictions(frame, expected_messages=12)


def test_small_fixture_requires_explicit_size_override(predictions):
    with pytest.raises(ValueError, match="600"):
        validate_predictions(predictions)


def test_cli_writes_generated_report(predictions, tmp_path, capsys):
    path = tmp_path / "predictions.parquet"
    predictions.to_parquet(path, index=False)
    main(["--predictions", str(path), "--out", str(tmp_path / "out"),
          "--expected-messages", "12"])
    assert json.loads(capsys.readouterr().out)["status"] == "validation_run_not_600"


def test_evaluation_does_not_write_partial_reports(predictions, tmp_path):
    path = tmp_path / "predictions.parquet"
    predictions.iloc[:-1].to_parquet(path, index=False)
    out = tmp_path / "invalid-output"
    with pytest.raises(ValueError):
        evaluate_predictions(path, out, expected_messages=12)
    assert not out.exists()


def scoring_receipt(path, n_messages):
    return {
        "status": "scored_not_evaluated", "n_messages": n_messages,
        "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "package_id": digest("unit-test-package"), "scoring_scope": SCORING_SCOPE,
        **{key: digest("unit-test-" + key) for key in RECEIPT_HASHES},
    }


def test_final600_requires_bound_scoring_evidence(predictions, tmp_path):
    # Synthetic unit fixture, confined to pytest's temporary directory.
    # This exercises the receipt gate, not a claim about actual benchmark quality.
    copies = []
    for number in range(50):
        frame = predictions.copy()
        for key in ("message_id", "variant_id"):
            frame[key] = frame[key].map(lambda value, n=number: f"copy-{n}:{value}")
        for key in ("source_sha256", "text_sha256"):
            frame[key] = frame[key].map(lambda value, n=number: digest(f"copy-{n}:{value}"))
        copies.append(frame)
    path = tmp_path / "predictions.parquet"
    pd.concat(copies, ignore_index=True).to_parquet(path, index=False)
    out = tmp_path / "final-evaluation"
    with pytest.raises(ValueError, match="requires the matching"):
        evaluate_predictions(path, out)
    assert not out.exists()
    receipt_path = tmp_path / "scoring_manifest.json"
    receipt = scoring_receipt(path, 600)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    result = evaluate_predictions(path, out)
    assert result["status"] == "measured"
    assert result["population"]["total"] == 600
    expected_receipt_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert result["scoring_manifest_sha256"] == expected_receipt_hash
    assert result["package_id"] == receipt["package_id"]
    assert result["scoring_evidence"] == {key: receipt[key] for key in RECEIPT_HASHES}
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "подтверждены владельцем" in report
    assert "не удостоверяет личность" in report
    for key, bad_value in [
        ("status", "not_scored"), ("n_messages", 599),
        ("predictions_sha256", digest("other-predictions")),
        ("package_id", "bad"), ("scoring_scope", "full_fusion"),
        *((key, "bad") for key in RECEIPT_HASHES),
    ]:
        receipt_path.write_text(json.dumps({**receipt, key: bad_value}), encoding="utf-8")
        invalid_out = tmp_path / ("invalid-" + key)
        with pytest.raises(ValueError, match="does not match"):
            evaluate_predictions(path, invalid_out)
        assert not invalid_out.exists()


def test_custom_scoring_receipt_path_keeps_fixture_status(predictions, tmp_path):
    path = tmp_path / "predictions.parquet"
    predictions.to_parquet(path, index=False)
    receipt_path = tmp_path / "custom-receipt.json"
    receipt_path.write_text(json.dumps(scoring_receipt(path, 12)), encoding="utf-8")
    result = evaluate_predictions(
        path, tmp_path / "out", expected_messages=12, scoring_manifest=receipt_path,
    )
    assert result["status"] == "validation_run_not_600"
    assert result["scoring_manifest_sha256"]


def test_evaluation_never_overwrites_nonempty_output(predictions, tmp_path):
    path = tmp_path / "predictions.parquet"
    predictions.to_parquet(path, index=False)
    out = tmp_path / "out"
    out.mkdir()
    marker = out / "previous-result.txt"
    marker.write_text("preserve this run", encoding="utf-8")
    with pytest.raises(ValueError, match="must be empty"):
        evaluate_predictions(path, out, expected_messages=12)
    assert marker.read_text(encoding="utf-8") == "preserve this run"
    assert list(out.iterdir()) == [marker]
