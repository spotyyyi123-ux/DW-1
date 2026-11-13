from __future__ import annotations

import csv
from email.message import EmailMessage
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from scripts.attacks_v2 import (
    PURPOSE,
    body_text,
    digest,
    load_benchmark,
    normalized_hash,
    prepare,
    read_json,
    read_rows,
    score,
    text_hash,
    validate_export,
    verify_package,
    write_json,
    write_rows,
)


@pytest.fixture
def benchmark(tmp_path):
    bench = tmp_path / "bench"
    bench.mkdir()
    labels = []
    for group, prov, intent in (
        ("ai", "ai_generated", "phishing"),
        ("spam", "human", "spam"),
        ("ham", "human", "legitimate"),
    ):
        for i in range(10):
            name = f"{group}-{i}.eml"
            msg = EmailMessage()
            msg["Subject"] = "Private subject excluded from body benchmark"
            msg.set_content(
                f"{group} original message number {i}. Please note that you should verify your "
                "account immediately. Several additional sentences provide a complete message "
                "for testing. Visit https://example.org/check to continue. Thank you for reading."
            )
            (bench / name).write_bytes(msg.as_bytes())
            labels.append(dict(file=name, provenance=prov, intent=intent, source="fixture"))
    with (bench / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(labels[0]))
        writer.writeheader()
        writer.writerows(labels)
    provenance = tmp_path / "provenance.json"
    write_json(provenance, dict(
        purpose=PURPOSE, external_allowed=True, internal_customer_data=False,
        approved_by="test fixture author", approved_at="2026-11-13",
        sources={"fixture": dict(public_dataset=True, external_allowed=True,
                                 license="CC0-1.0", evidence="synthetic test fixture")},
    ))
    return bench, provenance


def test_frozen_grid_is_deterministic_and_keeps_human_controls(benchmark, tmp_path):
    bench, provenance = benchmark
    first, second = tmp_path / "one", tmp_path / "two"
    result = prepare(bench, provenance, first, expected_messages=30)
    prepare(bench, provenance, second, expected_messages=30)
    assert result["status"] == "prepared_not_measured"
    assert result["package_id"] == verify_package(second)["package_id"]
    variants = read_rows(first / "variants.jsonl")
    requests = read_rows(first / "requests.jsonl")
    assert len(variants) + len(requests) == 30 * 33
    assert len(requests) == 10 * 4
    clean = {r["message_id"]: r for r in variants if r["attack"] == "clean"}
    assert {r["split"] for r in clean.values()} == {"validation", "test"}
    for row in variants:
        assert row["split"] == clean[row["message_id"]]["split"]
        if not row["y_provenance"]:
            assert row["text"] == clean[row["message_id"]]["text"]
            assert not row["changed"]
    with pytest.raises(ValueError, match="immutable"):
        prepare(bench, provenance, first, expected_messages=30)
    (second / "variants.jsonl").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Package changed"):
        verify_package(second)


def test_refuses_other_task_permission_or_unresolved_source(benchmark, tmp_path):
    bench, provenance = benchmark
    config = read_json(provenance)
    config["purpose"] = "bcubed"
    write_json(provenance, config)
    with pytest.raises(ValueError, match="specifically"):
        prepare(bench, provenance, tmp_path / "out", expected_messages=30)
    assert not (tmp_path / "out").exists()
    config["purpose"] = PURPOSE
    config["sources"]["fixture"]["license"] = "other"
    with pytest.raises(ValueError, match="Unresolved"):
        validate_export(config, {"fixture"})


def test_missing_email_and_duplicate_body_are_not_silently_counted(benchmark):
    bench, _ = benchmark
    (bench / "ham-1.eml").write_bytes((bench / "ham-0.eml").read_bytes())
    with pytest.raises(ValueError, match="Duplicate bodies"):
        load_benchmark(bench, 30)
    (bench / "ham-1.eml").unlink()
    with pytest.raises(FileNotFoundError):
        load_benchmark(bench, 30)


def test_only_inline_plain_body_is_exported(tmp_path):
    msg = EmailMessage()
    msg.set_content("Visible body only.")
    msg.add_attachment("PRIVATE ATTACHMENT", subtype="plain", filename="secret.txt")
    path = tmp_path / "multipart.eml"
    path.write_bytes(msg.as_bytes())
    assert body_text(path) == "Visible body only.\n"


def make_external_and_models(package: Path, tmp_path: Path):
    external = tmp_path / "external"
    external.mkdir()
    rewrites, reviews = [], []
    for row in read_rows(package / "requests.jsonl"):
        text = row["text"].replace("Please note that", "Kindly remember that")
        key = text_hash(text)
        rewrites.append(dict(variant_id=row["variant_id"], text=text, text_sha256=key))
        reviews.append(dict(variant_id=row["variant_id"], text_sha256=key,
                            package_id=read_json(package / "manifest.json")["package_id"],
                            original_text_sha256=row["text_sha256"],
                            approved=True, reviewer="synthetic fixture only",
                            meaning_preserved=True, language_preserved=True))
    write_rows(external / "external_variants.jsonl", rewrites)
    generated = {r["variant_id"]: r for r in rewrites}
    variants = read_rows(package / "variants.jsonl") + [
        dict(row, **{key: generated[row["variant_id"]][key] for key in ("text", "text_sha256")})
        for row in read_rows(package / "requests.jsonl")
    ]
    pd.DataFrame([dict(variant_id=r["variant_id"], text_sha256=r["text_sha256"],
                       criterion=1.0 + r["y_provenance"], model_version="fixture-fd-v1",
                       n_tokens=64, truncated=False) for r in variants]).to_parquet(
        external / "fd_scores.parquet", index=False
    )
    write_json(external / "external_result.json", dict(
        schema_version=1,
        package_id=read_json(package / "manifest.json")["package_id"], status="completed",
        runner_sha256=read_json(package / "manifest.json")["files"]["run_attacks_v2.py"],
        paraphrase_revision="a" * 40, detector_revision="b" * 40,
        detector_model="synthetic-fixture", paraphrase_model="synthetic-fixture",
        n_scored=len(variants), n_variants=len(variants), skipped=[], model_version="fixture-fd-v1",
        files={name: digest((external / name).read_bytes())
               for name in ("external_variants.jsonl", "fd_scores.parquet")},
    ))
    review = tmp_path / "semantic_review.jsonl"
    write_rows(review, reviews)
    training_texts = ["training human benign", "training generated phishing",
                      "training benign meeting", "training generated password"]
    model = make_pipeline(CountVectorizer(), LogisticRegression()).fit(training_texts, [0, 1, 0, 1])
    joblib.dump(model, tmp_path / "model.joblib")
    config = dict(schema_version=1, training_hashes_complete=True, models={},
                  training_hash_policy="exact-utf8+body-lower-whitespace-v1",
                  attested_by="synthetic fixture author", attested_at="2026-11-13")
    for axis in ("provenance", "intent"):
        config["models"][axis] = dict(
            path="model.joblib", sha256=digest((tmp_path / "model.joblib").read_bytes()),
            version="synthetic-tfidf-test-only",
            training_text_sha256=[text_hash(text) for text in training_texts],
            training_body_normalized_sha256=[normalized_hash(text) for text in training_texts],
            training_provenance="All four synthetic training bodies from this test fixture",
        )
    models = tmp_path / "models.json"
    write_json(models, config)
    return external, models, review


def test_score_binds_texts_and_reviews_and_saves_replayable_predictions(benchmark, tmp_path):
    bench, provenance = benchmark
    package = tmp_path / "package"
    prepare(bench, provenance, package, expected_messages=30)
    external, models, review = make_external_and_models(package, tmp_path)
    out = tmp_path / "scored"
    receipt = score(package, external, models, review, out)
    frame = pd.read_parquet(out / "predictions.parquet")
    assert receipt["n_messages"] == 30
    assert len(frame) == 30 * 33
    assert not {"text", "body", "prompt"} & set(frame)
    assert (frame.p_ensemble_provenance == (
        frame.p_tfidf_provenance + frame.p_fast_detectgpt
    ) / 2).all()
    assert all(cfg["fitted_on"] == "clean_validation_only"
               for cfg in read_json(out / "calibration.json").values())
    reviews = read_rows(review)
    reviews[0]["approved"] = False
    write_rows(review, reviews)
    with pytest.raises(ValueError, match="approval"):
        score(package, external, models, review, tmp_path / "unreviewed")
    assert not (tmp_path / "unreviewed").exists()


def test_score_rejects_training_overlap_before_model_inference(benchmark, tmp_path):
    bench, provenance = benchmark
    package = tmp_path / "package"
    prepare(bench, provenance, package, expected_messages=30)
    external, models, review = make_external_and_models(package, tmp_path)
    config = read_json(models)
    clean = next(r for r in read_rows(package / "variants.jsonl") if r["attack"] == "clean")
    config["models"]["provenance"]["training_text_sha256"] = [clean["text_sha256"]]
    write_json(models, config)
    with pytest.raises(ValueError, match="leakage"):
        score(package, external, models, review, tmp_path / "leaked")


def test_default_requires_real_600(benchmark):
    with pytest.raises(ValueError, match="600"):
        load_benchmark(benchmark[0])


def test_normalized_training_overlap_is_rejected(benchmark, tmp_path):
    bench, provenance = benchmark
    package = tmp_path / "package"
    prepare(bench, provenance, package, expected_messages=30)
    external, models, review = make_external_and_models(package, tmp_path)
    config = read_json(models)
    clean = next(r for r in read_rows(package / "variants.jsonl") if r["attack"] == "clean")
    # A MIME whitespace/case change must not make a training email appear independent.
    other_form = clean["text"].upper().replace(" ", "  ").replace("\n", "\r\n")
    config["models"]["provenance"]["training_text_sha256"] = [text_hash(other_form)]
    config["models"]["provenance"]["training_body_normalized_sha256"] = [
        normalized_hash(other_form)
    ]
    write_json(models, config)
    with pytest.raises(ValueError, match="leakage"):
        score(package, external, models, review, tmp_path / "normalized-leaked")


@pytest.mark.parametrize(("field", "value", "error"), [
    ("runner_sha256", "0" * 64, "runner differs"),
    ("model_version", "changed-fd-version", "version differs"),
    ("detector_revision", "main", "Pinned"),
    ("n_scored", 1, "coverage"),
])
def test_external_run_must_match_frozen_code_and_model(benchmark, tmp_path, field, value, error):
    bench, provenance = benchmark
    package = tmp_path / "package"
    prepare(bench, provenance, package, expected_messages=30)
    external, models, review = make_external_and_models(package, tmp_path)
    result = read_json(external / "external_result.json")
    result[field] = value
    write_json(external / "external_result.json", result)
    with pytest.raises(ValueError, match=error):
        score(package, external, models, review, tmp_path / "invalid")


def test_semantic_approval_cannot_move_between_packages(benchmark, tmp_path):
    bench, provenance = benchmark
    package = tmp_path / "package"
    prepare(bench, provenance, package, expected_messages=30)
    external, models, review = make_external_and_models(package, tmp_path)
    rows = read_rows(review)
    rows[0]["package_id"] = "0" * 64
    write_rows(review, rows)
    with pytest.raises(ValueError, match="approval"):
        score(package, external, models, review, tmp_path / "moved")
