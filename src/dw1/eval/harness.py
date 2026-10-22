"""Воспроизводимое обучение и проверка происхождения текста на CPU; назначение не оценивается."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from threadpoolctl import threadpool_limits

from dw1.models.baseline import MODEL_VERSION, build_model


def binary_metrics(y, score):
    y, score = np.asarray(y), np.asarray(score)
    if len(np.unique(y)) != 2:
        raise ValueError("Both ground-truth classes required for binary evaluation")
    pred = score >= 0.5
    return dict(
        n=len(y),
        accuracy=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y, score)),
        confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    )


def group_bootstrap(y, scores, groups, repeats=200):
    """Повторно выбирает целые доступные группы: вопросы HC3 или отдельные строки CoAT."""
    y, scores, groups = np.asarray(y), np.asarray(scores), np.asarray(groups)
    unique = np.unique(groups)
    indices = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(42)
    values = []
    for _ in range(repeats):
        ix = np.concatenate([indices[g] for g in rng.choice(unique, len(unique))])
        if len(np.unique(y[ix])) < 2:
            continue
        values.append(
            [f1_score(y[ix], scores[ix] >= 0.5, zero_division=0), roc_auc_score(y[ix], scores[ix])]
        )
    if not values:
        raise ValueError("No valid bootstrap draws")
    bounds = np.quantile(values, [0.025, 0.975], axis=0)
    return {key: bounds[:, i].tolist() for i, key in enumerate(["f1", "roc_auc"])}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="sprint1")
    ap.add_argument("--data", default="data/processed/w1_texts.parquet")
    ap.add_argument("--emails", default="data/processed/w1_emails.parquet")
    args = ap.parse_args(argv)
    if Path(args.run).name != args.run:
        ap.error("--run must be a simple directory name")
    out = Path("data/runs") / args.run
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    data = pd.read_parquet(args.data)
    train = data[data.split == "train"]
    if set(train.label.unique()) != {0, 1}:
        raise ValueError("Training requires human and AI")
    # Один и тот же текст или группа ответов не должны попадать в разные части данных.
    for key in ["fingerprint", "group_id"]:
        if data.groupby(key).split.nunique().max() > 1:
            raise ValueError(f"Cross-split leakage: {key}")
    model = build_model()
    print(f"Training on {len(train)} rows...", flush=True)
    with threadpool_limits(limits=2):
        model.fit(train.text, train.label)
    dummy = DummyClassifier(strategy="prior").fit(np.zeros((len(train), 1)), train.label)
    predictions = []
    summary = {}
    now = datetime.now(timezone.utc).isoformat()
    for split in ["validation", "test"]:
        part = data[data.split == split].copy()
        with threadpool_limits(limits=2):
            scores = model.predict_proba(part.text)[:, 1]
        part["p_provenance_ai"] = scores
        part["p_provenance_human"] = 1 - scores
        part["p_intent_phishing"] = np.nan
        part["p_intent_spam"] = np.nan
        part["p_intent_legit"] = np.nan
        part["y_provenance"] = part.ai_provenance
        part["y_intent"] = "unknown"
        part["campaign_id"] = None
        part["model_version"] = MODEL_VERSION
        part["predicted_at"] = now
        summary[split] = binary_metrics(part.label, scores)
        summary[split]["ci95_group_bootstrap"] = group_bootstrap(part.label, scores, part.group_id)
        summary[split]["by_source"] = {
            source: binary_metrics(g.label, g.p_provenance_ai)
            for source, g in part.groupby("source")
        }
        dummy_scores = dummy.predict_proba(np.zeros((len(part), 1)))[:, 1]
        summary[split]["dummy_prior"] = binary_metrics(part.label, dummy_scores)
        predictions.append(part.drop(columns=["text"]))
        errors = part[(scores >= 0.5).astype(int) != part.label].copy()
        errors["text_excerpt"] = errors.text.str.slice(0, 500)
        errors.drop(columns=["text"]).head(30).to_json(
            out / f"{split}_errors.jsonl", orient="records", lines=True, force_ascii=False
        )
    pd.concat(predictions).to_parquet(out / "predictions.parquet", index=False)
    emails = pd.read_parquet(args.emails)
    external = emails[(emails.source == "llmgen-phishing") & (emails.language == "en")].copy()
    if len(external):
        with threadpool_limits(limits=2):
            scores = model.predict_proba(external.text.fillna(""))[:, 1]
        external["p_provenance_ai"] = scores
        external.drop(columns="text").to_parquet(out / "external_predictions.parquet", index=False)
        summary["external_llmgen_en"] = dict(
            n=len(external),
            ai_detection_rate=float(np.mean(scores >= 0.5)),
            by_intent={
                intent: float((g.p_provenance_ai >= 0.5).mean())
                for intent, g in external.groupby("intent")
            },
            limitation="AI-only set: no human false-positive rate, binary F1 or ROC-AUC.",
        )
    joblib.dump(model, out / "model.joblib")
    summary.update(
        model_version=MODEL_VERSION,
        threshold=0.5,
        calibrated=False,
        seed=42,
        train_rows=len(train),
        elapsed_seconds=time.perf_counter() - start,
        data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        environment=dict(python=platform.python_version(), sklearn=sklearn.__version__),
        scope="General text provenance pilot, NOT validated email/campaign detector.",
        unavailable=[
            "intent metrics",
            "ai_polished class",
            "calibration",
            "full source-pair grouping for CoAT",
            "attack robustness",
        ],
    )
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
