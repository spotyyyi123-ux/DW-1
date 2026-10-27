"""Обучение модели намерения на письмах (Sprint 2, задача 4).

Задача: malicious (спам/фишинг) vs legitimate. Та же схема, что и для provenance:
TF-IDF → SVD → HistGB, бенчмарк исключается по хэшу текста.

Выход: data/runs/sprint2/intent_model.joblib + intent_model_meta.json

Запуск: python scripts/train_intent_model.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dw1.models.baseline import MODEL_VERSION, build_model  # noqa: E402

BENCH = ROOT / "data" / "sprint2" / "bench"
OUT = ROOT / "data" / "runs" / "sprint2"
SEED = 42


def _hash(text: str) -> str:
    return hashlib.sha256(" ".join((text or "").lower().split()).encode("utf-8")).hexdigest()[:16]


def _banned() -> set[str]:
    hashes = set()
    for eml in BENCH.glob("*.eml"):
        body = eml.read_text(encoding="utf-8", errors="ignore").split("\n\n", 1)[-1]
        hashes.add(_hash(body))
    return hashes


def _load() -> tuple[list[str], list[int]]:
    banned = _banned()
    texts: list[str] = []
    labels: list[int] = []

    seven = pd.read_parquet(ROOT / "data" / "raw" / "seven-phishing" / "test.parquet")
    for _, row in seven.iterrows():
        body = str(row.get("text", ""))
        if not body or _hash(body) in banned:
            continue
        texts.append(body)
        labels.append(int(row.get("label", 0)))

    enron = pd.read_csv(ROOT / "data" / "raw" / "enron_spam" / "enron_spam_data.csv").dropna(
        subset=["Message"]
    )
    for _, row in enron.iterrows():
        body = str(row["Message"])
        if not body or _hash(body) in banned:
            continue
        texts.append(body)
        labels.append(1 if str(row["Spam/Ham"]).lower() == "spam" else 0)
    return texts, labels


def main() -> int:
    rng = np.random.RandomState(SEED)
    texts, labels = _load()
    print(f"данные: n={len(texts)}, malicious={sum(labels)}")

    idx = np.arange(len(texts))
    rng.shuffle(idx)
    split = int(len(idx) * 0.8)
    tr, te = idx[:split], idx[split:]

    model = build_model(seed=SEED)
    model.fit([texts[i] for i in tr], [labels[i] for i in tr])

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    proba = model.predict_proba([texts[i] for i in te])[:, 1]
    y = [labels[i] for i in te]
    metrics = {
        "model_version": MODEL_VERSION + "-intent",
        "n_total": len(texts),
        "n_test": len(te),
        "accuracy": round(float(accuracy_score(y, (proba >= 0.5).astype(int))), 4),
        "f1_malicious": round(float(f1_score(y, (proba >= 0.5).astype(int))), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
    }
    print("holdout:", metrics)

    OUT.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(model, OUT / "intent_model.joblib")
    (OUT / "intent_model_meta.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"модель: {OUT / 'intent_model.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
