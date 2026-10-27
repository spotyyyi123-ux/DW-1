"""Мульти-генераторная модель происхождения (Sprint 2, задача 4).

Проблема, которую решаем: модель, обученная только на LLMGen, ловит стиль конкретного
генератора (TPR=0 на HC3/ChatGPT). Добавляем HC3 (ChatGPT, EN) и CoAT (русские генераторы)
и проверяем перенос.

Данные:
- письма: human = seven-phishing + enron, AI = LLMGen (бенчмарк исключён по хэшу);
- тексты EN: HC3 (human answers vs ChatGPT answers);
- тексты RU: CoAT binary (86k+86k, сэмплируем).

Выход: data/runs/sprint2/prov_model.joblib (заменяет предыдущую модель) + meta.

Запуск: python scripts/train_multi_gen.py
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
COAT_SAMPLE = 5000
HC3_SAMPLE = 2000


def _hash(text: str) -> str:
    return hashlib.sha256(" ".join((text or "").lower().split()).encode("utf-8")).hexdigest()[:16]


def _bench_hashes() -> set[str]:
    hashes = set()
    for eml in BENCH.glob("*.eml"):
        body = eml.read_text(encoding="utf-8", errors="ignore").split("\n\n", 1)[-1]
        hashes.add(_hash(body))
    return hashes


def _emails(banned: set[str]) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []

    seven = pd.read_parquet(ROOT / "data" / "raw" / "seven-phishing" / "test.parquet")
    for body in seven["text"].astype(str):
        if body and _hash(body) not in banned and len(texts) < 4000:
            texts.append(body)
            labels.append(0)

    enron = pd.read_csv(ROOT / "data" / "raw" / "enron_spam" / "enron_spam_data.csv").dropna(
        subset=["Message"]
    )
    for body in enron["Message"].astype(str):
        if body and _hash(body) not in banned and len(texts) < 8000:
            texts.append(body)
            labels.append(0)

    llmgen = pd.read_csv(ROOT / "data" / "raw" / "llmgen_phishing.csv")
    for content in llmgen["content"].astype(str):
        if len(content) >= 60 and _hash(content) not in banned:
            texts.append(content)
            labels.append(1)
    return texts, labels


def _hc3() -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    path = ROOT / "data" / "raw" / "hc3" / "all.jsonl"
    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(texts) >= HC3_SAMPLE * 2:
                break
            try:
                row = json.loads(line)
            except Exception:
                continue
            human = row.get("human_answers") or []
            ai = row.get("chatgpt_answers") or []
            if human and len(texts) < HC3_SAMPLE:
                texts.append(" ".join(human[0]))
                labels.append(0)
            if ai and len(texts) < HC3_SAMPLE * 2:
                texts.append(" ".join(ai[0]))
                labels.append(1)
    return texts, labels


def _coat() -> tuple[list[str], list[int]]:
    df = pd.read_parquet(ROOT / "data" / "raw" / "coat" / "binary" / "train-00000-of-00001.parquet")
    df = df.sample(n=min(len(df), COAT_SAMPLE * 2), random_state=SEED)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels


def _fit(train_texts, train_labels):
    model = build_model(seed=SEED)
    model.fit(train_texts, train_labels)
    return model


def _score(model, texts: list[str], labels: list[int]) -> dict:
    from sklearn.metrics import accuracy_score, roc_auc_score

    proba = model.predict_proba(texts)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = {"n": len(texts), "accuracy": round(float(accuracy_score(labels, pred)), 4)}
    if 0 < sum(labels) < len(labels):
        out["roc_auc"] = round(float(roc_auc_score(labels, proba)), 4)
    return out


def main() -> int:
    rng = np.random.RandomState(SEED)
    banned = _bench_hashes()
    e_texts, e_labels = _emails(banned)
    h_texts, h_labels = _hc3()
    c_texts, c_labels = _coat()
    texts = e_texts + h_texts + c_texts
    labels = e_labels + h_labels + c_labels
    print(f"данные: emails={len(e_texts)} hc3={len(h_texts)} coat={len(c_texts)}")

    idx = np.arange(len(texts))
    rng.shuffle(idx)
    split = int(len(idx) * 0.8)
    tr, te = idx[:split], idx[split:]

    model = _fit([texts[i] for i in tr], [labels[i] for i in tr])
    holdout = _score(model, [texts[i] for i in te], [labels[i] for i in te])
    print("holdout:", holdout)

    # кросс-генератор: модель без HC3 → проверка на HC3 (ChatGPT)
    texts_no_hc3 = e_texts + c_texts
    labels_no_hc3 = e_labels + c_labels
    model_no_hc3 = _fit(texts_no_hc3, labels_no_hc3)
    hc3_eval = _score(model_no_hc3, h_texts, h_labels)
    print("HC3 (ChatGPT) как невиданный генератор:", hc3_eval)

    # кросс-домен RU: модель без CoAT → проверка на CoAT (русские генераторы)
    texts_no_coat = e_texts + h_texts
    labels_no_coat = e_labels + h_labels
    model_no_coat = _fit(texts_no_coat, labels_no_coat)
    coat_eval = _score(model_no_coat, c_texts, c_labels)
    print("CoAT (RU) как невиданный домен/генератор:", coat_eval)

    OUT.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(model, OUT / "prov_model.joblib")
    meta = {
        "model_version": MODEL_VERSION + "-multi",
        "n_total": len(texts),
        "sources": {"emails": len(e_texts), "hc3": len(h_texts), "coat": len(c_texts)},
        "holdout": holdout,
        "cross_generator_hc3": hc3_eval,
        "cross_generator_coat": coat_eval,
    }
    (OUT / "prov_model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"модель: {OUT / 'prov_model.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
