"""Обучение модели происхождения на письмах (Sprint 2, задача 4).

Модель: TF-IDF → SVD → HistGradientBoosting (переиспользуется baseline.py).
Данные: письма (LLMGen = AI; seven-phishing ham + enron ham = human),
**бенчмарк-письма исключаются** (по нормализованному тексту) — без утечки.

Выход: data/runs/sprint2/prov_model.joblib + prov_model_meta.json

Запуск: python scripts/train_prov_model.py
"""
from __future__ import annotations

import hashlib
import json
import random
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


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def _hash(text: str) -> str:
    return hashlib.sha256(_norm(text).encode("utf-8")).hexdigest()[:16]


def _bench_texts() -> set[str]:
    """Хэши текстов бенчмарка, чтобы исключить их из обучения."""
    hashes: set[str] = set()
    if not BENCH.exists():
        return hashes
    for eml in BENCH.glob("*.eml"):
        raw = eml.read_text(encoding="utf-8", errors="ignore")
        body = raw.split("\n\n", 1)[-1]
        hashes.add(_hash(body))
    return hashes


def _load_seven(banned: set[str]) -> list[tuple[str, int]]:
    df = pd.read_parquet(ROOT / "data" / "raw" / "seven-phishing" / "test.parquet")
    rows = []
    for _, row in df.iterrows():
        body = str(row.get("text", ""))
        if not body or _hash(body) in banned:
            continue
        # спам и хам — оба человеческие тексты для оси происхождения
        rows.append((body, 0))
    return rows


def _load_enron(banned: set[str]) -> list[tuple[str, int]]:
    df = pd.read_csv(ROOT / "data" / "raw" / "enron_spam" / "enron_spam_data.csv")
    df = df.dropna(subset=["Message"])
    rows = []
    for body in df["Message"].astype(str):
        if not body or _hash(body) in banned:
            continue
        rows.append((body, 0))
    return rows


def _load_llmgen(banned: set[str]) -> list[tuple[str, int]]:
    df = pd.read_csv(ROOT / "data" / "raw" / "llmgen_phishing.csv")
    rows = []
    for content in df["content"].astype(str):
        if len(content) < 60 or _hash(content) in banned:
            continue
        rows.append((content, 1))
    return rows


def main() -> int:
    random.seed(SEED)
    rng = np.random.RandomState(SEED)
    banned = _bench_texts()
    print(f"исключаем бенчмарк-писем: {len(banned)}")

    human = _load_seven(banned) + _load_enron(banned)
    ai = _load_llmgen(banned)
    rng.shuffle(human)
    human = human[: min(len(human), 8000)]
    print(f"train: human={len(human)}, ai={len(ai)}")

    texts = [t for t, _ in human] + [t for t, _ in ai]
    labels = [y for _, y in human] + [y for _, y in ai]

    idx = np.arange(len(texts))
    rng.shuffle(idx)
    split = int(len(idx) * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]

    model = build_model(seed=SEED)
    model.fit([texts[i] for i in train_idx], [labels[i] for i in train_idx])

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    proba = model.predict_proba(test_texts)[:, 1]
    metrics = {
        "model_version": MODEL_VERSION,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "accuracy": round(float(accuracy_score(test_labels, (proba >= 0.5).astype(int))), 4),
        "f1_ai": round(float(f1_score(test_labels, (proba >= 0.5).astype(int))), 4),
        "roc_auc": round(float(roc_auc_score(test_labels, proba)), 4),
    }
    print("holdout:", metrics)

    OUT.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(model, OUT / "prov_model.joblib")

    # --- кросс-доменная проверка: модель учится на одном человеческом источнике,
    # тестируется на другом (видит ли она «ИИ», а не домен?)
    bench_human: dict[str, list[str]] = {}
    for eml in BENCH.glob("*.eml"):
        name = eml.name
        if name.startswith("enron-ham"):
            key = "enron"
        elif name.startswith("seven-ham"):
            key = "seven"
        else:
            continue
        raw = eml.read_text(encoding="utf-8", errors="ignore")
        bench_human.setdefault(key, []).append(raw.split("\n\n", 1)[-1])

    seven_human = [t for t, _ in _load_seven(banned)]
    enron_human = [t for t, _ in _load_enron(banned)]
    ai_rest = [t for t, _ in _load_llmgen(banned)]

    def _fpr_on(model_tmp, texts: list[str]) -> float | None:
        if not texts:
            return None
        proba = model_tmp.predict_proba(texts)[:, 1]
        return round(float((proba >= 0.5).mean()), 3)

    model_seven = build_model(seed=SEED)
    model_seven.fit(seven_human[:4000] + ai_rest[:4000], [0] * min(4000, len(seven_human)) + [1] * min(4000, len(ai_rest)))
    model_enron = build_model(seed=SEED)
    model_enron.fit(enron_human[:4000] + ai_rest[:4000], [0] * min(4000, len(enron_human)) + [1] * min(4000, len(ai_rest)))

    cross = {
        "fpr_enron_unseen_by_seven_model": _fpr_on(model_seven, bench_human.get("enron", [])),
        "fpr_seven_unseen_by_enron_model": _fpr_on(model_enron, bench_human.get("seven", [])),
    }

    # кросс-генераторная проверка на HC3 (другой генератор — ChatGPT, не GPT-LLMGen)
    hc3_path = ROOT / "data" / "raw" / "hc3" / "all.jsonl"
    if hc3_path.exists():
        import json as _json

        pairs = []
        with open(hc3_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = _json.loads(line)
                except Exception:
                    continue
                human = row.get("human_answers") or []
                ai = row.get("chatgpt_answers") or []
                if human and ai:
                    pairs.append((" ".join(human[0]), " ".join(ai[0])))
                if len(pairs) >= 1000:
                    break
        if pairs:
            human_txts = [h for h, _ in pairs]
            ai_txts = [a for _, a in pairs]
            proba_h = model.predict_proba(human_txts)[:, 1]
            proba_a = model.predict_proba(ai_txts)[:, 1]
            cross["hc3_chatgpt_tpr"] = round(float((proba_a >= 0.5).mean()), 3)
            cross["hc3_human_fpr"] = round(float((proba_h >= 0.5).mean()), 3)
    print("кросс-доменный/кросс-генераторный тест:", cross)

    meta = dict(metrics)
    meta["cross_domain"] = cross
    (OUT / "prov_model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"модель: {OUT / 'prov_model.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
