"""Внешняя валидация: cross-domain матрица «модель × домен» (Sprint 4, задача 4).

Источники (train): письма (LLMGen AI / seven+enron human), HC3-EN, CoAT-RU.
Мишени (test): те же домены. Модель происхождения: TF-IDF → SVD-64 → HistGB
(тот же пайплайн, что baseline/build_model).

Fast-DetectGPT: письма — скоры Алины (Qwen), тексты — метрики Sprint 3.

Выход: data/runs/sprint4/external_validation.json + таблица в stdout.

Запуск: python scripts/external_validation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from dw1.models.baseline import build_model  # noqa: E402


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    total = len(y)
    if total == 0:
        return 0.0
    score = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (p >= lo) & (p <= 1.0 if i == bins - 1 else p < hi)
        if mask.any():
            score += mask.sum() / total * abs(y[mask].mean() - p[mask].mean())
    return float(score)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    n = len(y)
    out = {"n": n, "f1": None, "auc": None, "ece": None}
    if n == 0:
        return out
    out["f1"] = round(float(f1_score(y, (p >= 0.5).astype(int), zero_division=0)), 4)
    out["ece"] = round(ece(y, p), 4)
    if len(np.unique(y)) == 2:
        out["auc"] = round(float(roc_auc_score(y, p)), 4)
    return out


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    emails = pd.read_json(ROOT / "data/sprint2/llm_benchmark_emails.jsonl", lines=True)
    texts = pd.read_json(ROOT / "data/sprint2/llm_benchmark.jsonl", lines=True)
    return emails, texts


def main() -> int:
    emails, texts = load()
    sources = {
        "emails": (emails[emails.split == "train"], emails[["text", "label"]]),
        "hc3": (texts[(texts.dataset == "hc3-en") & (texts.split == "train")], texts),
        "coat": (texts[(texts.dataset == "coat") & (texts.split == "train")], texts),
    }
    targets = {
        "emails": (emails[emails.split == "test"], emails),
        "hc3": (texts[(texts.dataset == "hc3-en") & (texts.split == "test")], texts),
        "coat": (texts[(texts.dataset == "coat") & (texts.split == "test")], texts),
    }

    rows = []
    for source, (train, _) in sources.items():
        model = build_model().fit(train["text"].tolist(), train["label"].astype(int).tolist())
        for target, (test, _) in targets.items():
            if test.empty:
                continue
            proba = model.predict_proba(test["text"].tolist())[:, 1]
            m = metrics(test["label"].to_numpy(), proba)
            rows.append({"source": f"TF-IDF ({source})", "target": target, **m})

    # Fast-DetectGPT: письма (скоры Алины, Platt на train её сплита)
    scores_path = ROOT / "data/runs/sprint3/alina_email_scores.parquet"
    if scores_path.exists():
        al = pd.read_parquet(scores_path)
        tr = al[al.split == "train"]
        te = al[al.split == "test"]
        clf = LogisticRegression().fit(tr[["criterion"]], tr["label"])
        proba = clf.predict_proba(te[["criterion"]])[:, 1]
        m = metrics(te["label"].to_numpy(), proba)
        rows.append({"source": "Fast-DetectGPT (Qwen)", "target": "emails", **m})

    out = ROOT / "data" / "runs" / "sprint4" / "external_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'источник':26}{'мишень':10}{'n':>5}{'AUC':>8}{'F1':>8}{'ECE':>8}")
    print("-" * 65)
    for r in rows:
        print(
            f"{r['source']:26}{r['target']:10}{r['n']:>5}"
            f"{str(r['auc']):>8}{str(r['f1']):>8}{str(r['ece']):>8}"
        )
    print(f"\nтаблица: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
