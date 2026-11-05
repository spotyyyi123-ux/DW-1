"""Сравнение детекторов по сохранённым скорам (AUC/F1/ECE после Platt-калибровки).

Вход: один или несколько parquet-файлов со скорами (формат llm_detectors.validate_scores:
message_id, text_sha256, group_id, dataset, language, label, split, criterion, ...).

Для каждого детектора: Platt (логистическая регрессия) на train → вероятность на test;
метрики: ROC-AUC, F1@0.5, ECE (10 бинов), по языкам en/ru и суммарно.

Запуск:
    python scripts/compare_detectors.py --scores gptneo=reports/sprint2-llm/llm_scores.parquet --scores qwen=data/runs/sprint2/qwen_scores.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "runs" / "sprint2" / "detector_comparison.json"


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    total = len(y)
    if total == 0:
        return 0.0
    score = 0.0
    for lo in range(bins):
        lo_edge, hi_edge = lo / bins, (lo + 1) / bins
        mask = (p >= lo_edge) & (p <= 1.0 if lo == bins - 1 else p < hi_edge)
        if not mask.any():
            continue
        score += mask.sum() / total * abs(y[mask].mean() - p[mask].mean())
    return float(score)


def evaluate(name: str, frame: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for lang in ("en", "ru", None):
        sub = frame if lang is None else frame[frame.language == lang]
        train = sub[sub.split == "train"]
        test = sub[sub.split == "test"]
        if train.empty or test.empty or len(np.unique(test.label)) < 2:
            continue
        clf = LogisticRegression().fit(train[["criterion"]], train["label"])
        proba = clf.predict_proba(test[["criterion"]])[:, 1]
        y = test["label"].to_numpy()
        pred = (proba >= 0.5).astype(int)
        rows.append(
            {
                "detector": name,
                "language": lang or "all",
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "roc_auc": round(float(roc_auc_score(y, proba)), 4),
                "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
                "ece": round(ece(y, proba), 4),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", action="append", required=True, help="name=path.parquet")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    all_rows: list[dict] = []
    for spec in args.scores:
        name, _, path = spec.partition("=")
        frame = pd.read_parquet(path)
        rows = evaluate(name, frame)
        if not rows:
            print(f"пропуск {name}: нет train/test разбивки")
            continue
        all_rows.extend(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'детектор':12}{'язык':6}{'n_test':>8}{'AUC':>8}{'F1':>8}{'ECE':>8}")
    print("-" * 50)
    for row in all_rows:
        print(
            f"{row['detector']:12}{row['language']:6}{row['n_test']:>8}"
            f"{row['roc_auc']:>8}{row['f1']:>8}{row['ece']:>8}"
        )
    print(f"\nтаблица: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
