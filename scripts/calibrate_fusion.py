"""Калибровка fusion: isotonic-регрессия по двум осям, ECE до/после (Sprint 2, задача 4).

Вход : data/sprint2/bench (labels.csv + *.eml)
Выход: data/runs/sprint2/calibration_metrics.json (ECE/Brier/AUC + пороги изотоники)

Запуск: python scripts/calibrate_fusion.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from dw1.pipeline import analyze  # noqa: E402

BENCH = ROOT / "data" / "sprint2" / "bench"
OUT = ROOT / "data" / "runs" / "sprint2"
SEED = 42
BINS = 10
TRAIN_SHARE = 0.7


def ece(y: np.ndarray, p: np.ndarray, bins: int = BINS) -> float:
    """Expected Calibration Error (10 равных бинов)."""
    total = len(y)
    if total == 0:
        return 0.0
    score = 0.0
    for lo in range(bins):
        lo_edge, hi_edge = lo / bins, (lo + 1) / bins
        if lo == bins - 1:
            mask = (p >= lo_edge) & (p <= 1.0)
        else:
            mask = (p >= lo_edge) & (p < hi_edge)
        if not mask.any():
            continue
        score += mask.sum() / total * abs(y[mask].mean() - p[mask].mean())
    return float(score)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.clip(p, 0.0, 1.0) - y) ** 2))


def load_rows() -> list[dict]:
    rows: list[dict] = []
    with open(BENCH / "labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def collect(rows: list[dict]) -> list[tuple]:
    data: list[tuple] = []
    for row in rows:
        ev = analyze(BENCH / row["file"])
        prov = ev.fusion.get("ai_provenance") or {}
        intent = ev.fusion.get("intent") or {}
        prov_raw = float((prov.get("components") or {}).get("ai", 0.0))
        int_raw = float((intent.get("components") or {}).get("malicious", 0.0))
        y_prov = 1 if row["provenance"] == "ai_generated" else 0
        y_int = 1 if row["intent"] in ("spam", "phishing") else 0
        data.append((row["file"], prov_raw, int_raw, y_prov, y_int))
    return data


def calibrate_axis(
    train: list[tuple], test: list[tuple], raw_idx: int, label_idx: int
) -> dict:
    x_tr = np.array([d[raw_idx] for d in train])
    y_tr = np.array([d[label_idx] for d in train])
    x_te = np.array([d[raw_idx] for d in test])
    y_te = np.array([d[label_idx] for d in test])

    iso = IsotonicRegression(out_of_bounds="clip").fit(x_tr, y_tr)
    p_before = np.clip(x_te, 0.0, 1.0)
    p_after = np.clip(iso.predict(x_te), 0.0, 1.0)

    has_both = len(set(y_te.tolist())) > 1
    return {
        "n_train": len(x_tr),
        "n_test": len(x_te),
        "n_pos_train": int(y_tr.sum()),
        "n_pos_test": int(y_te.sum()),
        "ece_before": round(ece(y_te, p_before), 4),
        "ece_after": round(ece(y_te, p_after), 4),
        "brier_before": round(brier(y_te, p_before), 4),
        "brier_after": round(brier(y_te, p_after), 4),
        "auc_before": round(float(roc_auc_score(y_te, p_before)), 4) if has_both else None,
        "auc_after": round(float(roc_auc_score(y_te, p_after)), 4) if has_both else None,
        "calibrator_x": [float(v) for v in iso.X_thresholds_],
        "calibrator_y": [float(v) for v in iso.y_thresholds_],
    }


def main() -> int:
    rows = load_rows()
    data = collect(rows)
    rng = np.random.RandomState(SEED)
    order = np.arange(len(data))
    rng.shuffle(order)
    split = int(len(data) * TRAIN_SHARE)
    train = [data[i] for i in order[:split]]
    test = [data[i] for i in order[split:]]

    result = {
        "bench": str(BENCH),
        "n_total": len(data),
        "n_train": len(train),
        "n_test": len(test),
        "train_share": TRAIN_SHARE,
        "seed": SEED,
        "ece_bins": BINS,
        "axes": {
            "ai_provenance": calibrate_axis(train, test, 1, 3),
            "intent": calibrate_axis(train, test, 2, 4),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "calibration_metrics.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    for axis, m in result["axes"].items():
        print(
            f"{axis:14} ECE {m['ece_before']:.3f} -> {m['ece_after']:.3f} | "
            f"Brier {m['brier_before']:.3f} -> {m['brier_after']:.3f} | "
            f"AUC {m['auc_before']} -> {m['auc_after']} | n_test={m['n_test']}"
        )
    print(f"метрики: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
