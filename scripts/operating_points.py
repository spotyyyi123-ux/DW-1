"""Operating points: TPR/FPR/F1 по порогам калиброванных вероятностей (Sprint 2).

Использует сохранённую калибровку (isotonic пороги из calibration_metrics.json)
и прогон analyze() по бенчмарку. Показывает, какой порог даёт какой баланс TP/FP,
и предлагает рабочую точку.

Запуск: python scripts/operating_points.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dw1.models.fusion import _apply_isotonic  # noqa: E402
from dw1.pipeline import analyze  # noqa: E402

BENCH = ROOT / "data" / "sprint2" / "bench"
CAL = ROOT / "data" / "runs" / "sprint2" / "calibration_metrics.json"
OUT = ROOT / "data" / "runs" / "sprint2" / "operating_points.json"
THRESHOLDS = [round(x * 0.05, 2) for x in range(8, 19)]  # 0.40 .. 0.90


def load_bench() -> list[dict]:
    rows: list[dict] = []
    with open(BENCH / "labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def proba(components: dict, cal_axis: dict, key: str) -> float:
    raw = float(components.get(key, 0.0))
    return float(np.clip(_apply_isotonic(raw, cal_axis), 0.0, 1.0))


def sweep(p: np.ndarray, y: np.ndarray) -> list[dict]:
    out = []
    for t in THRESHOLDS:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        precision = tp / max(1, tp + fp)
        f1 = 2 * precision * tpr / max(1e-9, precision + tpr)
        out.append(
            {
                "threshold": t,
                "tpr": round(tpr, 3),
                "fpr": round(fpr, 3),
                "precision": round(precision, 3),
                "f1": round(f1, 3),
                "tp": tp,
                "fp": fp,
            }
        )
    return out


def best_f1(sweep_rows: list[dict]) -> dict:
    return max(sweep_rows, key=lambda r: (r["f1"], -r["fpr"]))


def pick_target(sweep_rows: list[dict], max_fpr: float) -> dict | None:
    eligible = [r for r in sweep_rows if r["fpr"] <= max_fpr]
    return max(eligible, key=lambda r: r["f1"]) if eligible else None


def main() -> int:
    calibration = json.loads(CAL.read_text(encoding="utf-8"))["axes"]
    rows = load_bench()
    probs: dict[str, list[float]] = {"ai_provenance": [], "intent": []}
    labels: dict[str, list[int]] = {"ai_provenance": [], "intent": []}

    for row in rows:
        ev = analyze(BENCH / row["file"])
        prov = (ev.fusion.get("ai_provenance") or {}).get("components") or {}
        intent = (ev.fusion.get("intent") or {}).get("components") or {}
        probs["ai_provenance"].append(proba(prov, calibration["ai_provenance"], "ai"))
        probs["intent"].append(proba(intent, calibration["intent"], "malicious"))
        labels["ai_provenance"].append(1 if row["provenance"] == "ai_generated" else 0)
        labels["intent"].append(1 if row["intent"] in ("spam", "phishing") else 0)

    report = {}
    print(f"{'ось':14}{'порог':8}{'TPR':>7}{'FPR':>7}{'F1':>7}")
    print("-" * 44)
    for axis in ("ai_provenance", "intent"):
        p = np.array(probs[axis])
        y = np.array(labels[axis])
        rows_t = sweep(p, y)
        report[axis] = {
            "sweep": rows_t,
            "best_f1": best_f1(rows_t),
            "fpr_le_1pct": pick_target(rows_t, 0.01),
            "fpr_le_5pct": pick_target(rows_t, 0.05),
        }
        for r in rows_t:
            print(f"{axis:14}{r['threshold']:<8}{r['tpr']:>7}{r['fpr']:>7}{r['f1']:>7}")
        print(f"{axis}: best F1 {best_f1(rows_t)} | FPR<=1%: {pick_target(rows_t, 0.01)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"operating points: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
