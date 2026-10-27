"""Faithfulness фузии: comprehensiveness и sufficiency по элементам (Sprint 2, задача 4).

ERASER-подход на уровне элементов письма, без ручной разметки:
- comprehensiveness: убираем главный элемент → падение уверенности (должно падать);
- sufficiency: оставляем только главный элемент → насколько уверенность сохраняется.

Вход : data/runs/sprint1/demo/*.evidence.json
Выход: data/runs/sprint2/faithfulness.json + таблица в консоль.

Запуск: python scripts/faithfulness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dw1.models.fusion import INTENT_WEIGHTS, PROV_WEIGHTS, fuse  # noqa: E402

DEMO = ROOT / "data" / "runs" / "sprint1" / "demo"
OUT = ROOT / "data" / "runs" / "sprint2" / "faithfulness.json"


def _unknown() -> dict:
    return {"label": "unknown", "confidence": 0.0, "evidence": [], "model_version": "faithfulness"}


def _top_element(signals: dict, weights: dict, axis: str) -> str | None:
    best, best_score = None, 0.0
    for element, blob in signals.items():
        if not isinstance(blob, dict):
            continue
        sig = blob.get(axis) or {}
        if not isinstance(sig, dict):
            continue
        conf = float(sig.get("confidence") or 0.0)
        label = sig.get("label", "unknown")
        if label == "unknown" or conf <= 0:
            continue
        score = conf * weights.get(element, 0.0)
        if score > best_score:
            best, best_score = element, score
    return best


def _axis_conf(fused: dict, axis: str) -> float:
    return float((fused.get(axis) or {}).get("confidence") or 0.0)


def evaluate(signals: dict, axis: str, weights: dict) -> dict:
    full = _axis_conf(fuse(signals), axis)
    top = _top_element(signals, weights, axis)
    if top is None:
        return {"element": None, "full": round(full, 3), "without": round(full, 3), "only": round(full, 3)}

    without = dict(signals)
    without[top] = {**_unknown()}
    p_without = _axis_conf(fuse(without), axis)

    only = {k: {"ai_provenance": _unknown(), "intent": _unknown()} for k in signals}
    only[top] = signals[top]
    p_only = _axis_conf(fuse(only), axis)

    return {
        "element": top,
        "full": round(full, 3),
        "without": round(p_without, 3),
        "only": round(p_only, 3),
        "comprehensiveness": round(full - p_without, 3),
        "sufficiency_gap": round(full - p_only, 3),
    }


def main() -> int:
    paths = sorted(DEMO.glob("*.evidence.json"))
    if not paths:
        print("нет демо-карточек: запустите make_demo")
        return 1

    report = []
    print(f"{'письмо':22}{'ось':14}{'главный элемент':14}{'full':>6}{'без':>6}{'только':>8}{'comp':>6}{'suff':>6}")
    print("-" * 88)
    for path in paths:
        ev = json.loads(path.read_text(encoding="utf-8"))
        stem = path.name.replace(".evidence.json", "")
        row = {"file": stem}
        for axis, weights, name in (
            ("ai_provenance", PROV_WEIGHTS, "provenance"),
            ("intent", INTENT_WEIGHTS, "intent"),
        ):
            res = evaluate(ev.get("signals") or {}, axis, weights)
            row[name] = res
            print(
                f"{stem:22}{axis:14}{str(res['element']):14}{res['full']:>6}"
                f"{res['without']:>6}{res['only']:>8}"
                f"{res['comprehensiveness']:>6}{res['sufficiency_gap']:>6}"
            )
        report.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nfaithfulness сохранён: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
