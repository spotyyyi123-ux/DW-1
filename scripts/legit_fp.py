"""FP на легитимной почте (Sprint 2, задача 4): FPR + таксономия ошибок.

Вход : data/sprint2/bench (легит-письма Enron — enron-ham_*.eml)
Выход: data/runs/sprint2/legit_fp.json

Запуск: python scripts/legit_fp.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dw1.pipeline import analyze  # noqa: E402

BENCH = ROOT / "data" / "sprint2" / "bench"
OUT = ROOT / "data" / "runs" / "sprint2" / "legit_fp.json"
AI_LABELS = ("ai_generated", "ai_polished")
MALICIOUS_LABELS = ("phishing", "spam")


def main() -> int:
    rows = []
    with open(BENCH / "labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["source"] == "enron_spam" and row["intent"] == "legitimate":
                rows.append(row)
    if not rows:
        print("нет легитимных писем в бенчмарке")
        return 1

    fp_style = fp_intent = fp_both = clean = 0
    examples = []
    for row in rows:
        ev = analyze(BENCH / row["file"])
        prov = (ev.fusion.get("ai_provenance") or {}).get("label")
        intent = (ev.fusion.get("intent") or {}).get("label")
        style_fp = prov in AI_LABELS
        intent_fp = intent in MALICIOUS_LABELS
        fp_style += int(style_fp)
        fp_intent += int(intent_fp)
        fp_both += int(style_fp and intent_fp)
        clean += int(not style_fp and not intent_fp)
        if style_fp or intent_fp:
            examples.append(
                {
                    "file": row["file"],
                    "provenance": prov,
                    "intent": intent,
                    "prov_conf": (ev.fusion.get("ai_provenance") or {}).get("confidence"),
                    "intent_conf": (ev.fusion.get("intent") or {}).get("confidence"),
                }
            )

    n = len(rows)
    report = {
        "n_legit": n,
        "clean": clean,
        "fp_any": n - clean,
        "fp_style": fp_style,
        "fp_intent": fp_intent,
        "fp_both": fp_both,
        "fpr_any": round((n - clean) / n, 3),
        "fpr_style": round(fp_style / n, 3),
        "fpr_intent": round(fp_intent / n, 3),
        "taxonomy": {
            "style_only": fp_style - fp_both,
            "intent_only": fp_intent - fp_both,
            "both": fp_both,
        },
        "examples": examples[:10],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"легит-почта: {n} писем")
    print(f"чистых: {clean} | любой FP: {n - clean} (FPR {report['fpr_any']:.1%})")
    print(f"FP-стиль: {fp_style} | FP-интент: {fp_intent} | оба: {fp_both}")
    print(f"таксономия: {report['taxonomy']}")
    print(f"отчёт: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
