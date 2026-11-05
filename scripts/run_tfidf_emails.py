"""Считает TF-IDF-модель происхождения на email-бенчмарке (для сравнения детекторов).

Вход : data/sprint2/llm_benchmark_emails.jsonl
Выход: data/runs/sprint2/tfidf_email_scores.parquet (criterion = p_ai)

Запуск: python scripts/run_tfidf_emails.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    import joblib
    import pandas as pd

    model_path = ROOT / "data" / "runs" / "sprint2" / "prov_model.joblib"
    if not model_path.exists():
        raise SystemExit("нет модели происхождения: запустите scripts/train_prov_model.py")
    model = joblib.load(model_path)

    rows = []
    with open(ROOT / "data" / "sprint2" / "llm_benchmark_emails.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            proba = float(model.predict_proba([row["text"]])[0][1])
            rows.append(
                {
                    "message_id": row["message_id"],
                    "text_sha256": row["text_sha256"],
                    "group_id": row["group_id"],
                    "dataset": row["dataset"],
                    "language": row["language"],
                    "label": row["label"],
                    "split": row["split"],
                    "criterion": proba,
                    "n_tokens": len(row["text"].split()),
                    "truncated": False,
                    "model_version": "tfidf-svd64-histgb-w1-v1",
                }
            )
    frame = pd.DataFrame(rows)
    out = ROOT / "data" / "runs" / "sprint2" / "tfidf_email_scores.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    print(f"скоров TF-IDF: {len(frame)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
