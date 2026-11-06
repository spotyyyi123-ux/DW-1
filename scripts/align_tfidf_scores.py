"""Строит TF-IDF-скоры на тех же документах/сплитах, что и референсный parquet.

Нужно для честного сравнения детекторов: берём строки референса (например, скоры
Fast-DetectGPT по письмам), находим текст по text_sha256 в email-бенчмарке и считаем
p_ai нашей моделью происхождения.

Запуск:
    python scripts/align_tfidf_scores.py --reference data/runs/sprint3/alina_email_scores.parquet \
        --benchmark data/sprint2/llm_benchmark_emails.jsonl \
        --out data/runs/sprint3/tfidf_email_scores_aligned.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--benchmark", default=str(ROOT / "data/sprint2/llm_benchmark_emails.jsonl"))
    ap.add_argument("--model", default=str(ROOT / "data/runs/sprint2/prov_model.joblib"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = joblib.load(args.model)
    by_hash: dict[str, str] = {}
    with open(args.benchmark, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            by_hash[row["text_sha256"]] = row["text"]

    ref = pd.read_parquet(args.reference)
    matched = [h for h in ref.text_sha256 if h in by_hash]
    if len(matched) != len(ref):
        raise SystemExit(f"нет текстов для {len(ref) - len(matched)} строк референса")

    texts = [by_hash[h] for h in ref.text_sha256]
    proba = model.predict_proba(texts)[:, 1]
    out = ref[["message_id", "text_sha256", "group_id", "dataset", "language", "label", "split"]].copy()
    out["criterion"] = proba
    out["n_tokens"] = [len(t.split()) for t in texts]
    out["truncated"] = False
    out["model_version"] = "tfidf-svd64-histgb-w1-v1"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"TF-IDF на {len(out)} строках референса -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
