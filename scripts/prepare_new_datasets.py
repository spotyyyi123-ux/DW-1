"""Дополняет прежнюю выборку LLMTrace, сохраняя исходный эксперимент."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from prepare_w1 import clean_exact, remove_near_leaks

from dw1.ingest.text_datasets import sample_llmtrace

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2000)
    args = parser.parse_args()
    old = pd.read_parquet(ROOT / "data/processed/w1_texts.parquet")
    new, sampling = sample_llmtrace(ROOT / "data/raw/llmtrace", args.cap)
    combined = pd.concat([old, new], ignore_index=True)
    before = len(combined)
    combined = clean_exact(combined).reset_index(drop=True)
    removed_exact = before - len(combined)
    combined, removed_near = remove_near_leaks(combined)
    combined["ai_provenance"] = combined.label.map({0: "human", 1: "ai"})
    for key in ["group_id", "fingerprint"]:
        if combined.groupby(key).split.nunique().max() > 1:
            raise ValueError(f"Между частями пересекаются группы: {key}")
    path = ROOT / "data/processed/w1_expanded_texts.parquet"
    combined.to_parquet(path, index=False)
    audit = dict(
        seed=42,
        input_base="data/processed/w1_texts.parquet",
        sampling=sampling,
        rows=len(combined),
        empty_short_duplicate_conflict_removed=removed_exact,
        cross_split_near_duplicates_removed=removed_near,
        counts=combined.groupby(["source", "language", "split", "ai_provenance"])
        .size()
        .rename("count")
        .reset_index()
        .to_dict("records"),
        limitations=[
            "LLMTrace разделён по topic_id, исходные части сохранены в source_split.",
            "У CoAT исходные группы неизвестны; поиск близких повторов приближённый.",
            "Наборы общих текстов не заменяют проверку на реальных письмах.",
            "Сравнение со старой моделью проводится на общих отложенных строках.",
        ],
    )
    (path.parent / "expanded_preparation.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Готово: {path}, строк: {len(combined)}", flush=True)


if __name__ == "__main__":
    main()
