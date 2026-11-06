"""Прокси-разметка пакета review (2 «проверяющих» + финальный gold) для локального BCubed.

НЕ заменяет ручную разметку: это автоматическая прокси-разметка по TF-IDF-косинусу,
чтобы прогнать проверяемый расчёт BCubed и получить ориентир. В отчёте помечается явно.

Правило: группы = связные компоненты графа «cosine >= threshold» на 600 документах.
Reviewer A: 0.90; Reviewer B: 0.75; final gold: 0.85.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]


def components(sim: np.ndarray, threshold: float) -> dict[str, str]:
    n = sim.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    groups: dict[str, str] = {}
    for i in range(n):
        groups[str(i)] = f"g{find(i)}"
    return groups


def write_annotation(template: list[dict], labels: dict[str, str], annotator: str, path: Path) -> None:
    rows = []
    for row in template:
        rows.append(
            {
                **row,
                "gold_group": labels[row["id"]],
                "annotator": annotator,
                "review": "proxy",
                "notes": "автоматическая прокси-разметка по TF-IDF cosine (не ручная)",
            }
        )
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=str(ROOT / "data/runs/sprint3/campaigns/review"))
    args = ap.parse_args()
    package = Path(args.package)

    docs = [
        json.loads(line)
        for line in (package / "documents.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    template = [
        json.loads(line)
        for line in (package / "annotation.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    texts = [d["text"] for d in docs]
    vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)
    matrix = vectorizer.fit_transform(texts)
    sim = (matrix @ matrix.T).toarray()
    np.fill_diagonal(sim, 0.0)

    def labels_for(threshold: float) -> dict[str, str]:
        groups = components(sim, threshold)
        return {doc["id"]: groups[str(i)] for i, doc in enumerate(docs)}

    write_annotation(template, labels_for(0.90), "proxy-cosine-90", package / "proxy_reviewer_a.jsonl")
    write_annotation(template, labels_for(0.75), "proxy-cosine-75", package / "proxy_reviewer_b.jsonl")
    write_annotation(template, labels_for(0.85), "proxy-cosine-85-gold", package / "proxy_gold.jsonl")
    print("прокси-разметка: A=0.90, B=0.75, gold=0.85 ->", package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
