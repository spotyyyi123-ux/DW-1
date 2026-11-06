"""Validated human review + external embeddings -> paired B-Cubed report."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from dw1.campaigns.documents import document_hash, load_documents
from dw1.campaigns.review import canonical_hash
from dw1.campaigns.v2 import cluster_v2, load_embeddings
from dw1.eval.campaigns import bcubed, evaluate


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def validate_rows(rows: list[dict], manifest: dict, campaigns: dict) -> None:
    evaluate(campaigns, rows, required_clusters=manifest["top_n"])
    for row in rows:
        if row.get("package_id") != manifest["package_id"]:
            raise ValueError("Review from another package")
        expected = manifest["document_hashes"][row["id"]]
        if row.get("text_sha256") != expected or document_hash(row.get("text", "")) != expected:
            raise ValueError("Review text/hash mismatch")
        if not isinstance(row["gold_group"], str) or row["gold_group"] != row["gold_group"].strip():
            raise ValueError("gold_group must be a trimmed nonempty string")


def compare_reviewers(reviewers: list[list[dict]]) -> dict:
    if not 2 <= len(reviewers) <= 3:
        raise ValueError("Need 2 or 3 independent human reviewer files")
    names = []
    partitions = []
    universe = {r["id"] for r in reviewers[0]}
    for rows in reviewers:
        authors = {r["annotator"].strip() for r in rows}
        if len(authors) != 1 or not next(iter(authors)):
            raise ValueError("Each reviewer file must have one nonempty annotator")
        names.append(next(iter(authors)))
        if {r["id"] for r in rows} != universe:
            raise ValueError("Reviewer populations differ")
        groups: dict[str, set[str]] = defaultdict(set)
        for r in rows:
            groups[r["gold_group"]].add(r["id"])
        partitions.append({r["id"]: frozenset(groups[r["gold_group"]]) for r in rows})
    if len(set(names)) != len(names):
        raise ValueError("Independent reviewers must have distinct annotator IDs")
    disagreement = sorted(
        i for i in universe if any(p[i] != partitions[0][i] for p in partitions[1:])
    )
    return {
        "reviewers": names,
        "disagreement_ids": disagreement,
        "n_disagreements": len(disagreement),
        "method": "compare equivalence classes, invariant to gold label renaming",
    }


def verify_package(manifest: dict, campaigns: dict, docs: list[dict]) -> None:
    unsigned = {k: v for k, v in manifest.items() if k != "package_id"}
    if canonical_hash(unsigned) != manifest["package_id"]:
        raise ValueError("Manifest has changed")
    if manifest["top_n"] != 50 or manifest["provenance"].get("external_allowed") is not True:
        raise ValueError("Need an approved real top-50 review package")
    if canonical_hash(campaigns["clusters"]) != manifest["partition_sha256"]:
        raise ValueError("Frozen partition has changed")
    if campaigns.get("review_package_id") != manifest["package_id"]:
        raise ValueError("Campaign package ID mismatch")
    if {d["id"]: document_hash(d["text"]) for d in docs} != manifest["document_hashes"]:
        raise ValueError("Frozen document population has changed")


def report_text(metrics: dict) -> str:
    baseline, current = metrics["baseline_bcubed"], metrics["bcubed"]
    return f"""# Sprint 3 — кампании: BCubed F1

Отчёт создан из сохранённых predictions.parquet и metrics.json.
Набор: {metrics["dataset"]}. Проверяющие: {", ".join(metrics["review"]["reviewers"])}.

| Разбиение | Precision | Recall | BCubed F1 |
|---|---:|---:|---:|
| Исходные топ-50 | {baseline["precision"]:.6f} | {baseline["recall"]:.6f} | {baseline["f1"]:.6f} |
| LSH + HDBSCAN | {current["precision"]:.6f} | {current["recall"]:.6f} | {current["f1"]:.6f} |

Писем: {current["n"]}. Исходных кластеров: 50.
Модель эмбеддингов: {metrics["embedding_metadata"]["model"]}.
Ревизия: {metrics["embedding_metadata"]["revision"]}.
Расхождений между проверяющими: {metrics["review"]["n_disagreements"]}.
Согласовал: {metrics["adjudicated_by"] or "разбиения проверяющих совпали"}.

Выборка фиксируется до просмотра разметки. Истинные группы глобальны для всех писем.
Семантическая кластеризация повторяется только внутри отобранной выборки; это оценка
на участниках исходных топ-50, не измерение качества на всём корпусе. Шумовые одиночки
участвуют в метрике. Параметры не подбираются по gold-разметке.

Хеш пакета: {metrics["package_id"]}.
Хеш predictions.parquet: {metrics["predictions_sha256"]}.
Манифест, NPZ, исходная разметка и JSON-предсказания сохраняются рядом с метриками;
пути и хеши входов записаны в input_files. Тексты писем не добавлять в публичный Git.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True, help="Final human gold")
    ap.add_argument("--reviewers", nargs="+", type=Path, required=True)
    ap.add_argument("--adjudicated-by", default="")
    ap.add_argument("--out", type=Path, default=Path("data/runs/sprint3/campaigns/eval"))
    ap.add_argument("--report", type=Path, default=Path("docs/notes/sprint3-campaigns.md"))
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        ap.error("--out must be empty: preserve the previous evaluation")
    manifest = json.loads((args.package / "manifest.json").read_text(encoding="utf-8"))
    campaigns = json.loads((args.package / "campaigns.json").read_text(encoding="utf-8"))
    docs = load_documents(args.package / "documents.jsonl")
    verify_package(manifest, campaigns, docs)
    final = read_rows(args.annotations)
    validate_rows(final, manifest, campaigns)
    reviewers = [read_rows(p) for p in args.reviewers]
    for rows in reviewers:
        validate_rows(rows, manifest, campaigns)
    agreement = compare_reviewers(reviewers)
    # Final gold may also change unanimous labels; require explicit human adjudication then.
    with_final = compare_reviewers(
        [reviewers[0], [{**r, "annotator": "__final_comparison__"} for r in final]]
    )
    if (
        agreement["n_disagreements"] or with_final["n_disagreements"]
    ) and not args.adjudicated_by.strip():
        raise ValueError("Human adjudication required for disagreements or changed final groups")
    vectors, metadata = load_embeddings(args.embeddings, docs)
    if metadata.get("package_id") != manifest["package_id"]:
        raise ValueError("Embeddings are not linked to this frozen package")
    if not metadata.get("model") or not metadata.get("revision"):
        raise ValueError("Embedding model and revision are required")
    v2 = cluster_v2(docs, vectors)  # Fixed defaults, never tuned using gold.
    expected = {m: c["cluster_id"] for c in campaigns["clusters"] for m in c["members"]}
    predictions = [
        {
            "id": r["id"],
            "gold_group": r["gold_group"],
            "baseline_cluster": expected[r["id"]],
            "predicted_cluster": v2["assignments"][r["id"]],
        }
        for r in final
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    path = args.out / "predictions.parquet"
    pd.DataFrame(predictions).to_parquet(path, index=False)
    # Report numbers are reconstructed from the saved prediction artifact.
    saved = pd.read_parquet(path)
    metrics = {
        "status": "measured",
        "dataset": manifest["provenance"]["dataset"],
        "package_id": manifest["package_id"],
        "embedding_metadata": metadata,
        "model_version": v2["model_version"],
        "parameters": v2["parameters"],
        "review": agreement,
        "adjudicated_by": args.adjudicated_by.strip(),
        "bcubed": bcubed(saved.gold_group.tolist(), saved.predicted_cluster.tolist()),
        "baseline_bcubed": bcubed(saved.gold_group.tolist(), saved.baseline_cluster.tolist()),
        "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scope": "HDBSCAN rerun on frozen original top-50 message population",
        "input_files": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                args.package / "manifest.json",
                args.package / "campaigns.json",
                args.package / "documents.jsonl",
                args.embeddings,
                args.annotations,
                *args.reviewers,
            ]
        },
    }
    (args.out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "campaigns_v2.json").write_text(
        json.dumps(v2, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    inputs = args.out / "inputs"
    inputs.mkdir()
    for p, name in [
        (args.package / "manifest.json", "manifest.json"),
        (args.package / "campaigns.json", "campaigns.json"),
        (args.package / "documents.jsonl", "documents.jsonl"),
        (args.embeddings, "embeddings.npz"),
        (args.annotations, "annotations-final.jsonl"),
    ]:
        shutil.copyfile(p, inputs / name)
    for i, p in enumerate(args.reviewers):
        shutil.copyfile(p, inputs / f"reviewer-{i + 1}.jsonl")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text(metrics), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
