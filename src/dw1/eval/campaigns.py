"""B-Cubed over a fully reviewed top-50 partition; no inferred human annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def bcubed(gold: list[str], predicted: list[str]) -> dict:
    if not gold or len(gold) != len(predicted) or any(not g for g in gold):
        raise ValueError("Nonempty aligned partitions required")
    joint = Counter(zip(gold, predicted, strict=True))
    truth = Counter(gold)
    pred = Counter(predicted)
    precision = sum(n * n / pred[p] for (g, p), n in joint.items()) / len(gold)
    recall = sum(n * n / truth[g] for (g, p), n in joint.items()) / len(gold)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
        "n": len(gold),
    }


def evaluate(
    result: dict, rows: list[dict], required_clusters: int = 50
) -> tuple[dict, list[dict]]:
    top = [c for c in result["clusters"] if c["size"] > 1][:required_clusters]
    if len(top) != required_clusters:
        raise ValueError(f"Need {required_clusters} nonsingleton clusters, got {len(top)}")
    expected = {m: c["cluster_id"] for c in top for m in c["members"]}
    if len({r["id"] for r in rows}) != len(rows) or {r["id"] for r in rows} != set(expected):
        raise ValueError("Annotation IDs must cover exactly all members of selected clusters")
    for row in rows:
        if row.get("predicted_cluster") != expected[row["id"]]:
            raise ValueError("Annotation belongs to another clustering run")
        if not str(row.get("gold_group", "")).strip() or not row.get("annotator"):
            raise ValueError("Manual gold_group and annotator are mandatory")
        if row.get("review") not in ("near-dup", "different"):
            raise ValueError("review must be near-dup or different")
    predictions = [
        {"id": r["id"], "gold_group": r["gold_group"], "predicted_cluster": expected[r["id"]]}
        for r in rows
    ]
    metrics = bcubed(
        [r["gold_group"] for r in predictions], [r["predicted_cluster"] for r in predictions]
    )
    return {
        "bcubed": metrics,
        "reviewed_clusters": len(top),
        "scope": "Induced partition on all members of top reviewed clusters; global gold IDs",
        "model_version": result["model_version"],
    }, predictions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaigns", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = json.loads(args.campaigns.read_text(encoding="utf-8-sig"))
    rows = [
        json.loads(line)
        for line in args.annotations.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    metrics, predictions = evaluate(result, rows)
    metrics["campaigns_sha256"] = hashlib.sha256(args.campaigns.read_bytes()).hexdigest()
    metrics["annotations_sha256"] = hashlib.sha256(args.annotations.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(predictions).to_parquet(args.out / "predictions.parquet", index=False)
    (args.out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
