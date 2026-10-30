"""Campaigns v2: exact-verified LSH components, HDBSCAN over imported embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import hdbscan
import numpy as np
from datasketch import MinHashLSH

from dw1.campaigns.cluster import build_minhash, normalize_text, shingles
from dw1.campaigns.template import skeleton

VERSION = "minhash-hdbscan-2.0"


def document_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_documents(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not rows or any(
        not isinstance(r.get("id"), str)
        or not r["id"]
        or not isinstance(r.get("text"), str)
        or not r["text"].strip()
        for r in rows
    ):
        raise ValueError("Expected non-empty JSONL records with string id and text")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate document IDs")
    return rows


def load_embeddings(path: Path, docs: list[dict]) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        ids = data["ids"].tolist()
        hashes = data["text_sha256"].tolist()
        vectors = np.asarray(data["vectors"], dtype=np.float64)
        metadata = json.loads(str(data["metadata"].item()))
    if len(ids) != len(set(ids)) or set(ids) != {d["id"] for d in docs}:
        raise ValueError("Embedding IDs do not match documents")
    if vectors.ndim != 2 or vectors.shape[0] != len(ids) or vectors.shape[1] < 2:
        raise ValueError("Invalid embedding shape")
    if len(hashes) != len(ids) or not np.isfinite(vectors).all():
        raise ValueError("Invalid embedding values/hashes")
    lookup = {doc_id: i for i, doc_id in enumerate(ids)}
    order = [lookup[d["id"]] for d in docs]
    if any(hashes[i] != document_hash(d["text"]) for i, d in zip(order, docs, strict=True)):
        raise ValueError("Stale embeddings: text hashes differ")
    vectors = vectors[order]
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError("Zero embeddings")
    return vectors / norms[:, None], metadata


def cluster_v2(
    docs: list[dict], vectors: np.ndarray, threshold=0.8, min_cluster_size=2, min_samples=1
) -> dict:
    if len({d["id"] for d in docs}) != len(docs):
        raise ValueError("Duplicate IDs")
    if vectors.ndim != 2 or len(vectors) != len(docs) or not np.isfinite(vectors).all():
        raise ValueError("Invalid vectors")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError("Zero embeddings")
    vectors = vectors / norms[:, None]
    parent = list(range(len(docs)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    sets: list[set[str]] = []
    for i, doc in enumerate(docs):
        text = normalize_text(doc["text"])
        words = shingles(text)
        sig = build_minhash(text)
        for j in sorted(lsh.query(sig)) if words else []:
            overlap = len(words & sets[j]) / max(1, len(words | sets[j]))
            if overlap >= threshold:
                parent[root(i)] = root(j)
        if words:
            lsh.insert(i, sig)
        sets.append(words)
    groups: dict[int, list[int]] = {}
    for i in range(len(docs)):
        groups.setdefault(root(i), []).append(i)
    units = list(groups.values())
    centroids = np.asarray([vectors[g].mean(axis=0) for g in units])
    if len(units) >= max(3, min_cluster_size + 1):
        # Normalized centroid Euclidean distance corresponds to cosine ordering.
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
        labels = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            core_dist_n_jobs=1,
        ).fit_predict(centroids)
    else:
        labels = np.full(len(units), -1)
    merged: dict[str, list[int]] = {}
    for i, (unit, label) in enumerate(zip(units, labels, strict=True)):
        key = f"semantic-{label}" if label >= 0 else f"lsh-{i}"
        merged.setdefault(key, []).extend(unit)
    clusters = []
    assignments = {}
    for group in sorted(merged.values(), key=lambda g: (-len(g), min(docs[i]["id"] for i in g))):
        members = sorted(docs[i]["id"] for i in group)
        cid = "campaign-" + hashlib.sha256("\0".join(members).encode()).hexdigest()[:12]
        for member in members:
            assignments[member] = cid
        clusters.append(
            {
                "cluster_id": cid,
                "size": len(group),
                "members": members,
                "representative_text": docs[group[0]]["text"],
                "template_skeleton": skeleton([docs[i]["text"] for i in group]),
                "model_version": VERSION,
            }
        )
    return {
        "model_version": VERSION,
        "clusters": clusters,
        "assignments": assignments,
        "parameters": {
            "threshold": threshold,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
        },
        "lsh_components": len(units),
        "noise_policy": "retain LSH unit; singleton otherwise",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--documents", type=Path, required=True)
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--min-cluster-size", type=int, default=2)
    ap.add_argument("--min-samples", type=int, default=1)
    args = ap.parse_args()
    docs = load_documents(args.documents)
    vectors, metadata = load_embeddings(args.embeddings, docs)
    result = cluster_v2(docs, vectors, args.threshold, args.min_cluster_size, args.min_samples)
    result.update(
        embedding_metadata=metadata,
        documents_sha256=document_hash(args.documents.read_text(encoding="utf-8-sig")),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "campaigns.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "campaign_index.json").write_text(
        json.dumps([c for c in result["clusters"] if c["size"] > 1], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    by_id = {d["id"]: d["text"] for d in docs}
    top = [c for c in result["clusters"] if c["size"] > 1][:50]
    review = [
        {
            "id": m,
            "predicted_cluster": c["cluster_id"],
            "text": by_id[m],
            "gold_group": "",
            "review": "",
            "annotator": "",
        }
        for c in top
        for m in c["members"]
    ]
    (args.out / "annotation.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in review) + "\n", encoding="utf-8"
    )
    print(f"{len(docs)} documents; {len(top)} review clusters; output: {args.out}")


if __name__ == "__main__":
    main()
