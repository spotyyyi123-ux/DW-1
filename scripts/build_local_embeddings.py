"""Строит локальные эмбеддинги для пакета разметки (fallback без sentence-transformers).

TF-IDF → TruncatedSVD(256), L2-нормировка; формат NPZ совместим с dw1.eval.campaign_review
(ids, vectors, text_sha256, metadata).

Запуск: python scripts/build_local_embeddings.py --package data/runs/sprint3/campaigns/review
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]


def doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=str(ROOT / "data/runs/sprint3/campaigns/review"))
    ap.add_argument("--out", default="")
    ap.add_argument("--dim", type=int, default=256)
    args = ap.parse_args()
    package = Path(args.package)
    out = Path(args.out) if args.out else package / "embeddings.npz"

    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    docs = [
        json.loads(line)
        for line in (package / "documents.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    texts = [d["text"] for d in docs]

    vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)
    matrix = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=min(args.dim, matrix.shape[1] - 1), random_state=42)
    vectors = normalize(svd.fit_transform(matrix)).astype(np.float32)
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise SystemExit("некорректные векторы")

    metadata = {
        "model": "tfidf-svd-256",
        "revision": "sklearn-tfidf-svd-256",
        "normalized": True,
        "scope": "local fallback (public licensed data); sentence-transformers не использован",
        "n_documents": len(docs),
        "package_id": manifest["package_id"],
    }
    np.savez_compressed(
        out,
        ids=np.asarray([d["id"] for d in docs]),
        vectors=vectors,
        text_sha256=np.asarray([doc_hash(t) for t in texts]),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    print(f"эмбеддинги {vectors.shape} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
