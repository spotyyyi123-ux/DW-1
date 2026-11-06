"""Run only in an approved external environment on licensed public documents."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import time
from pathlib import Path

import numpy as np

from dw1.campaigns.documents import document_hash, load_documents


def export(args):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        raise ValueError("--revision must be a pinned 40-character model commit SHA")
    docs = load_documents(args.documents)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig")) if args.manifest else None
    if manifest:
        if manifest["provenance"].get("external_allowed") is not True:
            raise ValueError("Corpus is not approved for external processing")
        hashes = {d["id"]: document_hash(d["text"]) for d in docs}
        if hashes != manifest["document_hashes"]:
            raise ValueError("Documents differ from the frozen review manifest")
    # These imports must remain inside the external execution path.
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(42)
    start = time.perf_counter()
    model = SentenceTransformer(args.model, revision=args.revision, device=args.device)
    vectors = model.encode(
        [d["text"] for d in docs],
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=args.batch_size,
        convert_to_numpy=True,
    )
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Model returned non-finite or zero vectors")
    metadata = {
        "model": args.model,
        "revision": args.revision,
        "normalized": True,
        "scope": "approved public data",
        "seed": 42,
        "max_seq_length": model.max_seq_length,
        "text_policy": "model truncation at max_seq_length; no text edits",
        "n_documents": len(docs),
        "elapsed_seconds": time.perf_counter() - start,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("sentence-transformers", "torch", "numpy")
        },
        "package_id": manifest["package_id"] if manifest else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as target:
        np.savez_compressed(
            target,
            ids=np.asarray([d["id"] for d in docs]),
            vectors=vectors,
            text_sha256=np.asarray([document_hash(d["text"]) for d in docs]),
            metadata=np.asarray(json.dumps(metadata)),
        )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--documents", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, help="Frozen top-50 manifest")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    ap.add_argument("--revision", required=True, help="Pinned model commit SHA")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--public-data-approved", action="store_true", required=True)
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error("--batch-size must be positive")
    export(args)


if __name__ == "__main__":
    main()
