"""Run ONLY in an approved external environment on licensed public documents."""
import argparse
import json
from pathlib import Path

import numpy as np

from dw1.campaigns.v2 import document_hash, load_documents


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--documents", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--revision", required=True, help="Pinned model commit SHA")
    ap.add_argument("--public-data-approved", action="store_true", required=True)
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    docs = load_documents(args.documents)
    model = SentenceTransformer(args.model, revision=args.revision)
    vectors = model.encode([d["text"] for d in docs], normalize_embeddings=True,
                           show_progress_bar=True, batch_size=32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, ids=np.asarray([d["id"] for d in docs]), vectors=vectors,
                        text_sha256=np.asarray([document_hash(d["text"]) for d in docs]),
                        metadata=np.asarray(json.dumps({"model": args.model,
                            "revision": args.revision, "normalized": True,
                            "scope": "approved public data"})))


if __name__ == "__main__":
    main()
