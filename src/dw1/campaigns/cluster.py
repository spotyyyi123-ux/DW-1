# -*- coding: utf-8 -*-
"""Кампании (задача 4): группировка похожих писем в кампании через MinHash/LSH.

Вход: набор писем (subject + body_text). Выход: кластеры near-дубликатов.
Sprint 2: структурные шинглы DOM, pHash картинок, семантические эмбеддинги + HDBSCAN,
извлечение скелета шаблона (SPEC §6 W4).

CLI:
    python -m dw1.campaigns.cluster --data <parquet/jsonl> [--limit 10]
"""
from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path

from datasketch import MinHash, MinHashLSH

MODEL_VERSION = "minhash-lsh-0.1"

_STOP_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Нижний регистр, без пунктуации, схлопнутые пробелы — устойчивость к мелким правкам."""
    text = _STOP_RE.sub(" ", (text or "").lower())
    return _WS_RE.sub(" ", text).strip()


def shingles(text: str, k: int = 5) -> set[str]:
    """Шинглы по словам: последовательности из k слов."""
    words = text.split()
    if not words:
        return set()
    if len(words) < k:
        return {text}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def build_minhash(text: str, k: int = 5, num_perm: int = 128) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for s in shingles(text, k):
        mh.update(s.encode("utf-8"))
    return mh


def cluster_emails(
    documents: Iterable[tuple[str, str]],
    threshold: float = 0.8,
    num_perm: int = 128,
) -> list[dict]:
    """Группирует (doc_id, text) в кластеры похожих писем.

    Возвращает список кластеров (от большего к меньшему), в каждом — members с точным
    Jaccard-сходством и пара примеров.
    """
    minhashes: dict[str, MinHash] = {}
    texts: dict[str, str] = {}
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for doc_id, text in documents:
        mh = build_minhash(text, num_perm=num_perm)
        minhashes[doc_id] = mh
        texts[doc_id] = text
        lsh.insert(doc_id, mh)

    seen: set[str] = set()
    clusters: list[dict] = []
    for doc_id in minhashes:
        if doc_id in seen:
            continue
        near = lsh.query(minhashes[doc_id])
        if len(near) < 2:
            continue  # одиночки — не кампания
        seen.update(near)
        members = []
        for n in near:
            jaccard = minhashes[doc_id].jaccard(minhashes[n])
            members.append({"id": n, "jaccard": round(float(jaccard), 3)})
        members.sort(key=lambda m: -m["jaccard"])
        clusters.append(
            {
                "members": members,
                "size": len(members),
                "examples": [texts[members[0]["id"]][:160], texts[members[-1]["id"]][:160]],
            }
        )
    clusters.sort(key=lambda c: -c["size"])
    return clusters


def _load_documents(path: str) -> list[tuple[str, str]]:
    """Читает письма из parquet/jsonl: ищет колонки body_text/subject или text/body."""
    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
    elif path.suffix in (".jsonl", ".ndjson"):
        import pandas as pd

        df = pd.read_json(path, lines=True)
    else:
        raise SystemExit(f"поддерживаются только .parquet/.jsonl, получено: {path.suffix}")

    def pick(row, *cols):
        for c in cols:
            if c in row.index and isinstance(row[c], str):
                return row[c]
        return ""

    docs = []
    for i, row in df.iterrows():
        subject = pick(row, "subject", "Subject")
        body = pick(row, "body_text", "text", "body", "Body", "email")
        text = normalize_text(f"{subject} {body}")
        if text:
            docs.append((str(i), text))
    return docs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dw1.campaigns.cluster", description="Кластеризация писем в кампании")
    ap.add_argument("--data", required=True, help="путь к parquet/jsonl с письмами")
    ap.add_argument("--limit", type=int, default=10, help="сколько топ-кластеров показать")
    ap.add_argument("--threshold", type=float, default=0.8, help="порог Jaccard для LSH")
    ap.add_argument("--out", default=None, help="сохранить все кластеры в JSON")
    args = ap.parse_args(argv)

    docs = _load_documents(args.data)
    print(f"писем: {len(docs)}")
    clusters = cluster_emails(docs, threshold=args.threshold)
    in_clusters = sum(c["size"] for c in clusters)
    print(f"кластеров: {len(clusters)}, писем внутри кластеров: {in_clusters}")

    for c in clusters[: args.limit]:
        print(f"\n=== кластер: {c['size']} писем ===")
        for m in c["members"][:5]:
            print(f"  - id={m['id']} jaccard={m['jaccard']}")
        print(f"  пример A: {c['examples'][0]}")
        print(f"  пример B: {c['examples'][1]}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(clusters, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nвсе кластеры сохранены в {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
