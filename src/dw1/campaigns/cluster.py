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
import os
import re
from collections.abc import Iterable
from pathlib import Path

from datasketch import MinHash, MinHashLSH

MODEL_VERSION = "minhash-lsh-0.1"

DEFAULT_INDEX = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "runs"
    / "sprint1"
    / "campaign_index.json"
)

_STOP_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")

_PREPARED: list[dict] | None = None
_INDEX_MTIME: float = 0.0


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


def _index_path() -> Path:
    return Path(os.environ.get("DW1_CAMPAIGN_INDEX", str(DEFAULT_INDEX)))


def _prepared_index() -> list[dict]:
    """Индекс кампаний: cluster_id, размер, репрезентативный текст + шинглы."""
    global _PREPARED, _INDEX_MTIME
    path = _index_path()
    if not path.exists():
        return []
    mtime = path.stat().st_mtime
    if _PREPARED is None or mtime != _INDEX_MTIME:
        raw = json.loads(path.read_text(encoding="utf-8"))
        _PREPARED = [
            {
                **entry,
                "shingles": shingles(normalize_text(entry.get("representative_text", ""))),
            }
            for entry in raw
        ]
        _INDEX_MTIME = mtime
    return _PREPARED


def match_campaign(text: str, threshold: float = 0.8) -> dict | None:
    """Ищет письмо в сохранённых кампаниях по Jaccard с репрезентативным текстом."""
    text = normalize_text(text)
    if not text:
        return None
    probe = shingles(text)
    if not probe:
        return None
    best: dict | None = None
    for entry in _prepared_index():
        rep = entry.get("shingles") or set()
        if not rep:
            continue
        jaccard = len(probe & rep) / max(1, len(probe | rep))
        if jaccard >= threshold and (best is None or jaccard > best["jaccard"]):
            best = {
                "cluster_id": entry.get("cluster_id"),
                "member_count": int(entry.get("size", 1)),
                "jaccard": round(jaccard, 3),
                "template_skeleton": entry.get("template_skeleton"),
                "model_version": entry.get("model_version", MODEL_VERSION),
            }
    return best


def cluster(parts: dict) -> dict:
    """Контракт пайплайна: атрибуция кампании для одного письма.

    Если индекс кампаний (data/runs/sprint1/campaign_index.json) собран,
    письмо сопоставляется с ближайшим кластером по Jaccard.
    """
    subject = ""
    body = ""
    if isinstance(parts, dict):
        subject = str((parts.get("subject") or {}).get("text", ""))
        body = str((parts.get("body_text") or {}).get("text", ""))
    match = match_campaign(f"{subject} {body}")
    if match:
        return {
            "cluster_id": match["cluster_id"],
            "template_skeleton": match.get("template_skeleton"),
            "member_count": match["member_count"],
            "jaccard": match["jaccard"],
            "model_version": match.get("model_version", MODEL_VERSION),
        }
    return {
        "cluster_id": None,
        "template_skeleton": None,
        "member_count": 1,
        "model_version": MODEL_VERSION,
    }


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
                "representative_text": texts[members[0]["id"]],
            }
        )
    clusters.sort(key=lambda c: -c["size"])
    return clusters


def _load_documents(path: str) -> list[tuple[str, str]]:
    """Читает письма из parquet/jsonl: ищет колонки body_text/subject или text/body."""
    p = Path(path)
    if p.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(p)
    elif p.suffix in (".jsonl", ".ndjson"):
        import pandas as pd

        df = pd.read_json(p, lines=True)
    else:
        raise SystemExit(f"поддерживаются только .parquet/.jsonl, получено: {p.suffix}")

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
    ap = argparse.ArgumentParser(
        prog="dw1.campaigns.cluster",
        description="Кластеризация писем в кампании",
    )
    ap.add_argument("--data", required=True, help="путь к parquet/jsonl с письмами")
    ap.add_argument("--limit", type=int, default=10, help="сколько топ-кластеров показать")
    ap.add_argument("--threshold", type=float, default=0.8, help="порог Jaccard для LSH")
    ap.add_argument("--out", default=None, help="сохранить все кластеры в JSON")
    ap.add_argument(
        "--index",
        default=None,
        help="куда сохранить индекс кампаний (по умолчанию рядом с --out)",
    )
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

        index_path = (
            Path(args.index)
            if args.index
            else out_path.parent / "campaign_index.json"
        )
        index = [
            {
                "cluster_id": f"campaign-{i}",
                "size": c["size"],
                "representative_text": c["representative_text"],
            }
            for i, c in enumerate(clusters)
        ]
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"индекс кампаний сохранён в {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
