"""Собирает входные данные для оценки кампаний (Sprint 3, задача 3).

Из сохранённого разбиения кампаний (data/runs/sprint1/clusters.json) и исходных текстов
seven-phishing строит пакет:
  data/runs/sprint3/campaigns/inputs/{campaigns.json, documents.jsonl, provenance.json}

Разрешение на внешний экспорт для BCubed-оценки дано владельцем данных (тимлид)
2026-11-06 — фиксируется в provenance.json.

Запуск: python scripts/build_campaign_review_inputs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLUSTERS = ROOT / "data" / "runs" / "sprint1" / "clusters.json"
CORPUS = ROOT / "data" / "raw" / "seven-phishing" / "test.parquet"
OUT = ROOT / "data" / "runs" / "sprint3" / "campaigns" / "inputs"


def disjoint_partition(clusters: list[dict]) -> list[dict]:
    """Сохранённое разбиение частично пересекается (кластеры строились по совпадению
    с репрезентативными текстами). Для оценки нужен непересекающийся раздел: жадно
    по размеру кластера, каждый документ попадает только в первый (самый крупный) кластер."""
    result = []
    seen: set[str] = set()
    ordered = sorted(
        clusters,
        key=lambda c: (-int(c.get("size", len(c.get("members", [])))), str(c.get("cluster_id"))),
    )
    for i, cluster in enumerate(ordered):
        members = [str(m["id"]) if isinstance(m, dict) else str(m) for m in cluster["members"]]
        fresh = [m for m in members if m not in seen]
        if len(fresh) < 2:
            continue
        seen.update(fresh)
        result.append(
            {
                "cluster_id": f"campaign-{i}",
                "size": len(fresh),
                "members": [{"id": m} for m in fresh],
                "representative_text": cluster.get("representative_text", ""),
            }
        )
    return result


def main() -> int:
    clusters = json.loads(CLUSTERS.read_text(encoding="utf-8"))
    clusters = disjoint_partition(clusters)
    frame = pd.read_parquet(CORPUS)

    documents = []
    for idx, row in frame.iterrows():
        subject = str(row.get("subject") or "").strip()
        body = str(row.get("text") or "").strip()
        text = (subject + "\n\n" + body).strip() if subject else body
        if not text:
            continue
        documents.append({"id": str(idx), "text": text})

    members = {str(m["id"]) for c in clusters for m in c["members"]}
    have = {d["id"] for d in documents}
    missing = members - have
    if missing:
        raise SystemExit(f"нет текстов для {len(missing)} участников кластеров")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "campaigns.json").write_text(
        json.dumps(clusters, ensure_ascii=False), encoding="utf-8"
    )
    with open(OUT / "documents.jsonl", "w", encoding="utf-8") as f:
        for doc in documents:
            # ensure_ascii=True: load_documents() использует splitlines(), который режет
            # строку на \u0085/\u2028/\u2029 внутри текста — экранирование обязательно.
            f.write(json.dumps(doc, ensure_ascii=True) + "\n")
    provenance = {
        "dataset": "seven-phishing (объединённый публичный корпус фишинга)",
        "source": "https://monarch.ut.ee/ (seven-phishing); копия на D:/dw1/data/raw",
        "license": "owner-approved: использование для внешней BCubed-оценки разрешено "
        "владельцем данных (тимлид Сергей) 2026-11-06; исходная лицензия корпуса — «other»",
        "approved_by": "Сергей (тимлид, владелец данных)",
        "external_allowed": True,
        "approval_scope": "только embeddings для BCubed F1 и ручная разметка топ-50; "
        "сырые письма не публикуются",
    }
    (OUT / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"кластеров: {len(clusters)}, документов: {len(documents)} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
