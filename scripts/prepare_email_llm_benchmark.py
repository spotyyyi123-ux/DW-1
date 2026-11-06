"""Готовит benchmark-jsonl из email-subset для прогона LLM-детектором.

Вход : data/sprint2/bench_licensed (или любой bench с labels.csv)
Выход: data/sprint2/llm_benchmark_emails.jsonl (текст = body, без заголовков)

Разметка: provenance (ai_generated=1, human=0); split по хэшу тела (70/20/10).

Запуск: python scripts/prepare_email_llm_benchmark.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "sprint2" / "bench_licensed"
OUT = ROOT / "data" / "sprint2" / "llm_benchmark_emails.jsonl"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").strip().encode("utf-8")).hexdigest()


def split_for(group_id: str) -> str:
    bucket = int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16) % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "validation"
    return "test"


def body_of(eml_path: Path) -> str:
    from dw1.ingest.mime import parse_eml

    return parse_eml(eml_path)["parts"]["body_text"]["text"].strip()


def main() -> int:
    rows = []
    with open(BENCH / "labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["provenance"] not in {"ai_generated", "human"}:
                raise ValueError("Неизвестная разметка происхождения письма")
            if row["source"] not in {"llmgen", "nazario"}:
                raise ValueError("Для Colab разрешены только LLMGen и Nazario")
            path = BENCH / row["file"]
            text = body_of(path)
            if len(text) < 40:
                continue
            group_id = text_hash(text)
            rows.append(
                {
                    "message_id": path.stem,
                    "text_sha256": text_hash(text),
                    "group_id": group_id,
                    "dataset": "emails",
                    "language": "en",
                    "label": 1 if row["provenance"] == "ai_generated" else 0,
                    "split": split_for(group_id),
                    "text": text,
                }
            )

    # Одинаковые тела писем получают один split и один скор для всех копий.
    unique = {}
    for item in rows:
        key = item["text_sha256"]
        if key in unique and unique[key]["label"] != item["label"]:
            raise ValueError("Одинаковый текст имеет противоречивую разметку")
        unique.setdefault(key, item)
    print(f"Повторных тел исключено: {len(rows) - len(unique)}")
    rows = list(unique.values())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    counts: dict[str, int] = {}
    for r in rows:
        counts[f"{r['split']}/{r['label']}"] = counts.get(f"{r['split']}/{r['label']}", 0) + 1
    print(f"писем: {len(rows)} -> {OUT}")
    print("сплиты:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
