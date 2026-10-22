"""Готовит уникальные документы RoFT и их сегменты с известным происхождением."""

from __future__ import annotations

import json
from pathlib import Path

from dw1.ingest.text_datasets import prepare_roft

ROOT = Path(__file__).resolve().parents[1]


def main():
    docs, segments, audit = prepare_roft(ROOT / "data/raw/roft/roft.csv")
    out = ROOT / "data/processed"
    out.mkdir(exist_ok=True)
    docs.to_parquet(out / "roft_documents.parquet", index=False)
    segments.to_parquet(out / "roft_segments.parquet", index=False)
    (out / "roft_preparation.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
