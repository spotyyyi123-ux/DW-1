"""Публичная выборка HC3/CoAT для внешнего запуска; приватные письма не принимает."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from dw1.models.llm_detectors import text_hash
from prepare_w1 import clean_exact, load_hc3, remove_near_leaks

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=50)
    args = parser.parse_args()
    if args.per_class < 16:
        parser.error("Для пилотной калибровки нужно хотя бы 16 примеров каждого класса")
    frames = [load_hc3(ROOT / "data/raw/hc3-en/all.jsonl")]
    for original in ["train", "validation"]:
        df = pd.read_parquet(ROOT / "data/raw/coat/binary" / f"{original}-00000-of-00001.parquet")
        if not set(df.label) <= {0, 1}:
            raise ValueError("CoAT содержит скрытые метки")
        df["message_id"] = "coat-" + df.id.astype(str)
        df["group_id"] = df.message_id
        df["split"] = df.group_id.map(
            lambda g: ["test", "validation", *(["train"] * 8)][
                int(hashlib.sha256(g.encode()).hexdigest()[:8], 16) % 10
            ]
        )
        df["source"] = "coat"
        df["language"] = "ru"
        frames.append(df.drop(columns="id"))
    frame = clean_exact(pd.concat(frames, ignore_index=True))
    pieces = []
    for _, part in frame.groupby(["source", "label", "split"]):
        pieces.append(part.sample(min(len(part), args.per_class + 20), random_state=42))
    frame, removed = remove_near_leaks(pd.concat(pieces).reset_index(drop=True))
    frame = frame.groupby(["source", "label", "split"], group_keys=False).head(args.per_class)
    frame = frame.rename(columns={"source": "dataset"})
    frame["text"] = frame.text.map(lambda t: t.replace("\r\n", "\n").strip())
    frame["text_sha256"] = frame.text.map(text_hash)
    if frame.text_sha256.duplicated().any() or frame.groupby("group_id").split.nunique().max() != 1:
        raise ValueError("Не пройдена проверка разделения данных")
    columns = [
        "message_id",
        "text_sha256",
        "group_id",
        "dataset",
        "language",
        "label",
        "split",
        "text",
    ]
    out = ROOT / "data/sprint2"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "llm_benchmark.jsonl"
    frame[columns].to_json(path, orient="records", lines=True, force_ascii=False)
    metadata = dict(
        seed=42,
        public_data_only=True,
        benchmark_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        n_examples=len(frame),
        near_duplicates_removed=removed,
        counts=frame.groupby(["dataset", "language", "split", "label"])
        .size()
        .rename("n")
        .reset_index()
        .to_dict("records"),
        limitations=[
            "У CoAT неизвестны исходные группы вариантов.",
            "Поиск близких повторов приближённый.",
            "Публичный общий текст, не бенчмарк реальной почты.",
        ],
    )
    (out / "llm_benchmark_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Подготовлено {len(frame)} публичных текстов: {path}")


if __name__ == "__main__":
    main()
