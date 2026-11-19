"""Новая публичная RU-выборка с исключением прошлых экспериментальных текстов."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from prepare_w1 import clean_exact, fingerprint, remove_near_leaks

from dw1.models.llm_detectors import text_hash

ROOT = Path(__file__).resolve().parents[1]


def main():
    previous = pd.read_json(ROOT / "data/sprint2/llm_benchmark.jsonl", lines=True)
    previous = previous[previous.language == "ru"].copy()
    raw_path = ROOT / "data/raw/coat/binary/train-00000-of-00001.parquet"
    frame = clean_exact(pd.read_parquet(raw_path))
    frame = frame[frame.label.isin([0, 1])]
    frame = frame[~frame.fingerprint.isin(previous.text.map(fingerprint))]
    frame["message_id"] = "coat-" + frame.id.astype(str)
    frame["group_id"] = frame.fingerprint
    frame["split"] = frame.fingerprint.map(
        lambda key: ["train", "validation", "test"][int(key[:8], 16) % 3]
    )
    candidates = frame.groupby(["split", "label"], group_keys=False).sample(
        n=140, random_state=20261119
    )
    # Старые тексты индексируются первыми для проверки приближённых повторов.
    old = previous[["text", "label"]].assign(split="train", is_old=True)
    pooled = pd.concat([old, candidates.assign(split="validation", is_old=False)],
                       ignore_index=True)
    safe, removed_old = remove_near_leaks(pooled)
    safe = safe[~safe.is_old].drop(columns=["split", "is_old"])
    safe = safe.merge(candidates[["message_id", "split"]], on="message_id", validate="one_to_one")
    safe, removed_split = remove_near_leaks(safe.reset_index(drop=True))
    safe = safe.groupby(["split", "label"], group_keys=False).head(100).copy()
    if (safe.groupby(["split", "label"]).size() != 100).any() or len(safe) != 600:
        raise ValueError("Недостаточно независимых примеров после очистки")
    safe["text"] = safe.text.map(lambda text: text.replace("\r\n", "\n").strip())
    safe["text_sha256"] = safe.text.map(text_hash)
    safe["dataset"], safe["language"] = "coat", "ru"
    columns = ["message_id", "text_sha256", "group_id", "dataset", "language",
               "label", "split", "text"]
    out = ROOT / "data/runs/sprint5/ru"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "benchmark.jsonl"
    safe[columns].to_json(target, orient="records", lines=True, force_ascii=False)
    manifest = {
        "seed": 20261119, "n_input": len(safe),
        "benchmark_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "previous_sha256": hashlib.sha256(
            (ROOT / "data/sprint2/llm_benchmark.jsonl").read_bytes()).hexdigest(),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "near_old_removed": removed_old, "near_split_removed": removed_split,
        "counts": safe.groupby(["split", "label"]).size().rename("n").reset_index().to_dict("records"),
        "limitations": ["CoAT не содержит всех исходных групп вариантов.",
                        "Приближённый поиск близких повторов может пропустить совпадения."],
    }
    (out / "benchmark.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Подготовлено {len(safe)} новых публичных текстов CoAT")


if __name__ == "__main__":
    main()
