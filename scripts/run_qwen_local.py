"""Локальный прогон Fast-DetectGPT-аналитики на CPU (fallback без Colab).

Тот же критерий, что в scripts/external/run_fast_detectgpt.py, но с моделью Qwen
(по умолчанию Qwen2.5-0.5B — поддерживает RU) и без требования CUDA.

Запуск (после pip install torch --index-url .../cpu + transformers):
    python scripts/run_qwen_local.py --limit 120
Выход: data/runs/sprint2/qwen_scores.parquet + manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "external"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=str(ROOT / "data/sprint2/llm_benchmark.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="0 = все тексты бенчмарка")
    parser.add_argument("--language", default=None, help="фильтр: en/ru")
    parser.add_argument("--out", default=str(ROOT / "data/runs/sprint2/qwen_scores.parquet"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from run_fast_detectgpt import discrepancy
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dw1.models.llm_detectors import text_hash, validate_scores

    frame = pd.read_json(args.benchmark, lines=True)
    if not 33 <= args.max_tokens <= 512:
        raise ValueError("max-tokens должен быть от 33 до 512")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA GPU недоступен")
    if not set(frame.dataset) <= {"coat", "hc3-en", "emails"}:
        raise ValueError("Разрешены только публичные HC3/CoAT/emails (лицензионный subset)")
    if frame.text_sha256.tolist() != frame.text.map(text_hash).tolist():
        raise ValueError("Тексты изменились после подготовки выборки")
    if args.language:
        frame = frame[frame.language == args.language].reset_index(drop=True)
    if args.limit:
        frame = frame.head(args.limit).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, token=False)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.revision,
            torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
            use_safetensors=True,
            trust_remote_code=False,
            token=False,
        )
        .to(args.device)
        .eval()
    )
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    version = (
        f"fast-detectgpt-analytic-qwen:{args.model}@{(args.revision or 'main')}"
        f":max{args.max_tokens}:runner{runner_hash}"
    )

    records: list[dict] = []
    skipped: list[dict] = []
    for number, row in enumerate(frame.to_dict("records")):
        full_ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        ids = full_ids[: args.max_tokens]
        if len(ids) < 33:
            skipped.append({"message_id": row["message_id"], "reason": "less_than_32_target_tokens"})
            continue
        token_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
        with torch.inference_mode():
            logits = model(input_ids=token_ids).logits[:, :-1]
            criterion = discrepancy(logits, token_ids[:, 1:])
        if not np.isfinite(criterion):
            raise ValueError("Детектор вернул неконечный критерий")
        record = {
            key: row[key]
            for key in [
                "message_id",
                "text_sha256",
                "group_id",
                "dataset",
                "language",
                "label",
                "split",
            ]
        }
        record.update(
            criterion=criterion,
            n_tokens=len(ids) - 1,
            truncated=len(full_ids) > len(ids),
            model_version=version,
        )
        records.append(record)
        if (number + 1) % 20 == 0:
            print(f"Обработано {number + 1}/{len(frame)}", flush=True)

    scores = pd.DataFrame(records)
    validate_scores(scores)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out, index=False)
    manifest = {
        "status": "completed",
        "detector": version,
        "scoring_model": args.model,
        "model_revision": args.revision or "main",
        "max_tokens": args.max_tokens,
        "n_scored": len(scores),
        "n_input": len(frame),
        "benchmark_sha256": hashlib.sha256(Path(args.benchmark).read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "skipped": skipped,
        "device": args.device,
        "torch_version": torch.__version__,
        "limitations": [
            "Малая surrogate-модель; качество измеряется отдельно по языкам и источникам.",
            "Оцениваются первые max_tokens токенов.",
            "Критерий Fast-DetectGPT не является вероятностью.",
        ],
    }
    out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Сохранено скоров: {len(scores)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
