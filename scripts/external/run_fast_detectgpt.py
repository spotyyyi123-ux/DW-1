"""Внешний GPU-запуск аналитического Fast-DetectGPT. Локально не импортирует torch.

Формула адаптирована из baoguangsheng/fast-detect-gpt (Guangsheng Bao, MIT).
Лицензия сохранена рядом в FAST-DETECT-GPT-LICENSE.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def discrepancy(logits, labels):
    """Одна модель используется как scoring и sampling; считаем в float32."""
    import torch

    log_prob = torch.log_softmax(logits.float(), dim=-1)
    prob = torch.softmax(logits.float(), dim=-1)
    observed = log_prob.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    expected = (prob * log_prob).sum(-1)
    variance = (prob * log_prob.square()).sum(-1) - expected.square()
    denominator = variance.sum().clamp_min(1e-12).sqrt()
    return float(((observed.sum() - expected.sum()) / denominator).item())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data/sprint2/llm_benchmark.jsonl")
    parser.add_argument("--model", default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--out", default="data/runs/sprint2/llm_scores.parquet")
    args = parser.parse_args(argv)
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dw1.models.llm_detectors import text_hash, validate_scores

    if not torch.cuda.is_available():
        raise SystemExit("Нужен внешний CUDA GPU. В Colab выберите среду T4 GPU.")
    if not 33 <= args.max_tokens <= 512:
        raise ValueError("Для пилота max-tokens должен быть от 33 до 512")
    frame = pd.read_json(args.benchmark, lines=True)
    if not set(frame.dataset) <= {"coat", "hc3-en"}:
        raise ValueError("Разрешены только публичные HC3/CoAT")
    if frame.text_sha256.tolist() != frame.text.map(text_hash).tolist():
        raise ValueError("Тексты изменились после подготовки выборки")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.revision,
            torch_dtype=torch.float16,
            use_safetensors=True,
            trust_remote_code=False,
        )
        .to("cuda")
        .eval()
    )
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    version = (
        f"fast-detectgpt-analytic-v1:{args.model}@{args.revision}"
        f":max{args.max_tokens}:runner{runner_hash}"
    )
    records = []
    skipped = []
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(frame.to_dict("records")):
        full_ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        ids = full_ids[: args.max_tokens]
        if len(ids) < 33:
            skipped.append(dict(message_id=row["message_id"], reason="less_than_32_target_tokens"))
            continue
        token_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
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
        # Сохраняем завершённые скоры при разрыве сессии; неполный прогон явно помечен.
        if (number + 1) % 20 == 0:
            pd.DataFrame(records).to_parquet(output.with_suffix(".partial.parquet"), index=False)
            print(f"Обработано {number + 1}/{len(frame)}", flush=True)
    scores = pd.DataFrame(records)
    validate_scores(scores)
    scores.to_parquet(output, index=False)
    manifest = dict(
        status="completed",
        detector=version,
        scoring_model=args.model,
        sampling_model=args.model,
        model_revision=args.revision,
        max_tokens=args.max_tokens,
        n_scored=len(scores),
        n_input=len(frame),
        benchmark_sha256=hashlib.sha256(Path(args.benchmark).read_bytes()).hexdigest(),
        skipped=skipped,
        gpu=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        limitations=[
            "Англоязычная surrogate-модель; перенос на RU измеряется отдельно.",
            "Оцениваются первые max_tokens токенов текста.",
            "Критерий Fast-DetectGPT не является вероятностью.",
        ],
    )
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Сохранено скоров: {len(scores)}; исходные тексты в результат не включены.")


if __name__ == "__main__":
    main()
