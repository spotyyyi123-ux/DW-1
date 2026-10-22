"""Чтение LLMTrace без загрузки всего корпуса в оперативную память."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def topic_split(topic_id):
    """Все варианты одной темы получают одну и ту же часть выборки."""
    bucket = int(hashlib.sha256(topic_id.encode()).hexdigest()[:8], 16) % 10
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


def sample_llmtrace(directory, cap=2000, seed=42):
    """Равномерная резервуарная выборка по языку, классу и части данных.

    Сохраняем исходную часть в source_split, но делим по topic_id: это исключает
    пересечение родственных вариантов даже при их наличии в разных исходных файлах.
    """
    if cap < 1:
        raise ValueError("Размер выборки должен быть положительным")
    rng = random.Random(seed)
    pools: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    seen: Counter[tuple[str, int, str]] = Counter()
    totals: Counter[str] = Counter()
    for original_split in ["train", "valid", "test"]:
        path = Path(directory) / f"{original_split}.jsonl"
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream):
                item = json.loads(line)
                if item["label"] not in {"human", "ai"}:
                    raise ValueError("Неизвестная метка происхождения LLMTrace")
                if item["lang"] not in {"ru", "eng"}:
                    raise ValueError("Неизвестный язык LLMTrace")
                topic = item.get("topic_id")
                if not isinstance(topic, str) or not topic:
                    raise ValueError("Нет topic_id: нельзя гарантировать разделение тем")
                totals["raw_rows"] += 1
                text = item.get("text")
                if not isinstance(text, str) or len(text.strip()) < 30:
                    totals["short_or_empty"] += 1
                    continue
                group = "llmtrace-" + topic
                split = topic_split(group)
                lang = {"eng": "en", "ru": "ru"}[item["lang"]]
                label = {"human": 0, "ai": 1}[item["label"]]
                key = (lang, label, split)
                seen[key] += 1
                limit = cap if split == "train" else max(100, cap // 4)
                pool = pools.setdefault(key, [])
                # Запоминаем не весь корпус, а не более limit строк в каждой группе.
                position = len(pool) if len(pool) < limit else rng.randrange(seen[key])
                if position >= limit:
                    continue
                row = dict(
                    message_id=f"llmtrace-{original_split}-{number}",
                    text=text,
                    label=label,
                    source="llmtrace",
                    language=lang,
                    group_id=group,
                    split=split,
                    source_split=original_split,
                    intent="unknown",
                    generator=item.get("model"),
                    domain=item.get("data_type"),
                    prompt_type=item.get("prompt_type"),
                )
                if position == len(pool):
                    pool.append(row)
                else:
                    pool[position] = row
        print(f"LLMTrace: прочитан {original_split}", flush=True)
    frame = pd.DataFrame([row for key in sorted(pools) for row in pools[key]])
    audit: dict[str, Any] = dict(totals)
    audit["available_by_language_label_split"] = [
        dict(language=k[0], label=k[1], split=k[2], count=v) for k, v in sorted(seen.items())
    ]
    audit["sampled_rows"] = len(frame)
    return frame, audit


def prepare_roft(path):
    """Сохраняет реальные границы RoFT и размеченные сегменты, а не догадки аннотаторов.

    true_boundary_index — индекс последнего человеческого сегмента, начиная с нуля.
    Он равен числу сегментов prompt_body минус один (см. исходный обработчик RoFT).
    Смешанный документ не приравнивается к ai_polished: это продолжение, не редактура.
    """
    table = pd.read_csv(path, keep_default_na=False)
    required = {"prompt", "generation", "prompt_body", "gen_body", "model", "true_boundary_index"}
    if not required <= set(table):
        raise ValueError("В RoFT отсутствуют обязательные поля")
    documents = []
    segments = []
    skipped: Counter[str] = Counter()
    seen = set()
    for row in table.to_dict("records"):
        human = row["prompt_body"].split("_SEP_") if row["prompt_body"] else []
        generated = row["gen_body"].split("_SEP_") if row["gen_body"] else []
        truth = int(row["true_boundary_index"])
        key = hashlib.sha256((row["prompt_body"] + "\0" + row["gen_body"]).encode()).hexdigest()
        if truth != len(human) - 1 or not human or any(not x.strip() for x in human + generated):
            skipped["invalid_boundary_or_empty"] += 1
            continue
        if bool(generated) == (row["model"] == "human"):
            skipped["inconsistent_model_and_generation"] += 1
            continue
        if key in seen:
            skipped["repeated_annotation"] += 1
            continue
        seen.add(key)
        # Начало одного источника объединяет разные продолжения и длины префикса.
        first = " ".join(human[0].casefold().split())
        group = "roft-" + hashlib.sha256(first.encode()).hexdigest()
        doc_id = "roft-" + key[:24]
        texts = human + generated
        split = topic_split(group)
        documents.append(
            dict(
                message_id=doc_id,
                source="roft",
                language="en",
                group_id=group,
                split=split,
                prompt_id=str(row["prompt"]),
                generation_id=str(row["generation"]),
                generator=row["model"],
                text="\n".join(texts),
                true_boundary_index=truth,
                first_ai_segment_index=len(human) if generated else len(texts),
                n_segments=len(texts),
                human_segments=len(human),
                ai_segments=len(generated),
                composition="human_then_ai" if generated else "human_only",
                ai_provenance="unknown" if generated else "human",
                intent="unknown",
            )
        )
        for index, text in enumerate(texts):
            label = int(index >= len(human))
            segments.append(
                dict(
                    message_id=f"{doc_id}-{index}",
                    document_id=doc_id,
                    source="roft",
                    language="en",
                    group_id=group,
                    split=split,
                    segment_index=index,
                    text=text,
                    label=label,
                    ai_provenance="ai" if label else "human",
                    intent="unknown",
                )
            )
    audit = dict(
        raw_annotations=len(table),
        unique_documents=len(documents),
        segment_rows=len(segments),
        skipped=dict(skipped),
        boundary_definition="Последний человеческий сегмент, нумерация с нуля.",
        source_code="https://github.com/liamdugan/human-detection/blob/main/"
        "processing/process_database_dump.py",
        limitation="RoFT размечает продолжение текста, а не ИИ-редактуру. "
        "Человеческая догадка predicted_boundary_index не используется.",
    )
    return pd.DataFrame(documents), pd.DataFrame(segments), audit
