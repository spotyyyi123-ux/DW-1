"""Сравнение трёх Fast-DetectGPT детекторов на общих русских текстах CoAT."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from dw1.eval.llm_calibration import calibrate
from dw1.models.llm_detectors import validate_scores

METADATA = ["text_sha256", "label", "split", "language", "group_id", "dataset"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sprint3", type=Path, default=Path("data/runs/sprint3"))
    parser.add_argument("--out", type=Path, default=Path("data/runs/sprint4/ru-detector"))
    args = parser.parse_args()
    paths = {
        "GPT-Neo 2.7B": args.sprint3 / "gptneo_scores.parquet",
        "Qwen2.5-0.5B": args.sprint3 / "qwen_scores.parquet",
        "Qwen2.5-1.5B": args.out / "qwen15_ru_scores.parquet",
    }
    frames = {}
    for name, path in paths.items():
        frame = pd.read_parquet(path)
        validate_scores(frame)
        frames[name] = frame[(frame.language == "ru") & (frame.dataset == "coat")]

    common = set.intersection(*(set(frame.text_sha256) for frame in frames.values()))
    matched = {
        name: frame[frame.text_sha256.isin(common)].sort_values("text_sha256").reset_index(drop=True)
        for name, frame in frames.items()
    }
    reference = matched["GPT-Neo 2.7B"]
    for frame in matched.values():
        pd.testing.assert_frame_equal(frame[METADATA], reference[METADATA])

    metrics = {}
    predictions = []
    for name, frame in matched.items():
        config, result, test = calibrate(frame, languages=("ru",))
        predictions.append(test.assign(detector=name))
        metrics[name] = {
            **result["ru"],
            "n_scored_ru": len(frames[name]),
            "source_sha256": hashlib.sha256(paths[name].read_bytes()).hexdigest(),
            "model_version": config["model_version"],
            "platt_coefficient": config["platt_coefficient"],
        }
        if name == "Qwen2.5-1.5B":
            config["score_sha256"] = metrics[name]["source_sha256"]
            (args.out / "qwen15_ru_calibration.json").write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

    pd.concat(predictions).to_parquet(args.out / "predictions.parquet", index=False)
    current_auc = metrics["Qwen2.5-1.5B"]["after"]["roc_auc"]
    improved = current_auc > max(
        0.5, metrics["GPT-Neo 2.7B"]["after"]["roc_auc"],
        metrics["Qwen2.5-0.5B"]["after"]["roc_auc"]
    )
    summary = {
        "common": len(common),
        "split_counts": reference.split.value_counts().to_dict(),
        "improved_ru_discrimination": improved,
        "models": metrics,
    }
    (args.out / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = [
        "# Sprint 4 · Fast-DetectGPT на CoAT-RU",
        "",
        f"Общих текстов: {len(common)}. "
        + ", ".join(f"{part} {summary['split_counts'][part]}"
                    for part in ("train", "validation", "test")) + ".",
        "Для всех моделей заново обучили Platt на RU train и isotonic на RU validation.",
        "Метрики рассчитаны на одном RU test. Порог F1 равен 0.5.",
        "",
        "| Модель | RU скоров | Тест | ECE до → после | F1 до → после | AUC до → после |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in metrics.items():
        values = [f"{row['before'][key]:.4f} → {row['after'][key]:.4f}"
                  for key in ("ece", "f1", "roc_auc")]
        rows.append(f"| {name} | {row['n_scored_ru']} | {row['n_test']} | " +
                    " | ".join(values) + " |")
    rows.extend([
        "",
        "Критерий улучшения различения: AUC после калибровки выше 0.5 и обеих базовых моделей.",
        "Критерий выполнен." if improved else "Критерий не выполнен. RU остаётся ограничением.",
        "CoAT — публичный корпус текстов. Перенос выводов на русские письма требует проверки.",
    ])
    if all(row["platt_coefficient"] < 0 for row in metrics.values()):
        rows.extend([
            "",
            "На RU train коэффициент Platt отрицателен у всех трёх моделей. "
            "Поэтому Platt даёт более высокие оценки ИИ при меньшем исходном скоре. "
            "Изотоническая калибровка в текущем протоколе допускает только возрастающую зависимость.",
            "Следующий эксперимент: фиксировать направление на train, калибровать на validation "
            "и проверить обе схемы на новой отложенной RU-выборке.",
        ])
    report = "\n".join(rows) + "\n"
    (args.out / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
