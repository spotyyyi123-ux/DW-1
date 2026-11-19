"""Оценка прежней и train-направленной калибровки на новой RU-выборке."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dw1.eval.llm_calibration import calibrate
from dw1.models.llm_detectors import LLMScoreStore, validate_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/runs/sprint5/ru"))
    args = parser.parse_args()
    root = args.root
    source = root / "qwen05_new_scores.parquet"
    frame = pd.read_parquet(source)
    validate_scores(frame)
    benchmark = pd.read_json(root / "benchmark.jsonl", lines=True)
    manifest = json.loads((root / "qwen05_new_scores.manifest.json").read_text())
    benchmark_hash = hashlib.sha256((root / "benchmark.jsonl").read_bytes()).hexdigest()
    if (
        manifest.get("status") != "completed"
        or manifest.get("benchmark_sha256") != benchmark_hash
        or manifest.get("n_scored") != len(frame)
        or manifest.get("n_input") != len(benchmark)
        or manifest.get("detector") != frame.model_version.iloc[0]
    ):
        raise ValueError("Скоры и манифест не соответствуют новой выборке")
    old = pd.read_json("data/sprint2/llm_benchmark.jsonl", lines=True)
    if set(frame.text_sha256) & set(old.text_sha256):
        raise ValueError("В новой проверке найдены прежние экспериментальные тексты")
    keys = ["message_id", "text_sha256", "group_id", "dataset", "language", "label", "split"]
    pd.testing.assert_frame_equal(
        frame[keys].sort_values("message_id").reset_index(drop=True),
        benchmark[benchmark.text_sha256.isin(frame.text_sha256)][keys]
        .sort_values("message_id").reset_index(drop=True),
    )
    metrics = {}
    predictions = []
    for method, oriented in [("legacy_increasing", False), ("train_direction", True)]:
        config, result, test = calibrate(
            frame, languages=("ru",), direction_from_train=oriented
        )
        config["score_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        metrics[method] = result["ru"]
        predictions.append(test.assign(method=method))
        if oriented:
            calibration = root / "calibration.json"
            calibration.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
            store = LLMScoreStore(source, calibration)
            for row in test.itertuples():
                text = benchmark.loc[
                    benchmark.text_sha256 == row.text_sha256, "text"
                ].iloc[0]
                found = store.lookup(text)
                if found is None or not np.isclose(
                    found["evidence"][0]["value"]["p_ai"], row.p_ai_after
                ):
                    raise ValueError("Загрузка калибратора расходится с тестовыми прогнозами")

    summary = {
        "n_input": len(benchmark), "n_scored": len(frame),
        "split_counts": frame.groupby(["split", "label"]).size()
        .rename("n").reset_index().to_dict("records"),
        "score_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "benchmark_sha256": benchmark_hash,
        "score_direction": config["languages"]["ru"]["score_direction"],
        "metrics": metrics,
        "integration": "JSON-калибратор воспроизвёл все тестовые прогнозы",
    }
    pd.concat(predictions).to_parquet(root / "predictions.parquet", index=False)
    (root / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    rows = [
        "# Sprint 5 · Калибровка RU на новых текстах",
        "",
        f"Qwen2.5-0.5B, входных текстов {len(benchmark)}, скоров {len(frame)}.",
        "Модель выбрана по прежним экспериментам до просмотра нового теста.",
        "Направление определяется по знаку коэффициента Platt на train.",
        "Isotonic обучается на validation, итоговые показатели считаются на test.",
        "",
        "| Метод | test | ECE | F1 | AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    platt = metrics["train_direction"]
    rows.append(f"| Platt | {platt['n_test']} | {platt['before']['ece']:.4f} | "
                f"{platt['before']['f1']:.4f} | {platt['before']['roc_auc']:.4f} |")
    for method, result in metrics.items():
        m = result["after"]
        rows.append(f"| {method} | {result['n_test']} | {m['ece']:.4f} | "
                    f"{m['f1']:.4f} | {m['roc_auc']:.4f} |")
    rows += [
        "",
        "ECE-интервалы (групповой bootstrap 95%) приведены в metrics.json.",
        "Совпадения с прежними 300 RU-текстами исключены по нормализованным хэшам.",
        "Поиск близких повторов приближённый, исходные группы вариантов CoAT неизвестны.",
        "CoAT содержит общие русские тексты. Перенос на русские письма требует отдельной проверки.",
        "Исходные тексты и parquet передаются вне Git.",
    ]
    report = "\n".join(rows) + "\n"
    (root / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
