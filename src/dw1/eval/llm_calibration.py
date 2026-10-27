"""Калибровка Fast-DetectGPT по языкам на трёх непересекающихся частях."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score

from dw1.models.llm_detectors import validate_scores


def expected_calibration_error(y, probability, n_bins=10):
    """ECE положительного класса: средняя ошибка вероятности по равным интервалам.

    В каждом интервале сравниваются средняя p(ИИ) и фактическая доля ИИ.
    Это не метрика определения автора конкретного письма.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(probability, dtype=float)
    if (
        not len(y)
        or y.shape != p.shape
        or y.ndim != 1
        or n_bins < 1
        or not np.isfinite(p).all()
        or not np.isin(y, [0, 1]).all()
        or (p < 0).any()
        or (p > 1).any()
    ):
        raise ValueError("Некорректные входы ECE")
    ids = np.minimum((p * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for bin_id in range(n_bins):
        mask = ids == bin_id
        if mask.any():
            error += float(mask.mean() * abs(p[mask].mean() - y[mask].mean()))
    return error


def _summary(y, probability):
    return dict(
        ece=expected_calibration_error(y, probability),
        brier=float(brier_score_loss(y, probability)),
        f1=float(f1_score(y, probability >= 0.5, zero_division=0)),
        roc_auc=float(roc_auc_score(y, probability)),
    )


def calibrate(frame: pd.DataFrame, min_per_class=16):
    """Platt обучается на train; isotonic по языкам — на validation; отчёт только test."""
    validate_scores(frame)
    for language in ["en", "ru"]:
        for split in ["train", "validation", "test"]:
            part = frame[(frame.language == language) & (frame.split == split)]
            counts = part.label.value_counts()
            if any(counts.get(label, 0) < min_per_class for label in [0, 1]):
                raise ValueError(f"Недостаточно примеров классов: {language}, {split}")
    train = frame[frame.split == "train"]
    # Критерий Fast-DetectGPT не является вероятностью. Получаем базовую шкалу на train.
    platt = LogisticRegression(random_state=42).fit(train[["criterion"]], train.label)
    config: dict[str, Any] = dict(
        schema_version=1,
        model_version=str(frame.model_version.iloc[0]),
        calibration_version="isotonic-by-language-v1",
        before_definition="Platt(train), общий для языков; criterion не вероятность",
        platt_coefficient=float(platt.coef_[0, 0]),
        platt_intercept=float(platt.intercept_[0]),
        languages={},
    )
    evaluated = []
    metrics: dict[str, Any] = {}
    for language in ["en", "ru"]:
        val = frame[(frame.language == language) & (frame.split == "validation")]
        test = frame[(frame.language == language) & (frame.split == "test")].copy()
        iso = IsotonicRegression(out_of_bounds="clip").fit(val.criterion, val.label)
        before = platt.predict_proba(test[["criterion"]])[:, 1]
        after = iso.predict(test.criterion)
        test["p_ai_before"] = before
        test["p_ai_after"] = after
        evaluated.append(test)
        metrics[language] = dict(
            n_test=len(test), before=_summary(test.label, before), after=_summary(test.label, after)
        )
        # Простые доверительные интервалы с повторным выбором доступных групп.
        groups = {k: np.flatnonzero(test.group_id.to_numpy() == k) for k in test.group_id.unique()}
        keys = list(groups)
        rng = np.random.default_rng(42)
        intervals = []
        y = test.label.to_numpy()
        for _ in range(200):
            ix = np.concatenate([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
            intervals.append(
                [
                    expected_calibration_error(y[ix], before[ix]),
                    expected_calibration_error(y[ix], after[ix]),
                ]
            )
        bounds = np.quantile(intervals, [0.025, 0.975], axis=0)
        metrics[language]["ece_ci95"] = dict(
            before=bounds[:, 0].tolist(), after=bounds[:, 1].tolist()
        )
        config["languages"][language] = dict(
            isotonic_x=iso.X_thresholds_.tolist(),
            isotonic_y=iso.y_thresholds_.tolist(),
            n_calibration=len(val),
        )
    metrics["language_gap"] = dict(
        before_ru_minus_en=metrics["ru"]["before"]["ece"] - metrics["en"]["before"]["ece"],
        after_ru_minus_en=metrics["ru"]["after"]["ece"] - metrics["en"]["after"]["ece"],
    )
    return config, metrics, pd.concat(evaluated)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="data/runs/sprint2/llm_scores.parquet")
    parser.add_argument("--out", default="data/runs/sprint2")
    args = parser.parse_args(argv)
    source = Path(args.scores)
    if not source.is_file():
        parser.error("Нет настоящих LLM-скоров. Сначала выполните внешний GPU-ноутбук.")
    scores = pd.read_parquet(source)
    config, metrics, predictions = calibrate(scores)
    config["score_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "llm_calibration.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "llm_calibration_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    predictions.to_parquet(output / "llm_calibrated_predictions.parquet", index=False)
    rows = [
        "# Fast-DetectGPT: калибровка по языкам",
        "",
        "Исходный criterion не вероятность. До: Platt на train; после: isotonic на validation.",
        "Все показатели ниже рассчитаны на test, который не использовался при подгонке.",
        f"Скоров: {len(scores)}. Усечённых текстов: {int(scores.truncated.sum())}.",
        f"SHA-256 файла скоров: `{config['score_sha256']}`.",
        "",
        "| Язык | n test | ECE до | ECE после |",
        "|---|---:|---:|---:|",
    ]
    for language in ["en", "ru"]:
        row = metrics[language]
        rows.append(
            f"| {language} | {row['n_test']} | {row['before']['ece']:.4f} | "
            f"{row['after']['ece']:.4f} |"
        )
    rows += [
        "",
        "ECE измеряет согласованность вероятностей с частотами, а не точность классификации.",
        "",
        "| Язык | Brier до/после | F1 до/после | ROC-AUC до/после |",
        "|---|---:|---:|---:|",
    ]
    for language in ["en", "ru"]:
        before, after = metrics[language]["before"], metrics[language]["after"]
        rows.append(
            f"| {language} | {before['brier']:.4f}/{after['brier']:.4f} | "
            f"{before['f1']:.4f}/{after['f1']:.4f} | "
            f"{before['roc_auc']:.4f}/{after['roc_auc']:.4f} |"
        )
    rows += ["", "## Доверительные интервалы ECE (групповой bootstrap, 95%)", ""]
    for language in ["en", "ru"]:
        ci = metrics[language]["ece_ci95"]
        rows.append(
            f"- {language}: до [{ci['before'][0]:.4f}, {ci['before'][1]:.4f}], "
            f"после [{ci['after'][0]:.4f}, {ci['after'][1]:.4f}]."
        )
        if metrics[language]["after"]["roc_auc"] < 0.5:
            rows.append(
                f"- {language}: ROC-AUC после ниже 0.5; уменьшение ECE не означает "
                "улучшение различения человека и ИИ. Для применения нужен другой детектор."
            )
    gap = metrics["language_gap"]
    rows += [
        "",
        f"Разрыв ECE RU−EN: до {gap['before_ru_minus_en']:.4f}, "
        f"после {gap['after_ru_minus_en']:.4f}.",
        "",
        "Маленькая публичная выборка не доказывает перенос на почту заказчика.",
        "У CoAT нет исходных групп вариантов; исключение всех утечек не гарантируется.",
        "Доверительные интервалы и разрыв RU−EN сохранены в llm_calibration_metrics.json.",
    ]
    (output / "llm_calibration_report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
