"""Сравнивает две версии на одних текстах и проверяет границы RoFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from dw1.eval.harness import binary_metrics


def first_ai_boundary(scores):
    """Выбирает одну границу: до неё человек, после неё ИИ.

    Минимизируем сумму отрицательных логарифмов оценок. Это простое проверочное
    правило с некалиброванными оценками, а не обученный детектор ИИ-редактуры.
    Значение len(scores) означает отсутствие машинной части.
    """
    p = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    human_cost = np.r_[0, np.cumsum(-np.log1p(-p))]
    ai_cost = np.r_[np.cumsum(-np.log(p)[::-1])[::-1], 0]
    return int(np.argmin(human_cost + ai_cost))


def predict(model, texts):
    with threadpool_limits(limits=2):
        return model.predict_proba(texts)[:, 1]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", default="sprint1-v1")
    parser.add_argument("--new", default="sprint1-llmtrace-v2")
    args = parser.parse_args(argv)
    if any(Path(x).name != x for x in [args.old, args.new]):
        parser.error("Имена запусков должны быть простыми именами папок")
    old_run = Path("data/runs") / args.old
    new_run = Path("data/runs") / args.new
    models = {
        "old": joblib.load(old_run / "model.joblib"),
        "new": joblib.load(new_run / "model.joblib"),
    }
    data = pd.read_parquet("data/processed/w1_expanded_texts.parquet")
    old_pred = pd.read_parquet(old_run / "predictions.parquet")
    old_ids = set(old_pred[old_pred.split == "test"].message_id)
    shared = data[(data.split == "test") & data.message_id.isin(old_ids)]
    result = dict(old_run=args.old, new_run=args.new, shared_test={}, llmtrace_test={})
    prediction_frames = []
    for title, part in [
        ("shared_test", shared),
        ("llmtrace_test", data[(data.split == "test") & (data.source == "llmtrace")]),
    ]:
        frame = part[["message_id", "source", "language", "group_id", "label"]].copy()
        for version, model in models.items():
            scores = predict(model, part.text)
            frame[version + "_score"] = scores
            result[title][version] = binary_metrics(part.label, scores)
            if title == "llmtrace_test":
                result[title][version]["by_language"] = {
                    lang: binary_metrics(g.label, g[version + "_score"])
                    for lang, g in frame.groupby("language")
                }
        frame["evaluation"] = title
        prediction_frames.append(frame)
    pd.concat(prediction_frames).to_parquet(new_run / "comparison_predictions.parquet", index=False)

    docs = pd.read_parquet("data/processed/roft_documents.parquet")
    segments = pd.read_parquet("data/processed/roft_segments.parquet")
    docs = docs[docs.split == "test"].copy()
    segments = segments[segments.document_id.isin(docs.message_id)].copy()

    # Точные совпадения с учебными текстами исключаем целыми документами.
    def normalize(text):
        return " ".join(text.casefold().split())

    train_texts = set(data[data.split == "train"].text.map(normalize))
    overlapping = set(segments[segments.text.map(normalize).isin(train_texts)].document_id)
    docs = docs[~docs.message_id.isin(overlapping)]
    segments = segments[~segments.document_id.isin(overlapping)].copy()
    result["roft"] = dict(
        documents=len(docs),
        segments=len(segments),
        exact_overlap_documents_removed=len(overlapping),
        training_use=False,
        ai_polished_label_used=False,
    )
    boundary_rows = []
    for version, model in models.items():
        segments[version + "_score"] = predict(model, segments.text)
        boundary_by_doc = {}
        for doc_id, group in segments.groupby("document_id"):
            ordered = group.sort_values("segment_index")
            boundary_by_doc[doc_id] = first_ai_boundary(ordered[version + "_score"])
        predicted = docs.message_id.map(boundary_by_doc).to_numpy()
        truth = docs.first_ai_segment_index.to_numpy()
        errors = np.abs(predicted - truth)
        result["roft"][version] = dict(
            segment_metrics=binary_metrics(segments.label, segments[version + "_score"]),
            boundary_exact_accuracy=float(np.mean(errors == 0)),
            boundary_mae_segments=float(np.mean(errors)),
        )
        for doc_id, true, pred in zip(docs.message_id, truth, predicted, strict=True):
            boundary_rows.append(
                dict(
                    message_id=doc_id,
                    version=version,
                    true_first_ai_segment=int(true),
                    predicted_first_ai_segment=int(pred),
                )
            )
    segments.drop(columns="text").to_parquet(new_run / "roft_predictions.parquet", index=False)
    pd.DataFrame(boundary_rows).to_parquet(new_run / "roft_boundaries.parquet", index=False)
    result["limitations"] = [
        "Старый тест уже просматривался: сравнение предварительное, не новый слепой тест.",
        "RoFT — продолжение текста, а не редактирование; класс ai_polished не обучался.",
        "RoFT состоит из сегментов _SEP_, которые не всегда равны одному предложению.",
        "Для RoFT исключены точные пересечения с train; близкие совпадения могут остаться.",
        "Разделение LLMTrace выполнено по topic_id; у CoAT исходные пары неизвестны.",
        "Система пока не проверена на сбалансированном почтовом наборе с известным авторством.",
    ]
    (new_run / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    old_meta = json.loads((old_run / "metrics.json").read_text())
    new_meta = json.loads((new_run / "metrics.json").read_text())
    roft_audit = json.loads(Path("data/processed/roft_preparation.json").read_text())
    rows = [
        "# LLMTrace и RoFT: второй эксперимент",
        "",
        "LLMTrace добавлен к обучению. RoFT подготовлен для проверки смешанных текстов.",
        f"Обучающих текстов: {old_meta['train_rows']} → {new_meta['train_rows']}.",
        "",
        "| Одинаковая проверочная выборка | Старая модель | Новая модель |",
        "|---|---:|---:|",
    ]
    for title, display in [
        ("shared_test", "Прежние тексты, accuracy"),
        ("llmtrace_test", "LLMTrace EN+RU, accuracy"),
    ]:
        values = result[title]
        rows.append(
            f"| {display}, n={values['new']['n']} | "
            f"{values['old']['accuracy']:.1%} | {values['new']['accuracy']:.1%} |"
        )
    for metric, display in [("f1", "LLMTrace EN+RU, F1"), ("roc_auc", "LLMTrace EN+RU, ROC-AUC")]:
        before = result["llmtrace_test"]["old"][metric]
        after = result["llmtrace_test"]["new"][metric]
        rows.append(f"| {display} | {before:.3f} | {after:.3f} |")
    for lang in ["ru", "en"]:
        old = result["llmtrace_test"]["old"]["by_language"][lang]
        new = result["llmtrace_test"]["new"]["by_language"][lang]
        rows.append(
            f"| LLMTrace {lang}, accuracy, n={new['n']} | "
            f"{old['accuracy']:.1%} | {new['accuracy']:.1%} |"
        )
    rows += [
        f"| AI-письма LLMGen, доля обнаруженных | "
        f"{old_meta['external_llmgen_en']['ai_detection_rate']:.1%} | "
        f"{new_meta['external_llmgen_en']['ai_detection_rate']:.1%} |",
        "",
        "Для LLMGen измеряется только обнаружение ИИ-писем, а не общая accuracy.",
        "",
        "## RoFT",
        "",
        f"Из {roft_audit['raw_annotations']} аннотаций получено "
        f"{roft_audit['unique_documents']} уникальных документов и "
        f"{roft_audit['segment_rows']} размеченных сегментов.",
        "Повторные оценки одного текста удалены. Человеческая догадка о границе",
        "не используется вместо настоящей границы true_boundary_index.",
        "",
        f"Для проверки выбрано {len(docs)} документов с раздельными группами источников.",
        "",
        "| Показатель RoFT | Старая модель | Новая модель |",
        "|---|---:|---:|",
    ]
    for metric, display in [
        ("boundary_exact_accuracy", "Доля точно найденных границ"),
        ("boundary_mae_segments", "Средняя ошибка, сегментов"),
    ]:
        vals = [result["roft"][v][metric] for v in ["old", "new"]]
        fmt = ".1%" if metric.endswith("accuracy") else ".2f"
        rows.append(f"| {display} | {vals[0]:{fmt}} | {vals[1]:{fmt}} |")
    rows += ["", "## Ограничения", ""] + ["- " + s for s in result["limitations"]]
    rows += [
        "",
        "## Повторение",
        "",
        "Из настроенной среды в корне проекта:",
        "",
        "```bash",
        "export PYTHONPATH=src",
        "python scripts/fetch_datasets.py llmtrace",
        "python scripts/fetch_datasets.py roft",
        "python scripts/prepare_new_datasets.py",
        "python scripts/prepare_roft.py",
        "python -m dw1.eval.harness --run sprint1-llmtrace-repeat \\",
        "  --data data/processed/w1_expanded_texts.parquet \\",
        "  --preparation data/processed/expanded_preparation.json \\",
        "  --model-version tfidf-svd64-histgb-w1-v2",
        "python -m dw1.eval.compare --new sprint1-llmtrace-repeat",
        "```",
        "",
        "Для сравнения нужен прежний запуск sprint1-v1. Полный корпус читается",
        "построчно; в RAM и обучение попадает ограниченная воспроизводимая выборка.",
        "Лицензии и SHA-256 файлов записаны в data/datasheet.csv.",
        "Исходные данные, модель и прогнозы исключены из Git.",
    ]
    report = "\n".join(rows) + "\n"
    Path("docs/w1-new-datasets.md").write_text(report, encoding="utf-8")
    (new_run / "comparison.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
