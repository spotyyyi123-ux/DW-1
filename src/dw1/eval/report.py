"""Краткий отчёт по сохранённым результатам обучения."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="sprint1-v1")
    parser.add_argument("--out", help="Куда сохранить отчёт")
    args = parser.parse_args(argv)
    if Path(args.run).name != args.run:
        parser.error("--run must be a simple directory name")
    run = Path("data/runs") / args.run
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    audit_path = run / "preparation.json"
    if not audit_path.exists():
        audit_path = Path("data/processed/preparation.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    lines = [
        "# Данные и первая модель",
        "",
        "Цель — получить первую модель, различающую человеческие и ИИ-тексты,",
        "и проверить, подходит ли она для писем. Обучение и проверка выполнены локально.",
        "",
        "## Данные и метод",
        "",
        "Для обучения использованы CoAT (русские тексты) и HC3 (английские ответы).",
        "Метка 0 означает человека, 1 — ИИ. Скачаны также LLMGen и Nazario 2023–2025.",
        "В LLMGen все письма созданы ИИ, а метка обозначает фишинг/нормальное письмо.",
        "У Nazario происхождение неизвестно: его нельзя автоматически считать человеческим.",
        "",
        f"Обучение: {metrics['train_rows']} текстов. Модель: TF-IDF → SVD → HistGradientBoosting.",
        "TF-IDF преобразует слова в числовые признаки, SVD сокращает их число до 64,",
        "а классификатор учится различать два класса. Порог решения — 0.5.",
        "",
        "## Результаты",
        "",
        "| Проверка | Текстов | Правильных ответов | F1 для ИИ | ROC-AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, title in [("validation", "Промежуточная"), ("test", "Отложенная")]:
        row = metrics[split]
        lines.append(
            f"| {title} | {row['n']} | {row['accuracy']:.1%} | "
            f"{row['f1']:.3f} | {row['roc_auc']:.3f} |"
        )
    for source, row in metrics["test"]["by_source"].items():
        lines.append(
            f"| Отложенная: {source} | {row['n']} | {row['accuracy']:.1%} | "
            f"{row['f1']:.3f} | {row['roc_auc']:.3f} |"
        )
    if "llmtrace" in metrics["test"]["by_source"]:
        lines[
            lines.index(
                "Для обучения использованы CoAT (русские тексты) и HC3 (английские ответы)."
            )
        ] = "Для обучения использованы CoAT, HC3 и двуязычный LLMTrace."
        lines += ["", "LLMTrace разделён по topic_id: варианты одной темы не пересекают части."]
    external = metrics.get("external_llmgen_en")
    if external:
        lines += [
            "",
            f"На {external['n']} английских ИИ-письмах LLMGen обнаружено "
            f"{external['ai_detection_rate']:.1%}. Здесь нет человеческих писем,",
            "поэтому это доля обнаруженных ИИ-писем, а не общая точность.",
        ]
    lines += [
        "",
        "## Проверка и ограничения",
        "",
        "Обучение и проверка используют разные тексты. HC3 разделён по вопросам.",
        "В CoAT исходная validation используется как отложенная проверка, а исходный",
        "test исключён: его метки скрыты (-1). Исходные пары текстов CoAT неизвестны.",
        f"Приближённым поиском удалено близких повторов между частями: "
        f"{audit['cross_split_near_duplicates_removed']}. Все повторы исключить не гарантируется.",
        "TF-IDF и SVD обучались только на обучающей части; порог по тесту не подбирался.",
        "Сравнение с простой DummyClassifier и bootstrap-интервалы сохранены в metrics.json.",
        "Оценки модели не калиброваны. На письмах результат пока слабый.",
        "Нужны почтовые примеры с подтверждённым происхождением обоих классов.",
        "seven-phishing пока не скачан из-за неясной лицензии, CIFAKE — недоступного зеркала.",
        "Подключение к общей программе и оставшиеся данные — следующие шаги.",
        "",
        "## Файлы и запуск",
        "",
        "- `data/datasheet.csv` — источники, лицензии и хэши скачанных файлов.",
        "- `data/processed/` — подготовленные тексты и письма.",
        f"- `{run}/` — модель, прогнозы, метрики и примеры ошибок.",
        "- Данные и модель исключены из Git; для повторения есть скрипт загрузки.",
        "",
        "На Mac из корня проекта (проверено на Python 3.13.5):",
        "",
        "```bash",
        "python3 -m venv .venv",
        "source .venv/bin/activate",
        "pip install -r requirements-w1-lock.txt",
        "export PYTHONPATH=src",
        "python -m pytest",
        "python scripts/fetch_datasets.py all",
        "python scripts/prepare_w1.py",
        "python -m dw1.eval.harness --run my-first-run",
        "python -m dw1.eval.report --run my-first-run",
        "```",
        "",
        "Имя запуска должно быть новым: существующие результаты не перезаписываются.",
        "Для проверки своего UTF-8 файла:",
        "",
        "```bash",
        "python -m dw1.models.baseline --text-file example.txt",
        "```",
        "",
        "Автотесты проверяют разбор писем, расчёт метрик и разделение данных.",
        "Общие Ruff/mypy репозитория имеют замечания в прежнем коде; полный CI не зелёный.",
    ]
    path = Path(args.out) if args.out else Path("reports") / f"{args.run}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
