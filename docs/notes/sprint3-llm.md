# Sprint 3 · задача Алины: Qwen и письма

Задача выполнена на внешнем Colab Tesla T4; локально torch/transformers не устанавливались.
Qwen2.5-0.5B не дообучался: вычислен аналитический Fast-DetectGPT, затем обучены шкалы калибровки.
Модель Apache-2.0: https://huggingface.co/Qwen/Qwen2.5-0.5B
Ревизия: `060db6499f32faf8b98477b0a26969ef7d8b9987`.
Формула Fast-DetectGPT — MIT; лицензия в `scripts/external/FAST-DETECT-GPT-LICENSE.txt`.

## Данные и выводы

Исходные HC3/CoAT: те же 600 текстов из Sprint 2. GPT-Neo дал 576 скоров, Qwen — 501.
Для обоих детекторов калибровка и тест заново выполнены на пересечении 500 полных хэшей.
Разметка, группы и split на пересечении проверяются на равенство.
Поэтому числа GPT-Neo здесь отличаются от старого отчёта Sprint 2.
Platt обучен на train; isotonic по языкам — на validation; сравнение — на test.
Порог 0.5 не подбирался на test. Максимум 512, минимум 32 предсказываемых токена.

[Сгенерированный отчёт](../../reports/sprint3-llm/report.md) и
[полные метрики с bootstrap-интервалами](../../reports/sprint3-llm/metrics.json).
Qwen 0.5B не решил проблему RU: на общей тестовой части ECE ухудшилась,
AUC после калибровки осталась ниже 0.5. Это зафиксированное ограничение, а не улучшение.
Новый детектор нельзя рекомендовать для русской почты без дальнейших экспериментов.

Письма: предоставленный командой ZIP содержит 100 LLMGen (Apache-2.0) и 100 Nazario
(CC-BY-4.0). Seven-phishing и Enron не использовались. После MIME-разбора найдены
10 повторных тел: осталось 190 уникальных; ещё 5 слишком коротки для критерия.
Получено 185 скоров. Для писем применён EN-калибратор Qwen с HC3/CoAT без подгонки
на письмах; все 185 писем — независимая проверка переноса домена.
Значения вероятностей на почте не гарантированно калиброваны: это видно по ECE.
Все письма — фишинг; intent и перенос на легитимную почту здесь не оцениваются.
Метка human у Nazario — допущение исторического корпуса. Источник и возраст писем
связаны с классом, поэтому качество может отражать различия корпусов.

Проверены все 200 исходных .eml через `dw1.pipeline.analyze`: 195 карточек получили
`fusion.ai_provenance.components.model = fast-detectgpt`, включая повторные тексты.
Пять коротких писем сохранили прежний анализ. Исходные письма в Git не помещаются.
[Подтверждение интеграции без текстов](../../reports/sprint3-llm/integration_check.json).

## Воспроизведение

Ноутбук: [sprint3_qwen.ipynb](https://colab.research.google.com/github/spotyyyi123-ux/DW-1/blob/w1/ru-detector-email-scores/scripts/external/sprint3_qwen.ipynb).
Входной ZIP содержит ровно `llm_benchmark.jsonl`, `llm_benchmark_emails.jsonl`,
`gptneo_scores.parquet`; его нужно передать через ячейку загрузки, без публикации в Git.
Первый JSONL — сохранённые 600 текстов Sprint 2, второй собирается командой:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_email_llm_benchmark.py
```

Письма положить в `data/sprint2/bench_licensed/` вместе с `labels.csv`.
После Colab сохранить файлы выходного ZIP в `data/runs/sprint3/` и пересчитать отчёт:

```bash
make eval LLM=1 PY=.venv/bin/python
```

Для сквозного анализа письма:

```bash
export PYTHONPATH=src
export DW1_LLM_SCORES=data/runs/sprint3/email_scores.parquet
export DW1_LLM_CALIBRATION=data/runs/sprint3/email_calibration.json
.venv/bin/python -m dw1 data/sprint2/bench_licensed/llmgen_000.eml
```

Скоры ищутся по полному хэшу тела; заголовки не передаются детектору.
Калибратор привязан к SHA-256 таблицы скоров и версии модели.
Для перенесённого калибратора сохранён хэш исходных скоров HC3/CoAT и scope переноса.
На незнакомых текстах LLM-сигнал отсутствует; intent вычисляется отдельно.

## Артефакты

В Git: код, чистый ноутбук, отчёт, метрики, манифесты и проверка интеграции.
Parquet, исходные тексты, .eml, ZIP и веса — только локальные артефакты.
Для Сергея в архиве результатов: `gptneo_scores.parquet` (576), `qwen_scores.parquet` (501),
`email_scores.parquet` (185), `email_calibration.json`, тестовые прогнозы и отчёт.
Прогон выполнен кодом `a8547e4`; точный SHA скрипта входит в model_version манифестов.
Последующие изменения добавляют проверки входа и описание запуска, формулу не меняют.
