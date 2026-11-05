# Сравнение детекторов на письмах (Sprint 3, задача 4)

Цель: сравнить две ветки определения происхождения на **реальных письмах**:
нашу TF-IDF-модель и Fast-DetectGPT (LLM-детектор).

## Данные

Лицензионный subset (только данные с подтверждёнными лицензиями):

| Набор | Роль | Метка | Лицензия |
|---|---|---|---|
| LLMGen-Phishing (GPT) | AI-письма | `ai_generated` (1) | Apache-2.0 |
| Nazario phishing 2023–2025 | человеческие письма | `human` (0) | CC-BY-4.0 |

- 200 писем; email-бенчмарк: `data/sprint2/llm_benchmark_emails.jsonl`
  (текст = тело письма; сплиты по хэшу файла 70/15/15: train 141 / validation 41 / test 18).
- seven-phishing и enron сознательно **не** включены: лицензии «на проверке» / приватность.

## Метод

1. Для каждого детектора берётся его score: TF-IDF — `p_ai` модели; Fast-DetectGPT — criterion.
2. **Platt-калибровка** на train (`LogisticRegression` по score → P(AI)).
3. Метрики на test: ROC-AUC, F1@0.5, ECE (10 бинов). Реализация:
   `scripts/compare_detectors.py`.

## Текущие результаты (TF-IDF baseline)

| Детектор | Язык | n test | ROC-AUC | F1 | ECE |
|---|---|---|---|---|---|
| TF-IDF | en | 18 | 1.000 | 1.000 | 0.098 |

Команда:

```powershell
python scripts/run_tfidf_emails.py
python scripts/compare_detectors.py --scores "tfidf=data/runs/sprint2/tfidf_email_scores.parquet"
```

**Честные оговорки:** test всего 18 писем; классы разделены и по домену (LLMGen vs Nazario),
поэтому AUC 1.0 — оптимистичная верхняя оценка, а не качество в «дикой» почте.

## Ожидаемая строка Fast-DetectGPT

Когда Алина пришлёт скоры по письмам (`llm_scores.parquet`, формат совместим с
`llm_detectors.validate_scores`), таблица дополняется:

```powershell
python scripts/compare_detectors.py `
  --scores "tfidf=data/runs/sprint2/tfidf_email_scores.parquet" `
  --scores "fastdetectgpt=path/to/llm_scores_emails.parquet"
```

Ожидание: Fast-DetectGPT даёт менее «идеальный» результат из-за доменного сдвига,
но проверяет ту же ось независимым методом (вероятностная кривизна).

## Следующий шаг

1. Получить скоры Fast-DetectGPT для писем (задача 1, Алина).
2. Прогнать сравнение, зафиксировать таблицу в `docs/sprint3-report.md`.
3. Проверить карточку письма: `components.model = "fast-detectgpt"` при наличии скоров.
