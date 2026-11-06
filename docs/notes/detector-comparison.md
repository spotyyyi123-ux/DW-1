# Сравнение детекторов на письмах (Sprint 3, задача 4)

Итоговая таблица: TF-IDF-модель против Fast-DetectGPT (Qwen 2.5-0.5B) на **одних и тех же**
письмах и сплитах.

## Данные

- Лицензионный subset: **LLMGen (Apache-2.0, AI-письма) + Nazario (CC-BY-4.0, человеческие)**;
  200 писем, после удаления повторных тел — 185.
- Скоры Fast-DetectGPT по письмам: `data/runs/sprint3/alina_email_scores.parquet`
  (Алина; Qwen 2.5-0.5B, перенос шкалы с HC3/CoAT).
- Сплиты (по телу письма): train 131 / validation 31 / **test 23**; метки: 100 AI / 85 human.
- TF-IDF-скоры выровнены на те же строки: `data/runs/sprint3/tfidf_email_scores_aligned.parquet`
  (`scripts/align_tfidf_scores.py`).

## Метод

Для каждого детектора: Platt (логистическая регрессия по score) на train → вероятность на test;
метрики: ROC-AUC, F1@0.5, ECE (10 бинов). Реализация — `scripts/compare_detectors.py`.

## Результат (test = 23 письма, одни и те же строки)

| Детектор | Язык | n test | ROC-AUC | F1 | ECE |
|---|---|---|---|---|---|
| Fast-DetectGPT (Qwen 2.5-0.5B) | en | 23 | 0.992 | 0.966 | 0.097 |
| TF-IDF (наша модель) | en | 23 | 1.000 | 1.000 | 0.107 |

**Вывод:** на этом домене обе ветки разделяют классы почти идеально; TF-IDF даже чуть точнее,
Fast-DetectGPT — с меньшей ECE после калибровки. Это **не** доказательство превосходства:
классы разделены и по источнику (LLMGen vs Nazario), тест маленький (23 письма, все фишинг),
а Fast-DetectGPT переносит шкалу с HC3/CoAT. Настоящая проверка — на реальной почте с
неизвестным распределением (вне Sprint 3).

## Воспроизведение

```powershell
python scripts/run_tfidf_emails.py                     # TF-IDF на 200-письменном бенчмарке
python scripts/align_tfidf_scores.py --reference data/runs/sprint3/alina_email_scores.parquet `
  --out data/runs/sprint3/tfidf_email_scores_aligned.parquet
python scripts/compare_detectors.py `
  --scores "fastdetectgpt=data/runs/sprint3/alina_email_scores.parquet" `
  --scores "tfidf=data/runs/sprint3/tfidf_email_scores_aligned.parquet"
```

Сырые письма и архив разметки в Git не хранятся; в репозитории — только отчёт и метаданные,
архив передаётся напрямую (мессенджер/облако).
