# Sprint 3, задача 3 — эмбеддинги и BCubed F1

Статус: **в работе; ожидаются реальные исходные данные и ручная разметка**.
Итоговая метрика не измерена. Код подготовки/оценки и Colab-ноутбук подготовлены.

## Что найдено при проверке 2026-11-06

В origin/main отсутствуют исходные письма, полное разбиение кампаний, NPZ-эмбеддинги
и ручная gold-разметка. Путь D:/dw1 на этой машине отсутствует. Сведения о прежних
кампаниях есть в отчёте и datasheet, но восстановить тексты по ним нельзя.
Seven-phishing и Enron помечены как неразрешённые для внешнего прогона в docs/datasets.md.
Доступность корпуса в интернете не означает разрешение отправлять его в Colab.

Число BCubed не подставлено. Автоматические тестовые примеры не считаются ручной
разметкой, а тестовые векторы — результатами sentence-transformers.

## Что подготовлено

- `python -m dw1.campaigns.review`: принимает реальное полное разбиение и JSONL текстов,
  поддерживает legacy members=[{id,...}] и v2 members=[id,...]. Выбирает ровно топ-50
  по размеру с детерминированным порядком. Проверяет непересечение кластеров,
  полноту текстов, размеры и разрешение внешнего экспорта.
- Пакет review/: manifest.json с ID/хешами, замороженные campaigns.json,
  documents.jsonl, пустой annotation.jsonl, review.html, external_bundle.zip.
  Все участники выбранных кластеров включены без усечения и случайного подсэмплирования.
- `scripts/external/campaign_embeddings_colab.ipynb`: установка только на Colab,
  загрузка bundle, фиксация commit SHA модели, запуск экспортера, скачивание NPZ.
  В metadata — модель, ревизия, версии, ограничение длины и ID пакета.
  **На реальных данных ноутбук ещё не запускался.** Локально torch не установлен.
- review.html: локальное заполнение, поиск, навигация, импорт и сохранение JSONL.
  Имена истинных групп глобальны для всей выборки. Кнопки не назначают gold автоматически.
- `make eval REVIEW_PACKAGE=...`: требует 2–3 самостоятельные разметки, проверяет
  полное покрытие и неизменность текстов. Сравнивает разбиения независимо от названий
  групп; при расхождении требует явного human adjudication.
- Метрики исходного разбиения и LSH+HDBSCAN считаются по одной зафиксированной
  популяции. HDBSCAN повторяется на этой выборке; результат не объявляется оценкой
  всего корпуса. Синглтоны/шум учитываются в BCubed.
- Успешный прогон сохраняет predictions.parquet, metrics.json, campaigns_v2.json,
  копии входов в inputs/ и генерирует отчёт в docs/notes/ из сохранённых предсказаний.

## Какие файлы нужны от Сергея/владельца данных

1. Полные `campaigns.json` или `clusters.json` с ID участников каждого кластера.
   Индекса с одним representative_text на кластер недостаточно для оценки.
2. `documents.jsonl`: по строке `{"id":"тот же ID","text":"исходный subject + body"}`.
   Нужны все тексты участников кластеров; ID должны совпадать с кластеризацией.
3. `provenance.json` с источником, подтверждённой лицензией и подтверждением,
   что этот публичный набор разрешён для внешнего контура. Для mixed-корпуса
   семь-phishing требуется решение владельца данных; разрешение не подразумевается.

Образец provenance (до проверки external_allowed остаётся false):

```json
{
  "dataset": "НАЗВАНИЕ РЕАЛЬНОГО КОРПУСА",
  "source": "ССЫЛКА ИЛИ ВЕРСИЯ ИСТОЧНИКА",
  "license": "ПОДТВЕРЖДЁННАЯ ЛИЦЕНЗИЯ",
  "approved_by": "КТО ПОДТВЕРДИЛ ВНЕШНИЙ ЭКСПОРТ",
  "external_allowed": false
}
```

## Запуск на реальных данных

```powershell
python -m dw1.campaigns.review --campaigns clusters.json --documents documents.jsonl --provenance provenance.json --out data/runs/sprint3/campaigns/review
```

Открыть Colab-ноутбук и загрузить созданный external_bundle.zip. Вернувшийся
embeddings.npz сохранить в review/. Параллельно 2–3 человека независимо открывают
review.html и сохраняют каждый свой файл. Исходные письма не загружать в GitHub.

Для одного шаблона назначать одинаковый gold_group даже в разных кластерах.
Если внутри кластера разные шаблоны — разделить их по gold_group. Метка near-dup /
different сама по себе недостаточна для BCubed: нужны группы каждого письма.
После обсуждения разногласий сохранить итоговый annotations-final.jsonl.

```powershell
make eval PY=.venv/Scripts/python.exe REVIEW_PACKAGE=data/runs/sprint3/campaigns/review EMBEDDINGS=data/runs/sprint3/campaigns/review/embeddings.npz ANNOTATIONS=annotations-final.jsonl REVIEWERS="reviewer-a.jsonl reviewer-b.jsonl" ADJUDICATED_BY="имя согласовавшего"
```

Без make:

```powershell
python -m dw1.eval.campaign_review --package data/runs/sprint3/campaigns/review --embeddings data/runs/sprint3/campaigns/review/embeddings.npz --annotations annotations-final.jsonl --reviewers reviewer-a.jsonl reviewer-b.jsonl --adjudicated-by "имя согласовавшего"
```

Разметка с пропущенными письмами, повторами, другим ID пакета или изменёнными
текстами отклоняется. Имеющийся eval-каталог не перезаписывается; для нового
прогона задавать новый --out / EVAL_OUT.

## Что осталось для критерия готовности

- Получить реальную выгрузку и разрешённый для внешнего контура корпус.
- Фактически выполнить sentence-transformers на внешнем контуре и вернуть NPZ.
- Получить ручные разметки и разрешить расхождения.
- Запустить make eval и приложить сгенерированный результат.

До этих шагов статус не переводится в «выполнено», BCubed F1 остаётся null.
