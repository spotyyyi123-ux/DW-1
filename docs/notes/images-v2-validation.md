# Images v2 и валидация RU / визуальных сигналов

Срез 2026-11-20. Это методика и артефакты проверки, не финальный отчёт спринта.
По уточнению пользователя сайт, защита и тег m5 в объём работы не входят.

## Детектор изображений

Используется выбранная ветка **эмбеддингов**, не watermark: ResNet18-v2,
512 признаков после global pooling, 16 радиальных FFT-признаков и 64 бита pHash.
Классификатор StandardScaler + LogisticRegression обучен на clean train;
отдельный Platt — только на clean validation. Порог 0.5 и параметры остаются
фиксированными для всех атак test. Измерительные признаки сами по себе не
получают метку ИИ. Intent изображения всегда unknown; оси не смешиваются.

Прогон выполнен локально на CPU через OpenCV DNN / ONNX, без torch/transformers
и без отправки писем во внешний контур. 44.7-MB backbone скачивается отдельно
с проверкой SHA-256, во время API-анализа сеть не используется.

Источник: [авторский CIFAKE](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images),
revision `e112a942abaecd02b6b1f6f646c807d56be8fb62`, MIT по README автора.
ResNet: [ONNX Model Zoo](https://huggingface.co/onnxmodelzoo/resnet18-v2-7),
revision `0b071fb1cd97fc8f111036e48844b63dc480d5a2`, Apache-2.0.
Источники, лицензии и хэши зарегистрированы в `data/datasheet.csv` / manifest.

Поднабор: 1 200 изображений 32×32, поровну AI/real. Сначала фиксируются первые
1 000 файлов, доступных через GitHub contents API каждой исходной папки
train/test и класса. Из этого ограниченного списка выполняется выборка с
seed 20261120: 300 train + 100 validation и 200 test на класс. Используются
разные исходные train/test-папки. Всего 600 train / 200 validation / 400 test.
Точные повторы пикселей исключены глобально; поиск близких повторов не проводился.
Это не случайная выборка из полного CIFAKE и не проверка на новых генераторах.

Реальные значения и таблица устойчивости находятся в
`data/runs/sprint5/images/metrics.json`, `robustness.csv`, `report.md` и
`predictions.parquet` (4 000 прогнозов, без исходных изображений).
Все метрики получаются повторно через `make eval IMAGES_V2=1`.
Исходный test и семь искажений оцениваются на тех же 400 ID: JPEG 90/70/50/30,
ресайз 50/25%, центральный crop с сохранением 80% каждой стороны.
Для кропа сохраняется 26×26 пикселей из-за округления границ.
`model.json` связывает веса, encoder, split, dataset manifest и prediction hash.
`integration_check.json` проверяет совпадение runtime с сохранёнными прогнозами.

## Валидация QR и RU

`make eval VISUAL_V2=1` / `scripts/validate_sprint5.py` создают
`data/runs/sprint5/validation/`:

- `visual_predictions.parquet`: 5 созданных для проекта QR × 21 условие.
  Успех — полное совпадение декодированного URL с заранее заданным эталоном,
  а не просто наличие какой-либо строки. Сохраняются статусы, размеры,
  pHash distance и decoder version. `visual_metrics.json` содержит общий
  результат и отдельные доли для каждого искажения.
- Это **baseline текущего декодера**, без реализации новой задачи QR v2.
  Синтетические single-QR не доказывают качество на реальных вложениях,
  нескольких QR или новых устройствах. Старые 19/21 относятся к одному
  исходному QR, поэтому их нельзя трактовать как парное сравнение с новой
  выборкой из пяти QR.
- `ru_source.json` и `ru_validation.json`: показатели коллеги из
  `origin/w1/sprint5-ru-calibration`, фиксируется точный commit. Проверены
  структура, границы значений, общая исходная Platt-таблица.
  **Статус `reported_not_recomputed`**: исходные RU-скоры отсутствуют,
  по согласованию с пользователем метрики взяты из отчёта. Они не представлены
  как независимо воспроизведённые. Нельзя перепроверить подбор калибратора,
  утечки, исключения объектов и интервалы без исходных файлов.
- `validation.json` объединяет эти статусы и измеренный Images v2.

RU: из 600 входных текстов получены 467 скоров, test 158. В отчёте коллеги
train-direction isotonic даёт AUC 0.5968, F1 0.5147, ECE 0.0581;
Platt — AUC 0.5900, F1 0.6129, ECE 0.0251. Значит направление повышает AUC
относительно Platt, но не F1/ECE. Это CoAT-тексты, не русская почта.
Сравнение на общих объектах GPT-Neo / Qwen / новой модели без object-level
файлов не воспроизведено; отчёт одного детектора не заменяет это сравнение.

## Команды

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python scripts/fetch_images_v2.py
python scripts/images_v2.py --dataset data/raw/images-v2/dataset.json `
  --encoder data/raw/images-v2/resnet18-v2-7.onnx
make eval IMAGES_V2=1
make eval VISUAL_V2=1
# Без make:
python scripts/images_v2.py --predictions data/runs/sprint5/images/predictions.parquet
python scripts/validate_sprint5.py
```

Для элемента изображения в анализаторе:

```powershell
$env:DW1_IMAGE_MODEL = (Resolve-Path data/runs/sprint5/images/model.npz)
$env:DW1_IMAGE_ENCODER = (Resolve-Path data/raw/images-v2/resnet18-v2-7.onnx)
python -m dw1 path/to/email.eml
```

Без этих артефактов, при ошибке или несовпадении хэшей сохраняется unknown;
результат измерений не выдаётся за результат модели. Оценки выдаются отдельно
для каждой картинки. Сводка images использует максимальный p(AI), не вероятность
AI-происхождения всего письма; модель не добавлена в веса текстовой фузии.
Watermark не проверялся: отсутствие метки ничего не доказывает. Проверка на
CIFAKE не означает валидность для произвольного размера и почтового домена.

В Git не публикуются исходные картинки или письма, 44-MB encoder и feature cache.
Публикуются малые веса классификатора, manifest с хэшами и source ID, прогнозы
без пикселей и результаты валидации. Показатели тестовых фикстур не включены.
