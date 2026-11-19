# Sprint 4 · RU-детектор v2

Эксперимент использует Fast-DetectGPT с [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B).
Ревизия модели `8faed761d45a263340a0528343f099c05c9a4323`, лицензия Apache-2.0.
На Colab передаётся русская часть публичного [CoAT](https://huggingface.co/datasets/RussianNLP/coat)
(Apache-2.0) из бенчмарка Sprint 2. Базовые скоры GPT-Neo 2.7B и Qwen2.5-0.5B
берутся из артефактов Sprint 3.

Сравнение ограничено пересечением хэшей всех трёх моделей. Для каждого хэша проверяется
совпадение языка, класса, группы, набора данных и сплита. Для всех трёх моделей отдельно
обучаются Platt на RU train и isotonic на RU validation. В таблице приводятся метрики
на общем RU test. Отбор по наличию скоров ограничивает сравнение текстами, прошедшими
минимум токенов у всех детекторов. AUC сравнивается после калибровки. F1 считается
при фиксированном пороге 0.5, ECE использует 10 интервалов.

Критерий продолжения на письмах — AUC новой модели выше обеих базовых моделей и 0.5.
ECE и F1 показаны отдельно, поскольку измеряют другие свойства.

Прогон завершён на Tesla T4. У Qwen2.5-1.5B AUC после калибровки выше, чем у двух
базовых моделей, но ниже 0.5. Качество на RU остаётся ограничением, скоры писем сохранены
из Sprint 3. Следующий эксперимент проверит калибровку с направлением, выбранным на train,
на новой отложенной выборке. [Таблица метрик](../../reports/sprint4-ru/report.md)

## Воспроизведение

1. Из `data/sprint2/llm_benchmark.jsonl` взять строки с `dataset == "coat"` и
   `language == "ru"`, сохранив их в `coat_ru.jsonl` без изменения сплитов и хэшей.
2. Открыть [ноутбук Colab](https://colab.research.google.com/github/spotyyyi123-ux/DW-1/blob/w1/sprint4-ru-detector/scripts/external/sprint4_qwen_ru.ipynb),
   выбрать T4 GPU и загрузить `coat_ru.jsonl`. Используются первые 512 токенов.
3. Положить скачанные `qwen15_ru_scores.parquet` и манифест в
   `data/runs/sprint4/ru-detector/`. Базовые скоры должны лежать в
   `data/runs/sprint3/gptneo_scores.parquet` и `data/runs/sprint3/qwen_scores.parquet`.
4. Запустить `make eval LLM=4 PY=.venv/bin/python`. Скрипт сохранит калибратор,
   предсказания, метрики и отчёт в `data/runs/sprint4/ru-detector/`.
