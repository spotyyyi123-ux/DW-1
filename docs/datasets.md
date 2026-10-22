# Датасеты, которые мы используем

Всё скачиваем в `D:\dw1\data\raw\<набор>\` и записываем строку в `data\datasheet.csv`
(что, откуда, лицензия). В git данные не кладём.

## Тексты: человек vs ИИ

| Набор | Что даёт | Ссылка | Объём | Лицензия |
|---|---|---|---|---|
| **LLMTrace_classification** | human vs AI, **EN + RU**, свежий корпус 2025 | [HF](https://huggingface.co/datasets/iitolstykh/LLMTrace_classification) | ~3 ГБ | Apache-2.0 |
| **CoAT** | human vs AI, RU, 13 генераторов | [HF](https://huggingface.co/datasets/RussianNLP/coat) | 75 МБ | Apache-2.0 |
| **HC3 (EN)** | ChatGPT vs human, EN | [HF](https://huggingface.co/datasets/Hello-SimpleAI/HC3) | 74 МБ | CC-BY-SA-4.0 |
| **SemEval-2024 Task 8** (зеркало d0rj) | human vs AI, EN+RU и др. языки, много генераторов | [HF](https://huggingface.co/datasets/d0rj/SemEval2024-task8) | ~300–400 МБ | Apache-2.0 (зеркало) |
| **roft** (Real or Fake Text) | разметка, где текст переходит с человеческого на машинный — под класс «ИИ-причёсано» | [HF](https://huggingface.co/datasets/liamdugan/roft) | 62 МБ | MIT |
| **SeqXGPT** | разметка по предложениям: какое предложение сгенерировано ИИ | [HF](https://huggingface.co/datasets/TheItCrOw/SeqXGPT) | 33 МБ | не указана — уточнить |

## Реальные письма: фишинг, спам и легитимная почта

| Набор | Что даёт | Ссылка | Объём | Лицензия |
|---|---|---|---|---|
| **seven-phishing** | человеческий фишинг/спам + легитимные письма (7 корпусов) | [HF](https://huggingface.co/datasets/puyang2025/seven-phishing-email-datasets) | 230 МБ | «other» (на проверке) |
| **Nazario** | реальные фишинговые письма (годы 2023–2025) | [monkey.org](https://monkey.org/~jose/phishing/) | ~30 МБ | CC-BY-4.0 |
| **Enron** | легитимная корпоративная почта — сложные false positives | [CMU](https://www.cs.cmu.edu/~enron/) (или [SetFit/enron_spam](https://huggingface.co/datasets/SetFit/enron_spam), 160 МБ) | 1.7 ГБ | без явной лицензии (на проверке) |

## AI-фишинг

| Набор | Что даёт | Ссылка | Объём | Лицензия |
|---|---|---|---|---|
| **LLMGen-Phishing** | письма, сгенерированные LLM (EN/ZH) — наш позитивный класс | [HF](https://huggingface.co/datasets/Dizzzy0x00/LLMGen-Phishing-Email-Dataset) | 5 МБ | Apache-2.0 |

## Изображения: реальные vs ИИ

| Набор | Что даёт | Ссылка | Объём | Лицензия |
|---|---|---|---|---|
| **CIFAKE** | быстрый старт: 32×32 real/AI | [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | 106 МБ | MIT |
| **Defactify_Image** | 96k реальных/AI изображений 256+ — основной бенчмарк image-ветки | [HF](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset) | 7.5 ГБ | не указана — уточнить |

## Состязательные (проверка устойчивости к маскировке)

| Набор | Что даёт | Ссылка | Объём | Лицензия |
|---|---|---|---|---|
| **RAID** (test.csv) | human-vs-AI с атаками (перефраз и др.) — тест на устойчивость | [HF](https://huggingface.co/datasets/liamdugan/raid) | 1.2 ГБ | MIT |
| **NTIRE-RobustAI** (1 шард) | устойчивость детектора изображений к сжатию/перезаписи — только в Sprint 2+ | [HF](https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train) | ~19 ГБ (1 шард) | не указана |

## Приоритет загрузки (задача 1)

1. LLMTrace → seven-phishing → CoAT → HC3 → LLMGen → Nazario
2. roft → SeqXGPT → SemEval-8 → CIFAKE
3. Defactify → Enron/RAID (когда дойдём до image- и легит-веток)

Правило: если лицензия «не указана» — сначала уточнить (у оригинального источника),
потом качать; в datasheet писать реальную лицензию.
