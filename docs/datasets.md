# Аудит датасетов для бенчмарка DW-1

Дата аудита: 2026-10-15. Метод: живые проверки источников (API HuggingFace/Kaggle, GitHub, Zenodo,
страницы корпусов). Всё неподтверждённое честно помечено «не найдено».
**Ограничения среды:** свободно 1.29 ГБ на диске, GPU нет → целевой бюджет загрузки ≤ ~950 МБ.

## 1. Сводная таблица

| Название | Источник | Лицензия | Размер | Языки | Содержимое | Пригодность |
|---|---|---|---|---|---|---|
| SemEval-2024 Task 8 | Офиц. GitHub mbzuai-nlp/SemEval2024-task8 — HTTP 451; зеркало [d0rj/SemEval2024-task8](https://huggingface.co/datasets/d0rj/SemEval2024-task8) | зеркало: Apache-2.0 (оригинал не верифицирован) | ~665 МБ (subtask A_mono 239 МБ, A_multi 307 МБ, B 107 МБ, C 12 МБ) | EN, AR, RU, ZH, ID, UR, BG, DE | human vs AI (ChatGPT, GPT-4, davinci, cohere, bloomz, dolly…); B — атрибуция генератора; C — human-machine mixed | (а) ✅ ядро human-vs-AI EN+RU; частичная загрузка по parquet |
| M4 / M4GT-Bench | [github.com/mbzuai-nlp/M4](https://github.com/mbzuai-nlp/M4), [M4GT-Bench](https://github.com/mbzuai-nlp/M4GT-Bench) | не указана | не найдено (Google Drive) | EN+multi | то же семейство, что SemEval-8 | дублирует SemEval-8 — не брать |
| HC3 EN | [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) | CC-BY-SA-4.0 | ~147 МБ (all 74 МБ, finance 10 МБ) | EN | ответы человека vs ChatGPT (QA, вкл. finance/medicine) | (а) ✅ EN human-vs-AI; частичная по доменам |
| RAID | [liamdugan/raid](https://huggingface.co/datasets/liamdugan/raid) | MIT | 21.6 ГБ (test.csv 1.22 ГБ) | EN, CS, DE | 10M+ документов, 11 LLM, 12 атак | (а) ✅ эталон; test.csv 1.22 ГБ ≈ весь бюджет → вторым заходом/streaming |
| RuATD 2022 | Kaggle (конкурс закрыт), [github.com/dialogue-evaluation/RuATD](https://github.com/dialogue-evaluation/RuATD) | не найдено | не подтверждён | RU | human vs AI (ruGPT3, OPUS-MT, парафразеры) | заменён на CoAT |
| CoAT | [RussianNLP/coat](https://huggingface.co/datasets/RussianNLP/coat) | Apache-2.0 | ~150 МБ (download 75 МБ, 246k текстов) | RU | human + 13 генераторов, бинарная + атрибуция | (а) ✅ рекомендованная замена RuATD |
| MULTITuDE | [Zenodo 10013755](https://zenodo.org/records/10013755) | CC BY 4.0, но файлы restricted | 45.2 ГБ | 11 языков (RU 11 578) | 74 081 новостных текстов, 8 LLM | ⚠️ research-only по запросу → для индустрии непригодно |
| Enron | [cs.cmu.edu/~enron/](https://www.cs.cmu.edu/~enron/) | не указана (FERC, просьба уважать приватность) | 1.7 ГБ tar.gz | EN | реальная корпоративная почта (0.5M писем) | (в) ✅ эталон legit-почты; полный архив вне бюджета → замена SetFit/enron_spam 160 МБ или Enron-часть в агрегате |
| SpamAssassin public corpus | [spamassassin.apache.org/old/publiccorpus/](https://spamassassin.apache.org/old/publiccorpus/) | не указана (копирайт отправителей; корпус для тестов) | ~13 МБ (9 tar.bz2), 6 047 писем | EN | реальные spam + easy/hard ham 2002–2005 с заголовками | (б) ✅ классический spam/ham |
| CEAS 2008 | официальный сайт **мёртв (404)** | не найдено | ~39 154 письма (в агрегатах) | EN | spam + ham конкурса CEAS-2008 | (б) только через агрегаты |
| TREC 2007 Spam | официальный сайт **мёртв (404)**; Kaggle-копия | «© Original Authors» (не свободная) | 338 МБ CSV, 75 419 писем | EN | spam/ham 2007 | (б) ⚠️ лицензия; или subset в puyang2025 |
| Nazario phishing corpus | [monkey.org/~jose/phishing/](https://monkey.org/~jose/phishing/) (жив) | CC-BY-4.0 (README, коммерческое использование с атрибуцией) | ~150 МБ (отдельные mbox по годам, 3–31 МБ) | EN | реальные фишинговые письма 2005–2025, ручная классификация | (б) ✅✅ эталон человеческого фишинга; загрузка по годам |
| PhishTank | [phishtank.org](https://phishtank.org) | n/a | — | — | только URL, не письма; открытого дампа нет | для email-задачи непригоден |
| GenImage | [github](https://github.com/GenImage-Dataset/GenImage) | не указана; real = ImageNet (некоммерческая) | 100+ ГБ | изображения | real vs Midjourney/SD/ADM/GLIDE/Wukong/VQDM/BigGAN | ⚠️ лицензия и объём — вне бюджета |
| CIFAKE | [Kaggle birdy654/cifake](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images), HF yanbax | MIT | 106 МБ, 120k изображений 32×32 | изображения | 60k real (CIFAR-10) vs 60k SD 1.4 | ✅ идеален для старта real/AI |

## 2. Датасеты именно фишинговых писем (включая LLM-генерированные)

| Название | Источник | Лицензия | Размер | Содержимое |
|---|---|---|---|---|
| Dizzzy0x00/LLMGen-Phishing-Email-Dataset | [HF](https://huggingface.co/datasets/Dizzzy0x00/LLMGen-Phishing-Email-Dataset) | Apache-2.0 | ~5 МБ CSV | **LLM-генерированные** phishing/legit: EN (GPT), ZH (DeepSeek) — прямое попадание в задачу |
| zefang-liu/phishing-email-dataset | [HF](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) | LGPL-3.0 | 52 МБ, ~18 650 писем | человеческие phishing vs safe (PhishTank/Nazario + реальные) |
| puyang2025/seven-phishing-email-datasets | [HF](https://huggingface.co/datasets/puyang2025/seven-phishing-email-datasets) | **other** (микс 7 корпусов) | 230 МБ parquet, 203 017 писем | TREC-05/06/07, CEAS-08, Enron, Ling-Spam, SpamAssassin; есть subject/sender/date |
| it4lia/PhishingEmailCuratedDatasets_Cleaned | [HF](https://huggingface.co/datasets/it4lia/PhishingEmailCuratedDatasets_Cleaned) | CC-BY-4.0 | ~93 МБ | очищенная агрегация 11 корпусов 1995–2022 (исходник Zenodo, Champa et al. 2024) |
| luongnv89/phishing-email | [HF](https://huggingface.co/datasets/luongnv89/phishing-email) | MIT | ~2 МБ | инструкционные диалоги по CEAS-08 (для finetune, не бенчмарк) |
| darkknight25/phishing_benign_email_dataset | [HF](https://huggingface.co/datasets/darkknight25/phishing_benign_email_dataset) | MIT | <1K примеров | subject/body/intent/technique — мал |
| ealvaradob/phishing-dataset | [HF](https://huggingface.co/datasets/ealvaradob/phishing-dataset) | Apache-2.0 | ~13.4 ГБ | phishing text/URL/HTML — вне бюджета |

## 3. Красные флаги (юридический чек-лист перед индустриальным использованием)

1. **CEAS 2008** — официальный сайт мёртв (404); оригинал только через агрегаты.
2. **TREC 2007** — официальный сайт мёртв (404); Kaggle-копия «© Original Authors» — не свободная.
3. **SemEval-2024 Task 8** — официальный GitHub отдаёт HTTP 451; зеркало d0rj работает, но оригинальная лицензия не верифицирована.
4. **MULTITuDE** — research-only, запрос с академической почты, запрет перепубликации → для индустрии непригодно.
5. **PhishTank** — открытого дампа нет, хранит только URL, не письма.
6. **GenImage** — лицензии нет, real-часть = ImageNet (некоммерческая), 100+ ГБ.
7. **Enron / SpamAssassin / RuATD / M4GT / M4** — формальные лицензии не указаны → нужен юр. риск-анализ.
8. **RAID** — MIT, но test.csv 1.22 ГБ ≈ весь бюджет.
9. **puyang2025/seven** — license «other»: микс 7 корпусов без единой лицензии → юр. проверка; замена — it4lia (CC-BY-4.0).

## 4. Рекомендация: стартовый набор (обновлено: диск D — 138.6 ГБ свободно, бюджет C: снят)

| # | Датасет | Роль в бенчмарке | Объём |
|---|---|---|---|
| 1 | CoAT (RussianNLP/coat) | AI vs human **RU**, 13 генераторов | 75 МБ |
| 2 | SemEval-2024 Task 8 (зеркало d0rj, subtaskA_multilingual train + test) | AI vs human **EN+RU**+6 языков, мультигенератор | ~400 МБ |
| 3 | HC3 EN (all.jsonl) | ChatGPT vs human **EN** (QA, вкл. finance) | 74 МБ |
| 4 | puyang2025/seven (или it4lia CC-BY-4.0) | **человеческие phishing/spam + ham** (TREC/CEAS/Enron/Ling/SpamAssassin) | 230 МБ (или 93 МБ) |
| 5 | Nazario (свежие годы: 2023–2025) | **реальные фишинговые письма**, CC-BY-4.0 | ~30 МБ |
| 6 | Dizzzy0x00/LLMGen-Phishing-Email-Dataset | **LLM-генерированный фишинг** EN/ZH | 5 МБ |
| 7 | CIFAKE | изображения real/AI | 106 МБ |
| 8 | (опц.) SetFit/enron_spam | **легитимная корпоративная почта** Enron spam/ham | 160 МБ |

Покрытие ролей: AI-тексты RU+EN ✅ · человеческий фишинг ✅ · легит-корпоративная почта ✅
(Enron внутри п.4 и/или п.8) · изображения real/AI ✅ · LLM-фишинг ✅.
**Дополняется из нового бюджета (см. SPEC §5):** полный Enron (1.7 ГБ), RAID test.csv (1.22 ГБ)
и Nazario целиком — берём сразу; GenImage одним генератором (опц., десятки ГБ).
**Вне бюджета/доступа:** MULTITuDE (research-only), полный GenImage (100+ ГБ), TREC07-оригинал (мёртв).

## 5. Принципы обращения (из плана проекта)

- Внутренние образцы Doctor Web — только локальная обработка, не покидают контур; облачные API — только на публичных данных.
- Перед каждым скачиванием — фиксировать источник, лицензию и хэш в `data/datasheet.csv`.
- Сплиты строить по шаблонам/кампаниям, а не по строкам (защита от утечки через near-duplicates).
- Итог: для каждого набора — явная пометка «публичный/внутренний», «лицензия проверена/на юр. проверке».
