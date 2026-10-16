# SPEC.md — Техническое задание DW-1

> Единый источник истины по «что реализовать». Для агентов — краткая версия в `AGENTS.md`.
> Владелец: наставник («мозг»). Изменения — только через MR.

## 1. Задача

Разработать систему детекции писем и кампаний с признаками генеративного ИИ с прикладным
финалом — **explainable spam-enrichment** для аналитиков SOC: система объясняет, какие элементы
письма могли быть сгенерированы/модифицированы ИИ, и насколько она уверена.

Источники сигналов: темы, тела, CTA, HTML, изображения, QR-коды, ссылки, заголовки, шаблоны кампаний.

**Не-цели (то, что НЕ делаем):**
- не бинарный детектор «текст написан ИИ / человеком» как конечный продукт;
- не замена почтового антиспама (мы enrichment поверх вердикта антиспама);
- не генератор фишинга (офенсив-эксперименты — только синтетика для оценки, см. §11).

## 2. Центральный тезис и две оси

> **Происхождение ≠ злонамеренность.** Маркетинговый дайджест и HR-рассылка тоже написаны ИИ
> и тоже шаблонны. Поэтому две независимые оси:

| Ось | Значения | Что измеряет |
|---|---|---|
| `ai_provenance` | human · ai_generated · ai_polished · template · unknown | кем создан элемент |
| `intent` | phishing · spam · legitimate · unknown | зачем письмо |

Плюс третья ось масштаба: **элемент → письмо → кампания (кластер)**.

`ai_polished` — обязательный класс: атакующий пишет сам, LLM «причёсывает». Тема — ИИ, тело —
гибрид. Бинарная модель на таком ломается; позлементная — работает.

## 3. Архитектура

```
ИНГЕСТ           .eml/.msg → MIME-разбор → MessageParts (нормализованные части письма)
ЭКСТРАКТОРЫ      9 модулей, каждый со своей версией и кэшем по content-hash
                 text · subject · cta · html · images · qr · links · headers · template
ГРАФ ДОКАЗАТЕЛЬСТВ  каждое наблюдение = (элемент, сигнал, значение, вес, model_version)
ФУЗИЯ            ai_provenance(письмо) ⊕ intent(письмо) — раздельно,
                 калибровка (isotonic), доверительные интервалы, отказ от ответа
КАМПАНИИ         MinHash/SimHash LSH + эмбеддинги + HDBSCAN → кластер + скелет шаблона
ОБЪЯСНЕНИЯ       атрибуция по элементам, подсветка спанов, faithfulness, контр-свидетельства
API              /v1/analyze · /v1/message/{id} · /v1/campaign/{id} · /v1/explain/{id}
UI (SOC)         карточка enrichment + граф кампании + фидбек
```

Инженерные обязательства:
1. Каждый экстрактор — независимый модуль с `model_version`. Без этого невозможны аблации.
2. Кэш по контент-хэшу части (одинаковый HTML в 10 письмах считается один раз).
3. Тяжёлые детекторы — асинхронно, по бюджету времени на письмо; Binoculars — только офлайн-эталон.
4. Все доказательства сохраняются (parquet/JSONB), а не только итоговый score: объяснение
   нельзя восстановить постфактум.

## 4. Контракты (интерфейсы между воркстримами)

### 4.1 Evidence JSON v1 — единственный формат обмена между экстракторами и фузией

```json
{
  "schema_version": "1.0",
  "message_id": "sha256(raw)[:16]",
  "ingest": {"received_at": "…", "raw_sha256": "…", "source": "dataset|internal"},
  "parts": {
    "headers": {"from": "…", "reply_to": "…", "return_path": "…", "spf": "pass|fail|none", "dkim": "…", "dmarc": "…", "x_mailer": "…", "date_tz": "…", "message_id_domain": "…"},
    "subject": {"text": "…"},
    "body_text": {"text": "…"},
    "body_html": {"dom_fingerprint": "…", "has_mso_hacks": false, "esp_markers": ["sendgrid"], "obfuscation": []},
    "cta": [{"text": "…", "href": "…", "imperative": "confirm", "deadline": "24h"}],
    "links": [{"url": "…", "anchor_text": "…", "anchor_mismatch": false, "domain": "…", "punycode": false, "tld": "…", "tracker_like": false}],
    "images": [{"phash": "…", "forensic_features": {}, "ocr_text": "…", "exif_present": false}],
    "qr": [{"decoded_url": "…", "screenshot_like": false}],
    "attachments": [{"name": "…", "mime": "…", "sha256": "…"}]
  },
  "signals": {
    "subject": {"ai_provenance": {"label": "ai|human|ai_polished|template|unknown", "confidence": 0.0, "evidence": [{"signal": "…", "value": "…", "weight": 0.0}], "model_version": "…"}},
    "body_text": {"ai_provenance": {…}},
    "html": {"ai_provenance": {…}, "intent": {…}},
    "images": [{"ai_provenance": {…}}],
    "links": {"intent": {…}},
    "headers": {"intent": {…}}
  },
  "fusion": {
    "ai_provenance": {"label": "…", "confidence": 0.0, "calibrated": true, "ci95": [0.0, 0.0]},
    "intent": {"label": "…", "confidence": 0.0, "calibrated": true},
    "rejection": false
  },
  "campaign": {"cluster_id": null, "template_skeleton": null, "member_count": 1},
  "explanations": {
    "top_factors": [{"element": "…", "signal": "…", "importance": 0.0, "direction": "+|-"}],
    "highlights": [{"element": "body_text", "span": [12, 40], "reason": "curvature_peak"}],
    "counter_evidence": ["dkim_aligned", "esp_tracker_present", "typos_present"],
    "faithfulness": {"comprehensiveness": 0.0, "sufficiency": 0.0}
  }
}
```

Правила схемы: `src/dw1/evidence.py` валидирует и версионирует; новое поле — минорная версия
схемы и миграция; удаление поля — только в мажорной версии.

### 4.2 Интерфейс экстрактора (единый для всех 9)

```python
class Extractor(Protocol):
    version: str
    def extract(self, parts: MessageParts, cache: EvidenceCache) -> dict: ...
    # возвращает фрагмент секции "signals" для своего элемента; ошибки не роняет пайплайн —
    # вместо этого метка "unknown" + запись в лог ошибок
```

### 4.3 Артефакты оценки (контракт с eval-харнессом)

- `data/runs/<run_id>/predictions.parquet` — обязательные колонки: `message_id, split, campaign_id,
  y_provenance, y_intent, p_provenance_*, p_intent_*, model_version, predicted_at`.
- `data/runs/<run_id>/metrics.json` — все метрики §8 с доверительными интервалами.
- Отчёты (`reports/`) генерируются **только** из этих артефактов.

## 5. Данные

Стартовый набор из `docs/datasets.md` — берём целиком, плюс с новым бюджетом добавляем:

| Дополнение | Роль | Объём |
|---|---|---|
| Полный Enron (1.7 ГБ) | легит-корпоративная почта, hard negatives | 1.7 ГБ |
| RAID test.csv + streaming train | adversarial human-vs-AI | 1.2 ГБ+ |
| Nazario полностью (2005–2025) | человеческий фишинг | ~150 МБ |
| GenImage один генератор (SD1.4) | форензика изображений | ~10–20 ГБ (опц.) |

Протокол загрузки: `scripts/fetch_<dataset>.py` → `data/raw/<dataset>/` → строка в
`data/datasheet.csv` (URL, лицензия, sha256, дата, кто скачал). Лицензионные флаги из
`docs/datasets.md` §3 остаются в силе: Enron/SpamAssassin/RuATD — на юр. проверку.

Сплиты: **по шаблонам/кампаниям, не по строкам** (защита от утечки near-duplicates).
Схема сплита фиксируется в `data/splits.yaml` и применяется харнессом.

## 6. Модули: что реализовать в каждом

### W1. Ингрест и данные
- MIME-разбор: `email` stdlib + `mailparser`-эквивалент для .msg; нормализация кодировок; HTML→текст.
- Загрузчики датасетов с валидацией полей; datasheet; splits.yaml; checksum-контроль.
- **Классический ML-бейзлайн**: TF-IDF (word+char 3-5 n-grams) + стилометрия (burstiness, MTLD/TTR,
  профиль пунктуации, частота опечаток) → sklearn HistGradientBoosting. Это точка отсчёта всех аблаций.

### W2. Текстовые детекторы (внешний контур + локальные фичи)
- Локально: стилометрические фичи, TF-IDF-бейзлайн, лексические маркеры (RU+EN).
- Внешний контур (Colab/Kaggle): Fast-DetectGPT (основной curvature-детектор), Binoculars
  (офлайн-эталон), DetectGPT (эталон пертурбаций — на подвыборке). Артефакты: скоры по каждому
  письму → parquet в репо.
- **Калибровка по языкам**: замер ECE EN vs RU, перекалибровка isotonic — самостоятельный
  мини-результат, обязателен в отчёте.
- Класс `ai_polished`: правило/модель на смешанном тексте (SemEval-8 subtask C).

### W3. Структурные сигналы
- **headers**: SPF/DKIM/DMARC, выравнивание From/Return-Path/Reply-To, Received-цепочка, X-Mailer,
  Date/TZ, Message-ID → категориальные фичи → HistGB/CatBoost (если доступен).
- **html**: три группы признаков (генератор / обфускация / ESP-гасители FP) — см. lit.md;
  DOM-фингерпринт, zero-width, entity-плотность, mso-хаки, скрытые блоки, формы POST.
- **links**: URL-энтропия, punycode/homoglyph, TLD, редиректы, anchor↔href mismatch, трекеры ESP.
- **qr**: декодирование (Node jsqr/@zxing или python cv2), факт QR как интент-признак,
  скриншот-артефакты.
- **images**: спектральная форензика (FFT-пики, шум, JPEG-артефакты, EXIF), pHash, OCR→текстовая
  ветка, CLIP-эмбеддинги (внешний контур) → обучаемый детектор real/AI.

### W4. Кампании и фузия
- Near-dup: SimHash/MinHash + LSH (datasketch); структурная близость: DOM-шинглы, pHash, набор ссылок;
  семантическая близость: эмбеддинги (внешний контур) + HDBSCAN.
- **Скелет шаблона**: общий каркас + переменные слоты; агрегация на кластер: разброс слотов
  как признак машинной генерации.
- Фузия: раздельные модели provenance и intent; калибровка; доверительные интервалы (bootstrap);
  порог отказа (rejection) при нехватке сигналов.

### W5. Объяснимость, устойчивость, UI
- Атрибуция: SHAP (tree/linear) локально, permutation importance — везде; IG/attention — на
  внешнем контуре для нейросетевых веток.
- **Faithfulness (обязательно)**: ERASER comprehensiveness/sufficiency для каждой модели с
  объяснением; token-F1 против человеческой разметки рациональностей (разметить 200 писем).
- Набор атак маскировки (§9) + кривые деградации TPR@FPR по силе атаки.
- SOC-карточка (React или genui-вёрстка): таблица элементов, подсветка спанов, две шкалы
  provenance/intent, контр-свидетельства, граф кампании, фидбек-кнопки → данные фидбека в
  датасет для дообучения.

## 7. Вехи и критерии приёмки

| Веха | Артефакт | Критерий приёмки |
|---|---|---|
| M0 | скелет: ingest → 9 extractors-заглушек → evidence v1 → API-карточка | `make demo` открывает карточку по .eml; evidence валиден; CI зелёный |
| M1 | данные: стартовый набор + datasheet + splits | все наборы §5 скачаны, sha256 в datasheet, лицензии проставлены, сплиты по кампаниям |
| M2 | текст: стилометрия+TF-IDF локально, Fast-DetectGPT/Binoculars внешне | таблица метрик §8 для текстовой ветки; замер EN→RU калибровки |
| M3 | структура: headers/html/links/qr/images | аблация «только структура» ≥ текстового бейзлайна на интент-оси |
| M4 | кампании: LSH+HDBSCAN+скелет | BCubed F1 ≥ 0.8 на размеченных кампаниях; скелет читаем человеком |
| M5 | фузия: две оси + калибровка + CI | ECE < 0.05; rejection работает; FPR на легит-наборе зафиксирован |
| M6 | объяснения + атаки | comprehensiveness/sufficiency отчитаны; кривые деградации по 7 атакам |
| M7 | UI + фидбек | карточка открывается на 3 примерах; фидбек пишется в БД |
| M8 | отчёт | все таблицы из make eval; таксономия FP; ethics-раздел |

## 8. Метрики (обязательный набор)

- Классика: precision/recall/F1/ROC-AUC — по осям, элементам и языкам раздельно.
- **TPR@FPR=1% и 0.1%** (рабочие точки SOC) с bootstrap-CI по письмам.
- Калибровка: ECE, reliability curves.
- Объяснения: comprehensiveness, sufficiency, token-F1 рациональностей.
- Устойчивость: TPR@FPR как функция силы атаки.
- FP на легит-корпоративной почте: FPR отдельным отчётом + таксономия (FP-стиль / FP-интент).
- Кампании: BCubed F1, ARI, homogeneity/completeness; точность скелета шаблона.
- Стоимость: латентность p50/p95, токены на письмо.
- **Аблации (обязательны):** только текст · только структура · только картинки+QR · без curvature ·
  без стилометрии · полная фузия · без кластеризации.

## 9. Устойчивость к маскировке (7 атак)

| Атака | Что ломает |
|---|---|
| LLM-перефраз | curvature, watermark |
| Обратный перевод RU→EN→RU | стилометрия, watermark |
| Синонимическая замена / перестановка предложений | кривизна |
| Внесение опечаток и «человеческого шума» | признак отсутствия опечаток |
| Zero-width / гомоглифы | токенизация, обфускация |
| Разбиение/склейка предложений | burstiness |
| Watermark spoofing | ложные срабатывания watermark-детектора |

Вывод для защиты: структурные и интент-признаки (HTML-каркас, ссылки, заголовки, QR, шаблон
кампании) маскировать почти невозможно — стиль даёт покрытие, структура даёт устойчивость.

## 10. Definition of Done (каждый MR)

1. Код: ruff + mypy + pytest проходят; покрытие новых модулей ≥ 80%.
2. Каждый экстрактор/модель имеет `model_version`; артефакты запуска зафиксированы.
3. `make eval` прогнан: metrics.json приложен, изменения метрик описаны в MR.
4. Данные: источник и лицензия в datasheet; внутренние данные не попали в облако.
5. Документация: модуль описан в README/docs, пример вывода в tests/fixtures.
6. Ревью подопечного → мёрж наставником.

## 11. Безопасность и этика

- Внутренние образцы Doctor Web не покидают контур; облачные API — только на публичных данных.
- Персональные данные маскируются до логирования; логи с телами писем — TTL.
- Раздел dual-use в отчёте: генерация фишинга для оценки — только синтетические получатели.
- Watermark spoofing-атака: детектор не должен становиться оружием против невиновных.

## 12. Стек (финальный)

| Слой | Выбор |
|---|---|
| Язык/среда | Python 3.12, venv, ruff, mypy, pytest, pre-commit |
| ML локально | sklearn (HistGB, linear), scipy; опц. LightGBM/CatBoost по месту |
| Признаки | pandas, lxml, beautifulsoup4, tldextract, dnspython |
| Изображения | Pillow, OpenCV (по месту), pyzbar или Node jsqr |
| Кластеризация | datasketch, sklearn, hdbscan |
| Эмбеддинги/LLM-детекторы | внешний контур: Fast-DetectGPT, Binoculars, sentence-transformers, CLIP |
| Объяснения | SHAP (tree), permutation importance, ERASER-метрики своими руками |
| API/UI | FastAPI + Pydantic v2; React (Vite) или genui-карточка в DSH |
| Хранение | parquet (артефакты), SQLite/Postgres для фидбека и кампаний |
| Оценка | Makefile: make data / eval / report / test |
