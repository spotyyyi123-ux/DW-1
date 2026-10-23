# Неделя 2 — промежуточный отчёт (M1)

## Что готово

| Задача | Что сделано | Статус |
|---|---|---|
| 1 — данные + бейзлайн (Alina) | 9 наборов данных, datasheet, TF-IDF→SVD→HistGB, eval-харнесс с метриками и сравнением | ✅ в main (PR #3) |
| 2 — текстовые признаки | body_text (стилометрия, эвристики), subject, cta, text_utils, 242 строки тестов | ✅ в main |
| 3 — структура + карточка | headers, html, links, render_card.py | ✅ в main |
| 4 — кампании + демо (тимлид) | кластеризация MinHash/LSH (427 кластеров на 20 304 письмах), индекс кампаний, карточка → кампания, демо-пакет из 4 писем | ✅ в main |

## Ключевые цифры

- **Кампании**: 20 304 письма → 427 кластеров; топ-кластеры: CNN-рассылка (102 письма),
  «ваш платёж не прошёл» (15 писем), спам-софт (16). Артефакты: `data/runs/sprint1/clusters.json`
  + `campaign_index.json` (427 записей, 732 КБ).
- **Бейзлайн (Alina)**: F1 0.734 / ROC-AUC 0.867 на отложенной выборке; по языкам: CoAT 0.512 /
  HC3-en 0.937; на 3746 ИИ-письмах LLMGen обнаруживается 40.4% (см. `docs/w1-results.md` в её ветке).
- **Фузия v1** (эвристика, не калибрована): две оси + усилители (SPF/DMARC fail, punycode,
  anchor mismatch) и гасители (ESP-маркеры, List-Unsubscribe, DKIM pass) → контр-свидетельства.

## Демо-пакет (4 письма)

| Письмо | Происхождение | Злонамеренность | Кампания |
|---|---|---|---|
| `sample.eml` (чистое, SPF pass) | ИИ (0.32) | phishing (0.17) | — |
| `sample_suspicious.eml` (SPF fail, голая ссылка) | ИИ (0.47) | phishing (0.28) | — |
| `campaign_email.eml` (реальный спам из CEAS-08) | ai_polished (0.12) | phishing (0.08) | **campaign-5, 15 писем** |
| `llmgen_phishing.eml` (письмо от LLM, LLMGen) | ИИ (0.10) | phishing (0.30) | — |

Карточки: `data/runs/sprint1/demo/<name>.html` (Автономный HTML: две шкалы, бейдж
«fusion v1 — эвристика, не калибровано», таблица признаков, контр-свидетельства).

## Как воспроизвести

```cmd
python scripts\prepare_demo_emails.py
python -m dw1.campaigns.cluster --data data\raw\seven-phishing\test.parquet --out data\runs\sprint1\clusters.json
python scripts\make_demo.py tests\fixtures\sample.eml tests\fixtures\sample_suspicious.eml tests\fixtures\campaign_email.eml tests\fixtures\llmgen_phishing.eml --run sprint1
python scripts\render_card.py data\runs\sprint1\demo\campaign_email.evidence.json -o data\runs\sprint1\demo\campaign_email.html
```

## Ограничения (честно)

1. Эвристики не калиброваны — уверенности низкие и не сопоставимы между элементами (калибровка в M5).
2. Русский бейзлайн слабый (CoAT 0.512) — нужен LLMTrace-RU и отдельная калибровка по языкам.
3. Письма LLMGen — plain-text; короткие тексты дают мало стилометрических сигналов
   (провенанс 0.10) — исправляется большим корпусом писем с обеими метками.
4. `images`, `qr`, `template` — пока заглушки (Sprint 2).
5. `campaign_email.eml` — единственный демо-образец из CEAS-08 (лицензия «на проверке»),
   только для демонстрации, не для обучения.

## Следующие шаги

1. Мёрж ветки `w1/data-baseline` (Alina) → тег `m1`.
2. Sprint 2: внешние LLM-детекторы (Fast-DetectGPT), калибровка фузии, скелет шаблона кампаний,
   атаки маскировки, faithfulness-метрики.
