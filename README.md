# DW-1 — Do Spammers Dream of Electric Sheep?

Система детекции писем и кампаний с признаками генеративного ИИ + **explainable
spam-enrichment** для аналитиков SOC. Заказчик: Doctor Web.

## Что это

Конвейер: письмо `.eml` → разбор частей (тема, текст, HTML, ссылки, заголовки, изображения,
QR) → 9 экстракторов → две независимые оси (**происхождение** и **намерение**) → группировка
в кампании → карточка-объяснение для аналитика.

Главный принцип: **«написано ИИ» ≠ «злонамеренно»** — маркетинговая рассылка тоже написана
ИИ, поэтому оси не смешиваются. Объяснения позлементные, уверенности калиброванные,
контр-свидетельства показываются явно.

## Команда

| Имя | Роль |
|---|---|
| Алина | задача 1 — LLM-детекторы, данные, бейзлайн |
| Ева | задача 2 — устойчивость к атакам маскировки |
| Жека | задача 3 — кампании v2, структура, images/QR |
| Сергей | тимлид — калибровка, фузия, легит-почта, отчёт |

## Быстрый старт

```powershell
git clone <repo-url> dw1
cd dw1
python -m venv .venv
.venv\Scripts\pip install -e ".[dev,api,images]"
python -m dw1 tests\fixtures\sample.eml   # .eml -> Evidence JSON v1
python scripts\smoke.py                   # сквозная проверка
python -m pytest                          # тесты
```

## Основные команды

| Команда | Что делает |
|---|---|
| `python -m dw1 <file.eml> --out out.json` | разбор письма → Evidence JSON v1 |
| `python scripts\smoke.py` | сквозная проверка пайплайна |
| `python scripts\make_demo.py <eml>... --run sprint1` | демо-карточки писем |
| `python scripts\make_report.py` | дашборд по артефактам |
| `python scripts\make_sprint2_site.py` | HTML-отчёт Sprint 2 (`reports/sprint2-site.html`) |
| `uvicorn dw1.api.app:app` | API (extras `[api]`) |

## Тесты и CI

```powershell
python -m pytest      # 72+ теста
python -m ruff check src tests
python -m mypy src
```

CI (GitHub Actions) гоняет ruff + mypy + pytest на каждый push.

## Статус

- **Sprint 1** — закрыт, тег `m1`: сквозной пайплайн, кластеризация кампаний, бейзлайн.
- **Sprint 2** — закрыт, тег `m2`: Fast-DetectGPT (EN), 7 атак устойчивости,
  кампании v2 + images/QR, калиброванная фузия, **FP 0% на легит-почте**.

Ключевые метрики:

| Метрика | Значение |
|---|---|
| Калибровка (ECE) | provenance 0.007→0.006, intent 0.361→0.007 |
| Intent TPR @ FPR 1.5% | 0.993 (порог 0.85 → TPR 0.96 при FPR 0.5%) |
| FP на легит-почте (Enron) | 0% (100 писем) |
| Fast-DetectGPT EN | ECE 0.243→0.024, F1 0.89→0.99, AUC 0.988 |
| Атаки (intent) | деградация ≤1.2% |
| Кампании | 20 304 письма → 427 кластеров |

## Структура репозитория

```
├── src/dw1/          # пайплайн: ingest, extractors (9), models (fusion, llm_detectors),
│                     #   campaigns, explain, eval, api
├── tests/            # 72+ тестов + фикстуры писем
├── scripts/          # демо, отчёты, калибровка, обучение моделей, атаки, внешний Colab-ноутбук
├── docs/             # SPEC (ТЗ), SPRINT-1/2/3, TEAM, lit, datasets, отчёты задач
├── reports/          # артефакты: HTML-отчёт Sprint 2, калибровка Fast-DetectGPT
├── data/             # данные и артефакты прогонов (в git — только datasheet и splits)
├── AGENTS.md         # инструкции для DSH-агентов
└── README.md
```

## Документация

- [SPEC.md](docs/SPEC.md) — ТЗ, архитектура, контракты, метрики;
- [SPRINT-2.md](docs/SPRINT-2.md) — закрытый спринт; [SPRINT-3.md](docs/SPRINT-3.md) — план;
- [sprint2-report.md](docs/sprint2-report.md) — отчёт Sprint 2 (текст + шпаргалка);
- [TEAM.md](docs/TEAM.md) — роли и регламент; [lit.md](docs/lit.md) — литература;
- [datasets.md](docs/datasets.md) — датасеты и лицензии;
- [notes/llm-detectors.md](docs/notes/llm-detectors.md) — Fast-DetectGPT;
- [notes/attacks.md](docs/notes/attacks.md) — устойчивость к атакам;
- [task3-campaigns-images.md](docs/task3-campaigns-images.md) — кампании v2 и images/QR.
