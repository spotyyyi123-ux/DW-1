# DW-1 — Do Spammers Dream of Electric Sheep?

Детекция писем и кампаний с признаками генеративного ИИ + **explainable spam-enrichment**
для аналитиков SOC. Заказчик: Doctor Web.

**Документация:** [SPEC.md](docs/SPEC.md) (ТЗ и контракты) · [TEAM.md](docs/TEAM.md) (роли и задачи) ·
[lit.md](docs/lit.md) (литература) · [datasets.md](docs/datasets.md) (данные).
Для агентов: [AGENTS.md](AGENTS.md) — читается DSH-агентами автоматически.

Результат задачи 1: [данные, первая модель и запуск](docs/w1-results.md).
Обновление: [LLMTrace и RoFT, сравнение моделей](docs/w1-new-datasets.md).

## Быстрый старт

```powershell
git clone <repo-url> dw1
cd dw1
python -m venv .venv
.venv\Scripts\pip install -e ".[dev,api,images]"
python -m dw1 tests\fixtures\sample.eml   # M0: .eml -> Evidence JSON v1
python scripts\smoke.py                   # smoke-проверка без установки зависимостей
python -m pytest                          # тесты
```

## Основные команды

| Команда | Что делает |
|---|---|
| `python -m dw1 <file.eml> --out out.json` | разбор письма → Evidence JSON v1 (контракт SPEC §4.1) |
| `python scripts\smoke.py` | сквозная проверка M0 на фикстуре |
| `make eval RUN=bench-v1` | прогон оценки → `data/runs/<run>/metrics.json` |
| `make report RUN=bench-v1` | отчёт, сгенерированный из артефактов |
| `make data` | план загрузки датасетов (W1) |
| `uvicorn dw1.api.app:app` | API (нужны extras `[api]`) |

## Статус

**M0 (скелет):** ингест .eml рабочий; 9 экстракторов — заглушки (`label=unknown`, честный
`rejection=true` в фузии); контракт Evidence v1 валидируется; CI настроен.
Следующие вехи — SPEC §7. Кто за что отвечает — TEAM.md.

## Две оси (главный принцип)

`ai_provenance` (кем написано) и `intent` (зачем) — отдельные предсказания, отдельная
калибровка. Происхождение ≠ злонамеренность: HR-рассылка тоже написана ИИ.
