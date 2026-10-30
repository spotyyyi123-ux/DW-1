# Проверки задачи 3

- pytest: 69 passed.
- ruff check src tests: passed.
- mypy src: passed (36 source files).
- scripts/smoke.py: passed.
- scripts/demo_task3.py: passed; QR decoded, skeleton matched.
- make eval: интерфейс кампаний проверен на искусственном разбиении
  из 50 пар (100 ID), predictions.parquet и metrics.json созданы.
  Это проверка вычислительного пути, а не измерение качества модели;
  разметка фикстуры автоматическая, не ручная.
- Браузерная проверка card.html: обе оси, каркас, слоты, QR и признаки отображаются.
  Скриншот: card-screenshot.png.

Реальный корпус, внешний прогон sentence-transformers, ручная разметка и
BCubed F1 на реальных топ-50: НЕ ВЫПОЛНЕНО, входные данные не предоставлены.
Прежний текстовый make eval на обучающих данных не запускался: данных в клоне нет.
