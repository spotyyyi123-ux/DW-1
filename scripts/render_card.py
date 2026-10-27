"""Рендер Evidence JSON в автономную HTML-карточку."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

PROVENANCE_LABELS = {
    "human": "Человек",
    "ai_generated": "Сгенерировано ИИ",
    "ai_polished": "Отредактировано ИИ",
    "template": "Шаблон",
    "unknown": "Не определено",
}

INTENT_LABELS = {
    "phishing": "Фишинг",
    "spam": "Спам",
    "legitimate": "Легитимное письмо",
    "unknown": "Не определено",
}


def esc(value) -> str:
    """Безопасное HTML-экранирование."""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value))


def collect_signals(data: dict, key: str) -> list[dict]:
    result = []
    for extractor, output in data.get("signals", {}).items():
        signal = output.get(key, {})
        if isinstance(signal, dict):
            item = dict(signal)
            item["extractor"] = extractor
            result.append(item)
    return result


def summarize(items: list[dict], labels: dict) -> tuple[str, float]:
    """Определяет наиболее частую метку и среднюю уверенность."""
    known = [item for item in items if item.get("label") in labels
             and item.get("label") != "unknown"]

    if not known:
        return "unknown", 0.0

    counts = {}
    for item in known:
        label = item["label"]
        counts[label] = counts.get(label, 0) + 1

    label = max(counts, key=counts.get)
    confidence_values = [
        float(item.get("confidence", 0))
        for item in known if item["label"] == label
    ]
    confidence = sum(confidence_values) / len(confidence_values)
    return label, max(0.0, min(1.0, confidence))


def collect_evidence(data: dict) -> list[dict]:
    rows = []
    for extractor, output in data.get("signals", {}).items():
        for key in ("ai_provenance", "intent"):
            signal = output.get(key, {})
            for item in signal.get("evidence", []):
                if isinstance(item, dict):
                    rows.append({
                        "extractor": extractor,
                        "category": key,
                        "signal": item.get("signal", ""),
                        "value": item.get("value", ""),
                        "weight": item.get("weight", ""),
                    })
    return rows


def render_scale(title: str, label: str, confidence: float) -> str:
    percent = round(confidence * 100)
    return f"""
    <section class="scale-card">
      <h2>{esc(title)}</h2>
      <div class="scale-value">{esc(label)}</div>
      <div class="bar"><div class="bar-fill" style="width:{percent}%"></div></div>
      <div class="scale-percent">Уверенность: {percent}%</div>
    </section>
    """


def fusion_axes(data: dict):
    """Берёт готовые оси из fusion v1 (в приоритете над голосованием по сигналам)."""
    fusion = data.get("fusion") or {}
    result = []
    for axis in (fusion.get("ai_provenance") or {}, fusion.get("intent") or {}):
        label = axis.get("label", "unknown")
        if isinstance(label, str) and label != "unknown":
            result.append((label, float(axis.get("confidence", 0) or 0)))
        else:
            result.append(None)
    return result[0], result[1]


def render(data: dict) -> str:
    provenance_items = collect_signals(data, "ai_provenance")
    intent_items = collect_signals(data, "intent")

    fused_prov, fused_intent = fusion_axes(data)
    provenance, provenance_conf = (
        fused_prov if fused_prov else summarize(provenance_items, PROVENANCE_LABELS)
    )
    intent, intent_conf = (
        fused_intent if fused_intent else summarize(intent_items, INTENT_LABELS)
    )

    fusion = data.get("fusion") or {}
    if fusion.get("calibrated"):
        heuristic_note = "fusion v2 — калиброванная вероятность (isotonic, Sprint 2)"
    elif fusion.get("heuristic"):
        heuristic_note = "fusion v1 — эвристика, не калибровано"
    else:
        heuristic_note = "по сигналам экстракторов"
    counter_items = (data.get("explanations") or {}).get("counter_evidence") or []
    counter_html = ""
    if counter_items:
        counter_html = (
            '<section class="meta"><h2>Контр-свидетельства</h2><ul>'
            + "".join(f"<li>{esc(item)}</li>" for item in counter_items)
            + "</ul></section>"
        )

    evidence_rows = collect_evidence(data)
    parts = data.get("parts", {})
    headers = parts.get("headers", {})
    subject = str(parts.get("subject", {}).get("text", ""))
    sender = headers.get("from", "")

    table_rows = ""
    for row in evidence_rows:
        table_rows += f"""
        <tr>
          <td>{esc(row['extractor'])}</td>
          <td>{esc(row['category'])}</td>
          <td>{esc(row['signal'])}</td>
          <td>{esc(row['value'])}</td>
          <td>{esc(row['weight'])}</td>
        </tr>
        """

    if not table_rows:
        table_rows = '<tr><td colspan="5">Признаки не найдены</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DW-1 — карточка письма</title>
<style>
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 32px;
  background: #f3f5f8; color: #202938;
  font-family: Arial, sans-serif;
}}
main {{ max-width: 1100px; margin: auto; }}
h1 {{ margin-top: 0; }}
.meta, .scale-card, .table-wrap {{
  background: white; border: 1px solid #dce2ea;
  border-radius: 10px; padding: 20px; margin-bottom: 18px;
}}
.meta p {{ margin: 8px 0; overflow-wrap: anywhere; }}
.scales {{
  display: grid; grid-template-columns: repeat(2, 1fr);
  gap: 18px;
}}
.scale-card h2 {{ font-size: 18px; margin-top: 0; }}
.scale-value {{ font-size: 22px; font-weight: bold; margin: 16px 0; }}
.bar {{ height: 12px; background: #e8edf3; border-radius: 8px; overflow: hidden; }}
.bar-fill {{ height: 100%; background: #3977c3; }}
.scale-percent {{ color: #687386; margin-top: 9px; font-size: 13px; }}
.table-wrap {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th, td {{ text-align: left; padding: 11px; border-bottom: 1px solid #e5e9ef; }}
th {{ background: #f5f7fa; }}
td {{ overflow-wrap: anywhere; }}
@media (max-width: 700px) {{
  body {{ padding: 14px; }}
  .scales {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<main>
  <h1>DW-1 — карточка письма</h1>

  <section class="meta">
    <h2>Письмо</h2>
    <p><b>Отправитель:</b> {esc(sender or "Не указан")}</p>
    <p><b>Тема:</b> {esc(subject or "Не указана")}</p>
    <p><b>Версия схемы:</b> {esc(data.get("schema_version", "не указана"))}</p>
    <p><b>Источник оценок:</b> {esc(heuristic_note)}</p>
  </section>

  <div class="scales">
    {render_scale("Происхождение письма", PROVENANCE_LABELS[provenance], provenance_conf)}
    {render_scale("Злонамеренность", INTENT_LABELS[intent], intent_conf)}
  </div>

  {counter_html}

  <section class="table-wrap">
    <h2>Что известно про каждый элемент</h2>
    <table>
      <thead>
        <tr>
          <th>Экстрактор</th>
          <th>Категория</th>
          <th>Признак</th>
          <th>Значение</th>
          <th>Вес</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
  </section>
</main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Создание HTML-карточки из Evidence JSON"
    )
    parser.add_argument("input", help="Путь к Evidence JSON")
    parser.add_argument("-o", "--output", help="Путь к HTML-файлу")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = (
        Path(args.output) if args.output
        else input_path.with_suffix(".html")
    )

    data = json.loads(input_path.read_text(encoding="utf-8"))
    output_path.write_text(render(data), encoding="utf-8")
    print(f"Карточка создана: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())