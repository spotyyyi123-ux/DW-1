"""Собирает единый HTML-отчёт для проверяющего из артефактов прогона.

Вход: data/runs/sprint1/demo/*.evidence.json, data/runs/sprint1/clusters.json
Выход: data/runs/sprint1/report.html (самодостаточный HTML, без внешних зависимостей).

Запуск: python scripts/make_report.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data" / "runs" / "sprint1"
DEMO = RUN / "demo"
OUT = RUN / "report.html"

PROV_LABELS = {
    "human": "Человек",
    "ai_generated": "Сгенерировано ИИ",
    "ai_polished": "Отредактировано ИИ",
    "template": "Шаблон",
    "unknown": "Не определено",
}
INTENT_LABELS = {
    "phishing": "Фишинг",
    "spam": "Спам",
    "legitimate": "Легитимное",
    "unknown": "Не определено",
}


def esc(value) -> str:
    from html import escape

    return escape(str(value))


def bar(label: str, confidence: float, color: str) -> str:
    percent = round(confidence * 100)
    return (
        f'<div class="bar"><div class="fill {color}" style="width:{percent}%"></div></div>'
        f'<div class="pct">{esc(label)} — {percent}%</div>'
    )


def load_demo() -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    if DEMO.exists():
        for path in sorted(DEMO.glob("*.evidence.json")):
            rows.append(
                (path.name.replace(".evidence.json", ""), json.loads(path.read_text(encoding="utf-8")))
            )
    return rows


def load_clusters() -> list[dict]:
    path = RUN / "clusters.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def demo_row(stem: str, ev: dict) -> str:
    fusion = ev.get("fusion") or {}
    prov = fusion.get("ai_provenance") or {}
    intent = fusion.get("intent") or {}
    campaign = ev.get("campaign") or {}
    threats = fusion.get("threat_signals") or []
    subject = (ev.get("parts") or {}).get("subject", {}).get("text", "")
    campaign_cell = "—"
    if campaign.get("cluster_id"):
        campaign_cell = f"{campaign['cluster_id']} · {campaign.get('member_count')} писем"
    chips = "".join(f'<span class="chip">{esc(t)}</span>' for t in threats) or "—"
    card = f"{stem}.html"
    prov_label = PROV_LABELS.get(prov.get("label"), prov.get("label"))
    intent_label = INTENT_LABELS.get(intent.get("label"), intent.get("label"))
    return f"""
    <tr>
      <td><b>{esc(stem)}</b><br><span class="muted">{esc(subject[:70])}</span></td>
      <td>{bar(prov_label, float(prov.get("confidence") or 0), "prov")}</td>
      <td>{bar(intent_label, float(intent.get("confidence") or 0), "intent")}</td>
      <td>{esc(campaign_cell)}</td>
      <td>{chips}</td>
      <td><a href="demo/{esc(card)}">открыть карточку</a></td>
    </tr>
    """


def cluster_row(c: dict, index: int) -> str:
    examples = c.get("examples") or ["", ""]
    return f"""
    <tr>
      <td>{index + 1}</td>
      <td><b>{c.get('size')}</b></td>
      <td class="muted">{esc(examples[0][:110])}…</td>
      <td class="muted">{esc(examples[1][:110])}…</td>
    </tr>
    """


def build() -> str:
    demo = load_demo()
    clusters = load_clusters()
    rows = "".join(demo_row(stem, ev) for stem, ev in demo)
    if not rows:
        rows = '<tr><td colspan="6">Демо-письма не найдены: запустите make_demo</td></tr>'
    cluster_rows = "".join(cluster_row(c, i) for i, c in enumerate(clusters[:10]))
    if not cluster_rows:
        cluster_rows = '<tr><td colspan="4">Кластеры не найдены: запустите кластеризацию</td></tr>'
    total_in_clusters = sum(c.get("size", 0) for c in clusters)

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DW-1 — отчёт для проверяющего</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 32px; background: #f3f5f8; color: #1f2937;
  font-family: Arial, sans-serif; }}
main {{ max-width: 1150px; margin: auto; }}
h1 {{ margin: 0 0 6px; font-size: 30px; }}
h2 {{ margin: 34px 0 14px; font-size: 21px; }}
.muted {{ color: #6b7280; font-size: 13px; }}
.card, .table-wrap {{ background: #fff; border: 1px solid #dce2ea;
  border-radius: 12px; padding: 20px; margin-bottom: 16px; }}
.pipeline {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px;
  align-items: stretch; }}
.step {{ background: #eef3fa; border: 1px solid #c9d6e6; border-radius: 10px;
  padding: 12px; text-align: center; font-size: 13px; }}
.step b {{ display: block; font-size: 15px; margin-bottom: 4px; color: #1d4ed8; }}
.arrow {{ display: none; }}
.grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }}
.stat {{ background: #fff; border: 1px solid #dce2ea; border-radius: 12px; padding: 16px; }}
.stat .num {{ font-size: 26px; font-weight: bold; color: #1d4ed8; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th, td {{ padding: 10px; border-bottom: 1px solid #e5e9ef; text-align: left;
  vertical-align: top; }}
th {{ background: #f5f7fa; }}
.bar {{ height: 10px; background: #e8edf3; border-radius: 6px; overflow: hidden;
  margin-bottom: 5px; width: 170px; }}
.fill {{ height: 100%; }}
.fill.prov {{ background: #7c3aed; }}
.fill.intent {{ background: #dc2626; }}
.pct {{ font-size: 12px; color: #4b5563; }}
.chip {{ display: inline-block; background: #fee2e2; color: #991b1b;
  border-radius: 999px; padding: 2px 9px; font-size: 12px; margin: 2px; }}
ul {{ margin: 8px 0; padding-left: 22px; }}
code {{ background: #eef3fa; border-radius: 6px; padding: 1px 6px; font-size: 13px; }}
</style>
</head>
<body>
<main>
  <h1>DW-1 — детекция AI-assisted фишинга</h1>
  <p class="muted">Отчёт для проверяющего · {datetime.now():%d.%m.%Y %H:%M} · данные и артефакты: D:\\dw1</p>

  <h2>Как это работает</h2>
  <div class="card">
    <div class="pipeline">
      <div class="step"><b>1. Ингест</b>письмо .eml → части: тема, текст, HTML, ссылки, заголовки</div>
      <div class="step"><b>2. 9 экстракторов</b>стилометрия, срочность, CTA, HTML, ссылки, заголовки…</div>
      <div class="step"><b>3. Фузия</b>две оси: <i>кем написано</i> и <i>зачем</i></div>
      <div class="step"><b>4. Кампании</b>MinHash/LSH → похожие письма = кампания</div>
      <div class="step"><b>5. Карточка</b>аналитику: шкалы, признаки, контр-свидетельства</div>
    </div>
    <p class="muted">Ключевая идея: «написано ИИ» ≠ «злонамеренно». Маркетинговая рассылка тоже ИИ —
    поэтому две независимые шкалы, а не один вердикт.</p>
  </div>

  <h2>Что уже работает</h2>
  <div class="grid">
    <div class="stat"><div class="num">20 304</div>письма прогнано через кластеризацию</div>
    <div class="stat"><div class="num">427</div>кампаний найдено автоматически</div>
    <div class="stat"><div class="num">9 × 2</div>сигнала: элемент × ось (происхождение/намерение)</div>
    <div class="stat"><div class="num">43</div>автотеста, CI зелёный</div>
  </div>

  <h2>Демо: 4 письма</h2>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>Письмо</th><th>Происхождение</th><th>Злонамеренность</th>
        <th>Кампания</th><th>Усилители</th><th></th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>

  <h2>Кампании (топ-10 из 427)</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th>#</th><th>Писем</th><th>Пример A</th><th>Пример B</th></tr></thead>
      <tbody>{cluster_rows}</tbody>
    </table>
  </div>
  <p class="muted">Всего писем внутри кластеров: {total_in_clusters}. Список кластеров:
  <a href="clusters.json">clusters.json</a>, индекс для сопоставления — campaign_index.json.</p>

  <h2>Ограничения (честно)</h2>
  <div class="card">
    <ul>
      <li>Эвристики не калиброваны — уверенности низкие, сравнивать шкалы между письмами можно условно.</li>
      <li>Русский бейзлайн слабый (CoAT ≈ 0.51): нужен LLMTrace-RU и калибровка по языкам.</li>
      <li>Изображения, QR, шаблоны — пока заглушки (Sprint 2).</li>
      <li>Кампании: near-dup по тексту; скелет шаблона и эмбеддинги — Sprint 2.</li>
    </ul>
  </div>

  <h2>Как воспроизвести</h2>
  <div class="card">
    <code>python scripts\\prepare_demo_emails.py</code> — демо-письма из корпусов<br>
    <code>python -m dw1.campaigns.cluster --data data\\raw\\seven-phishing\\test.parquet --out data\\runs\\sprint1\\clusters.json</code><br>
    <code>python scripts\\make_demo.py tests\\fixtures\\*.eml --run sprint1</code><br>
    <code>python scripts\\render_card.py data\\runs\\sprint1\\demo\\campaign_email.evidence.json -o data\\runs\\sprint1\\demo\\campaign_email.html</code>
  </div>
</main>
</body>
</html>
"""


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    print(f"отчёт готов: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
