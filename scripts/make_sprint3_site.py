"""Собирает наглядный HTML-отчёт по Sprint 3 (одним файлом) для проверяющего.

Вход: reports/sprint3-llm/metrics.json, data/runs/sprint2/attacks/attack_summary.csv,
      data/runs/sprint3/detector_comparison.json, data/runs/sprint3/campaigns/eval/metrics.json,
      демо-карточки data/runs/sprint1/demo/*.html.
Выход: reports/sprint3-site.html.

Запуск: python scripts/make_sprint3_site.py
"""
from __future__ import annotations

import csv
import html
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "sprint3-site.html"


def esc(v) -> str:
    return html.escape(str(v))


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def llm_rows() -> str:
    m = read_json(ROOT / "reports" / "sprint3-llm" / "metrics.json", {})
    rows = ""
    for det, label in (("gptneo", "GPT-Neo 2.7B"), ("qwen", "Qwen2.5-0.5B")):
        for lang, lang_label in (("en", "EN"), ("ru", "RU")):
            cell = (m.get(det) or {}).get(lang) or {}
            if not cell:
                continue
            before, after = cell.get("before") or {}, cell.get("after") or {}
            rows += (
                f"<tr><td>{label}</td><td>{lang_label}</td><td>{cell.get('n_test')}</td>"
                f"<td>{fmt(before.get('ece'))} → {fmt(after.get('ece'))}</td>"
                f"<td>{fmt(after.get('f1'))}</td><td>{fmt(after.get('roc_auc'))}</td></tr>"
            )
    return rows


def detector_rows() -> str:
    rows = ""
    for r in read_json(ROOT / "data/runs/sprint3/detector_comparison.json", []):
        rows += (
            f"<tr><td>{esc(r['detector'])}</td><td>{esc(r['language'])}</td><td>{r['n_test']}</td>"
            f"<td>{r['roc_auc']}</td><td>{r['f1']}</td><td>{r['ece']}</td></tr>"
        )
    return rows


def attack_rows() -> str:
    path = ROOT / "data/runs/sprint2/attacks/attack_summary.csv"
    rows = ""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows += (
                    f"<tr><td>{esc(r['attack'])}</td>"
                    f"<td>{r['baseline_tpr_prov']} → {r['attacked_tpr_prov_max']}</td>"
                    f"<td>{r['degradation_prov_pct']}%</td>"
                    f"<td>{r['baseline_tpr_intent']} → {r['attacked_tpr_intent_max']}</td>"
                    f"<td>{r['degradation_intent_pct']}%</td></tr>"
                )
    return rows


def bcubed() -> tuple[str, str]:
    m = read_json(ROOT / "data/runs/sprint3/campaigns/eval/metrics.json", {})
    b = m.get("baseline_bcubed") or {}
    v = m.get("bcubed") or {}
    rows = (
        f"<tr><td>Исходные топ-50 (LSH)</td><td>{fmt(b.get('precision'))}</td>"
        f"<td>{fmt(b.get('recall'))}</td><td><b>{fmt(b.get('f1'))}</b></td></tr>"
        f"<tr><td>LSH + HDBSCAN (v2)</td><td>{fmt(v.get('precision'))}</td>"
        f"<td>{fmt(v.get('recall'))}</td><td><b>{fmt(v.get('f1'))}</b></td></tr>"
    )
    review = m.get("review") or {}
    meta = m.get("embedding_metadata") or {}
    note = (
        f"Модель: {esc(meta.get('model', '—'))} · ревизия {esc(str(meta.get('revision', '—'))[:12])} · "
        f"писем {m.get('scope', '') and 600}. Проверяющие: {', '.join(review.get('reviewers', []))}; "
        f"расхождений {review.get('n_disagreements', '—')}; согласовал: {esc(m.get('adjudicated_by', '—'))}."
    )
    return rows, note


def demo_section() -> str:
    blocks = []
    demo = ROOT / "data/runs/sprint1/demo"
    prov = {
        "human": "Человек",
        "ai_generated": "ИИ-генерировано",
        "ai_polished": "ИИ-причёсано",
        "template": "Шаблон",
        "unknown": "Не определено",
    }
    intent = {
        "phishing": "Фишинг",
        "spam": "Спам",
        "legitimate": "Легитимно",
        "unknown": "Не определено",
    }
    for path in sorted(demo.glob("*.evidence.json")):
        stem = path.name.replace(".evidence.json", "")
        ev = read_json(path, {})
        fusion = ev.get("fusion") or {}
        p = fusion.get("ai_provenance") or {}
        i = fusion.get("intent") or {}
        card = demo / f"{stem}.html"
        iframe = ""
        if card.exists():
            iframe = (
                f'<iframe title="{esc(stem)}" srcdoc="{html.escape(card.read_text(encoding="utf-8"), quote=True)}"></iframe>'
            )
        blocks.append(
            f"""
    <div class="demo-row">
      <b>{esc(stem)}</b>
      <div class="kv">Происхождение: <b>{esc(prov.get(p.get('label'), p.get('label')))}</b>
        (calib {fmt(p.get('calibrated_confidence'))}) · Намерение:
        <b>{esc(intent.get(i.get('label'), i.get('label')))}</b>
        (calib {fmt(i.get('calibrated_confidence'))})</div>
      <details><summary>открыть карточку письма</summary>{iframe}</details>
    </div>"""
        )
    return "".join(blocks)


def build() -> str:
    bcubed_rows, bcubed_note = bcubed()
    llm = read_json(ROOT / "reports" / "sprint3-llm" / "metrics.json", {})
    transfer = llm.get("emails_transfer") or {}

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DW-1 — Sprint 3: отчёт для проверяющего</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 32px; background: #f3f5f8; color: #1f2937; font-family: Arial, sans-serif; }}
main {{ max-width: 1150px; margin: auto; }}
h1 {{ margin: 0 0 6px; font-size: 30px; }}
h2 {{ margin: 34px 0 14px; font-size: 21px; border-bottom: 2px solid #dce2ea; padding-bottom: 8px; }}
.badge {{ display: inline-block; background: #dcfce7; color: #166534; border-radius: 8px;
  padding: 4px 12px; font-size: 13px; font-weight: bold; margin-left: 10px; }}
.muted {{ color: #6b7280; font-size: 13px; }}
.card, .table-wrap {{ background: #fff; border: 1px solid #dce2ea; border-radius: 12px; padding: 20px; margin-bottom: 16px; }}
.grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }}
.stat {{ background: #fff; border: 1px solid #dce2ea; border-radius: 12px; padding: 16px; }}
.stat .num {{ font-size: 18px; font-weight: bold; color: #1d4ed8; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th, td {{ padding: 9px; border-bottom: 1px solid #e5e9ef; text-align: left; vertical-align: top; }}
th {{ background: #f5f7fa; }}
.pipeline {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; }}
.step {{ background: #eef3fa; border: 1px solid #c9d6e6; border-radius: 10px; padding: 12px; text-align: center; font-size: 13px; }}
.step b {{ display: block; font-size: 15px; margin-bottom: 4px; color: #1d4ed8; }}
.demo-row {{ border-bottom: 1px solid #e5e9ef; padding: 12px 0; }}
.kv {{ font-size: 13px; color: #374151; margin-top: 4px; }}
details summary {{ cursor: pointer; color: #1d4ed8; font-size: 13px; margin-top: 6px; }}
iframe {{ width: 100%; height: 420px; border: 1px solid #dce2ea; border-radius: 8px; margin-top: 8px; background: #fff; }}
.ok {{ color: #166534; font-weight: bold; }}
.warn {{ color: #b45309; font-weight: bold; }}
</style>
</head>
<body>
<main>
  <h1>DW-1 — Sprint 3 «Доказательная база»<span class="badge">ЗАКРЫТ · тег m3</span></h1>
  <p class="muted">Отчёт для проверяющего · {datetime.now():%d.%m.%Y} · репозиторий D:\\dw1</p>

  <h2>Как это работает</h2>
  <div class="card">
    <div class="pipeline">
      <div class="step"><b>1. Ингест</b>.eml → тема, текст, HTML, ссылки, заголовки, картинки, QR</div>
      <div class="step"><b>2. 9 экстракторов</b>стилометрия, CTA, HTML, links, headers, images, QR, template</div>
      <div class="step"><b>3. Детекторы</b>TF-IDF-модель + Fast-DetectGPT (LLM)</div>
      <div class="step"><b>4. Фузия</b>две оси, калибровка, порог отказа</div>
      <div class="step"><b>5. Карточка</b>шкалы, признаки, кампания, контр-свидетельства</div>
    </div>
  </div>

  <h2>Итоги по ролям</h2>
  <div class="grid">
    <div class="stat"><div class="num">Алина — задача 1</div>
      <b>Qwen vs GPT-Neo + скоры по письмам</b>
      <ul>
        <li><span class="ok">Готово:</span> Qwen 2.5-0.5B на HC3/CoAT; 185 писем по скорам;
          195/200 карточек подхватывают LLM-сигнал</li>
        <li><span class="warn">Ограничение:</span> Qwen не улучшил RU (AUC 0.361 против 0.382)</li>
      </ul>
    </div>
    <div class="stat"><div class="num">Ева — задача 2</div>
      <b>7 атак на 600 письмах</b>
      <ul>
        <li><span class="ok">Готово:</span> `make attacks`, кривые деградации</li>
        <li><span class="ok">Intent TPR 0.97</span>, деградация ≤ 0.8%</li>
      </ul>
    </div>
    <div class="stat"><div class="num">Жека — задача 3</div>
      <b>BCubed F1 кампаний</b>
      <ul>
        <li><span class="ok">Измерено:</span> sentence-transformers + HDBSCAN;
          baseline F1 0.900, v2 F1 0.689</li>
        <li><span class="warn">Оговорка:</span> gold — прокси-разметка, не ручная</li>
      </ul>
    </div>
    <div class="stat"><div class="num">Сергей — задача 4</div>
      <b>Сравнение детекторов + отчёт</b>
      <ul>
        <li><span class="ok">Готово:</span> Fast-DetectGPT vs TF-IDF на одних письмах;
          precision/recall/F1/ECE</li>
        <li><span class="ok">Готово:</span> пакет разметки, отчёт, сайт, тег m3</li>
      </ul>
    </div>
  </div>

  <h2>Qwen vs GPT-Neo (HC3/CoAT, общие тексты)</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Детектор</th><th>Язык</th><th>n test</th><th>ECE до → после</th>
      <th>F1 после</th><th>ROC-AUC после</th></tr></thead>
      <tbody>{llm_rows()}</tbody>
    </table>
  </div>
  <p class="muted">перенос шкалы на письма ({llm.get('n_emails', '—')} писем):
  AUC {fmt(transfer.get('roc_auc'))}, F1 {fmt(transfer.get('f1'))}, ECE {fmt(transfer.get('ece'))}.
  Qwen 0.5B не улучшил RU — ограничение зафиксировано.</p>

  <h2>Fast-DetectGPT vs TF-IDF на одних письмах</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Детектор</th><th>Язык</th><th>n test</th><th>ROC-AUC</th><th>F1</th><th>ECE</th></tr></thead>
      <tbody>{detector_rows()}</tbody>
    </table>
  </div>
  <p class="muted">Обе ветки почти идеальны на этом домене (LLMGen vs Nazario — конфаунд источника);
  тест 23 письма — ориентир, не измерение «дикой» почты.</p>

  <h2>Устойчивость к 7 атакам (600 писем)</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Атака</th><th>Provenance TPR</th><th>Деградация</th>
      <th>Intent TPR</th><th>Деградация</th></tr></thead>
      <tbody>{attack_rows()}</tbody>
    </table>
  </div>

  <h2>BCubed F1 кампаний (топ-50, 600 писем)</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Разбиение</th><th>Precision</th><th>Recall</th><th>BCubed F1</th></tr></thead>
      <tbody>{bcubed_rows}</tbody>
    </table>
  </div>
  <p class="muted">{bcubed_note} Gold — прокси по TF-IDF cosine; для человеческого BCubed
  нужна ручная разметка 2–3 человек (workflow готов).</p>

  <h2>Демо: карточки писем</h2>
  <div class="card">{demo_section()}</div>

  <h2>Ограничения (честно)</h2>
  <div class="card">
    <ul>
      <li><b>RU-детектор:</b> Qwen 0.5B не улучшил RU (AUC 0.361; GPT-Neo 0.382) — нужен более крупный RU-детектор.</li>
      <li><b>Fast-DetectGPT на письмах:</b> скоры получены, но перенос шкалы с HC3/CoAT — калибровка на почте не гарантируется.</li>
      <li><b>BCubed gold:</b> прокси-разметка (TF-IDF cosine), 131 расхождение между «проверяющими» — цифра ориентировочная.</li>
      <li><b>Сравнение детекторов:</b> тест 23 письма, доменный конфаунд — не доказательство превосходства.</li>
    </ul>
  </div>

  <h2>Что дальше</h2>
  <div class="card">
    <ul>
      <li>Защита: 3-минутный сценарий + шпаргалка (`docs/sprint3-report.md`);</li>
      <li>Ручная разметка BCubed (2–3 человека) — заменить прокси-цифру;</li>
      <li>RU-capable детектор (крупнее/специализированный) — следующая итерация;</li>
      <li>Реальная почта заказчика — отдельный контур, вне публичного бенчмарка.</li>
    </ul>
  </div>

  <h2>Как воспроизвести</h2>
  <div class="card">
    <code>python -m pytest</code> — тесты<br>
    <code>python scripts/make_report.py</code> — дашборд Sprint 1/2<br>
    <code>python scripts/compare_detectors.py --scores ...</code> — сравнение детекторов<br>
    <code>make attacks</code> — 7 атак на 600 письмах<br>
    <code>python -m dw1.eval.campaign_review ...</code> — BCubed<br>
    <code>python scripts/make_sprint3_site.py</code> — этот отчёт
  </div>
</main>
</body>
</html>
"""


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    print(f"сайт-отчёт готов: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
