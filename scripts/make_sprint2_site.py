"""Собирает наглядный HTML-отчёт по Sprint 2 (одним файлом) для проверяющего.

Читает артефакты (калибровка, FP, operating points, Fast-DetectGPT, атаки,
кампании, демо-карточки) и собирает самодостаточный файл reports/sprint2-site.html.

Запуск: python scripts/make_sprint2_site.py
"""
from __future__ import annotations

import csv
import html
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPRINT2 = ROOT / "data" / "runs" / "sprint2"
LLM_DIR = ROOT / "reports" / "sprint2-llm"
DEMO = ROOT / "data" / "runs" / "sprint1" / "demo"
CLUSTERS = ROOT / "data" / "runs" / "sprint1" / "clusters.json"
OUT = ROOT / "reports" / "sprint2-site.html"

PROV = {
    "human": "Человек",
    "ai_generated": "ИИ-генерировано",
    "ai_polished": "ИИ-причёсано",
    "template": "Шаблон",
    "unknown": "Не определено",
}
INTENT = {
    "phishing": "Фишинг",
    "spam": "Спам",
    "legitimate": "Легитимно",
    "unknown": "Не определено",
}


def esc(value) -> str:
    return html.escape(str(value))


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def card(value: str, body: str) -> str:
    return f'<div class="card">{body}</div>'


def demo_section() -> str:
    blocks = []
    for path in sorted(DEMO.glob("*.evidence.json")):
        stem = path.name.replace(".evidence.json", "")
        ev = read_json(path, {})
        fusion = ev.get("fusion") or {}
        prov = fusion.get("ai_provenance") or {}
        intent = fusion.get("intent") or {}
        campaign = ev.get("campaign") or {}
        subject = (ev.get("parts") or {}).get("subject", {}).get("text", "")
        card_html = DEMO / f"{stem}.html"
        iframe = ""
        if card_html.exists():
            inner = card_html.read_text(encoding="utf-8")
            iframe = (
                f'<iframe title="{esc(stem)}" srcdoc="{html.escape(inner, quote=True)}"></iframe>'
            )
        campaign_txt = (
            f"{campaign.get('cluster_id')} · {campaign.get('member_count')} писем"
            if campaign.get("cluster_id")
            else "—"
        )
        blocks.append(
            f"""
    <div class="demo-row">
      <div class="demo-meta">
        <b>{esc(stem)}</b><br><span class="muted">{esc(subject[:90])}</span>
        <div class="kv">Происхождение: <b>{esc(PROV.get(prov.get('label'), prov.get('label')))}</b>
          (calib {fmt(prov.get('calibrated_confidence'))})</div>
        <div class="kv">Намерение: <b>{esc(INTENT.get(intent.get('label'), intent.get('label')))}</b>
          (calib {fmt(intent.get('calibrated_confidence'))})</div>
        <div class="kv">Кампания: {esc(campaign_txt)}</div>
      </div>
      <details>
        <summary>открыть карточку письма</summary>
        {iframe}
      </details>
    </div>"""
        )
    return "".join(blocks)


def llm_rows() -> str:
    metrics = read_json(LLM_DIR / "llm_calibration_metrics.json", {})
    rows = ""
    for lang, label in (("en", "Английский"), ("ru", "Русский")):
        m = metrics.get(lang) or {}
        if not m:
            continue
        before, after = m.get("before") or {}, m.get("after") or {}
        rows += (
            f"<tr><td>{label}</td><td>{m.get('n_test')}</td>"
            f"<td>{fmt(before.get('ece'))} → {fmt(after.get('ece'))}</td>"
            f"<td>{fmt(before.get('f1'))} → {fmt(after.get('f1'))}</td>"
            f"<td>{fmt(before.get('roc_auc'))} → {fmt(after.get('roc_auc'))}</td></tr>"
        )
    return rows


def attack_rows() -> str:
    path = SPRINT2 / "attacks" / "attack_summary.csv"
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


def cluster_rows() -> str:
    clusters = read_json(CLUSTERS, [])
    rows = ""
    for i, c in enumerate(clusters[:10]):
        examples = c.get("examples") or ["", ""]
        rows += (
            f"<tr><td>{i + 1}</td><td>{c.get('size')}</td>"
            f"<td class='muted'>{esc(examples[0][:100])}…</td>"
            f"<td class='muted'>{esc(examples[1][:100])}…</td></tr>"
        )
    total = sum(c.get("size", 0) for c in clusters)
    return rows, total


def build() -> str:
    cal = read_json(SPRINT2 / "calibration_metrics.json", {})
    fp = read_json(SPRINT2 / "legit_fp.json", {})
    op = read_json(SPRINT2 / "operating_points.json", {})
    clusters_rows, clusters_total = cluster_rows()

    calibration_rows = ""
    for axis, label in (
        ("ai_provenance", "Происхождение (AI)"),
        ("intent", "Намерение (malicious)"),
    ):
        m = (cal.get("axes") or {}).get(axis) or {}
        if m:
            calibration_rows += (
                f"<tr><td>{label}</td><td>{fmt(m.get('ece_before'))}</td>"
                f"<td>{fmt(m.get('ece_after'))}</td><td>{fmt(m.get('auc_before'))}</td>"
                f"<td>{fmt(m.get('auc_after'))}</td></tr>"
            )

    fp_row = ""
    if fp:
        fp_row = (
            f"<tr><td>{fp.get('n_legit')}</td><td>{fp.get('fp_style')}</td>"
            f"<td>{fp.get('fp_intent')}</td><td>{fp.get('fpr_any')}</td></tr>"
        )

    op_rows = ""
    for r in ((op.get("intent") or {}).get("sweep") or []):
        if r.get("threshold") in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9):
            op_rows += (
                f"<tr><td>{r['threshold']}</td><td>{r['tpr']}</td>"
                f"<td>{r['fpr']}</td><td>{r['f1']}</td></tr>"
            )

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DW-1 — Sprint 2: отчёт для проверяющего</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 32px; background: #f3f5f8; color: #1f2937;
  font-family: Arial, sans-serif; }}
main {{ max-width: 1150px; margin: auto; }}
h1 {{ margin: 0 0 6px; font-size: 30px; }}
h2 {{ margin: 34px 0 14px; font-size: 21px; border-bottom: 2px solid #dce2ea; padding-bottom: 8px; }}
.badge {{ display: inline-block; background: #dcfce7; color: #166534; border-radius: 8px;
  padding: 4px 12px; font-size: 13px; font-weight: bold; margin-left: 10px; }}
.muted {{ color: #6b7280; font-size: 13px; }}
.card, .table-wrap {{ background: #fff; border: 1px solid #dce2ea;
  border-radius: 12px; padding: 20px; margin-bottom: 16px; }}
.grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }}
.stat {{ background: #fff; border: 1px solid #dce2ea; border-radius: 12px; padding: 16px; }}
.stat .num {{ font-size: 26px; font-weight: bold; color: #1d4ed8; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th, td {{ padding: 9px; border-bottom: 1px solid #e5e9ef; text-align: left; vertical-align: top; }}
th {{ background: #f5f7fa; }}
.pipeline {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; }}
.step {{ background: #eef3fa; border: 1px solid #c9d6e6; border-radius: 10px;
  padding: 12px; text-align: center; font-size: 13px; }}
.step b {{ display: block; font-size: 15px; margin-bottom: 4px; color: #1d4ed8; }}
.demo-row {{ border-bottom: 1px solid #e5e9ef; padding: 12px 0; }}
.kv {{ font-size: 13px; color: #374151; margin-top: 4px; }}
details summary {{ cursor: pointer; color: #1d4ed8; font-size: 13px; margin-top: 6px; }}
iframe {{ width: 100%; height: 420px; border: 1px solid #dce2ea; border-radius: 8px;
  margin-top: 8px; background: #fff; }}
.ok {{ color: #166534; font-weight: bold; }}
.warn {{ color: #b45309; font-weight: bold; }}
.bad {{ color: #b91c1c; font-weight: bold; }}
</style>
</head>
<body>
<main>
  <h1>DW-1 — Sprint 2 «Измеримая система»<span class="badge">ЗАКРЫТ · тег m2</span></h1>
  <p class="muted">Отчёт для проверяющего · {datetime.now():%d.%m.%Y} · репозиторий D:\\dw1</p>

  <h2>Как это работает</h2>
  <div class="card">
    <div class="pipeline">
      <div class="step"><b>1. Ингест</b>.eml → часть: тема, текст, HTML, ссылки, заголовки, картинки, QR</div>
      <div class="step"><b>2. 9 экстракторов</b>стилометрия, CTA, HTML, links, headers, images, QR, template</div>
      <div class="step"><b>3. Модели</b>происхождение (TF-IDF/LLM-детектор) + намерение</div>
      <div class="step"><b>4. Фузия</b>две оси, калибровка, порог отказа</div>
      <div class="step"><b>5. Карточка</b>аналитику: шкалы, признаки, кампания, контр-свидетельства</div>
    </div>
  </div>

  <h2>Что реализовано в Sprint 2 (по ролям)</h2>
  <div class="grid">
    <div class="stat"><div class="num">Задача 1 · w1</div>
      <b>Fast-DetectGPT + калибровка EN/RU</b>
      <ul>
        <li><span class="ok">Готово:</span> GPU-прогон (T4), 576 скоров, калибраторы EN/RU, отчёты, локальная интеграция</li>
        <li><span class="warn">Не доделано:</span> RU-детектор (AUC 0.43), скоры для писем</li>
      </ul>
    </div>
    <div class="stat"><div class="num">Задача 2 · w2</div>
      <b>Устойчивость к 7 атакам</b>
      <ul>
        <li><span class="ok">Готово:</span> 7 атак, TPR@FPR=1%, кривые деградации, тесты 99% coverage</li>
        <li><span class="ok">Критерий:</span> intent устойчив (≤1.2%), provenance ≤1.7%</li>
      </ul>
    </div>
    <div class="stat"><div class="num">Задача 3 · w3</div>
      <b>Кампании v2 + images/QR</b>
      <ul>
        <li><span class="ok">Готово:</span> LSH+HDBSCAN, скелет шаблона, pHash/FFT/EXIF, QR-декодер, карточки</li>
        <li><span class="warn">Не доделано:</span> BCubed F1 на реальных топ-50 (нужны эмбеддинги + разметка)</li>
      </ul>
    </div>
    <div class="stat"><div class="num">Задача 4 · тимлид</div>
      <b>Калиброванная фузия + объяснения + легит-почта</b>
      <ul>
        <li><span class="ok">Готово:</span> модели prov/intent, калибровка (ECE 0.006/0.007), operating points, FP 0% на Enron</li>
        <li><span class="ok">Готово:</span> faithfulness, дашборд v2, интеграция Fast-DetectGPT в fusion</li>
      </ul>
    </div>
  </div>

  <h2>Метрики калибровки фузии (публичный бенчмарк, 600 писем)</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Ось</th><th>ECE до</th><th>ECE после</th><th>AUC до</th><th>AUC после</th></tr></thead>
      <tbody>{calibration_rows}</tbody>
    </table>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Легит-почта (Enron)</th><th>FP-стиль</th><th>FP-интент</th><th>FPR (любой)</th></tr></thead>
      <tbody>{fp_row}</tbody>
    </table>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Порог intent</th><th>TPR</th><th>FPR</th><th>F1</th></tr></thead>
      <tbody>{op_rows}</tbody>
    </table>
  </div>

  <h2>Fast-DetectGPT: калибровка EN/RU</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Язык</th><th>n test</th><th>ECE до → после</th>
      <th>F1 до → после</th><th>ROC-AUC до → после</th></tr></thead>
      <tbody>{llm_rows()}</tbody>
    </table>
  </div>
  <p class="muted">RU-часть ниже случайного (англоязычный surrogate GPT-Neo) — честно
  зафиксированное ограничение: нужен RU-детектор.</p>

  <h2>Устойчивость к 7 атакам маскировки</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Атака</th><th>Provenance TPR</th><th>Деградация</th>
      <th>Intent TPR</th><th>Деградация</th></tr></thead>
      <tbody>{attack_rows()}</tbody>
    </table>
  </div>

  <h2>Демо: 4 письма (карточки встроены)</h2>
  <div class="card">{demo_section()}</div>

  <h2>Кампании (топ-10 из 427)</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th>#</th><th>Писем</th><th>Пример A</th><th>Пример B</th></tr></thead>
      <tbody>{clusters_rows}</tbody>
    </table>
  </div>
  <p class="muted">Всего писем внутри кластеров: {clusters_total} (20 304 письма в корпусе).</p>

  <h2>Ограничения (честно)</h2>
  <div class="card">
    <ul>
      <li><b>RU Fast-DetectGPT не работает</b> (AUC 0.43) — нужен RU-capable детектор.</li>
      <li><b>Fast-DetectGPT не скорил письма</b> — live-пайплайн на письмах использует TF-IDF-модели.</li>
      <li><b>BCubed F1 кампаний не измерен</b> — нужны эмбеддинги и ручная разметка топ-50.</li>
      <li>TF-IDF-детекторы ловят конкретный генератор/домен (HC3/ChatGPT TPR≈0).</li>
    </ul>
  </div>

  <h2>Будущее проекта (Sprint 3)</h2>
  <div class="card">
    <ul>
      <li><b>w1:</b> RU-детектор (Qwen/mGPT surrogate) + скоры для 100–200 писем;</li>
      <li><b>w2:</b> атаки на полном бенчмарке (600) + интеграция в make eval;</li>
      <li><b>w3 + Alina:</b> реальные эмбеддинги + BCubed F1 (ручная разметка топ-50);</li>
      <li><b>тимлид:</b> сравнение LLM-детектор vs TF-IDF на письмах, финальный отчёт и защита.</li>
    </ul>
  </div>

  <h2>Как воспроизвести</h2>
  <div class="card">
    <code>python -m pytest</code> — полный набор тестов<br>
    <code>python scripts/make_report.py</code> — дашборд Sprint 1/2<br>
    <code>python scripts/calibrate_fusion.py</code> — калибровка фузии<br>
    <code>python scripts/operating_points.py</code> — рабочие точки<br>
    <code>python scripts/legit_fp.py</code> — FP на легит-почте<br>
    <code>python scripts/make_sprint2_site.py</code> — этот отчёт
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
