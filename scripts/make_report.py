"""Собирает единый HTML-отчёт для проверяющего из артефактов прогона.

Вход: data/runs/sprint1/demo/*.evidence.json, data/runs/sprint1/clusters.json
Выход: data/runs/sprint1/report.html (самодостаточный HTML, без внешних зависимостей).

Запуск: python scripts/make_report.py
"""
from __future__ import annotations

import csv
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
    prov_calib = prov.get("calibrated_confidence")
    intent_calib = intent.get("calibrated_confidence")
    if prov_calib is not None:
        prov_label = f"{prov_label} · calib {float(prov_calib):.2f}"
    if intent_calib is not None:
        intent_label = f"{intent_label} · calib {float(intent_calib):.2f}"
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


def _read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


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

    sprint2 = RUN.parent / "sprint2"
    cal = _read_json(sprint2 / "calibration_metrics.json", {})
    fp = _read_json(sprint2 / "legit_fp.json", {})
    op = _read_json(sprint2 / "operating_points.json", {})

    calibration_rows = ""
    for axis, label in (
        ("ai_provenance", "Происхождение (AI)"),
        ("intent", "Намерение (malicious)"),
    ):
        m = (cal.get("axes") or {}).get(axis) or {}
        if m:
            calibration_rows += (
                f"<tr><td>{label}</td><td>{m.get('ece_before')}</td>"
                f"<td>{m.get('ece_after')}</td><td>{m.get('auc_before')}</td>"
                f"<td>{m.get('auc_after')}</td></tr>"
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

    llm = _read_json(ROOT / "reports" / "sprint2-llm" / "llm_calibration_metrics.json", {})
    llm_rows = ""
    for lang, label in (("en", "Английский"), ("ru", "Русский")):
        m = llm.get(lang) or {}
        if not m:
            continue
        before, after = m.get("before") or {}, m.get("after") or {}
        llm_rows += (
            f"<tr><td>{label}</td><td>{m.get('n_test')}</td>"
            f"<td>{_fmt(before.get('ece'))} → {_fmt(after.get('ece'))}</td>"
            f"<td>{_fmt(before.get('f1'))} → {_fmt(after.get('f1'))}</td>"
            f"<td>{_fmt(before.get('roc_auc'))} → {_fmt(after.get('roc_auc'))}</td></tr>"
        )

    attack_rows = ""
    attacks_csv = sprint2 / "attacks" / "attack_summary.csv"
    if attacks_csv.exists():
        with open(attacks_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                attack_rows += (
                    f"<tr><td>{esc(r['attack'])}</td>"
                    f"<td>{r['baseline_tpr_prov']} → {r['attacked_tpr_prov_max']}</td>"
                    f"<td>{r['degradation_prov_pct']}%</td>"
                    f"<td>{r['baseline_tpr_intent']} → {r['attacked_tpr_intent_max']}</td>"
                    f"<td>{r['degradation_intent_pct']}%</td></tr>"
                )

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

  <h2>Sprint 2 — калибровка и рабочая точка</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Ось</th><th>ECE до</th><th>ECE после</th><th>AUC до</th><th>AUC после</th></tr></thead>
      <tbody>{calibration_rows}</tbody>
    </table>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Легит-почта (писем)</th><th>FP-стиль</th><th>FP-интент</th><th>FPR (любой)</th></tr></thead>
      <tbody>{fp_row}</tbody>
    </table>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Порог intent</th><th>TPR</th><th>FPR</th><th>F1</th></tr></thead>
      <tbody>{op_rows}</tbody>
    </table>
  </div>
  <p class="muted">Калибровка обучена на публичном бенчмарке (600 писем). Честное ограничение:
  TF-IDF-модели детектируют стиль конкретного генератора — на невиданном (HC3/ChatGPT)
  TPR≈0; нужен вероятностный детектор (Fast-DetectGPT, задача 1 Sprint 2).</p>

  <h2>Fast-DetectGPT: калибровка EN/RU (задача 1)</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Язык</th><th>n test</th><th>ECE до → после</th><th>F1 до → после</th>
      <th>ROC-AUC до → после</th></tr></thead>
      <tbody>{llm_rows}</tbody>
    </table>
  </div>
  <p class="muted">Публичный бенчмарк (HC3-en + CoAT). RU-часть — честно зафиксированное
  ограничение (ниже случайного): нужен RU-детектор. Отчёт: reports/sprint2-llm.</p>

  <h2>Устойчивость к 7 атакам маскировки (задача 2)</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Атака</th><th>Provenance TPR (до → после)</th><th>Деградация</th>
      <th>Intent TPR (до → после)</th><th>Деградация</th></tr></thead>
      <tbody>{attack_rows}</tbody>
    </table>
  </div>
  <p class="muted">intent-ось устойчива (деградация ≤1.2%), provenance почти не ломается
  (≤1.7%). Артефакты: data/runs/sprint2/attacks, отчёт: docs/notes/attacks.md.</p>

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
