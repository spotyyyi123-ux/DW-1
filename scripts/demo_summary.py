"""Консольная демо-сводка: 4 письма + топ кампаний (быстро, без пересчёта).

Использует уже сохранённые артефакты:
- data/runs/sprint1/demo/*.evidence.json
- data/runs/sprint1/clusters.json

Запуск: python scripts/demo_summary.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "runs" / "sprint1" / "demo"
CLUSTERS = ROOT / "data" / "runs" / "sprint1" / "clusters.json"

PROV = {
    "human": "Человек",
    "ai_generated": "ИИ-генерировано",
    "ai_polished": "ИИ-причёсано",
    "template": "Шаблон",
    "unknown": "не определено",
}
INTENT = {
    "phishing": "Фишинг",
    "spam": "Спам",
    "legitimate": "Легитимно",
    "unknown": "не определено",
}


def demo_rows() -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    for path in sorted(DEMO.glob("*.evidence.json")):
        rows.append(
            (path.name.replace(".evidence.json", ""), json.loads(path.read_text(encoding="utf-8")))
        )
    return rows


def main() -> int:
    print("=" * 96)
    print("DW-1: детекция AI-assisted фишинга — демо-сводка (Sprint 1, тег m1)")
    print("=" * 96)

    rows = demo_rows()
    if not rows:
        print("Нет демо-артефактов: запустите scripts\\make_demo.py --run sprint1")
        return 1

    print("\nДЕМО-ПИСЬМА (две независимые оси + кампания):\n")
    print(f"{'письмо':<24}{'происхождение':<26}{'злонамеренность':<24}{'кампания':<20}усилители")
    print("-" * 118)
    for stem, ev in rows:
        fusion = ev.get("fusion") or {}
        prov = fusion.get("ai_provenance") or {}
        intent = fusion.get("intent") or {}
        campaign = ev.get("campaign") or {}
        threats = fusion.get("threat_signals") or []
        prov_s = f"{PROV.get(prov.get('label'), prov.get('label'))} ({float(prov.get('confidence') or 0):.2f})"
        intent_s = f"{INTENT.get(intent.get('label'), intent.get('label'))} ({float(intent.get('confidence') or 0):.2f})"
        camp_s = (
            f"{campaign.get('cluster_id')} · {campaign.get('member_count')} писем"
            if campaign.get("cluster_id")
            else "—"
        )
        print(f"{stem:<24}{prov_s:<26}{intent_s:<24}{camp_s:<20}{', '.join(threats) or '—'}")

    print("\nТОП-3 КАМПАНИИ (из 427, кластеризация MinHash/LSH на 20 304 письмах):\n")
    if CLUSTERS.exists():
        clusters = json.loads(CLUSTERS.read_text(encoding="utf-8"))
        for c in clusters[:3]:
            example = (c.get("examples") or [""])[0][:85]
            print(f"  {c.get('size'):>4} писем — {example}...")
    else:
        print("  clusters.json не найден: запустите кластеризацию")

    print("\nКАК ВОСПРОИЗВЕСТИ:")
    print("  python -m dw1 tests\\fixtures\\sample.eml        — карточка письма (JSON)")
    print("  python scripts\\make_demo.py ... --run sprint1   — собрать демо")
    print("  python scripts\\make_report.py                   — HTML-дашборд")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
