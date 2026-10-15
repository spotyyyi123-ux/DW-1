# -*- coding: utf-8 -*-
"""Сборка демо (задача 4, тимлид): прогоняет пайплайн на списке писем и сохраняет
Evidence JSON каждой карточки в data/runs/<run>/demo/.

Использование:
    python scripts/make_demo.py письмо1.eml письмо2.eml ... [--run sprint1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dw1.pipeline import analyze  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_demo", description="Сборка демо: Evidence JSON по каждому письму")
    ap.add_argument("emails", nargs="+", help=".eml файлы для демо")
    ap.add_argument("--run", default="sprint1", help="имя прогона (папка в data/runs)")
    args = ap.parse_args(argv)

    out_dir = Path(__file__).resolve().parents[1] / "data" / "runs" / args.run / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    for eml in args.emails:
        ev = analyze(eml)
        out = out_dir / f"{Path(eml).stem}.evidence.json"
        out.write_text(ev.to_json(), encoding="utf-8")
        print(f"{eml} -> {out} (rejection={ev.fusion['rejection']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
