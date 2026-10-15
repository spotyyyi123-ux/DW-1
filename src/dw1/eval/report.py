# -*- coding: utf-8 -*-
"""Генератор отчётов из артефактов data/runs (SPEC §4.3: никаких чисел руками). M0: заглушка."""
from __future__ import annotations

import argparse


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dw1.eval.report")
    ap.add_argument("--run", default="bench-v1", help="имя прогона")
    args = ap.parse_args(argv)
    print(f"[M0 stub] run={args.run}: отчёты генерируются из data/runs/<run>/ (с M2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
