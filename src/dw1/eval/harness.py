"""Eval-харнесс (SPEC §4.3, §8). M0: заглушка — артефакты появятся с M2."""
from __future__ import annotations

import argparse


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dw1.eval.harness")
    ap.add_argument("--run", default="bench-v1", help="имя прогона (data/runs/<run>)")
    args = ap.parse_args(argv)
    print(f"[M0 stub] run={args.run}: контракт артефактов — SPEC §4.3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
