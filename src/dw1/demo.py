"""CLI M0: python -m dw1 <file.eml> [--out out.json] -> Evidence JSON v1."""
from __future__ import annotations

import argparse

from dw1.pipeline import analyze


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dw1", description="DW-1: .eml -> Evidence JSON v1")
    ap.add_argument("eml", help="путь к .eml")
    ap.add_argument("--out", default=None, help="записать JSON в файл вместо stdout")
    args = ap.parse_args(argv)

    evidence = analyze(args.eml)
    data = evidence.to_json()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(data)
        print(f"Evidence записан в {args.out}")
    else:
        print(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
