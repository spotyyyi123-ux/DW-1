# -*- coding: utf-8 -*-
"""Загрузка датасетов (W1). Полный аудит и красные флаги: docs/datasets.md.

Протокол: каждый набор -> data/raw/<dataset>/ + строка в data/datasheet.csv
(URL, лицензия, sha256, дата, кто скачал). Лицензия неясна = блокер (AGENTS.md).
"""
from __future__ import annotations

import argparse

# (имя, источник, лицензия, прим. объём) — стартовый набор из docs/datasets.md §4
PLAN = [
    ("coat", "https://huggingface.co/datasets/RussianNLP/coat", "Apache-2.0", "~150 МБ"),
    ("semeval2024-task8", "https://huggingface.co/datasets/d0rj/SemEval2024-task8", "Apache-2.0 (зеркало; оригинал не верифицирован)", "~665 МБ"),
    ("hc3-en", "https://huggingface.co/datasets/Hello-SimpleAI/HC3", "CC-BY-SA-4.0", "~147 МБ"),
    ("seven-phishing", "https://huggingface.co/datasets/puyang2025/seven-phishing-email-datasets", "other (микс 7 корпусов)", "230 МБ"),
    ("nazario", "https://monkey.org/~jose/phishing/", "CC-BY-4.0", "~150 МБ"),
    ("llmgen-phishing", "https://huggingface.co/datasets/Dizzzy0x00/LLMGen-Phishing-Email-Dataset", "Apache-2.0", "~5 МБ"),
    ("cifake", "https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images", "MIT", "106 МБ"),
    ("enron-full", "https://www.cs.cmu.edu/~enron/", "не указана — на юр. проверку", "1.7 ГБ"),
    ("raid-test", "https://huggingface.co/datasets/liamdugan/raid", "MIT", "test.csv 1.22 ГБ"),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fetch_datasets")
    ap.add_argument("--plan", action="store_true", help="показать план загрузки")
    ap.add_argument("dataset", nargs="?", help="имя набора для загрузки")
    args = ap.parse_args(argv)

    if args.plan or not args.dataset:
        print(f"{'имя':<18}{'лицензия':<45}{'объём':<12}источник")
        for name, url, lic, size in PLAN:
            print(f"{name:<18}{lic:<45}{size:<12}{url}")
        return 0

    print(f"[TODO W1] загрузка '{args.dataset}' не реализована: "
          "см. scripts/fetch_datasets.py и docs/datasets.md §5")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
