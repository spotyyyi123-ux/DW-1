"""Сборка размеченного бенчмарка писем для калибровки (Sprint 2, задача 4).

Источники:
- LLMGen-Phishing (GPT)      — AI-фишинг: provenance=ai_generated, intent=phishing
- seven-phishing test.parquet — spam/ham: provenance=human, intent=spam/legitimate
- enron_spam                  — spam/ham: provenance=human, intent=spam/legitimate

Выход: data/sprint2/bench/*.eml + labels.csv (file, provenance, intent, source).

Запуск: python scripts/prepare_bench.py
"""
from __future__ import annotations

import csv
import random
import re
from datetime import datetime
from email.utils import format_datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "sprint2" / "bench"
LLMGEN = ROOT / "data" / "raw" / "llmgen_phishing.csv"
SEVEN = ROOT / "data" / "raw" / "seven-phishing" / "test.parquet"
ENRON = ROOT / "data" / "raw" / "enron_spam" / "enron_spam_data.csv"

SAMPLES = {
    "llmgen": 200,
    "seven_spam": 150,
    "seven_ham": 100,
    "enron_spam": 50,
    "enron_ham": 100,
}
SEED = 42


def _write_eml(path: Path, headers: dict[str, str], body: str) -> None:
    lines = [f"{k}: {v}" for k, v in headers.items()]
    text = "\n".join(lines) + "\nMIME-Version: 1.0\n"
    text += "Content-Type: text/plain; charset=utf-8\nContent-Transfer-Encoding: 8bit\n\n" + body
    path.write_text(text, encoding="utf-8")


def _short(text: str, limit: int = 4000) -> str:
    return " ".join((text or "").split())[:limit]


def load_llmgen(n: int) -> list[dict]:
    import pandas as pd

    df = pd.read_csv(LLMGEN)
    out = []
    pattern = re.compile(
        r"Subject:\s*(?P<subject>.+?)\s+From:\s*(?P<sender>\S+@\S+)\s+(?P<body>.*)",
        re.DOTALL,
    )
    for content in df["content"].astype(str):
        m = pattern.match(content)
        if not m:
            continue
        subject, sender, body = m.group("subject"), m.group("sender"), m.group("body")
        if len(_short(body)) < 30:
            continue
        out.append({"subject": subject, "sender": sender, "body": body})
        if len(out) >= n:
            break
    return out


def load_seven(n: int, wanted: int) -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(SEVEN)
    df = df[df["label"] == wanted]
    out = []
    for _, row in df.iterrows():
        text = _short(str(row.get("text", "")), 4000)
        if len(text) < 30:
            continue
        out.append(
            {
                "subject": str(row.get("subject", ""))[:200],
                "sender": str(row.get("sender", "unknown@demo.example")),
                "body": text,
            }
        )
        if len(out) >= n:
            break
    return out


def load_enron(n: int, kind: str) -> list[dict]:
    import pandas as pd

    df = pd.read_csv(ENRON)
    df = df[df["Spam/Ham"].astype(str).str.lower() == kind]
    df = df.dropna(subset=["Message"])
    out = []
    for _, row in df.iterrows():
        body = _short(str(row["Message"]), 4000)
        if len(body) < 30:
            continue
        out.append(
            {
                "subject": str(row.get("Subject", ""))[:200],
                "sender": "unknown@enron.example",
                "body": body,
            }
        )
        if len(out) >= n:
            break
    return out


def _dump(
    items: list[dict], prefix: str, prov: str, intent: str, source: str, rows: list[tuple]
) -> None:
    for i, item in enumerate(items):
        name = f"{prefix}_{i:03d}.eml"
        _write_eml(
            BENCH / name,
            {
                "From": item["sender"],
                "To": "user@example.org",
                "Subject": item["subject"] or f"[{prefix}]",
                "Date": format_datetime(datetime(2026, 10, 27, 12, 0, 0)),
                "Message-ID": f"<{prefix}-{i}@bench.example>",
                "X-Demo-Source": prefix,
            },
            item["body"],
        )
        rows.append((name, prov, intent, source))


def main() -> int:
    random.seed(SEED)
    BENCH.mkdir(parents=True, exist_ok=True)
    rows: list[tuple] = []

    for prefix, items, prov, intent, source in (
        ("llmgen", load_llmgen(SAMPLES["llmgen"]), "ai_generated", "phishing", "llmgen"),
        ("seven-spam", load_seven(SAMPLES["seven_spam"], 1), "human", "spam", "seven-phishing"),
        ("seven-ham", load_seven(SAMPLES["seven_ham"], 0), "human", "legitimate", "seven-phishing"),
        ("enron-spam", load_enron(SAMPLES["enron_spam"], "spam"), "human", "spam", "enron_spam"),
        ("enron-ham", load_enron(SAMPLES["enron_ham"], "ham"), "human", "legitimate", "enron_spam"),
    ):
        _dump(items, prefix, prov, intent, source, rows)

    with open(BENCH / "labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "provenance", "intent", "source"])
        writer.writerows(rows)

    print(f"бенчмарк готов: {BENCH} ({len(rows)} писем)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
