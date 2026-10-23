"""Готовит демо-письма из скачанных корпусов (задача 4).

Генерирует два фикстура в tests/fixtures/:
- campaign_email.eml — реальное письмо из кластера seven-phishing
  («Your payment didn't succeed…», CEAS-08, label=1), для демонстрации кампаний;
- llmgen_phishing.eml — письмо, сгенерированное LLM (LLMGen-Phishing, label=1),
  для демонстрации AI-фишинга.

Запуск: python scripts/prepare_demo_emails.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from email.utils import format_datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIXTURES = ROOT / "tests" / "fixtures"
PARQUET = ROOT / "data" / "raw" / "seven-phishing" / "test.parquet"
LLMGEN_CSV = ROOT / "data" / "raw" / "llmgen_phishing.csv"
ROW_ID = 480  # член кластера «Your payment didn't succeed…» из прогона кластеризации


def _write_eml(path: Path, headers: dict[str, str], body: str) -> None:
    lines = [f"{key}: {value}" for key, value in headers.items()]
    text = "\n".join(lines) + "\nMIME-Version: 1.0\n" + \
        "Content-Type: text/plain; charset=utf-8\nContent-Transfer-Encoding: 8bit\n\n" + body
    path.write_text(text, encoding="utf-8")


def make_campaign_email() -> Path:
    import pandas as pd

    df = pd.read_parquet(PARQUET)
    row = df.loc[ROW_ID]
    dt = row["date"].to_pydatetime() if hasattr(row["date"], "to_pydatetime") else datetime(2008, 8, 1)
    out = FIXTURES / "campaign_email.eml"
    _write_eml(
        out,
        {
            "From": row["sender"],
            "To": row["receiver"],
            "Subject": row["subject"],
            "Date": format_datetime(dt),
            "Message-ID": f"<demo-{ROW_ID}@demo.example>",
            "X-Demo-Source": f"seven-phishing test.parquet row {ROW_ID} ({row['dataset_name']}, label={row['label']})",
        },
        str(row["text"]),
    )
    return out


def make_llmgen_email() -> Path:
    import re

    import pandas as pd

    df = pd.read_csv(LLMGEN_CSV)
    row = df.iloc[0]
    content = str(row["content"]).strip()
    subject = ""
    sender = "Bank of America Security Team <security@bankofamerica.com>"
    body = content
    match = re.match(
        r"Subject:\s*(?P<subject>.+?)\s+From:\s*(?P<from>\S+@\S+)\s+(?P<body>.*)",
        content,
        re.DOTALL,
    )
    if match:
        subject = match.group("subject").strip()
        sender = match.group("from").strip()
        body = match.group("body").strip()
    out = FIXTURES / "llmgen_phishing.eml"
    _write_eml(
        out,
        {
            "From": sender,
            "To": "user@example.org",
            "Subject": subject or "Security Alert: Account Compromised",
            "Date": format_datetime(datetime(2026, 10, 23, 10, 0, 0)),
            "Message-ID": "<llmgen-demo@demo.example>",
            "X-Demo-Source": f"LLMGen-Phishing (GPT, label={row['label']})",
        },
        body,
    )
    return out


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    a = make_campaign_email()
    b = make_llmgen_email()
    print(f"готово: {a}")
    print(f"готово: {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
