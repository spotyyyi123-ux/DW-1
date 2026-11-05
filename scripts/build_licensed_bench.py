"""Сборка лицензионного subset писем (LLMGen Apache-2.0 + Nazario CC-BY-4.0).

Для внешнего прогона (Colab) разрешены только данные с подтверждёнными лицензиями.
seven-phishing и enron сюда НЕ входят (лицензия «на проверке»).

Выход: data/sprint2/bench_licensed/*.eml + labels.csv + bench_licensed.zip

Запуск: python scripts/build_licensed_bench.py
"""
from __future__ import annotations

import csv
import html
import mailbox
import random
import re
from datetime import datetime
from email.header import decode_header
from email.utils import format_datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sprint2" / "bench_licensed"
ZIP = ROOT / "data" / "sprint2" / "bench_licensed.zip"
LLMGEN = ROOT / "data" / "raw" / "llmgen_phishing.csv"
NAZARIO = ROOT / "data" / "raw" / "nazario"
SEED = 42
PER_CLASS = 100

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def _clean(text: str, limit: int = 4000) -> str:
    text = TAG_RE.sub(" ", text or "")
    text = html.unescape(text)
    return WS_RE.sub(" ", text).strip()[:limit]


def _decode(value) -> str:
    if not value:
        return ""
    parts = decode_header(str(value))
    out = []
    for chunk, charset in parts:
        if isinstance(chunk, bytes):
            out.append(chunk.decode(charset or "utf-8", errors="replace"))
        else:
            out.append(chunk)
    return "".join(out)


def _eml(path: Path, headers: dict[str, str], body: str) -> None:
    lines = [f"{k}: {v}" for k, v in headers.items()]
    text = "\n".join(lines) + "\nMIME-Version: 1.0\n"
    text += "Content-Type: text/plain; charset=utf-8\nContent-Transfer-Encoding: 8bit\n\n" + body
    path.write_text(text, encoding="utf-8")


def load_llmgen(n: int) -> list[dict]:
    import pandas as pd

    df = pd.read_csv(LLMGEN)
    pattern = re.compile(
        r"Subject:\s*(?P<subject>.+?)\s+From:\s*(?P<sender>\S+@\S+)\s+(?P<body>.*)", re.DOTALL
    )
    out = []
    for content in df["content"].astype(str):
        m = pattern.match(content.strip())
        if not m:
            continue
        body = _clean(m.group("body"))
        if len(body) < 40:
            continue
        out.append(
            {"subject": m.group("subject")[:200], "sender": m.group("sender"), "body": body}
        )
        if len(out) >= n:
            break
    return out


def _mbox_text(msg) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    return _clean(payload.decode(part.get_content_charset() or "utf-8", "replace"))
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    return _clean(payload.decode(part.get_content_charset() or "utf-8", "replace"))
        return ""
    payload = msg.get_payload(decode=True)
    if payload:
        return _clean(payload.decode(msg.get_content_charset() or "utf-8", "replace"))
    return ""


def load_nazario(n: int) -> list[dict]:
    out = []
    for name in ("phishing-2023", "phishing-2024", "phishing-2025"):
        path = NAZARIO / name
        if not path.exists():
            continue
        box = mailbox.mbox(path, create=False)
        for msg in box:
            subject = _decode(msg.get("Subject"))[:200]
            sender = _decode(msg.get("From")) or "unknown@nazario.example"
            body = _mbox_text(msg)
            if len(body) < 40:
                continue
            out.append({"subject": subject, "sender": sender, "body": body})
            if len(out) >= n:
                break
        if len(out) >= n:
            break
    return out


def main() -> int:
    random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[tuple] = []

    llmgen = load_llmgen(PER_CLASS)
    for i, item in enumerate(llmgen):
        name = f"llmgen_{i:03d}.eml"
        _eml(
            OUT / name,
            {
                "From": item["sender"],
                "To": "user@example.org",
                "Subject": item["subject"],
                "Date": format_datetime(datetime(2026, 10, 30, 12, 0, 0)),
                "Message-ID": f"<licensed-llmgen-{i}@bench.example>",
            },
            item["body"],
        )
        rows.append((name, "ai_generated", "phishing", "llmgen"))

    nazario = load_nazario(PER_CLASS)
    random.shuffle(nazario)
    for i, item in enumerate(nazario):
        name = f"nazario_{i:03d}.eml"
        _eml(
            OUT / name,
            {
                "From": item["sender"],
                "To": "user@example.org",
                "Subject": item["subject"],
                "Date": format_datetime(datetime(2026, 10, 30, 12, 0, 0)),
                "Message-ID": f"<licensed-nazario-{i}@bench.example>",
            },
            item["body"],
        )
        rows.append((name, "human", "phishing", "nazario"))

    with open(OUT / "labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "provenance", "intent", "source"])
        writer.writerows(rows)

    import shutil

    shutil.make_archive(str(ZIP.with_suffix("")), "zip", root_dir=OUT)
    print(f"LLMGen: {len(llmgen)}, Nazario: {len(nazario)} -> {OUT}")
    print(f"zip: {ZIP}")
    print("Лицензии: LLMGen Apache-2.0; Nazario CC-BY-4.0. seven-phishing/enron не включены.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
