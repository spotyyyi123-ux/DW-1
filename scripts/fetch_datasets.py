"""Скачивает фиксированные версии данных через curl, записывает хэши и источники."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = "https://huggingface.co/datasets"
SOURCES = {
    "coat": (
        "RussianNLP/coat",
        "a2698e9e441144d39bedbc1d5812bb6351102d06",
        "Apache-2.0",
        ["README.md"]
        + [f"binary/{s}-00000-of-00001.parquet" for s in ("train", "validation", "test")],
    ),
    "hc3-en": (
        "Hello-SimpleAI/HC3",
        "4d0ff18143b5a7e1b1e79beb540c04549d1e59d3",
        "CC-BY-SA-4.0",
        ["README.md", "all.jsonl"],
    ),
    "llmgen-phishing": (
        "Dizzzy0x00/LLMGen-Phishing-Email-Dataset",
        "d717e5457b71cb44dd038dafe6a81cea5e0badab",
        "Apache-2.0",
        ["README.md", "GPT_Phishing_Email_dataset.csv", "CN_Phishing_Email_dataset.csv"],
    ),
    "nazario": (
        "https://monkey.org/~jose/phishing",
        "",
        "CC-BY-4.0",
        ["README.txt", "LICENSE.txt", "phishing-2023", "phishing-2024", "phishing-2025"],
    ),
}
FIELDS = ["dataset", "source_url", "license", "sha256", "downloaded_at", "downloaded_by", "notes"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name):
    source, revision, license_name, files = SOURCES[name]
    manifest = ROOT / "data/datasheet.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8"))) if manifest.exists() else []
    for filename in files:
        url = f"{HF}/{source}/resolve/{revision}/{filename}" if revision else f"{source}/{filename}"
        target = ROOT / "data/raw" / name / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        previous = next((r for r in rows if r["source_url"] == url), None)
        if target.exists() and previous:
            if sha256(target) != previous["sha256"]:
                raise ValueError(f"Hash mismatch: {target}")
            print(f"verified: {name}/{filename}", flush=True)
            continue
        tmp = target.with_name(target.name + ".part")
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "3",
                "--connect-timeout",
                "30",
                "--max-time",
                "600",
                url,
                "--output",
                str(tmp),
            ],
            check=True,
        )
        tmp.replace(target)
        rows = [r for r in rows if r["source_url"] != url]
        rows.append(
            dict(
                zip(
                    FIELDS,
                    [
                        name,
                        url,
                        license_name,
                        sha256(target),
                        datetime.now(timezone.utc).isoformat(),
                        "w1-script",
                        f"file={filename}; revision={revision or 'unversioned upstream'}",
                    ],
                    strict=True,
                )
            )
        )
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"downloaded: {name}/{filename} ({target.stat().st_size} bytes)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", choices=[*SOURCES, "all"])
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.plan or not args.dataset:
        for name, (source, _, license_name, _) in SOURCES.items():
            print(name, license_name, source)
        print("seven-phishing: BLOCKED, mixed license 'other'; needs team review")
        print("CIFAKE: pending access and image-data license review; not needed for text baseline")
        return 0
    for name in SOURCES if args.dataset == "all" else [args.dataset]:
        fetch(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
