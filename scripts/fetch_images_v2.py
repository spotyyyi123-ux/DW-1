"""Fetch the pinned public MIT CIFAKE subset and Apache-2.0 ResNet ONNX encoder.

Only explicit public URLs; no user authentication, emails or torch imports.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image

REPO = "jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images"
REV = "e112a942abaecd02b6b1f6f646c807d56be8fb62"
MODEL = "onnxmodelzoo/resnet18-v2-7"
MODEL_REV = "0b071fb1cd97fc8f111036e48844b63dc480d5a2"
MODEL_SHA = "3b4d048057854022232aa20bf68262e31718ab7119a17f920b8a2dc1433f824b"


def get(url):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DW1-public-CIFAKE-evaluation"})
            with urllib.request.urlopen(req, timeout=90) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/raw/images-v2"))
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    encoder = out / "resnet18-v2-7.onnx"
    if not encoder.exists():
        encoder.write_bytes(get(f"https://huggingface.co/{MODEL}/resolve/{MODEL_REV}/{encoder.name}"))
    if hashlib.sha256(encoder.read_bytes()).hexdigest() != MODEL_SHA:
        raise ValueError("Pinned ONNX SHA mismatch")
    (out / "model-source.json").write_text(json.dumps(dict(model=MODEL, revision=MODEL_REV,
        license="Apache-2.0", sha256=MODEL_SHA), indent=2), encoding="utf-8", newline="\n")
    license_text = get(f"https://raw.githubusercontent.com/{REPO}/{REV}/README.md")
    if b"MIT" not in license_text:
        raise ValueError("Upstream license assertion missing; review source before download")
    (out / "CIFAKE-README.md").write_bytes(license_text)
    jobs = []
    for partition in ("train", "test"):
        for label, folder in ((0, "REAL"), (1, "FAKE")):
            entries = json.loads(get(f"https://api.github.com/repos/{REPO}/contents/DATASET/{partition}/{folder}?ref={REV}"))
            entries = sorted([r for r in entries if r["type"] == "file"], key=lambda r: r["name"])
            if len(entries) < 400:
                raise ValueError("Upstream listing does not contain expected candidates")
            random.Random(20261120 + label).shuffle(entries)
            for i, row in enumerate(entries[:400 if partition == "train" else 200]):
                split = ("train" if i < 300 else "validation") if partition == "train" else "test"
                path = f"{partition}/{folder}/{row['name']}"
                jobs.append(dict(path=path, split=split, label=label,
                                 url=f"https://raw.githubusercontent.com/{REPO}/{REV}/DATASET/{quote(path, safe='/()')}"))

    def download(row):
        path = out / "cifake" / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(get(row["url"]))
        with Image.open(path) as source:
            if source.size != (32, 32):
                raise ValueError("Expected CIFAKE 32x32 input")
            pixels = np.asarray(source.convert("RGB")).tobytes()
        return dict(row, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    pixel_sha256=hashlib.sha256(pixels).hexdigest())

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(download, jobs))
    seen, unique = set(), []
    for row in rows:
        if row["pixel_sha256"] not in seen:
            unique.append(row)
            seen.add(row["pixel_sha256"])
    result = dict(dataset="CIFAKE", license="MIT",
        license_evidence=f"https://github.com/{REPO}/blob/{REV}/README.md", revision=REV,
        selection="seed20261120; GitHub contents API first1000 per official partition/class; train300+val100 and test200 per class; exact pixel dedup globally",
        n_downloaded=len(rows), n_unique=len(unique), rows=unique)
    (out / "dataset.json").write_text(json.dumps(result, indent=2), encoding="utf-8", newline="\n")
    print(f"CIFAKE images: {len(unique)}; encoder SHA verified; raw files remain gitignored")


if __name__ == "__main__":
    main()
