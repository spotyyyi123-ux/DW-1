"""Compare raw QR decoding with QR v2 on identical synthetic images.

Run with make eval QR=1 or directly with Python after installing [dev,images].
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import cv2
import qrcode
from PIL import Image, ImageOps
from validate_visual import payload, variants

from dw1.extractors.image_features import decode_qr

DISTORTIONS = (
    "original", "jpeg_q90", "jpeg_q70", "jpeg_q50", "resize_50", "resize_25",
    "rotate_5", "rotate_15", "grayscale", "blur_1", "blur_2", "inverted",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("data/runs/sprint5/qr_validation.json"),
    )
    args = parser.parse_args()
    rows = []
    for box_size in (3, 5, 8):
        expected = f"https://example.invalid/qr/{box_size}"
        image = qrcode.make(expected, box_size=box_size).convert("RGB")
        samples = variants(image)
        samples["inverted"] = payload(ImageOps.invert(image))
        for name in DISTORTIONS:
            with Image.open(io.BytesIO(samples[name])) as sample:
                row = {"box_size": box_size, "distortion": name, "expected": expected}
                for label, preprocess in (("raw", False), ("v2", True)):
                    result = decode_qr(sample, preprocess=preprocess)
                    decoded = [code["text"] for code in result["codes"] if code["decoded"]]
                    row[label] = {
                        "status": result["status"],
                        "payloads": decoded,
                        "correct": expected in decoded,
                    }
                rows.append(row)

    summary = {
        label: {
            "correct": sum(row[label]["correct"] for row in rows),
            "total": len(rows),
        }
        for label in ("raw", "v2")
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"opencv": cv2.__version__, "summary": summary, "cases": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    for label, counts in summary.items():
        print(f"{label}: {counts['correct']}/{counts['total']} correct decodes")
    print(args.out)


if __name__ == "__main__":
    main()
