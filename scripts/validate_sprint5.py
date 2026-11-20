"""RU report audit + reproducible synthetic QR validation + measured image robustness.

RU aggregates are explicitly not re-computed when object-level predictions are absent.
No final site, defense materials or milestone tag are produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from dw1.extractors.image_features import inspect_image  # noqa: E402
from scripts.validate_visual import phash_distance, variants  # noqa: E402


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def audit_ru(source, commit):
    required = {"legacy_increasing", "train_direction"}
    if not required <= set(source["metrics"]):
        raise ValueError("Missing RU comparison")
    rows = []
    for method, values in source["metrics"].items():
        if values["n_test"] <= 0:
            raise ValueError("Missing RU population")
        for stage in ("before", "after"):
            metrics = values[stage]
            if any(
                not math.isfinite(metrics[k]) or not 0 <= metrics[k] <= 1
                for k in ("ece", "f1", "roc_auc")
            ):
                raise ValueError("Invalid RU aggregate")
            rows.append(dict(method=method, stage=stage, n_test=values["n_test"], **metrics))
    if (
        source["metrics"]["legacy_increasing"]["before"]
        != source["metrics"]["train_direction"]["before"]
    ):
        raise ValueError("RU baseline differs across methods")
    return dict(
        status="reported_not_recomputed",
        source_commit=commit,
        source_path="reports/sprint5-ru/metrics.json",
        score_sha256=source["score_sha256"],
        n_input=source["n_input"],
        n_scored=source["n_scored"],
        limitation="Owner-reported metrics; raw scores unavailable. Bounds/schema/common baseline checked, predictions/calibration/leakage not independently reproduced. CoAT texts, not Russian emails.",
        rows=rows,
    )


def validate_qr(out):
    import qrcode

    rows = []
    for sample in range(5):
        expected = f"https://benchmark.example/validation/{sample}"
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=5 + sample,
            border=4,
        )
        qr.add_data(expected)
        qr.make(fit=True)
        original = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        reference = None
        for attack, data in variants(original).items():
            features = inspect_image(data)
            status = features["qr"]["status"]
            codes = sorted({c["text"] for c in features["qr"]["codes"] if c.get("decoded")})
            if reference is None:
                reference = features["phash"]
            rows.append(
                dict(
                    sample_id=f"generated-{sample}",
                    variant=attack,
                    image_sha256=hashlib.sha256(data).hexdigest(),
                    expected_payload=expected,
                    decoded_payloads=json.dumps(codes),
                    decoder_status=status,
                    payload_match=codes == [expected],
                    phash_distance=phash_distance(reference, features["phash"]),
                    width=features["width"],
                    height=features["height"],
                )
            )
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "visual_predictions.parquet", index=False)
    statuses = frame.decoder_status.value_counts().to_dict()
    import cv2

    metrics = dict(
        status="measured_synthetic_baseline",
        n_sources=5,
        n_variants=len(frame),
        payload_match_rate=float(frame.payload_match.mean()),
        n_correct=int(frame.payload_match.sum()),
        statuses=statuses,
        opencv_version=cv2.__version__,
        decoder="current production QRCodeDetector; no new QR-v2 preprocessing",
        phash_mean=float(frame.phash_distance.mean()),
        predictions_sha256=hashlib.sha256(
            (out / "visual_predictions.parquet").read_bytes()
        ).hexdigest(),
        per_attack=frame.groupby("variant")
        .payload_match.agg(["count", "sum", "mean"])
        .reset_index()
        .to_dict("records"),
        scope="Five generated single QR payloads; not real mail attachments; exact payload match, not merely any decoded string",
    )
    save(out / "visual_metrics.json", metrics)
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ru-ref", default="origin/w1/sprint5-ru-calibration")
    ap.add_argument("--images", type=Path, default=ROOT / "data/runs/sprint5/images/metrics.json")
    ap.add_argument("--out", type=Path, default=ROOT / "data/runs/sprint5/validation")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", a.ru_ref], cwd=ROOT, text=True).strip()
    raw = subprocess.check_output(
        ["git", "show", commit + ":reports/sprint5-ru/metrics.json"], cwd=ROOT
    )
    source = json.loads(raw)
    save(a.out / "ru_source.json", source)
    ru = audit_ru(source, commit)
    save(a.out / "ru_validation.json", ru)
    qr = validate_qr(a.out)
    images = json.loads(a.images.read_text()) if a.images.exists() else {"status": "not_measured"}
    result = dict(
        ru_status=ru["status"],
        qr_status=qr["status"],
        images_status=images["status"],
        ru=ru,
        qr=qr,
        images=images,
        final_report_site_defense_tag="not_requested",
    )
    save(a.out / "validation.json", result)
    print(json.dumps({k: result[k] for k in ("ru_status", "qr_status", "images_status")}, indent=2))


if __name__ == "__main__":
    main()
