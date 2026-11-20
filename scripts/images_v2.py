"""Train on clean CIFAKE train, calibrate on clean validation, freeze under test attacks."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dw1.models.image_detector import VERSION, ImageEncoder, sha256  # noqa: E402

ATTACKS = ("clean", "jpeg90", "jpeg70", "jpeg50", "jpeg30", "resize50", "resize25", "crop80")


def distort(image, name):
    rgb = image.convert("RGB")
    if name.startswith("jpeg"):
        stream = io.BytesIO()
        rgb.save(stream, format="JPEG", quality=int(name[4:]))
        stream.seek(0)
        return Image.open(stream).convert("RGB")
    if name.startswith("resize"):
        factor = int(name[6:]) / 100
        return rgb.resize(
            (max(1, round(rgb.width * factor)), max(1, round(rgb.height * factor))),
            Image.Resampling.LANCZOS,
        )
    if name == "crop80":
        w, h = rgb.size
        return rgb.crop((round(w * 0.1), round(h * 0.1), round(w * 0.9), round(h * 0.9)))
    if name != "clean":
        raise ValueError("Unknown image attack")
    return rgb


def ece(y, p):
    return float(
        sum(
            mask.mean() * abs(y[mask].mean() - p[mask].mean())
            for k in range(10)
            if (mask := ((p >= k / 10) & (p < (k + 1) / 10 if k < 9 else p <= 1))).any()
        )
    )


def evaluate(predictions: Path, out: Path):
    frame = pd.read_parquet(predictions)
    required = {
        "image_id",
        "original_sha256",
        "pixel_sha256",
        "split",
        "label",
        "attack",
        "p_ai",
        "model_version",
    }
    if not required <= set(frame) or frame[list(required)].isna().any().any():
        raise ValueError("Incomplete image predictions")
    if not set(frame.label) <= {0, 1} or not set(frame.split) <= {"train", "validation", "test"}:
        raise ValueError("Invalid labels/splits")
    if not np.isfinite(frame.p_ai).all() or not frame.p_ai.between(0, 1).all():
        raise ValueError("Invalid probabilities")
    if (frame.groupby("original_sha256").split.nunique() != 1).any() or (
        frame.groupby("pixel_sha256").split.nunique() != 1
    ).any():
        raise ValueError("Cross-split image leakage")
    if frame.duplicated(["image_id", "attack"]).any() or frame.model_version.nunique() != 1:
        raise ValueError("Duplicate predictions/model mismatch")
    test = frame[frame.split == "test"]
    ids = set(test[test.attack == "clean"].image_id)
    if set(test.attack) != set(ATTACKS) or not ids:
        raise ValueError("Incomplete robustness grid")
    if (frame.groupby("image_id").label.nunique() != 1).any():
        raise ValueError("Changed labels")
    for key in ("split", "original_sha256", "pixel_sha256"):
        if (frame.groupby("image_id")[key].nunique() != 1).any():
            raise ValueError("Changed source identity/split")
    if (frame.loc[frame.split != "test", "attack"] != "clean").any():
        raise ValueError("Only test images may be attacked")
    if (
        "variant_pixel_sha256" in frame
        and (frame.groupby("variant_pixel_sha256").split.nunique() > 1).any()
    ):
        raise ValueError("Transformed pixels cross split boundary")
    receipt = predictions.with_name("model.json")
    provenance = json.loads(receipt.read_text()) if receipt.exists() else {}
    verified = provenance.get("predictions_sha256") == sha256(predictions)
    if provenance and not verified:
        raise ValueError("Predictions do not match the saved training receipt")
    if verified and frame.model_version.iloc[0] != VERSION + ":" + provenance["model_sha256"][:12]:
        raise ValueError("Predictions/model version mismatch")
    rows = []
    for attack in ATTACKS:
        sub = test[test.attack == attack]
        if set(sub.image_id) != ids or set(sub.label) != {0, 1}:
            raise ValueError("Attack population differs from clean")
        y, p = sub.label.to_numpy(), sub.p_ai.to_numpy()
        rows.append(
            dict(
                attack=attack,
                n=len(y),
                auc=float(roc_auc_score(y, p)),
                f1=float(f1_score(y, p >= 0.5)),
                ece=ece(y, p),
            )
        )
    for row in rows:
        row.update(auc_delta=row["auc"] - rows[0]["auc"], f1_delta=row["f1"] - rows[0]["f1"])
    result: dict = dict(
        status="measured" if verified else "unverified_predictions",
        predictions_sha256=sha256(predictions),
        feature_version=VERSION,
        threshold=0.5,
        calibration="clean validation only; frozen under all attacks",
        scope=(
            "CIFAKE official repository API candidate subset; 32x32; one generator, not email images"
            if verified
            else "Provided predictions only; training/data provenance not verified"
        ),
        n_train=int(((frame.split == "train") & (frame.attack == "clean")).sum()),
        n_validation=int(((frame.split == "validation") & (frame.attack == "clean")).sum()),
        n_test=len(ids),
        metrics=rows,
        watermark="not evaluated; absence is not evidence of real origin",
        training_manifest_sha256=sha256(receipt) if verified else None,
    )
    if "p_ai_uncalibrated" in test:
        clean = test[test.attack == "clean"]
        y = clean.label.to_numpy()
        p = clean.p_ai_uncalibrated.to_numpy()
        result["clean_before_calibration"] = dict(
            auc=float(roc_auc_score(y, p)), f1=float(f1_score(y, p >= 0.5)), ece=ece(y, p)
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8", newline="\n")
    pd.DataFrame(rows).to_csv(out / "robustness.csv", index=False)
    report: list[str] = [
        "# Images v2 — измеренный поднабор CIFAKE",
        "",
        str(result["scope"]),
        "",
        f"Train {result['n_train']}, validation {result['n_validation']}, test {len(ids)}. Порог 0.5 фиксирован.",
        "",
        "| Условие | n | AUC | F1 | ECE | ΔAUC | ΔF1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    report += [
        f"| {r['attack']} | {r['n']} | {r['auc']:.4f} | {r['f1']:.4f} | {r['ece']:.4f} | {r['auc_delta']:+.4f} | {r['f1_delta']:+.4f} |"
        for r in rows
    ]
    report += [
        "",
        "Метрики относятся к происхождению изображения, не к злонамеренности. Измерительные pHash/FFT/EXIF не являются самостоятельным детектором. Watermark не измерен. Перенос на другие генераторы и реальные вложения не доказан.",
    ]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
    return result


def train(dataset: Path, encoder_path: Path, out: Path):
    cfg = json.loads(dataset.read_text())
    if cfg["license"] != "MIT" or not cfg["revision"] or not cfg["license_evidence"]:
        raise ValueError("Missing licensed source provenance")
    rows = cfg["rows"]
    if len({r["pixel_sha256"] for r in rows}) != len(rows):
        raise ValueError("Duplicate pixels across evaluation population")
    source = json.loads(encoder_path.with_name("model-source.json").read_text())
    encoder = ImageEncoder(encoder_path, source["sha256"])
    cache = out / "feature-cache"
    cache.mkdir(parents=True, exist_ok=True)
    features, records = [], []
    for i, row in enumerate(rows):
        path = dataset.parent / "cifake" / row["path"]
        if (
            not path.resolve().is_relative_to((dataset.parent / "cifake").resolve())
            or sha256(path) != row["sha256"]
        ):
            raise ValueError("Dataset image changed or escaped dataset root")
        with Image.open(path) as original:
            for attack in ATTACKS if row["split"] == "test" else ("clean",):
                rgb = distort(original, attack)
                pixels = hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
                key = hashlib.sha256(
                    (source["sha256"] + VERSION + pixels + str(rgb.size)).encode()
                ).hexdigest()
                saved = cache / (key + ".npy")
                if saved.exists():
                    vector = np.load(saved, allow_pickle=False)
                else:
                    vector = encoder.features(rgb)
                    np.save(saved, vector)
                features.append(vector)
                records.append(
                    dict(
                        image_id=row["path"],
                        original_sha256=row["sha256"],
                        pixel_sha256=row["pixel_sha256"],
                        variant_pixel_sha256=pixels,
                        label=row["label"],
                        split=row["split"],
                        attack=attack,
                    )
                )
        if i % 100 == 0:
            print(f"Features {i}/{len(rows)}", flush=True)
    x = np.asarray(features)
    f = pd.DataFrame(records)
    training, validation = f.split == "train", f.split == "validation"
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=42))
    with threadpool_limits(limits=2):
        model.fit(x[training], f.loc[training, "label"])
        margins = model.decision_function(x)
        platt = LogisticRegression(C=1, random_state=42).fit(
            margins[validation, None], f.loc[validation, "label"]
        )
    scaler, classifier = model.steps[0][1], model.steps[1][1]
    out.mkdir(parents=True, exist_ok=True)
    artifact = out / "model.npz"
    np.savez(
        artifact,
        mean=scaler.mean_,
        scale=scaler.scale_,
        coef=classifier.coef_[0],
        intercept=classifier.intercept_,
        platt_coef=platt.coef_[0],
        platt_intercept=platt.intercept_,
    )
    manifest = dict(
        feature_version=VERSION,
        model_sha256=sha256(artifact),
        encoder_sha256=source["sha256"],
        encoder_source=source,
        dataset_sha256=sha256(dataset),
        dataset_revision=cfg["revision"],
        license="MIT",
        license_evidence=cfg["license_evidence"],
        selection=cfg["selection"],
        training_pixel_sha256=f.loc[training, "pixel_sha256"].tolist(),
        threshold=0.5,
        calibration="Platt on clean validation; test never used for fitting or selection",
        feature_source_sha256=sha256(ROOT / "src/dw1/models/image_detector.py"),
        training_script_sha256=sha256(Path(__file__)),
    )
    artifact.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    f["raw_margin"] = margins
    f["p_ai_uncalibrated"] = expit(margins)
    f["p_ai"] = expit(margins * platt.coef_[0, 0] + platt.intercept_[0])
    f["model_version"] = VERSION + ":" + manifest["model_sha256"][:12]
    f.to_parquet(out / "predictions.parquet", index=False)
    manifest["predictions_sha256"] = sha256(out / "predictions.parquet")
    artifact.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    return evaluate(out / "predictions.parquet", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--encoder", type=Path)
    ap.add_argument("--predictions", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "data/runs/sprint5/images")
    a = ap.parse_args()
    if a.predictions:
        result = evaluate(a.predictions, a.out)
    elif a.dataset and a.encoder:
        result = train(a.dataset, a.encoder, a.out)
    else:
        ap.error("Supply --dataset + --encoder, or --predictions for evaluation")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
