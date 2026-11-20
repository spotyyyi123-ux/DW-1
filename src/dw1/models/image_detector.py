"""CPU ResNet embeddings + measured FFT/pHash, never treating metadata as authorship.

No torch/transformers, network requests or model downloads at runtime.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn
from scipy.special import expit

VERSION = "resnet18-pool-fft16-phash64-v1"
LAYER = "resnetv22_pool1_fwd"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ImageEncoder:
    def __init__(self, path: Path, expected_sha: str):
        import cv2

        if sha256(path) != expected_sha:
            raise ValueError("Encoder SHA-256 mismatch")
        cv2.setNumThreads(2)
        try:
            self.net = cv2.dnn.readNetFromONNX(str(path))
        except cv2.error as exc:
            raise ValueError("Invalid ONNX encoder") from exc
        self.model_sha = expected_sha

    def features(self, image: Image.Image) -> np.ndarray:
        import cv2

        # ONNX Model Zoo ResNet preprocessing: RGB, 256 short edge, 224 centre crop.
        rgb = image.convert("RGB")
        w, h = rgb.size
        if max(w, h) / min(w, h) > 32:
            raise ValueError("Extreme aspect ratio outside encoder resource limits")
        resized = rgb.resize(
            (max(256, round(w * 256 / min(w, h))), max(256, round(h * 256 / min(w, h)))),
            Image.Resampling.BILINEAR,
        )
        x, y = (resized.width - 224) // 2, (resized.height - 224) // 2
        pixels = np.asarray(resized.crop((x, y, x + 224, y + 224)), dtype=np.float32) / 255
        pixels = (pixels - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        # OpenCV Net mutates input state; each caller owns its encoder instance.
        try:
            self.net.setInput(pixels.transpose(2, 0, 1)[None].copy())
            embedding = self.net.forward(LAYER).ravel().astype(float)
        except cv2.error as exc:
            raise ValueError("Image encoder failed") from exc
        if len(embedding) != 512:
            raise ValueError("Expected 512-dimensional pooled ResNet embedding")
        gray = np.asarray(rgb.convert("L").resize((32, 32)), dtype=float)
        low = dctn(gray, norm="ortho")[:8, :8].ravel()
        bits = (low > np.median(low[1:])).astype(float)
        power = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean()))) ** 2
        yy, xx = np.indices(power.shape)
        radius = np.minimum((np.sqrt((xx - 16) ** 2 + (yy - 16) ** 2) / 23 * 16).astype(int), 15)
        rings = np.bincount(radius.ravel(), weights=power.ravel(), minlength=16)
        rings = rings / max(float(rings.sum()), 1e-12)
        result = np.concatenate([embedding, rings, bits])
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite image features")
        return result


class ImageDetector:
    def __init__(self, artifact: Path, encoder_path: Path):
        self.config = json.loads(artifact.with_suffix(".json").read_text(encoding="utf-8"))
        if self.config.get("feature_version") != VERSION or self.config.get(
            "model_sha256"
        ) != sha256(artifact):
            raise ValueError("Image detector artifact/version mismatch")
        with np.load(artifact, allow_pickle=False) as saved:
            self.params = {key: saved[key].copy() for key in saved.files}
        for key in ("mean", "scale", "coef"):
            if self.params[key].shape != (592,) or not np.isfinite(self.params[key]).all():
                raise ValueError("Invalid image classifier dimensions/parameters")
        if (self.params["scale"] <= 0).any():
            raise ValueError("Invalid image feature scale")
        for key in ("intercept", "platt_coef", "platt_intercept"):
            if self.params[key].size != 1 or not np.isfinite(self.params[key]).all():
                raise ValueError("Invalid image calibration parameters")
        self.encoder = ImageEncoder(encoder_path, self.config["encoder_sha256"])

    def predict(self, image: Image.Image) -> dict:
        values = self.encoder.features(image)
        p = self.params
        margin = float(((values - p["mean"]) / p["scale"]) @ p["coef"] + p["intercept"].item())
        probability = float(expit(margin * p["platt_coef"].item() + p["platt_intercept"].item()))
        return dict(
            status="scored",
            p_ai=probability,
            label="ai_generated" if probability >= 0.5 else "human",
            confidence=probability if probability >= 0.5 else 1 - probability,
            model_version=VERSION + ":" + self.config["model_sha256"][:12],
            calibrated=True,
            evaluation_scope="CIFAKE_32px_single_generator_subset",
            watermark_status="not_evaluated",
            intent="unknown",
        )


@lru_cache(maxsize=2)
def _detector(
    artifact: str, encoder: str, artifact_mtime: int, encoder_mtime: int, manifest_mtime: int
):
    return ImageDetector(Path(artifact), Path(encoder))


def image_signal(payload: bytes) -> dict:
    artifact = os.environ.get("DW1_IMAGE_MODEL")
    encoder = os.environ.get("DW1_IMAGE_ENCODER")
    if not artifact or not encoder:
        return {
            "status": "model_unavailable",
            "label": "unknown",
            "watermark_status": "not_evaluated",
        }
    try:
        a, e = Path(artifact), Path(encoder)
        detector = _detector(
            str(a),
            str(e),
            a.stat().st_mtime_ns,
            e.stat().st_mtime_ns,
            a.with_suffix(".json").stat().st_mtime_ns,
        )
        # Cached OpenCV Net is shared by API requests: protect mutable input/output state.
        with _INFERENCE_LOCK, Image.open(io.BytesIO(payload)) as source:
            return detector.predict(source)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        return {"status": "model_error", "label": "unknown", "error": type(exc).__name__}


from threading import Lock  # noqa: E402

_INFERENCE_LOCK = Lock()
