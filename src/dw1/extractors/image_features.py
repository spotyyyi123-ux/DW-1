"""Local image measurements; no remote fetching or provenance claims."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image, UnidentifiedImageError
from scipy.fft import dctn


def decode_qr(image: Image.Image) -> dict:
    try:
        import cv2
    except ImportError:
        return {"status": "decoder_unavailable", "codes": []}
    resized = image.copy()
    resized.thumbnail((2048, 2048))
    gray = cv2.cvtColor(np.asarray(resized), cv2.COLOR_RGB2GRAY)
    try:
        ok, values, points, _ = cv2.QRCodeDetector().detectAndDecodeMulti(gray)
        codes = []
        if ok:
            for value, polygon in zip(values, points, strict=True):
                codes.append({"text": value, "points": polygon.tolist(), "decoded": bool(value)})
        if not codes:
            value, polygon, _ = cv2.QRCodeDetector().detectAndDecode(gray)
            if polygon is not None:
                codes.append(
                    {
                        "text": value,
                        "points": polygon.reshape(-1, 2).tolist(),
                        "decoded": bool(value),
                    }
                )
        return {
            "status": "decoded"
            if any(c["decoded"] for c in codes)
            else "detected_undecoded"
            if codes
            else "not_found",
            "codes": codes,
            "coordinate_size": list(resized.size),
        }
    except cv2.error:
        return {"status": "decode_error", "codes": []}


def inspect_image(payload: bytes) -> dict:
    result: dict = {"sha256": hashlib.sha256(payload).hexdigest()}
    if len(payload) > 20 * 1024 * 1024:
        return {**result, "status": "too_large"}
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.width * source.height > 25_000_000:
                return {**result, "status": "too_large"}
            result.update(width=source.width, height=source.height, format=source.format)
            result["exif"] = {
                ExifTags.TAGS.get(k, str(k)): str(v)[:500] for k, v in source.getexif().items()
            }
            rgb = source.convert("RGB")
            gray = np.asarray(rgb.convert("L").resize((32, 32)), dtype=float)
            low = dctn(gray, norm="ortho")[:8, :8].ravel()
            bits = low > np.median(low[1:])
            result["phash"] = f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"
            pixels = np.asarray(rgb.convert("L").resize((256, 256)), dtype=float)
            pixels -= pixels.mean()
            power = np.abs(np.fft.fftshift(np.fft.fft2(pixels))) ** 2
            power[126:131, 126:131] = 0
            total = float(power.sum())
            peaks = np.argsort(power.ravel())[-5:][::-1]
            result["fft_peaks"] = [
                {
                    "fx": (int(i % 256) - 128) / 256,
                    "fy": (int(i // 256) - 128) / 256,
                    "power_fraction": float(power.ravel()[i] / total) if total else 0.0,
                }
                for i in peaks
            ]
            result.update(status="ok", qr=decode_qr(rgb))
            if root_name := os.environ.get("DW1_IMAGE_ARTIFACTS"):
                root = Path(root_name)
                root.mkdir(parents=True, exist_ok=True)
                preview = rgb.copy()
                preview.thumbnail((1200, 1200))
                path = root / (result["sha256"] + ".png")
                preview.save(path)
                result["preview_path"] = str(path)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        result.update(status="invalid", error=type(exc).__name__)
    return result
