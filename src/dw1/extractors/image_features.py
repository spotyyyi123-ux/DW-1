"""Local image measurements; no remote fetching or provenance claims."""

from __future__ import annotations

import hashlib
import io
import os
from collections.abc import Iterator
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import ExifTags, Image, UnidentifiedImageError
from scipy.fft import dctn


def _qr_variants(
    cv2: Any, gray: NDArray[np.uint8],
) -> Iterator[tuple[NDArray[np.uint8], NDArray[np.float64]]]:
    """Yield bounded decoder inputs with the transform back to the base image."""
    identity = np.eye(2, 3)
    yield cv2.bitwise_not(gray), identity
    height, width = gray.shape
    for scale in (1, 2, 4):
        if max(width, height) * scale > 2048:
            break
        scaled = gray
        transform = identity.copy()
        if scale != 1:
            scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            transform[:, :2] /= scale
            transform[:, 2] = (1 / scale - 1) / 2
            yield scaled, transform
            yield cv2.bitwise_not(scaled), transform
        binary = cv2.adaptiveThreshold(
            scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5,
        )
        yield binary, transform
        yield cv2.bitwise_not(binary), transform

    for angle in (-5, 5, -15, 15):
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        # Reserve space for QR codes near the edge when rotating.
        a, b = np.abs(matrix[0, :2])
        rotated_width = int(np.ceil(width * a + height * b))
        rotated_height = int(np.ceil(height * a + width * b))
        if max(rotated_width, rotated_height) > 2048:
            continue
        matrix[:, 2] += [(rotated_width - width) / 2, (rotated_height - height) / 2]
        rotated = cv2.warpAffine(gray, matrix, (rotated_width, rotated_height), borderValue=255)
        yield rotated, cv2.invertAffineTransform(matrix)


def _read_qr(detector: Any, image: NDArray[np.uint8]) -> Iterator[tuple[str, NDArray]]:
    _, values, points, _ = detector.detectAndDecodeMulti(image)
    if points is not None:
        for index, polygon in enumerate(points):
            yield values[index] if index < len(values) else "", polygon
    if not values or not all(values):
        value, polygon, _ = detector.detectAndDecode(image)
        if polygon is not None:
            yield value, polygon


def _add_qr(codes: list[dict], value: str, points: NDArray) -> None:
    """Merge repeat detections at the same location, keeping successful decodes."""
    center = points.mean(axis=0)
    size = min(np.ptp(points, axis=0))
    for code in codes:
        previous = np.asarray(code["points"])
        if np.linalg.norm(center - previous.mean(axis=0)) <= max(2.0, size * 0.25):
            if value and not code["decoded"]:
                code.update(text=value, points=points.tolist(), decoded=True)
            return
    codes.append({"text": value, "points": points.tolist(), "decoded": bool(value)})


def decode_qr(image: Image.Image, *, preprocess: bool = True) -> dict:
    try:
        import cv2
    except (ImportError, OSError):
        return {"status": "decoder_unavailable", "codes": []}
    resized = image.copy()
    resized.thumbnail((2048, 2048))
    gray = np.asarray(resized.convert("L"))
    codes: list[dict] = []
    completed = False
    try:
        detector = cv2.QRCodeDetector()
        variants = _qr_variants(cv2, gray) if preprocess else iter(())
        for pixels, transform in chain(((gray, np.eye(2, 3)),), variants):
            try:
                for value, polygon in _read_qr(detector, pixels):
                    points = polygon.reshape(-1, 2) @ transform[:, :2].T + transform[:, 2]
                    _add_qr(codes, value, points)
                completed = True
            except (cv2.error, UnicodeError):
                continue
    except cv2.error:
        pass
    if any(code["decoded"] for code in codes):
        status = "decoded"
    elif codes:
        status = "detected_undecoded"
    else:
        status = "not_found" if completed else "decode_error"
    return {"status": status, "codes": codes, "coordinate_size": list(resized.size)}


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
