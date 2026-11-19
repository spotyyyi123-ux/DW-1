"""Визуальная валидация: устойчивость QR/изображений к искажениям (Sprint 5, задача 4).

Берёт эталонное изображение с QR, применяет искажения (JPEG, ресайз, поворот, grayscale,
blur, затемнение), прогоняет через inspect_image (pHash/FFT/EXIF + decode_qr) и считает:
- decode rate (доля изображений, где QR декодирован);
- совпадение payload с эталоном;
- Hamming-расстояние pHash до эталона (64 бита).

Выход: data/runs/sprint5/visual_validation.json + таблица в stdout.

Запуск: python scripts/validate_visual.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dw1.extractors.image_features import inspect_image  # noqa: E402

SRC = ROOT / "docs" / "examples" / "task3" / "images"
OUT = ROOT / "data" / "runs" / "sprint5" / "visual_validation.json"


def payload(image: Image.Image, fmt: str = "PNG", quality: int = 95) -> bytes:
    buf = io.BytesIO()
    if fmt == "JPEG":
        image.convert("RGB").save(buf, format="JPEG", quality=quality)
    else:
        image.save(buf, format="PNG")
    return buf.getvalue()


def variants(image: Image.Image) -> dict[str, bytes]:
    import numpy as np

    rgb = image.convert("RGB")
    out = {"original": payload(rgb)}
    for q in (90, 70, 50, 30):
        out[f"jpeg_q{q}"] = payload(rgb, "JPEG", q)
    for scale, name in ((0.5, "resize_50"), (0.25, "resize_25"), (0.15, "resize_15")):
        out[name] = payload(rgb.resize((int(rgb.width * scale), int(rgb.height * scale))))
    for deg, name in ((5, "rotate_5"), (15, "rotate_15"), (45, "rotate_45")):
        out[name] = payload(rgb.rotate(deg, expand=True, fillcolor="white"))
    out["grayscale"] = payload(ImageOps.grayscale(rgb))
    for radius, name in ((1, "blur_1"), (2, "blur_2"), (3, "blur_3")):
        out[name] = payload(rgb.filter(ImageFilter.GaussianBlur(radius)))
    out["darken_0.7"] = payload(ImageEnhance.Brightness(rgb).enhance(0.7))
    out["contrast_0.5"] = payload(ImageEnhance.Contrast(rgb).enhance(0.5))

    # перспектива: сдвиг четырёх углов
    w, h = rgb.size
    quad = (0.05 * w, 0.10 * h, 0.95 * w, 0.0, 0.90 * w, h, 0.0, 0.90 * h)
    out["perspective"] = payload(rgb.transform((w, h), Image.QUAD, quad, fillcolor="white"))

    # шум
    arr = np.asarray(rgb).astype(np.int16) + np.random.default_rng(42).normal(0, 25, (h, w, 3))
    out["noise"] = payload(Image.fromarray(np.clip(arr, 0, 255).astype("uint8")))

    # окклюзия: закрываем часть QR (нижний правый угол)
    occ = rgb.copy()
    side = int(min(w, h) * 0.12)
    occ.paste((255, 255, 255), (w - side, h - side, w, h))
    out["occlusion"] = payload(occ)

    # даунскейл→апскейл (потеря деталей)
    small = rgb.resize((max(8, int(w * 0.3)), max(8, int(h * 0.3))))
    out["downup"] = payload(small.resize((w, h), Image.LANCZOS))
    return out


def phash_distance(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def main() -> int:
    sources = sorted(SRC.glob("*.png"))
    if not sources:
        raise SystemExit("нет эталонных изображений")
    image = Image.open(sources[0]).convert("RGB")

    base = inspect_image(payload(image))
    base_codes = [c["text"] for c in (base.get("qr") or {}).get("codes", []) if c.get("decoded")]

    rows = []
    for name, data in variants(image).items():
        result = inspect_image(data)
        qr = result.get("qr") or {}
        codes = [c["text"] for c in qr.get("codes", []) if c.get("decoded")]
        rows.append(
            {
                "variant": name,
                "image_status": result.get("status"),
                "qr_status": qr.get("status"),
                "decoded": bool(codes),
                "payload_match": codes[:1] == base_codes[:1] if base_codes else None,
                "phash": result.get("phash"),
                "phash_distance": phash_distance(base["phash"], result["phash"])
                if result.get("phash") and base.get("phash")
                else None,
                "width": result.get("width"),
                "height": result.get("height"),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(rows)
    decoded = sum(r["decoded"] for r in rows)
    print(f"эталон: {sources[0].name} | QR payload: {base_codes[:1]}")
    print(f"{'искажение':14}{'QR статус':20}{'декод':>6}{'pHash dist':>11}")
    print("-" * 55)
    for r in rows:
        print(
            f"{r['variant']:14}{str(r['qr_status']):20}{'да' if r['decoded'] else 'нет':>6}"
            f"{str(r['phash_distance']):>11}"
        )
    print(f"\ndecode rate: {decoded}/{total} = {decoded / total:.0%} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
