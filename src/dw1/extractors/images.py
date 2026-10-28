"""Measured image features; no trained provenance classifier."""

from dw1.extractors import BaseExtractor, register


def measured_signal(evidence: list[dict], version: str) -> dict:
    return {
        axis: {
            "label": "unknown",
            "confidence": 0.0,
            "evidence": evidence,
            "model_version": version,
        }
        for axis in ("ai_provenance", "intent")
    }


@register
class ImagesExtractor(BaseExtractor):
    name = "images"
    version = "image-forensics-1.0"

    def extract(self, parts: dict) -> dict:
        evidence = [
            {"signal": "images_count", "value": len(parts.get("images", [])), "weight": 0.0}
        ]
        for i, image in enumerate(parts.get("images", [])):
            for key in ("status", "width", "height", "format", "phash", "fft_peaks", "exif"):
                if key in image:
                    evidence.append(
                        {"signal": f"image_{i}_{key}", "value": image[key], "weight": 0.0}
                    )
        return measured_signal(evidence, self.version)
