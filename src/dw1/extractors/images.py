"""Image measurements and an optional, separately identified trained classifier."""

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
        result = measured_signal(evidence, self.version)
        scored = []
        for i, image in enumerate(parts.get("images", [])):
            detector = image.get("ai_detector", {"status": "model_unavailable"})
            evidence.append({"signal": f"image_{i}_ai_detector", "value": detector, "weight": 0.0})
            if detector.get("status") == "scored":
                scored.append(detector)
        # Each image keeps its own model result. The element-level summary uses
        # the highest p_ai; it is not a calibrated probability for the whole email.
        if scored:
            selected = max(scored, key=lambda row: row["p_ai"])
            if selected["label"] == "human" and len(scored) != len(parts.get("images", [])):
                return result
            result["ai_provenance"] = {
                "label": selected["label"],
                "confidence": selected["confidence"],
                "model_version": selected["model_version"],
                "evidence": evidence,
            }
        return result
