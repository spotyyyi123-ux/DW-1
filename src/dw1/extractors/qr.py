"""QR payloads are evidence, never executable navigation instructions."""

from dw1.extractors import BaseExtractor, register
from dw1.extractors.images import measured_signal


@register
class QrExtractor(BaseExtractor):
    name = "qr"
    version = "opencv-qr-1.0"

    def extract(self, parts: dict) -> dict:
        evidence = [{"signal": "qr_codes", "value": parts.get("qr", []), "weight": 0.0}]
        evidence.append(
            {
                "signal": "qr_decoder_status",
                "value": [
                    im.get("qr", {}).get("status", im.get("status", "unprocessed"))
                    for im in parts.get("images", [])
                ],
                "weight": 0.0,
            }
        )
        return measured_signal(evidence, self.version)
