from __future__ import annotations

import re
from urllib.parse import unquote

from dw1.extractors import BaseExtractor, register


@register
class HeadersExtractor(BaseExtractor):
    name = "headers"
    version = "stub-1.0"

    @staticmethod
    def _domain(value: str) -> str:
        """Извлекает домен из email-адреса."""
        value = unquote((value or "").strip())

        match = re.search(r"@([^>\s,]+)", value)
        if not match:
            return ""

        return match.group(1).strip().lower().rstrip(".")

    @staticmethod
    def _auth_evidence(headers: dict) -> list[dict]:
        evidence = []

        for name in ("spf", "dkim", "dmarc"):
            value = str(headers.get(name, "none")).lower()

            evidence.append(
                {
                    "signal": f"{name}_{value}",
                    "value": value,
                    "weight": 0.0,
                }
            )

        return evidence

    def extract(self, parts: dict) -> dict:
        headers = parts.get("headers", {})

        from_domain = self._domain(headers.get("from", ""))
        reply_domain = self._domain(headers.get("reply_to", ""))

        reply_mismatch = bool(
            from_domain
            and reply_domain
            and from_domain != reply_domain
        )

        has_x_mailer = bool(str(headers.get("x_mailer", "")).strip())

        evidence = self._auth_evidence(headers)

        evidence.append(
            {
                "signal": "from_reply_to_mismatch",
                "value": reply_mismatch,
                "weight": 0.0,
            }
        )

        evidence.append(
            {
                "signal": "x_mailer_present",
                "value": has_x_mailer,
                "weight": 0.0,
            }
        )

        # Заголовки сами по себе недостаточны для определения
        # происхождения текста как AI-generated.
        provenance = {
            "label": "unknown",
            "confidence": 0.0,
            "evidence": evidence,
            "model_version": self.version,
        }

        # Здесь пока также не делаем вывод о phishing/spam:
        # заголовки — только один источник evidence.
        intent = {
            "label": "unknown",
            "confidence": 0.0,
            "evidence": evidence,
            "model_version": self.version,
        }

        return {
            "ai_provenance": provenance,
            "intent": intent,
        }