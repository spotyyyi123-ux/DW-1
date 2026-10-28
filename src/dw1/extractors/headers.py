from __future__ import annotations

import re
from email.utils import parsedate_to_datetime
from urllib.parse import unquote

from dw1.extractors import BaseExtractor, register


@register
class HeadersExtractor(BaseExtractor):
    name = "headers"
    version = "headers-1.1"

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

        reply_mismatch = bool(from_domain and reply_domain and from_domain != reply_domain)

        has_x_mailer = bool(str(headers.get("x_mailer", "")).strip())

        evidence = self._auth_evidence(headers)
        received = headers.get("received", [])
        dates = []
        malformed = 0
        for hop in received:
            try:
                stamp = parsedate_to_datetime(hop.rsplit(";", 1)[-1].strip())
                if stamp.tzinfo is None:
                    raise ValueError("timezone missing")
                dates.append(stamp)
            except (ValueError, TypeError, IndexError):
                malformed += 1
        date_valid = False
        date_timezone_minutes = None
        try:
            date = parsedate_to_datetime(headers.get("date_tz", ""))
            offset = date.utcoffset()
            if offset is not None:
                date_valid = True
                date_timezone_minutes = int(offset.total_seconds() / 60)
        except (ValueError, TypeError, IndexError):
            pass
        extra = {
            "received_count": len(received),
            "received_malformed": malformed,
            "received_chronology_anomaly": any(
                a < b for a, b in zip(dates, dates[1:], strict=False)
            ),
            "date_valid_with_timezone": date_valid,
            "date_timezone_minutes": date_timezone_minutes,
            "from_return_path_mismatch": bool(
                from_domain
                and self._domain(headers.get("return_path", ""))
                and from_domain != self._domain(headers.get("return_path", ""))
            ),
            "from_message_id_mismatch": bool(
                from_domain
                and headers.get("message_id_domain")
                and from_domain != str(headers["message_id_domain"]).lower()
            ),
            "authentication_source": "unverified message headers; not cryptographic verification",
        }
        evidence.extend(
            {"signal": name, "value": value, "weight": 0.0} for name, value in extra.items()
        )

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
