from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup

from dw1.extractors import BaseExtractor, register


@register
class HtmlExtractor(BaseExtractor):
    name = "html"
    version = "html-1.1"

    ZERO_WIDTH_RE = re.compile(
        r"[\u200b-\u200f\u2060\ufeff]"
    )

    HIDDEN_RE = re.compile(
        r"display\s*:\s*none|"
        r"visibility\s*:\s*hidden|"
        r"opacity\s*:\s*0(?:\.0+)?(?![\d.])|"
        r"font-size\s*:\s*0(?:px|pt|em|rem)?(?![\d.])",
        re.IGNORECASE,
    )

    ESP_RE = re.compile(
        r"sendgrid|mailchimp|mandrill|"
        r"constantcontact|campaignmonitor|"
        r"mailgun|hubspot",
        re.IGNORECASE,
    )

    SEMANTIC_RE = re.compile(
        r"<\s*(header|footer|main|article|section|nav)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _signal(name: str, value, weight: float = 0.0) -> dict:
        return {
            "signal": name,
            "value": value,
            "weight": weight,
        }

    def extract(self, parts: dict) -> dict:
        html_data = parts.get("body_html", {})
        html_text = str(html_data.get("text", ""))
        lowered = html_text.lower()
        dom = BeautifulSoup(html_text, "html.parser")

        # A. Признаки генерации HTML
        generator: dict[str, Any] = {
            "has_flex": bool(re.search(r"display\s*:\s*flex", lowered)),
            "has_grid": bool(re.search(r"display\s*:\s*grid", lowered)),
            "semantic_tags": len(self.SEMANTIC_RE.findall(html_text)),
            "has_css_variables": bool(re.search(r"--[\w-]+\s*:", html_text)),
            "has_mso_hacks": "<!--[if mso]" in lowered,
        }

        # B. Признаки обфускации и вредоносной структуры
        obfuscation: dict[str, Any] = {
            "zero_width_count": len(self.ZERO_WIDTH_RE.findall(html_text)),
            "hidden_blocks": len(self.HIDDEN_RE.findall(html_text)),
            "entity_count": len(re.findall(r"&(?:#\d+|#x[\da-f]+|\w+);", html_text, re.I)),
            "comment_count": len(re.findall(r"<!--.*?-->", html_text, re.S)),
            "external_post_forms": sum(
                str(form.get("method", "get")).lower() == "post"
                and str(form.get("action", "")).lower().startswith(("http://", "https://"))
                for form in dom.find_all("form")
            ),
        }

        # C. Признаки легитимных массовых рассылок
        esp_markers = sorted(set(self.ESP_RE.findall(lowered)))
        mailing: dict[str, Any] = {
            "esp_markers": esp_markers,
            "list_unsubscribe": bool(
                str(parts.get("headers", {}).get("list_unsubscribe", "")).strip()
            ),
            "precedence_bulk": (
                str(parts.get("headers", {}).get("precedence", "")).lower() == "bulk"
            ),
        }

        evidence = []

        for name, val in generator.items():
            evidence.append(self._signal(f"html_generator_{name}", val))

        for name, val in obfuscation.items():
            evidence.append(self._signal(f"html_obfuscation_{name}", val))

        for name, val in mailing.items():
            evidence.append(self._signal(f"html_mailing_{name}", val))

        provenance = {
            "label": "unknown",
            "confidence": 0.0,
            "evidence": evidence,
            "model_version": self.version,
        }

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
