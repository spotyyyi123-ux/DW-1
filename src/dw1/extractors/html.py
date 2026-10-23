from __future__ import annotations

import re
from html import unescape


from dw1.extractors import BaseExtractor, register


@register
class HtmlExtractor(BaseExtractor):
    name = "html"
    version = "stub-1.0"

    ZERO_WIDTH_RE = re.compile(
        r"[\u200b-\u200f\u2060\ufeff]"
    )

    HIDDEN_RE = re.compile(
        r"display\s*:\s*none|"
        r"visibility\s*:\s*hidden|"
        r"opacity\s*:\s*0|"
        r"font-size\s*:\s*0(?:px|pt|em|rem)?",
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

        # A. Признаки генерации HTML
        generator = {
            "has_flex": bool(re.search(r"display\s*:\s*flex", lowered)),
            "has_grid": bool(re.search(r"display\s*:\s*grid", lowered)),
            "semantic_tags": len(self.SEMANTIC_RE.findall(html_text)),
            "has_css_variables": bool(re.search(r"--[\w-]+\s*:", html_text)),
            "has_mso_hacks": "<!--[if mso]" in lowered,
        }

        # B. Признаки обфускации и вредоносной структуры
        obfuscation = {
            "zero_width_count": len(self.ZERO_WIDTH_RE.findall(html_text)),
            "hidden_blocks": len(self.HIDDEN_RE.findall(html_text)),
            "entity_count": len(re.findall(r"&(?:#\d+|#x[\da-f]+|\w+);", html_text, re.I)),
            "comment_count": len(re.findall(r"<!--.*?-->", html_text, re.S)),
            "external_post_forms": len(
                re.findall(
                    r"<form\b[^>]*action\s*=\s*['\"]https?://",
                    html_text,
                    re.I,
                )
            ),
        }

        # C. Признаки легитимных массовых рассылок
        esp_markers = sorted(set(self.ESP_RE.findall(html_text)))
        mailing = {
            "esp_markers": esp_markers,
            "list_unsubscribe": bool(
                parts.get("headers", {}).get("list-unsubscribe")
            ),
            "precedence_bulk": (
                parts.get("headers", {}).get("precedence", "").lower()
                == "bulk"
            ),
        }

        evidence = []

        for name, value in generator.items():
            evidence.append(self._signal(f"html_generator_{name}", value))

        for name, value in obfuscation.items():
            evidence.append(self._signal(f"html_obfuscation_{name}", value))

        for name, value in mailing.items():
            evidence.append(self._signal(f"html_mailing_{name}", value))

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