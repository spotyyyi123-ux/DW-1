# -*- coding: utf-8 -*-
"""Извлечение признаков URL и ссылок в письмах."""

from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlsplit

from dw1.extractors import BaseExtractor, register


@register
class LinksExtractor(BaseExtractor):
    name = "links"
    version = "stub-1.0"

    ESP_DOMAINS = (
        "sendgrid.net",
        "mailchimp.com",
        "list-manage.com",
        "mandrillapp.com",
        "mailgun.org",
        "hubspot.com",
        "constantcontact.com",
    )

    @staticmethod
    def _host(url: str) -> str:
        try:
            host = urlsplit(url if "://" in url else "//" + url).hostname or ""
            return host.lower().rstrip(".")
        except ValueError:
            return ""

    @staticmethod
    def _entropy(value: str) -> float:
        if not value:
            return 0.0
        counts = Counter(value)
        length = len(value)
        return round(
            -sum((n / length) * math.log2(n / length) for n in counts.values()),
            3,
        )

    @staticmethod
    def _anchor_mismatch(anchor: str, host: str) -> bool:
        anchor = (anchor or "").strip().lower()
        if not anchor or not host:
            return False

        # Сравниваем только текст, похожий на URL или домен.
        visible_host = LinksExtractor._host(anchor)
        if visible_host:
            return visible_host != host

        return False

    def extract(self, parts: dict) -> dict:
        links = parts.get("links", [])
        evidence = []
        analyzed = []

        for index, link in enumerate(links):
            url = str(link.get("url", ""))
            anchor = str(link.get("anchor_text", ""))
            host = self._host(url)

            labels = host.split(".") if host else []
            punycode = any(label.startswith("xn--") for label in labels)
            unicode_domain = any(not label.isascii() for label in labels)

            mismatch = self._anchor_mismatch(anchor, host)
            tracker = any(
                host == domain or host.endswith("." + domain)
                for domain in self.ESP_DOMAINS
            )

            item = {
                "url_length": len(url),
                "url_entropy": self._entropy(url),
                "domain": host,
                "subdomain_depth": max(0, len(labels) - 2),
                "punycode": punycode,
                "unicode_domain": unicode_domain,
                "tld": labels[-1] if labels else "",
                "anchor_mismatch": mismatch,
                "tracker_like": tracker,
            }
            analyzed.append(item)

            for name, value in item.items():
                evidence.append({
                    "signal": f"link_{index}_{name}",
                    "value": value,
                    "weight": 0.0,
                })

        evidence.append({
            "signal": "links_count",
            "value": len(links),
            "weight": 0.0,
        })

        signal = {
            "label": "unknown",
            "confidence": 0.0,
            "evidence": evidence,
            "model_version": self.version,
        }

        return {
            "ai_provenance": dict(signal),
            "intent": dict(signal),
        }