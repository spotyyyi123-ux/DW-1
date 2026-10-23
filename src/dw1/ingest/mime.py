"""M0-ингест: разбор .eml в MessageParts (stdlib email).

Развитие (SPEC §6 W1): .msg, нормализация кодировок, HTML→текст через bs4,
DOM-фингерпринт, ESP-маркеры, MSO-хаки.
"""
from __future__ import annotations

import hashlib
import re
from email import policy
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import urlparse

AUTH_RE = re.compile(r"(spf|dkim|dmarc)=(\S+)", re.IGNORECASE)
LINK_RE = re.compile(r"""href=["']([^"']+)["']""", re.IGNORECASE)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _domain(s: str) -> str:
    """Извлекает домен из email-адреса, Message-ID или URL."""
    s = (s or "").strip().strip("<>")
    if s.startswith(("http://", "https://")):
        s = urlparse(s).netloc
    if "@" in s:
        s = s.rsplit("@", 1)[1]
    return s.strip().strip("/")


def _auth_results(msg) -> dict:
    out = {"spf": "none", "dkim": "none", "dmarc": "none"}
    for value in msg.get_all("Authentication-Results") or []:
        for m in AUTH_RE.finditer(value):
            out[m.group(1).lower()] = m.group(2).lower()
    return out


def parse_eml(path: str | Path) -> dict:
    """Разбирает .eml -> {raw_sha256, parts} (схема parts: SPEC §4.1)."""
    raw = Path(path).read_bytes()
    msg = BytesParser(policy=policy.default).parsebytes(raw)

    flat: dict[str, str] = {}
    for key in msg.keys():
        values = msg.get_all(key)
        flat[key.lower()] = ", ".join(values) if isinstance(values, list) else str(values)

    text_parts: list[str] = []
    html_parts: list[str] = []
    attachments: list[dict] = []
    images: list[dict] = []
    for part in msg.walk():
        ctype = part.get_content_type()
        disp = (part.get("Content-Disposition") or "").lower()
        if ctype == "text/plain" and "attachment" not in disp:
            text_parts.append(part.get_content())
        elif ctype == "text/html" and "attachment" not in disp:
            html_parts.append(part.get_content())
        elif disp.startswith("attachment"):
            raw_payload = part.get_payload(decode=True)
            payload = raw_payload if isinstance(raw_payload, bytes) else b""
            attachments.append(
                {"name": part.get_filename() or "", "mime": ctype, "sha256": _sha256(payload)}
            )
        elif ctype.startswith("image/"):
            raw_payload = part.get_payload(decode=True)
            payload = raw_payload if isinstance(raw_payload, bytes) else b""
            images.append(
                {"mime": ctype, "sha256": _sha256(payload), "cid": part.get("Content-ID", "")}
            )

    body_text = "\n\n".join(t for t in text_parts if t)
    body_html = "\n\n".join(h for h in html_parts if h)
    links = [
        {"url": u, "anchor_text": "", "domain": _domain(u)}
        for u in LINK_RE.findall(body_html)
    ]

    parts = {
        "headers": {
            "from": flat.get("from", ""),
            "to": flat.get("to", ""),
            "reply_to": flat.get("reply-to", ""),
            "return_path": flat.get("return-path", ""),
            "date_tz": flat.get("date", ""),
            "x_mailer": flat.get("x-mailer", ""),
            "message_id_domain": _domain(flat.get("message-id", "")),
            "list_unsubscribe": flat.get("list-unsubscribe", ""),
            "precedence": flat.get("precedence", ""),
            "in_reply_to": flat.get("in-reply-to", ""),
            "references": flat.get("references", ""),
            **_auth_results(msg),
        },
        "subject": {"text": msg.get("Subject", "")},
        "body_text": {"text": body_text},
        "body_html": {
            "text": body_html,
            "raw_len": len(body_html),
            "dom_fingerprint": "",
            "links_count": len(links),
        },
        "cta": [],  # M2-M3
        "links": links,
        "images": images,
        "qr": [],  # M3
        "attachments": attachments,
    }
    return {"raw_sha256": _sha256(raw), "parts": parts}
