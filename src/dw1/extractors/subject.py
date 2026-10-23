from __future__ import annotations

import logging
import re
from typing import Any

from dw1.extractors import BaseExtractor, register
from dw1.extractors.text_utils import (
    extract_emojis,
    tokenize_words,
)

log = logging.getLogger("dw1.extractors.subject")

VERSION = "subject-0.2"

# Слова срочности (urgency) в теме письма RU + EN
URGENCY_PATTERNS_RU = [
    r"\bсрочно\b",
    r"\bсрочный\b",
    r"\bнемедленно\b",
    r"\bвнимание\b",
    r"\bважно\b",
    r"\b24\s*час(?:а|ов)?\b",
    r"\b48\s*час(?:а|ов)?\b",
    r"\bистекает\b",
    r"\bистек(?:ает|ающий|ает срок)\b",
    r"\bзаблокирован(?:а|о)?\b",
    r"\bблокировк(?:а|е|у)\b",
    r"\bприостановлен(?:о)?\b",
    r"\bпоследн(?:ий|ее)\s+(?:шанс|предупреждение|уведомление)\b",
    r"\bтребуется\s+действие\b",
    r"\bсегодня\s+до\b",
    r"\bоплат(?:ите|ить)\b",
    r"\bдедлайн\b",
]

URGENCY_PATTERNS_EN = [
    r"\burgent\b",
    r"\burgently\b",
    r"\bimmediate\b",
    r"\baction\s+required\b",
    r"\b24\s*hours?\b",
    r"\b48\s*hours?\b",
    r"\bexpir(?:es|ing|ed)\b",
    r"\bsuspended\b",
    r"\bsuspension\b",
    r"\bfinal\s+notice\b",
    r"\bimportant\b",
    r"\battention\b",
    r"\bsecurity\s+alert\b",
    r"\bverify\s+now\b",
    r"\btoday\s+only\b",
    r"\bdeadline\b",
]

RE_THREAD_PREFIX = re.compile(
    r"^(?:[^\w]*)(?:re|fwd|fw|отв|ответ):\s*",
    re.IGNORECASE,
)

NAME_ADDRESS_PATTERNS = [
    # "Иван, ..." или "Иван Петров, ..."
    re.compile(r"^([А-ЯЁA-Z][а-яёa-z]+(?:\s+[А-ЯЁA-Z][а-яёa-z]+)?)\s*,\s*"),
    # "Уважаемый(-ая) Иван ..."
    re.compile(r"^уважаем(?:ый|ая)\s+([А-ЯЁA-Z][а-яёa-z]+)", re.IGNORECASE),
    # "Dear John ..."
    re.compile(r"^dear\s+([A-Z][a-z]+)", re.IGNORECASE),
]


@register
class SubjectExtractor(BaseExtractor):
    """Экстрактор темы письма (SPEC §6 W2, Задача 2).

    Анализирует тему письма:
    - длину (символы, слова);
    - наличие эмодзи;
    - лексику срочности (urgency) RU + EN;
    - персональное обращение по имени;
    - фальшивые префиксы треда (Re:/Fwd: без реального диалога в заголовках).
    """

    name = "subject"
    version = VERSION

    def extract(self, parts: dict) -> dict:
        try:
            subj_info = parts.get("subject", {})
            text = subj_info.get("text", "") if isinstance(subj_info, dict) else ""
            if not isinstance(text, str):
                text = str(text or "")
            text = text.strip()

            if not text:
                return {
                    "ai_provenance": {
                        "label": "unknown",
                        "confidence": 0.0,
                        "evidence": [
                            {"signal": "empty_subject", "value": True, "weight": 0.0}
                        ],
                        "model_version": self.version,
                    },
                    "intent": {
                        "label": "unknown",
                        "confidence": 0.0,
                        "evidence": [
                            {"signal": "empty_subject", "value": True, "weight": 0.0}
                        ],
                        "model_version": self.version,
                    },
                }

            char_len = len(text)
            words = tokenize_words(text)
            word_count = len(words)

            # 1. Эмодзи
            emojis = extract_emojis(text)
            has_emoji = len(emojis) > 0

            # 2. Слова срочности (urgency)
            urgency_matches: list[str] = []
            for pat in URGENCY_PATTERNS_RU + URGENCY_PATTERNS_EN:
                found = re.findall(pat, text, re.IGNORECASE)
                if found:
                    urgency_matches.extend(found if isinstance(found[0], str) else [pat])

            # 3. Обращение по имени
            detected_name: str | None = None
            for name_pat in NAME_ADDRESS_PATTERNS:
                m = name_pat.search(text)
                if m:
                    candidate = m.group(1).strip()
                    # Исключаем служебные слова, похожие на имя
                    if candidate.lower() not in {"коллега", "клиент", "пользователь", "друг"}:
                        detected_name = candidate
                        break

            # Проверка имени получателя из заголовка To/From при наличии
            headers = parts.get("headers", {})
            to_header = headers.get("to", "") if isinstance(headers, dict) else ""
            if not detected_name and to_header:
                for token in words:
                    if len(token) >= 3 and token.lower() in to_header.lower():
                        detected_name = token
                        break

            has_personal_address = detected_name is not None

            # 4. Проверка фальшивого Re:/Fwd:
            has_thread_prefix = bool(RE_THREAD_PREFIX.match(text))
            # Если префикс Re:/Fwd: есть, но нет References/In-Reply-To — признак фишинга
            is_fake_thread = False
            if has_thread_prefix and isinstance(headers, dict):
                has_in_reply_to = bool(headers.get("in-reply-to") or headers.get("references"))
                if not has_in_reply_to:
                    is_fake_thread = True

            # 5. Вычисление сигналов
            # Intent: urgency лексика в теме — сильнейший признак фишинга / спама
            if urgency_matches:
                is_dangerous = (
                    has_personal_address
                    or is_fake_thread
                    or "оплат" in text.lower()
                    or "pay" in text.lower()
                )
                intent_label = "phishing" if is_dangerous else "spam"
                intent_conf = min(0.95, 0.70 + 0.10 * len(urgency_matches))
            else:
                intent_label = "unknown"
                intent_conf = 0.0

            # Provenance:
            # - Эмодзи-спам, КАПС-крик, фальшивый тред -> human spam (0.75+)
            # - Чистая грамматика, спокойный стиль, без КАПС -> ai_generated / ai_polished (0.60)
            is_shouting = (
                word_count >= 2
                and sum(1 for w in words if w.isupper() and len(w) >= 2) / max(1, word_count) >= 0.5
            )

            if has_emoji or is_shouting or is_fake_thread:
                prov_label = "human"
                prov_conf = 0.75
            elif 5 <= word_count <= 15 and text[0].isupper() and not has_emoji:
                prov_label = "ai_polished"
                prov_conf = 0.60
            else:
                prov_label = "unknown"
                prov_conf = 0.0

            evidence_items: list[dict[str, Any]] = [
                {"signal": "char_length", "value": char_len, "weight": 0.10},
                {"signal": "word_count", "value": word_count, "weight": 0.10},
                {
                    "signal": "has_emoji",
                    "value": has_emoji,
                    "emoji_count": len(emojis),
                    "emojis": emojis[:5],
                    "weight": 0.20,
                },
                {
                    "signal": "urgency_cues",
                    "value": urgency_matches,
                    "count": len(urgency_matches),
                    "weight": 0.40,
                },
                {
                    "signal": "personal_address",
                    "value": has_personal_address,
                    "name": detected_name or "",
                    "weight": 0.20,
                },
                {
                    "signal": "fake_thread_prefix",
                    "value": is_fake_thread,
                    "weight": 0.25,
                },
            ]

            return {
                "ai_provenance": {
                    "label": prov_label,
                    "confidence": prov_conf,
                    "evidence": evidence_items,
                    "model_version": self.version,
                },
                "intent": {
                    "label": intent_label,
                    "confidence": intent_conf,
                    "evidence": evidence_items,
                    "model_version": self.version,
                },
            }
        except Exception as exc:
            log.warning("Ошибка в SubjectExtractor: %s", exc)
            return {
                "ai_provenance": self._unknown("ai_provenance"),
                "intent": self._unknown("intent"),
            }
