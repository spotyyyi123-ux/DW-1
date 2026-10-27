from __future__ import annotations

import logging
import re
from typing import Any

from dw1.extractors import BaseExtractor, register

log = logging.getLogger("dw1.extractors.cta")

VERSION = "cta-0.2"

# Шаблоны императивов и целевых действий RU + EN
CTA_IMPERATIVES_RU = [
    # Подтверждение / Верификация
    (r"\bподтверд(?:ите|ить)\b", "confirm"),
    (r"\bверифицир(?:уйте|овать)\b", "confirm"),
    (r"\bпродл(?:ите|ить)\b", "confirm"),
    # Переход по ссылке / Нажатие
    (r"\bперейдит(?:е|ь)\s+по\s+ссылке\b", "navigate"),
    (r"\bперейдит(?:е|ь)\s+к\b", "navigate"),
    (r"\bпройдит(?:е|ь)\s+по\s+ссылке\b", "navigate"),
    (r"\bнажмит(?:е|ь)\s+(?:сюда|на\s+кнопку|по\s+ссылке)?\b", "navigate"),
    (r"\bкликнит(?:е|ь)\b", "navigate"),
    # Авторизация / Вход
    (r"\bавторизуйт(?:есь|ся)\b", "login"),
    (r"\bвойдит(?:е|ь)\s+в\s+аккаунт\b", "login"),
    (r"\bвведит(?:е|ь)\s+(?:логин|пароль|данные)\b", "login"),
    # Обновление данных
    (r"\bобновит(?:е|ь)\s+данные\b", "update"),
    (r"\bобновит(?:е|ь)\s+доступ\b", "update"),
    (r"\bвосстановит(?:е|ь)\s+доступ\b", "update"),
    # Оплата
    (r"\bоплат(?:ите|ить)\s+счёт\b", "pay"),
    (r"\bпроизвест(?:и|ите)\s+оплату\b", "pay"),
    (r"\bпогас(?:ите|ить)\s+задолженность\b", "pay"),
    # Скачивание / Вложения
    (r"\bскачайт(?:е|ь)\s+(?:файл|документ|вложение)?\b", "download"),
    (r"\bоткройт(?:е|ь)\s+(?:файл|вложение)\b", "download"),
    # Проверка
    (r"\bпровер(?:ьте|ить)\s+(?:статус|данные|информацию)?\b", "review"),
    (r"\bознаком(?:ьтесь|иться)\b", "review"),
]

CTA_IMPERATIVES_EN = [
    # Confirmation / Verification — объект обязателен: голое «please confirm»
    # в деловой переписке не является высокорисковым действием (FP на легит-почте).
    (r"\bconfirm\s+(?:your\s+)?(?:account|identity|email|access|details)\b", "confirm"),
    (r"\bverify\s+(?:your\s+)?(?:account|identity|email|credentials|password)\b", "confirm"),
    (r"\bvalidate\s+(?:your\s+)?account\b", "confirm"),
    # Navigation / Clicks
    (r"\bclick\s+(?:the\s+link\s+below|here|to\s+verify|to\s+confirm|the\s+button)\b", "navigate"),
    (r"\bfollow\s+the\s+link\b", "navigate"),
    (r"\bproceed\s+to\s+(?:payment|verification|login)\b", "navigate"),
    # Login / Auth
    (r"\blog\s*in\s+(?:to\s+your\s+account)?\b", "login"),
    (r"\bsign\s*in\s+(?:to\s+verify)?\b", "login"),
    (r"\bauthenticate\s+now\b", "login"),
    # Updates
    (r"\bupdate\s+(?:your\s+)?(?:account|profile|payment\s+info|password)\b", "update"),
    (r"\breactivate\s+your\s+account\b", "update"),
    (r"\brestore\s+access\b", "update"),
    # Payment
    (r"\bpay\s+(?:the\s+)?(?:invoice|bill|now)\b", "pay"),
    (r"\bmake\s+(?:a\s+)?payment\b", "pay"),
    (r"\bsettle\s+(?:your\s+)?account\b", "pay"),
    # Download
    (r"\bdownload\s+(?:the\s+)?(?:attached|file|document|invoice)\b", "download"),
    (r"\bopen\s+(?:the\s+)?attachment\b", "download"),
    # Review
    (r"\breview\s+(?:the\s+)?(?:changes|document|account|activity)\b", "review"),
    (r"\bcheck\s+(?:your\s+)?details\b", "review"),
]

# Дедлайны и временные рамки RU + EN
DEADLINE_PATTERNS_RU = [
    r"\bв\s+течение\s+\d+\s*(?:часов|часа|дней|дня|минут|суток)\b",
    r"\bчерез\s+\d+\s*(?:часов|часа|дней|минут)\b",
    r"\bсегодня\s+до\s+\d{1,2}(?:[:.]\d{2})?\b",
    r"\bдо\s+конца\s+дня\b",
    r"\bв\s+кратчайшие\s+сроки\b",
    r"\bнемедленно\b",
    r"\bпрямо\s+сейчас\b",
]

DEADLINE_PATTERNS_EN = [
    r"\bwithin\s+\d+\s*(?:hours|hour|days|day|minutes|mins)\b",
    r"\bin\s+\d+\s*(?:hours|hour|days|day|minutes)\b",
    r"\bby\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b",
    r"\bby\s+the\s+end\s+of\s+the\s+day\b",
    r"\bimmediately\b",
    r"\bright\s+now\b",
    r"\bwithout\s+delay\b",
]

# Высокорисковые действия для классификации фишинга
HIGH_RISK_ACTIONS = {"confirm", "login", "pay", "update"}


@register
class CtaExtractor(BaseExtractor):
    """Экстрактор Call-to-Action (SPEC §6 W2, Задача 2).

    Извлекает призывы к действию:
    - императивы и целевые действия (подтвердить, оплатить, перейти по ссылке);
    - дедлайны и временные рамки («в течение 24 часов», «сегодня до 18:00»);
    - анкорные тексты ссылок с призывами;
    - наполняет структуру `parts['cta']` при её отсутствии;
    - формирует сигналы намерений (intent) и происхождения (ai_provenance).
    """

    name = "cta"
    version = VERSION

    def extract(self, parts: dict) -> dict:
        try:
            body_info = parts.get("body_text", {})
            body_text = body_info.get("text", "") if isinstance(body_info, dict) else ""
            if not isinstance(body_text, str):
                body_text = str(body_text or "")

            links = parts.get("links", [])
            if not isinstance(links, list):
                links = []

            # 1. Поиск дедлайнов в тексте
            found_deadlines: list[str] = []
            for pat in DEADLINE_PATTERNS_RU + DEADLINE_PATTERNS_EN:
                for match in re.finditer(pat, body_text, re.IGNORECASE):
                    found_deadlines.append(match.group(0))

            # 2. Поиск призывов к действию (CTA) в тексте и ссылках
            extracted_ctas: list[dict[str, Any]] = []
            found_imperatives: list[str] = []
            matched_actions: set[str] = set()

            # Сканирование предложений тела
            sentences = re.split(r"[.!?\n]+", body_text)
            for s in sentences:
                s_clean = s.strip()
                if not s_clean:
                    continue
                for pat, action in CTA_IMPERATIVES_RU + CTA_IMPERATIVES_EN:
                    m = re.search(pat, s_clean, re.IGNORECASE)
                    if m:
                        imperative_text = m.group(0)
                        found_imperatives.append(imperative_text)
                        matched_actions.add(action)

                        # Ищем дедлайн в том же или соседнем предложении
                        deadline_in_s = ""
                        for dl in found_deadlines:
                            if dl.lower() in s_clean.lower():
                                deadline_in_s = dl
                                break

                        extracted_ctas.append({
                            "text": s_clean[:120],
                            "imperative": action,
                            "deadline": deadline_in_s,
                            "href": "",
                        })
                        break

            # Сканирование анкоров ссылок
            for link in links:
                if not isinstance(link, dict):
                    continue
                anchor = link.get("anchor_text", "") or ""
                url = link.get("url", "") or ""
                # Проверяем анкор или текст вокруг ссылки
                text_to_check = f"{anchor} {url}"
                for pat, action in CTA_IMPERATIVES_RU + CTA_IMPERATIVES_EN:
                    if re.search(pat, text_to_check, re.IGNORECASE):
                        matched_actions.add(action)
                        extracted_ctas.append({
                            "text": anchor or url[:60],
                            "imperative": action,
                            "deadline": found_deadlines[0] if found_deadlines else "",
                            "href": url,
                        })
                        break

            # Если parts['cta'] был пустым — наполняем его найденными элементами
            if not parts.get("cta") and extracted_ctas:
                parts["cta"] = extracted_ctas

            # 3. Вычисление сигналов
            cta_count = len(extracted_ctas)
            has_high_risk = bool(matched_actions & HIGH_RISK_ACTIONS)
            has_deadlines = len(found_deadlines) > 0

            # Оценка Intent:
            # Высокорисковое действие (подтвердите доступ) + дедлайн = фишинг
            if cta_count > 0:
                if has_high_risk and has_deadlines:
                    intent_label = "phishing"
                    intent_conf = min(0.95, 0.80 + 0.05 * cta_count)
                elif has_high_risk:
                    intent_label = "phishing"
                    intent_conf = 0.75
                else:
                    intent_label = "spam"
                    intent_conf = 0.65
            else:
                intent_label = "unknown"
                intent_conf = 0.0

            # Оценка Provenance:
            # Вежливые, формальные формулировки («Пожалуйста, подтвердите учётные данные...»)
            # характерны для LLM-шаблонов. Агрессивный КАПС («СРОЧНО ПЕРЕЙДИ») — человек.
            if cta_count > 0:
                is_polite = any(
                    word in body_text.lower()
                    for word in ("пожалуйста", "просим", "please", "kindly", "уважаемый", "dear")
                )
                is_shouting = any(c["text"].isupper() for c in extracted_ctas)
                if is_shouting:
                    prov_label = "human"
                    prov_conf = 0.70
                elif is_polite and not has_deadlines:
                    prov_label = "ai_polished"
                    prov_conf = 0.65
                elif is_polite and has_deadlines:
                    prov_label = "ai_generated"
                    prov_conf = 0.70
                else:
                    prov_label = "unknown"
                    prov_conf = 0.0
            else:
                prov_label = "unknown"
                prov_conf = 0.0

            evidence_items: list[dict[str, Any]] = [
                {"signal": "cta_count", "value": cta_count, "weight": 0.25},
                {
                    "signal": "imperatives",
                    "value": list(set(found_imperatives)),
                    "count": len(found_imperatives),
                    "weight": 0.35,
                },
                {
                    "signal": "deadlines",
                    "value": list(set(found_deadlines)),
                    "count": len(found_deadlines),
                    "weight": 0.30,
                },
                {
                    "signal": "high_risk_actions",
                    "value": sorted(matched_actions & HIGH_RISK_ACTIONS),
                    "weight": 0.40,
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
            log.warning("Ошибка в CtaExtractor: %s", exc)
            return {
                "ai_provenance": self._unknown("ai_provenance"),
                "intent": self._unknown("intent"),
            }
