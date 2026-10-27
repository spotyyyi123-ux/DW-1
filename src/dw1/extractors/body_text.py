from __future__ import annotations

import logging
import re
from typing import Any

from dw1.extractors import BaseExtractor, register
from dw1.extractors.text_utils import (
    analyze_punctuation,
    compute_burstiness,
    compute_mtld,
    compute_ttr,
    find_typos_and_anomalies,
    split_sentences,
    tokenize_words,
)
from dw1.models.llm_detectors import cached_llm_signal

log = logging.getLogger("dw1.extractors.body_text")

VERSION = "body_text-0.2"

# Ключевые слова фишингового намерения в теле письма
PHISHING_KEYWORDS_RU = [
    r"учётн(?:ые|ых|ым)\s+данн(?:ые|ых|ым)",
    r"введите\s+(?:ваш\s+)?пароль",
    r"доступ\s+истекает",
    r"будет\s+(?:заблокирован|расторгнут|приостановлен)",
    r"подтвердите\s+(?:доступ|учётные|личность)",
    r"оплат(?:ите|ить)\s+счёт",
    r"реквизиты\s+и\s+ссылка\s+на\s+оплату",
]

PHISHING_KEYWORDS_EN = [
    r"account\s+(?:will\s+be\s+)?(?:suspended|terminated|locked)",
    r"verify\s+your\s+(?:identity|account|credentials|password)",
    r"confirm\s+your\s+(?:login|credentials|password)",
    r"unauthorized\s+access\s+detected",
    r"click\s+(?:the\s+link\s+)?below\s+to\s+confirm",
    r"pay\s+(?:the\s+)?invoice\s+immediately",
]


@register
class BodyTextExtractor(BaseExtractor):
    """Экстрактор признаков текста тела (SPEC §6 W2, Задача 2).

    Реализует стилометрический анализ:
    - разброс длин предложений (burstiness / CV);
    - лексическое разнообразие (MTLD, TTR);
    - профиль пунктуации (восклицания, КАПС, плотность знаков);
    - эвристический детектор опечаток и аномалий (смешение алфавитов, спам-шум);
    - сигналы намерений (угрозы блокировки, запрос паролей/оплаты).
    """

    name = "body_text"
    version = VERSION

    def extract(self, parts: dict) -> dict:
        try:
            body_info = parts.get("body_text", {})
            text = body_info.get("text", "") if isinstance(body_info, dict) else ""
            if not isinstance(text, str):
                text = str(text or "")

            tokens = tokenize_words(text)
            sentences = split_sentences(text)

            # Если текста нет или он слишком короткий (< 5 слов)
            if len(tokens) < 5:
                return {
                    "ai_provenance": {
                        "label": "unknown",
                        "confidence": 0.0,
                        "evidence": [
                            {
                                "signal": "text_length",
                                "value": f"слишком короткий текст ({len(tokens)} слов)",
                                "weight": 0.0,
                            }
                        ],
                        "model_version": self.version,
                    },
                    "intent": {
                        "label": "unknown",
                        "confidence": 0.0,
                        "evidence": [
                            {"signal": "text_length", "value": len(tokens), "weight": 0.0}
                        ],
                        "model_version": self.version,
                    },
                }

            # 1. Burstiness (разброс длин предложений)
            sent_lengths = [len(tokenize_words(s)) for s in sentences if s.strip()]
            b_stats = compute_burstiness(sent_lengths)
            burstiness = b_stats["burstiness"]
            cv = b_stats["cv"]

            # 2. Лексическое разнообразие: MTLD и TTR
            ttr = compute_ttr(tokens)
            mtld = compute_mtld(tokens)

            # 3. Пунктуационный профиль
            punct_stats = analyze_punctuation(text)

            # 4. Опечатки и аномалии (включая смешанный алфавит)
            typo_count, anomalies = find_typos_and_anomalies(tokens)
            typo_rate = typo_count / len(tokens)

            # 5. Оценка происхождения (ai_provenance)
            # Признаки ИИ: низкий burstiness, zero typos, сбалансированная пунктуация,
            # средний/высокий MTLD. Признаки человека: высокий burstiness, опечатки, КАПС
            ai_score = 0.5

            # Влияние burstiness
            if burstiness < -0.2:
                ai_score += 0.20
            elif burstiness > 0.15:
                ai_score -= 0.20

            # Влияние опечаток
            if typo_count == 0:
                ai_score += 0.15
            else:
                ai_score -= min(0.35, 0.15 * typo_count)

            # Влияние пунктуации (экспрессивность, КАПС)
            if punct_stats["multiple_punct_count"] > 0 or punct_stats["caps_words_count"] > 1:
                ai_score -= 0.25
            elif punct_stats["expressive_ratio"] == 0.0:
                ai_score += 0.10

            # Класс ai_polished: чистый текст без опечаток, но с лёгкой
            # структурной неравномерностью
            if 0.50 <= ai_score <= 0.65 and typo_count == 0:
                prov_label = "ai_polished"
                prov_conf = round(ai_score, 2)
            elif ai_score > 0.65:
                prov_label = "ai_generated"
                prov_conf = min(0.95, round(ai_score, 2))
            elif ai_score < 0.35:
                prov_label = "human"
                prov_conf = min(0.95, round(1.0 - ai_score, 2))
            else:
                prov_label = "unknown"
                prov_conf = 0.50

            prov_evidence: list[dict[str, Any]] = [
                {
                    "signal": "burstiness",
                    "value": round(burstiness, 3),
                    "interpretation": "low_uniform" if burstiness < -0.1 else "high_bursty",
                    "weight": 0.25,
                },
                {"signal": "sentence_length_cv", "value": round(cv, 3), "weight": 0.20},
                {"signal": "mtld", "value": round(mtld, 2), "weight": 0.20},
                {"signal": "ttr", "value": round(ttr, 3), "weight": 0.15},
                {
                    "signal": "typo_rate",
                    "value": round(typo_rate, 4),
                    "typo_count": typo_count,
                    "anomalies": anomalies[:5],
                    "weight": 0.20,
                },
                {
                    "signal": "punctuation_profile",
                    "value": {
                        "caps_words": punct_stats["caps_words_count"],
                        "multiple_punct": punct_stats["multiple_punct_count"],
                        "expressive_ratio": round(punct_stats["expressive_ratio"], 3),
                    },
                    "weight": 0.15,
                },
            ]

            # 6. Оценка намерения (intent)
            phishing_matches: list[str] = []
            for pat in PHISHING_KEYWORDS_RU + PHISHING_KEYWORDS_EN:
                m = re.findall(pat, text, re.IGNORECASE)
                if m:
                    phishing_matches.extend(m if isinstance(m[0], str) else [pat])

            if phishing_matches:
                intent_label = "phishing"
                intent_conf = min(0.95, 0.70 + 0.05 * len(phishing_matches))
            else:
                intent_label = "unknown"
                intent_conf = 0.0

            intent_evidence: list[dict[str, Any]] = [
                {
                    "signal": "phishing_keywords",
                    "value": phishing_matches,
                    "count": len(phishing_matches),
                    "weight": 0.5 if phishing_matches else 0.0,
                }
            ]

            result = {
                "ai_provenance": {
                    "label": prov_label,
                    "confidence": prov_conf,
                    "evidence": prov_evidence,
                    "model_version": self.version,
                },
                "intent": {
                    "label": intent_label,
                    "confidence": intent_conf,
                    "evidence": intent_evidence,
                    "model_version": self.version,
                },
            }
            llm_signal = cached_llm_signal(text)
            if llm_signal is not None:
                # Эвристики сохраняются как контекст, но не подменяют LLM-скор.
                llm_signal["evidence"].append(
                    {
                        "signal": "style_heuristic_context",
                        "value": {"label": prov_label, "confidence": prov_conf},
                        "weight": 0.0,
                    }
                )
                result["ai_provenance"] = llm_signal
            return result
        except Exception as exc:
            log.warning("Ошибка в BodyTextExtractor: %s", exc)
            return {
                "ai_provenance": self._unknown("ai_provenance"),
                "intent": self._unknown("intent"),
            }
