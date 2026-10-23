"""Фузия v1 (задача 4): две независимые оси из позлементных сигналов.

Взвешенное голосование по элементам письма. Это ЭВРИСТИКА — уверенность
не калибрована (``calibrated: false``); isotonic-калибровка на бенчмарке — M5.

- ``ai_provenance``: тело 0.50 · тема 0.20 · CTA 0.15 · HTML 0.15
- ``intent``:       заголовки 0.35 · ссылки 0.25 · CTA 0.20 · тема 0.10 · тело 0.10

Постобработка intent:
- «усилители» (SPF/DMARC/DKIM fail, punycode, anchor↔href mismatch, high-risk CTA)
  повышают уверенность;
- «гасители» (ESP-трейсеры, List-Unsubscribe, Precedence: bulk, DKIM pass) её
  снижают и попадают в контр-свидетельства (SPEC §4.1, enrich-карточка).

Отказ от ответа: ``rejection=true``, только если ни одна ось не получила голосов.
"""
from __future__ import annotations

FUSION_VERSION = "fusion-heuristic-v1"

PROV_WEIGHTS = {"body_text": 0.50, "subject": 0.20, "cta": 0.15, "html": 0.15}
INTENT_WEIGHTS = {"headers": 0.35, "links": 0.25, "cta": 0.20, "subject": 0.10, "body_text": 0.10}

# Подстроки имён сигналов, которые считаем «усилителями» intent.
THREAT_PREFIXES = (
    "spf_fail",
    "dmarc_fail",
    "dkim_fail",
    "high_risk_actions",
    "fake_thread_prefix",
)
THREAT_SUFFIXES = ("_punycode", "_anchor_mismatch", "_external_post_forms")
THREAT_NUMERIC = ("html_obfuscation_zero_width_count", "html_obfuscation_hidden_blocks")

# Подстроки имён сигналов-«гасителей» (легитимные массовые рассылки).
SUPPRESSOR_PREFIXES = (
    "html_mailing_list_unsubscribe",
    "html_mailing_precedence_bulk",
    "html_mailing_esp_markers",
    "dkim_pass",
    "dmarc_pass",
    "spf_pass",
)


def _collect_evidence(signals: dict) -> dict[str, object]:
    """Плоская карта имя_сигнала → значение по всем элементам и обеим осям."""
    flags: dict[str, object] = {}
    for blob in signals.values():
        blobs = blob if isinstance(blob, list) else [blob]
        for b in blobs:
            if not isinstance(b, dict):
                continue
            for kind in ("ai_provenance", "intent"):
                sig = b.get(kind) or {}
                if not isinstance(sig, dict):
                    continue
                for item in sig.get("evidence", []) or []:
                    if isinstance(item, dict) and item.get("signal"):
                        flags[str(item["signal"])] = item.get("value")
    return flags


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set)):
        return len(value) > 0
    return bool(value)


def _find_threats(flags: dict[str, object]) -> list[str]:
    found = []
    for name, value in flags.items():
        if not _truthy(value):
            continue
        if name in THREAT_PREFIXES or name in THREAT_NUMERIC:
            found.append(name)
        elif name.endswith(THREAT_SUFFIXES):
            found.append(name)
    return sorted(set(found))


def _find_suppressors(flags: dict[str, object]) -> list[str]:
    return sorted(
        name for name, value in flags.items() if name in SUPPRESSOR_PREFIXES and _truthy(value)
    )


def _vote(votes: dict, label: str, confidence: float, weight: float, element: str) -> None:
    if label in ("", None, "unknown") or confidence <= 0 or weight <= 0:
        return
    slot = votes.setdefault(label, {"score": 0.0, "elements": []})
    slot["score"] += weight * confidence
    slot["elements"].append(element)


def _axis(votes: dict, threats: list[str], suppressors: list[str]) -> dict:
    if not votes:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "calibrated": False,
            "ci95": [],
            "model_version": FUSION_VERSION,
        }

    label, data = max(votes.items(), key=lambda kv: kv[1]["score"])
    confidence = min(0.95, data["score"])

    if label in ("phishing", "spam") and threats:
        confidence = min(0.95, confidence + 0.05 * len(threats))
    if suppressors:
        # Легитимные маркеры массовой рассылки — повод не завышать уверенность.
        confidence *= max(0.35, 1.0 - 0.15 * len(suppressors))

    return {
        "label": label,
        "confidence": round(float(confidence), 2),
        "calibrated": False,
        "ci95": [],
        "model_version": FUSION_VERSION,
    }


def fuse(signals: dict) -> dict:
    """Собирает две оси из позлементных сигналов (SPEC §3).

    Раздельные модели для provenance и intent (без смешивания), эвристические
    веса, отказ от ответа при полном отсутствии сигналов.
    """
    votes_prov: dict = {}
    votes_int: dict = {}

    for element, blob in signals.items():
        sig = blob if isinstance(blob, dict) else {}
        for kind, votes, weights in (
            ("ai_provenance", votes_prov, PROV_WEIGHTS),
            ("intent", votes_int, INTENT_WEIGHTS),
        ):
            signal = sig.get(kind) or {}
            if not isinstance(signal, dict):
                continue
            _vote(
                votes,
                str(signal.get("label", "unknown")),
                float(signal.get("confidence", 0.0) or 0.0),
                weights.get(element, 0.0),
                element,
            )

    flags = _collect_evidence(signals)
    threats = _find_threats(flags)
    suppressors = _find_suppressors(flags)

    counter = [f"гаситель: {name}" for name in suppressors]
    return {
        "ai_provenance": _axis(votes_prov, threats, suppressors),
        "intent": _axis(votes_int, threats, suppressors),
        "rejection": not votes_prov and not votes_int,
        "heuristic": True,
        "threat_signals": threats,
        "counter_evidence": counter,
    }
