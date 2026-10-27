"""Фузия (задача 4): две независимые оси из позлементных сигналов.

Взвешенное голосование по элементам письма. База — эвристика (``heuristic: true``);
после Sprint 2 к ней добавляется isotonic-калибровка уверенностей на бенчмарке
письменных данных (`data/runs/sprint2/calibration_metrics.json`). Если калибровка
загружена, ``calibrated: true`` и ``confidence`` становится калиброванной вероятностью.

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

import json
import os
from pathlib import Path

import numpy as np

FUSION_VERSION = "fusion-heuristic-v2"
CALIBRATION_VERSION = "isotonic-sprint2"

CALIBRATION_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "runs"
    / "sprint2"
    / "calibration_metrics.json"
)

_CALIBRATION: dict | None = None
_CALIBRATION_MTIME: float = 0.0

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


def _components(
    votes: dict,
    positive: tuple[str, ...],
    negative: tuple[str, ...],
    pos_key: str,
    neg_key: str,
) -> dict:
    """Сырые компоненты оси: доля веса в «положительной» и «отрицательной» группах.

    Нужны для калибровки (isotonic): score = share(positive), P(y=positive | score).
    """
    total = sum(slot["score"] for slot in votes.values())
    if total <= 0:
        return {pos_key: 0.0, neg_key: 0.0}
    positive_score = sum(slot["score"] for label, slot in votes.items() if label in positive)
    negative_score = sum(slot["score"] for label, slot in votes.items() if label in negative)
    return {
        pos_key: round(positive_score / total, 4),
        neg_key: round(negative_score / total, 4),
    }


def _calibration() -> dict | None:
    """Загружает isotonic-калибровку, собранную на бенчмарке писем (если есть)."""
    global _CALIBRATION, _CALIBRATION_MTIME
    path = Path(os.environ.get("DW1_FUSION_CALIBRATION", str(CALIBRATION_PATH)))
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if _CALIBRATION is None or mtime != _CALIBRATION_MTIME:
        _CALIBRATION = json.loads(path.read_text(encoding="utf-8"))
        _CALIBRATION_MTIME = mtime
    return _CALIBRATION


def _apply_isotonic(value: float, axis: dict) -> float:
    """Кусочно-линейное применение сохранённых порогов isotonic-регрессии."""
    xs = [float(v) for v in (axis.get("calibrator_x") or [])]
    ys = [float(v) for v in (axis.get("calibrator_y") or [])]
    if len(xs) < 2 or len(ys) != len(xs):
        return value
    return float(np.interp(value, xs, ys))


def _apply_calibration(axis: dict, raw_positive: float, calibration: dict, key: str) -> None:
    """Считает калиброванную вероятность и кладёт её отдельным полем.

    Важно: ``confidence`` (эвристическая) НЕ перезаписывается — калибровка
    измерена на публичном бенчмарке и на других доменах (карточки демо) может
    «сплющиваться»; поэтому храним её как ``calibrated_confidence``,
    а решение о подмене — за потребителем (API/карточка/Sprint 3).
    """
    cfg = (calibration.get("axes") or {}).get(key) or {}
    if not cfg:
        return
    p_positive = float(np.clip(_apply_isotonic(raw_positive, cfg), 0.0, 1.0))
    label = axis.get("label")
    if label in ("ai_generated", "ai_polished", "phishing", "spam"):
        axis["calibrated_confidence"] = round(p_positive, 3)
    elif label in ("human", "legitimate"):
        axis["calibrated_confidence"] = round(1.0 - p_positive, 3)
    else:
        return
    axis["calibrated"] = True
    axis["calibration_version"] = CALIBRATION_VERSION


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
    prov_axis = _axis(votes_prov, threats, suppressors)
    prov_axis["components"] = _components(
        votes_prov, ("ai_generated", "ai_polished"), ("human",), "ai", "human"
    )
    intent_axis = _axis(votes_int, threats, suppressors)
    intent_axis["components"] = _components(
        votes_int, ("phishing", "spam"), ("legitimate",), "malicious", "benign"
    )

    calibration = _calibration()
    if calibration:
        _apply_calibration(
            prov_axis, prov_axis["components"]["ai"], calibration, "ai_provenance"
        )
        _apply_calibration(
            intent_axis, intent_axis["components"]["malicious"], calibration, "intent"
        )

    return {
        "ai_provenance": prov_axis,
        "intent": intent_axis,
        "rejection": not votes_prov and not votes_int,
        "heuristic": True,
        "calibrated": bool(calibration),
        "threat_signals": threats,
        "counter_evidence": counter,
    }
