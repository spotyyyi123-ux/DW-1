"""Офлайн-подключение скоров Fast-DetectGPT без загрузки LLM на локальную машину."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
SCORE_COLUMNS = {
    "message_id",
    "text_sha256",
    "group_id",
    "dataset",
    "language",
    "label",
    "split",
    "criterion",
    "model_version",
    "n_tokens",
    "truncated",
}


def text_hash(text: str) -> str:
    """Хэш полного текста; одинаково обрабатываем края и переводы строк."""
    return hashlib.sha256(text.replace("\r\n", "\n").strip().encode("utf-8")).hexdigest()


def validate_scores(frame: pd.DataFrame) -> None:
    """Проверяет типы, происхождение и отсутствие явной утечки между частями."""
    missing = SCORE_COLUMNS - set(frame)
    if missing:
        raise ValueError(f"В таблице скоров отсутствуют поля: {sorted(missing)}")
    if {"text", "body", "prompt"} & set(frame):
        raise ValueError("Таблица скоров не должна содержать исходные тексты")
    if frame.empty or frame[list(SCORE_COLUMNS)].isna().any().any():
        raise ValueError("Таблица скоров пуста или содержит пропуски")
    if not set(frame.label) <= {0, 1} or not set(frame.language) <= {"en", "ru"}:
        raise ValueError("Некорректный язык или метка")
    if not set(frame.split) <= {"train", "validation", "test"}:
        raise ValueError("Неизвестная часть выборки")
    if not set(frame.dataset) <= {"coat", "hc3-en"}:
        raise ValueError("Для этого эксперимента допустимы только публичные CoAT и HC3")
    if frame.model_version.nunique() != 1 or not str(frame.model_version.iloc[0]):
        raise ValueError("Один файл должен содержать скоры одной версии детектора")
    if not np.isfinite(frame.criterion.to_numpy(dtype=float)).all():
        raise ValueError("Критерий детектора должен быть конечным числом")
    if not frame.text_sha256.map(
        lambda x: isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)
    ).all():
        raise ValueError("Некорректный SHA-256 текста")
    if frame.text_sha256.duplicated().any() or frame.message_id.duplicated().any():
        raise ValueError("В файле есть дубли текстов или идентификаторов")
    if frame.groupby("group_id").split.nunique().max() != 1:
        raise ValueError("Родственные тексты попали в разные части")
    if (frame.n_tokens < 32).any():
        raise ValueError("Для скоров требуется минимум 32 предсказываемых токена")


class LLMScoreStore:
    """Ищет точное совпадение текста; на незнакомых письмах не выдумывает результат."""

    def __init__(self, scores: Path, calibration: Path):
        self.frame = pd.read_parquet(scores)
        validate_scores(self.frame)
        self.config = json.loads(calibration.read_text(encoding="utf-8"))
        self.model_version = str(self.frame.model_version.iloc[0])
        digest = hashlib.sha256(scores.read_bytes()).hexdigest()
        if self.config.get("score_sha256") != digest:
            raise ValueError("Калибровка относится к другому файлу скоров")
        if self.config.get("model_version") != self.model_version:
            raise ValueError("Не совпадает версия LLM-детектора")
        if self.config.get("schema_version") != 1:
            raise ValueError("Неизвестная версия формата калибровки")
        for cfg in self.config["languages"].values():
            xs = np.asarray(cfg["isotonic_x"], dtype=float)
            ys = np.asarray(cfg["isotonic_y"], dtype=float)
            if (
                len(xs) < 1
                or len(xs) != len(ys)
                or not np.isfinite(xs).all()
                or not np.isfinite(ys).all()
                or (np.diff(xs) <= 0).any()
                or (np.diff(ys) < 0).any()
                or (ys < 0).any()
                or (ys > 1).any()
            ):
                raise ValueError("Некорректная изотоническая калибровка")
        self.rows = self.frame.set_index("text_sha256")

    def lookup(self, text: str) -> dict | None:
        key = text_hash(text)
        if key not in self.rows.index:
            return None
        row = self.rows.loc[key]
        language = str(row.language)
        cfg = self.config["languages"].get(language)
        if cfg is None:
            return None
        probability = float(np.interp(float(row.criterion), cfg["isotonic_x"], cfg["isotonic_y"]))
        label = "ai_generated" if probability >= 0.5 else "human"
        return {
            "label": label,
            "confidence": probability if label == "ai_generated" else 1 - probability,
            "model_version": self.model_version + ":" + self.config["calibration_version"],
            "evidence": [
                {
                    "signal": "fast_detectgpt",
                    "weight": 1.0,
                    "value": dict(
                        criterion=float(row.criterion),
                        p_ai=probability,
                        language=language,
                        calibrated=True,
                        n_tokens=int(row.n_tokens),
                        truncated=bool(row.truncated),
                        matched_by="full_text_sha256",
                        evaluation_scope="public_benchmark_only",
                    ),
                }
            ],
        }


@lru_cache(maxsize=4)
def _load(scores: str, calibration: str, score_mtime: int, calibration_mtime: int):
    return LLMScoreStore(Path(scores), Path(calibration))


def cached_llm_signal(text: str) -> dict | None:
    """Пути задаются переменными среды; без артефактов сохраняются прежние эвристики."""
    score_path = Path(
        os.environ.get("DW1_LLM_SCORES", ROOT / "data/runs/sprint2/llm_scores.parquet")
    )
    calibration_path = Path(
        os.environ.get("DW1_LLM_CALIBRATION", ROOT / "data/runs/sprint2/llm_calibration.json")
    )
    if not score_path.is_file() or not calibration_path.is_file():
        return None
    try:
        store = _load(
            str(score_path),
            str(calibration_path),
            score_path.stat().st_mtime_ns,
            calibration_path.stat().st_mtime_ns,
        )
        return store.lookup(text)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        log.warning("LLM-скоры не подключены: %s", exc)
        return None
