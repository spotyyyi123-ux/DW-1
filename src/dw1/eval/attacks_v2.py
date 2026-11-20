"""Strict attack evaluation from saved probabilities; thresholds use clean validation only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ATTACKS = (
    "paraphrase", "back_translation", "synonym_replacement", "sentence_reordering",
    "typo_injection", "homoglyphs_zerowidth", "burstiness_split_merge", "llm_paraphrase",
)
STRENGTHS = (0.25, 0.5, 0.75, 1.0)
CONDITIONS = [("clean", 0.0)] + [(a, s) for a in ATTACKS for s in STRENGTHS]
SIGNALS = {
    "tfidf": ("provenance", "y_provenance", "p_tfidf_provenance"),
    "fast_detectgpt": ("provenance", "y_provenance", "p_fast_detectgpt"),
    "ensemble": ("provenance", "y_provenance", "p_ensemble_provenance"),
    "intent": ("intent", "y_intent", "p_intent"),
}
VERSIONS = ("tfidf_model_version", "fd_model_version", "intent_model_version")
RECEIPT_HASHES = (
    "model_manifest_sha256", "external_result_sha256", "semantic_review_sha256",
    "calibration_sha256",
)
SCORING_SCOPE = "body_text_predictors_not_full_multielement_fusion"
REQUIRED = {
    "variant_id", "message_id", "attack", "strength", "text_sha256", "source_sha256",
    "y_provenance", "y_intent", "split", "changed", *VERSIONS,
    *(s[2] for s in SIGNALS.values()),
}


def _sha(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def validate_predictions(frame: pd.DataFrame, expected_messages: int = 600) -> None:
    """Reject partial joins, altered labels, cross-split duplicates and silent fallbacks."""
    if expected_messages < 1:
        raise ValueError("expected_messages must be positive")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate prediction columns")
    if REQUIRED - set(frame):
        raise ValueError(f"Missing prediction columns: {sorted(REQUIRED - set(frame))}")
    if {"text", "body", "prompt", "raw", "subject"} & set(frame):
        raise ValueError("Public predictions must not contain source text")
    if frame.empty or frame[list(REQUIRED)].isna().any().any():
        raise ValueError("Empty or incomplete prediction rows")
    for key in ("message_id", "variant_id"):
        if not frame[key].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError(f"Invalid {key}")
    if frame.variant_id.duplicated().any():
        raise ValueError("Duplicate variant_id")
    if frame.duplicated(["message_id", "attack", "strength"]).any():
        raise ValueError("Duplicate message/condition")
    if not set(frame.split) <= {"validation", "test"}:
        raise ValueError("Only validation/test splits are allowed")
    for key in ("y_provenance", "y_intent"):
        if not set(frame[key]) <= {0, 1}:
            raise ValueError(f"Invalid binary label {key}")
    for key in ("text_sha256", "source_sha256"):
        if not frame[key].map(_sha).all():
            raise ValueError(f"Invalid {key}")
        if (frame.groupby(key).split.nunique() > 1).any():
            raise ValueError(f"Cross-split duplicate {key}")
    if not frame.changed.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise ValueError("changed must be boolean")
    if any(not pd.api.types.is_numeric_dtype(frame[s[2]]) for s in SIGNALS.values()):
        raise ValueError("Probabilities must be numeric")
    try:
        probabilities = frame[[s[2] for s in SIGNALS.values()]].to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid probability") from exc
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (
        probabilities > 1
    ).any():
        raise ValueError("Probabilities must be finite in [0,1]")
    if not np.allclose(
        frame.p_ensemble_provenance.to_numpy(dtype=float),
        0.5 * (frame.p_tfidf_provenance.to_numpy(dtype=float)
               + frame.p_fast_detectgpt.to_numpy(dtype=float)),
        rtol=0, atol=1e-12,
    ):
        raise ValueError("Ensemble must average the two provenance probabilities only")
    for key in VERSIONS:
        if frame[key].nunique() != 1 or not isinstance(frame[key].iloc[0], str) or (
            not frame[key].iloc[0].strip()
        ):
            raise ValueError(f"A single nonempty {key} is required")
    clean = frame[(frame.attack == "clean") & (frame.strength == 0.0)]
    if len(clean) != expected_messages or clean.message_id.nunique() != expected_messages:
        raise ValueError(f"Need exactly {expected_messages} unique clean messages")
    if set(zip(frame.attack, frame.strength, strict=True)) != set(CONDITIONS):
        raise ValueError("Complete grid of clean + eight attacks at four strengths is required")
    ids = set(clean.message_id)
    for attack, strength in CONDITIONS:
        part = frame[(frame.attack == attack) & (frame.strength == strength)]
        if len(part) != expected_messages or set(part.message_id) != ids:
            raise ValueError(f"Incomplete population for {attack}/{strength}")
    for key in ("source_sha256", "y_provenance", "y_intent", "split"):
        if (frame.groupby("message_id")[key].nunique() != 1).any():
            raise ValueError(f"Message metadata changed between conditions: {key}")
    for split in ("validation", "test"):
        subset = clean[clean.split == split]
        if any(set(subset[key]) != {0, 1} for key in ("y_provenance", "y_intent")):
            raise ValueError("Both axes require both classes in validation and test")
    originals = clean.set_index("message_id").text_sha256
    actually_changed = frame.text_sha256 != frame.message_id.map(originals)
    if not (frame.changed == actually_changed).all():
        raise ValueError("changed flag does not match the clean text hash")
    if frame.loc[frame.y_provenance == 0, "changed"].any():
        raise ValueError("Attack scope is AI originals only; human messages must stay unchanged")
    llm_ai = frame[(frame.attack == "llm_paraphrase") & (frame.y_provenance == 1)]
    if not llm_ai.changed.all():
        raise ValueError("An LLM attack must actually change every targeted text")
    # Identical text must have identical scores, including untouched controls.
    for key in (s[2] for s in SIGNALS.values()):
        if (frame.groupby("text_sha256")[key].max()
            - frame.groupby("text_sha256")[key].min() > 1e-12).any():
            raise ValueError(f"Identical texts have different {key}")


def threshold_at_fpr(y: np.ndarray, p: np.ndarray, target_fpr: float = 0.01) -> float:
    """Lowest finite threshold meeting the empirical negative budget, including ties."""
    if not 0 <= target_fpr < 1 or set(y) != {0, 1} or not np.isfinite(p).all():
        raise ValueError("Finite scores, both classes and a target FPR in [0,1) are required")
    negatives = np.sort(p[y == 0])[::-1]
    allowed = math.floor(len(negatives) * target_fpr)
    # Strictly exceed the first disallowed negative. Tied negatives are never split.
    return float(np.nextafter(negatives[allowed], np.inf))


def _read_scoring_manifest(path: Path, predictions_sha256: str, n_messages: int) -> dict:
    """Bind final metrics to the scorer's input/semantic-review/calibration receipt."""
    if not path.is_file():
        raise ValueError("Final evaluation requires the matching scoring_manifest.json")
    content = path.read_bytes()
    try:
        receipt = json.loads(content)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid scoring_manifest.json") from exc
    if not isinstance(receipt, dict) or (
        receipt.get("status") != "scored_not_evaluated"
        or receipt.get("predictions_sha256") != predictions_sha256
        or receipt.get("n_messages") != n_messages
        or receipt.get("scoring_scope") != SCORING_SCOPE
        or not _sha(receipt.get("package_id"))
        or any(not _sha(receipt.get(key)) for key in RECEIPT_HASHES)
    ):
        raise ValueError("Scoring manifest does not match predictions or required evidence")
    return {
        "scoring_manifest_sha256": hashlib.sha256(content).hexdigest(),
        "package_id": receipt["package_id"],
        "scoring_scope": receipt["scoring_scope"],
        "scoring_evidence": {key: receipt[key] for key in RECEIPT_HASHES},
    }


def _ece(y: np.ndarray, p: np.ndarray) -> float:
    total = 0.0
    for index in range(10):
        mask = (p >= index / 10) & (p < (index + 1) / 10 if index < 9 else p <= 1)
        if mask.any():
            total += float(mask.mean() * abs(y[mask].mean() - p[mask].mean()))
    return total


def _recall(y: np.ndarray, predicted: np.ndarray) -> float | None:
    return float(predicted[y == 1].mean()) if (y == 1).any() else None


def _metrics(part: pd.DataFrame, label: str, score: str, threshold: float) -> dict:
    y = part[label].to_numpy(dtype=int)
    p = part[score].to_numpy(dtype=float)
    predicted = p >= threshold
    ai = part.y_provenance.to_numpy(dtype=int) == 1
    return {
        "n": len(y), "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()), "n_changed": int(part.changed.sum()),
        "tpr": _recall(y, predicted),
        "fpr": float(predicted[y == 0].mean()) if (y == 0).any() else None,
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "ece": _ece(y, p),
        "mean_positive_confidence": float(p[y == 1].mean()) if (y == 1).any() else None,
        "ai_subset_n_positive": int(((y == 1) & ai).sum()),
        "ai_subset_tpr": _recall(y[ai], predicted[ai]),
    }


def _svg(curves: list[dict]) -> str:
    """Portable SVG; no plotting runtime is needed on the offline machine."""
    colors = ("#2563eb", "#c2410c", "#15803d", "#9333ea",
              "#be123c", "#0f766e", "#854d0e", "#334155")
    chunks = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1040 740" '
        'role="img" aria-label="Test recall under attacks at frozen validation thresholds">',
        '<rect width="1040" height="740" fill="white"/>',
        '<g font-family="sans-serif" font-size="12" fill="#17212c">',
        '<text x="30" y="23" font-size="17">Test TPR; thresholds frozen on clean validation</text>',
    ]
    for number, signal in enumerate(SIGNALS):
        left, top = 58 + (number % 2) * 510, 65 + (number // 2) * 270
        chunks.append(f'<text x="{left}" y="{top - 12}" font-size="15">{signal}</text>')
        for tick in (0.0, 0.5, 1.0):
            yy = top + 190 * (1 - tick)
            chunks.append(f'<path d="M {left} {yy} h 420" stroke="#e2e8f0"/>')
            chunks.append(f'<text x="{left - 30}" y="{yy + 4}">{tick:g}</text>')
            xx = left + tick * 420
            chunks.append(f'<text x="{xx - 4}" y="{top + 210}">{tick:g}</text>')
        chunks.append(f'<path d="M {left} {top} v 190 h 420" fill="none" stroke="#475569"/>')
        for attack, color in zip(ATTACKS, colors, strict=True):
            line = next(c for c in curves if c["signal"] == signal and c["attack"] == attack)
            points = " ".join(
                f'{left + row["strength"] * 420:.2f},{top + (1 - row["tpr"]) * 190:.2f}'
                for row in line["points"] if row["tpr"] is not None
            )
            chunks.append(f'<polyline points="{points}" fill="none" stroke="{color}" '
                          'stroke-width="2"/>')
    for index, (attack, color) in enumerate(zip(ATTACKS, colors, strict=True)):
        x, y = 45 + (index % 2) * 510, 635 + (index // 2) * 24
        chunks.append(f'<path d="M{x} {y - 4} h24" stroke="{color}" stroke-width="3"/>')
        chunks.append(f'<text x="{x + 31}" y="{y}">{attack}</text>')
    chunks.append('</g></svg>')
    return "\n".join(chunks)


def _report(result: dict) -> str:
    counts = result["population"]
    lines = [
        "# Sprint 4 — атаки v2: сохранённый прогон", "",
        f'Статус: **{result["status"]}**. Источник: `predictions.parquet`, '
        f'SHA-256 `{result["predictions_sha256"]}`.', "",
        f'Всего исходных писем: {counts["total"]}; чистая validation: '
        f'{counts["validation"]}; отложенный test: {counts["test"]}. '
        "Кривые и таблицы рассчитаны только на test, а не на всей сумме validation + test.", "",
        "Порог каждой модели выбран на чистой validation при эмпирическом FPR ≤ 1% "
        "и зафиксирован для всех атак. Фактический FPR на test измерен отдельно; "
        "гарантия ≤ 1% на test из этого не следует. Калибровка и обучающие данные "
        "моделей проверяются подготовкой предсказаний, не этим отчётом.", "",
        "Ансамбль усредняет только вероятности происхождения TF-IDF и Fast-DetectGPT. "
        "Ось intent оценивается отдельно по замороженной модели тела письма "
        "(predict_proba), а не по фьюжну всех элементов письма. "
        "Метка provenance описывает происхождение "
        "исходного письма; LLM-перефраз меняет стиль уже AI-писем.", "",
        "Атакуются только исходные AI-письма. Человеческий спам и легитимные письма "
        "остаются контрольными: устойчивость ко всем видам вредоносной почты этим "
        "экспериментом не доказана. URL/числа и смысл требуют отдельного контроля "
        "качества перефраза; одной проверки хэшей для смысловой эквивалентности мало.", "",
        "| Модель / ось | Чистый TPR | Худший TPR | Максимальная потеря, п.п. | "
        "Максимальный FPR |", "|---|---:|---:|---:|---:|",
    ]
    for signal in SIGNALS:
        records = [r for r in result["conditions"] if r["signal"] == signal]
        baseline = next(r for r in records if r["attack"] == "clean")
        worst = min(r["tpr"] for r in records)
        fpr = max(r["fpr"] for r in records)
        lines.append(f'| {signal} | {baseline["tpr"]:.4f} | {worst:.4f} | '
                     f'{100 * (baseline["tpr"] - worst):.2f} | {fpr:.4f} |')
    intent = [r for r in result["conditions"] if r["signal"] == "intent"]
    clean = next(r for r in intent if r["attack"] == "clean")
    worst = min(r["tpr"] for r in intent)
    lines.extend([
        "", f'На данной test-выборке максимальное снижение общего intent TPR: '
        f'{100 * (clean["tpr"] - worst):.2f} п.п. '
        "Это наблюдение ограничено атакой AI-части. В JSON дополнительно сохранён "
        "TPR на AI-подмножестве, чтобы неизменный человеческий спам не скрывал потери.", "",
        "Артефакты: `metrics.json`, `degradation_curves.json`, `summary.csv`, `curves.svg`. "
        "Точные значения хранятся в JSON; округление в таблице служит только для чтения.",
    ])
    if counts["total"] != 600:
        lines.extend([
            "", "**Это проверочный прогон другого размера; критерий 600 писем не выполнен.**",
        ])
    if result.get("scoring_manifest_sha256"):
        lines.extend([
            "", f'Квитанция scorer: SHA-256 `{result["scoring_manifest_sha256"]}`; '
            f'пакет `{result["package_id"]}`. Квитанция связывает предсказания с хэшами '
            "манифеста моделей, внешнего результата, смысловой проверки и калибровки. "
            "Происхождение входов и полнота списков обучения подтверждены владельцем; "
            "хэш-связь не удостоверяет личность и не заменяет независимый аудит этих утверждений.",
        ])
    return "\n".join(lines) + "\n"


def evaluate_predictions(
    predictions: Path, out: Path, *, expected_messages: int = 600,
    scoring_manifest: Path | None = None,
) -> dict:
    """Save all reports from one validated, complete prediction artifact."""
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Evaluation output must be empty; choose a new run directory")
    payload = predictions.read_bytes()
    predictions_sha256 = hashlib.sha256(payload).hexdigest()
    evidence = {}
    if expected_messages == 600 or scoring_manifest is not None:
        evidence = _read_scoring_manifest(
            scoring_manifest or predictions.with_name("scoring_manifest.json"),
            predictions_sha256, expected_messages,
        )
    frame = pd.read_parquet(io.BytesIO(payload))
    validate_predictions(frame, expected_messages)
    clean = frame[frame.attack == "clean"]
    validation = clean[clean.split == "validation"]
    test = frame[frame.split == "test"]
    thresholds = {
        signal: threshold_at_fpr(
            validation[label].to_numpy(dtype=int), validation[score].to_numpy(dtype=float)
        ) for signal, (_, label, score) in SIGNALS.items()
    }
    records = []
    baselines = {}
    for attack, strength in CONDITIONS:
        part = test[(test.attack == attack) & (test.strength == strength)]
        for signal, (axis, label, score) in SIGNALS.items():
            values = _metrics(part, label, score, thresholds[signal])
            if attack == "clean":
                baselines[signal] = values
            base = baselines[signal]
            values.update(
                signal=signal, axis=axis, attack=attack, strength=strength,
                threshold=thresholds[signal], tpr_delta=values["tpr"] - base["tpr"],
                fpr_delta=values["fpr"] - base["fpr"],
                degradation_pct=(100 * (base["tpr"] - values["tpr"]) / base["tpr"]
                                 if base["tpr"] else None),
            )
            records.append(values)
    curves = [
        {
            "signal": signal, "axis": axis, "attack": attack,
            "threshold": thresholds[signal],
            "points": [{k: r[k] for k in (
                "strength", "tpr", "fpr", "degradation_pct", "ai_subset_tpr", "n_changed",
            )} for r in records if r["signal"] == signal and r["attack"] in ("clean", attack)],
        } for signal, (axis, _, _) in SIGNALS.items() for attack in ATTACKS
    ]
    result = {
        "schema_version": 1,
        "status": "measured" if expected_messages == 600 else "validation_run_not_600",
        "predictions_sha256": predictions_sha256,
        "population": {
            "total": len(clean), "validation": len(validation),
            "test": int((clean.split == "test").sum()),
            "ai_test": int(((clean.split == "test") & (clean.y_provenance == 1)).sum()),
        },
        "attack_scope": "ai_generated_originals_only; human controls unchanged",
        "evaluation_split": "test", "threshold_split": "clean_validation",
        "target_validation_fpr": 0.01, "thresholds": thresholds,
        "model_versions": {key: frame[key].iloc[0] for key in VERSIONS},
        "conditions": records,
        **evidence,
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, content in (("metrics.json", result), ("degradation_curves.json", curves)):
        (out / name).write_text(
            json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (out / "curves.svg").write_text(_svg(curves), encoding="utf-8")
    (out / "report.md").write_text(_report(result), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-messages", type=int, default=600)
    parser.add_argument("--scoring-manifest", type=Path)
    args = parser.parse_args(argv)
    result = evaluate_predictions(
        args.predictions, args.out, expected_messages=args.expected_messages,
        scoring_manifest=args.scoring_manifest,
    )
    print(json.dumps({"status": result["status"], "population": result["population"]}))


if __name__ == "__main__":
    main()
