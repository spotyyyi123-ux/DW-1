"""Базовая модель определения происхождения текста: TF-IDF → SVD → HistGradientBoosting.

SVD создаёт небольшую плотную матрицу: классификатор не принимает разреженную TF-IDF.
Метки: 0 — человек, 1 — ИИ. Назначение письма не определяется; оценки не калиброваны.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

MODEL_VERSION = "tfidf-svd64-histgb-w1-v1"


def build_model(seed=42):
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=15000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            ("svd", TruncatedSVD(n_components=64, random_state=seed)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=100,
                    max_leaf_nodes=15,
                    learning_rate=0.1,
                    early_stopping=False,
                    random_state=seed,
                ),
            ),
        ]
    )


def main():
    """Проверяет текст из UTF-8 файла с помощью ранее обученной модели."""
    import argparse
    import json
    from pathlib import Path

    import joblib
    from threadpoolctl import threadpool_limits

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/runs/sprint1-v1/model.joblib")
    parser.add_argument("--text-file", required=True)
    args = parser.parse_args()
    model = joblib.load(args.model)
    text = Path(args.text_file).read_text(encoding="utf-8")
    with threadpool_limits(limits=2):
        score = float(model.predict_proba([text])[0, 1])
    print(
        json.dumps(
            {
                "ai_score_uncalibrated": score,
                "label": "ai" if score >= 0.5 else "human",
                "intent": "unknown",
                "model_version": getattr(model, "dw1_model_version_", MODEL_VERSION),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
