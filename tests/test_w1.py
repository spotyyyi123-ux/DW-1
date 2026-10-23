"""Проверки меток происхождения, разделения ответов по вопросам и расчёта метрик."""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dw1.eval.harness import binary_metrics

PREP = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/prepare_w1.py"))


def test_conflicting_provenance_is_removed_not_arbitrarily_assigned():
    frame = pd.DataFrame(
        [
            {"text": "A sufficiently long human message with the same contents.", "label": 0},
            {"text": "A sufficiently long human message with the same contents.", "label": 1},
            {"text": "A different sufficiently long original human message.", "label": 0},
        ]
    )
    clean = PREP["clean_exact"](frame)
    assert len(clean) == 1
    assert clean.iloc[0].text.startswith("A different")


def test_question_answers_share_split(tmp_path):
    path = tmp_path / "hc3.jsonl"
    path.write_text(
        '{"question":"Question?", "human_answers":["Human"], '
        '"chatgpt_answers":["AI", "Other AI"]}\n'
    )
    frame = PREP["load_hc3"](path)
    assert frame.group_id.nunique() == 1
    assert frame.split.nunique() == 1
    assert frame.label.tolist() == [0, 1, 1]


def test_near_duplicate_does_not_leak_into_test():
    words = " ".join("word" + chr(97 + i // 26) + chr(97 + i % 26) for i in range(60))
    frame = pd.DataFrame(
        [
            {"text": words, "split": "train"},
            {"text": words + " thank you", "split": "test"},
            {"text": "a completely unrelated example in the test set", "split": "test"},
        ]
    )
    result, removed = PREP["remove_near_leaks"](frame)
    assert removed == 1
    assert result.index.tolist() == [0, 2]


def test_metrics_use_ai_as_positive_class():
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9])
    assert metrics["f1"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["roc_auc"] == 0.75


def test_ai_only_data_cannot_be_claimed_as_binary_evaluation():
    with pytest.raises(ValueError, match="Both"):
        binary_metrics(np.ones(3), [0.2, 0.4, 0.9])
