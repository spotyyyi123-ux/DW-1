"""Проверки новых источников: группировка тем и настоящая граница RoFT."""

import json

import pandas as pd
import pytest

from dw1.ingest.text_datasets import prepare_roft, sample_llmtrace


def test_llmtrace_topic_never_crosses_splits(tmp_path):
    for split in ["train", "valid", "test"]:
        rows = [
            dict(
                topic_id="same-topic",
                lang=lang,
                label=label,
                text=f"Достаточно длинный текст для проверки {split} {lang} {label}",
            )
            for lang in ["ru", "eng"]
            for label in ["human", "ai"]
        ]
        (tmp_path / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )
    data, audit = sample_llmtrace(tmp_path, cap=2)
    assert data.group_id.nunique() == 1
    assert data.split.nunique() == 1
    assert set(data.language) == {"ru", "en"}
    assert set(data.label) == {0, 1}
    assert audit["raw_rows"] == 12
    second, _ = sample_llmtrace(tmp_path, cap=2)
    pd.testing.assert_frame_equal(data, second)


def test_llmtrace_refuses_missing_group_id(tmp_path):
    row = dict(lang="ru", label="ai", text="Достаточно длинный текст без идентификатора темы.")
    (tmp_path / "train.jsonl").write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="topic_id"):
        sample_llmtrace(tmp_path)


def roft_row(**overrides):
    row = dict(
        prompt=1,
        generation=2,
        prompt_body="Человек первый._SEP_Человек второй.",
        gen_body="Машинный текст.",
        model="gpt2",
        true_boundary_index=1,
        predicted_boundary_index=0,
    )
    row.update(overrides)
    return row


def test_roft_uses_true_boundary_not_human_guess(tmp_path):
    path = tmp_path / "roft.csv"
    pd.DataFrame([roft_row(), roft_row(predicted_boundary_index=2)]).to_csv(path, index=False)
    docs, segments, audit = prepare_roft(path)
    assert len(docs) == 1
    assert segments.label.tolist() == [0, 0, 1]
    assert docs.iloc[0].first_ai_segment_index == 2
    assert docs.iloc[0].ai_provenance == "unknown"
    assert docs.iloc[0].composition == "human_then_ai"
    assert audit["skipped"]["repeated_annotation"] == 1


def test_roft_does_not_invent_ai_for_human_only_document(tmp_path):
    path = tmp_path / "roft.csv"
    pd.DataFrame([roft_row(gen_body="", model="human")]).to_csv(path, index=False)
    docs, segments, _ = prepare_roft(path)
    assert docs.iloc[0].composition == "human_only"
    assert segments.label.tolist() == [0, 0]
    assert docs.iloc[0].first_ai_segment_index == 2


def test_roft_rejects_inconsistent_boundary(tmp_path):
    path = tmp_path / "roft.csv"
    pd.DataFrame([roft_row(true_boundary_index=5)]).to_csv(path, index=False)
    docs, _, audit = prepare_roft(path)
    assert len(docs) == 0
    assert audit["skipped"]["invalid_boundary_or_empty"] == 1


def test_boundary_rule_finds_single_transition():
    from dw1.eval.compare import first_ai_boundary

    assert first_ai_boundary([0.01, 0.05, 0.95, 0.99]) == 2
    assert first_ai_boundary([0.01, 0.02, 0.05]) == 3
    assert first_ai_boundary([0.95, 0.98]) == 0
