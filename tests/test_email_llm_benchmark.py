"""Проверки подготовки разрешённых писем для внешнего LLM-прогона."""
from __future__ import annotations

import csv
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
import pytest

from scripts import prepare_email_llm_benchmark as prep


def benchmark(tmp_path: Path, monkeypatch, sources, labels):
    rows = []
    for i, (source, label) in enumerate(zip(sources, labels, strict=True)):
        msg = EmailMessage()
        msg.set_content('Public benchmark body. ' * 20, charset='utf-8', cte='base64')
        name = f'{i}.eml'
        (tmp_path / name).write_bytes(msg.as_bytes())
        rows.append({'file': name, 'source': source, 'provenance': label})
    with (tmp_path / 'labels.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['file', 'source', 'provenance'])
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / 'out.jsonl'
    monkeypatch.setattr(prep, 'BENCH', tmp_path)
    monkeypatch.setattr(prep, 'OUT', out)
    return out


def test_mime_decoding_and_duplicate_bodies(tmp_path, monkeypatch):
    out = benchmark(tmp_path, monkeypatch, ['nazario', 'nazario'], ['human', 'human'])
    prep.main()
    frame = pd.read_json(out, lines=True)
    assert len(frame) == 1
    assert frame.iloc[0].text.startswith('Public benchmark body.')
    assert frame.iloc[0].group_id == frame.iloc[0].text_sha256


def test_conflicting_provenance_rejected(tmp_path, monkeypatch):
    benchmark(tmp_path, monkeypatch, ['nazario', 'llmgen'], ['human', 'ai_generated'])
    with pytest.raises(ValueError, match='противоречивую'):
        prep.main()


def test_unapproved_email_source_rejected(tmp_path, monkeypatch):
    benchmark(tmp_path, monkeypatch, ['enron'], ['human'])
    with pytest.raises(ValueError, match='LLMGen и Nazario'):
        prep.main()
