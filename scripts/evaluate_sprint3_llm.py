"""Сравнение на общих текстах и перенос калибратора на независимые письма."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from dw1.eval.llm_calibration import calibrate
from dw1.models.llm_detectors import LLMScoreStore, validate_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/runs/sprint3')
    args = ap.parse_args()
    root = Path(args.root)
    frames = {name: pd.read_parquet(root / f'{name}_scores.parquet')
              for name in ['gptneo', 'qwen']}
    for frame in frames.values():
        validate_scores(frame)
    common = set(frames['gptneo'].text_sha256) & set(frames['qwen'].text_sha256)
    metadata = ['text_sha256', 'label', 'split', 'language', 'group_id']
    left, right = [f[f.text_sha256.isin(common)][metadata].sort_values('text_sha256')
                   .reset_index(drop=True) for f in frames.values()]
    pd.testing.assert_frame_equal(left, right)
    results = {}
    configs = {}
    for name, frame in frames.items():
        matched = frame[frame.text_sha256.isin(common)].copy()
        cfg, metrics, predictions = calibrate(matched)
        configs[name] = cfg
        results[name] = metrics
        predictions.to_parquet(root / f'{name}_test_predictions.parquet', index=False)
    emails = pd.read_parquet(root / 'email_scores.parquet')
    validate_scores(emails)
    if set(emails.text_sha256) & set(frames['qwen'].text_sha256):
        raise ValueError('Письма пересекаются с обучением калибратора')
    if emails.model_version.iloc[0] != frames['qwen'].model_version.iloc[0]:
        raise ValueError('Письма и калибратор требуют одной версии детектора')
    config = configs['qwen']
    config['calibration_source_sha256'] = hashlib.sha256(
        (root / 'qwen_scores.parquet').read_bytes()).hexdigest()
    config['score_sha256'] = hashlib.sha256((root / 'email_scores.parquet').read_bytes()).hexdigest()
    config['evaluation_scope'] = 'public_emails_transfer_from_hc3_coat'
    calibration = root / 'email_calibration.json'
    calibration.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    store = LLMScoreStore(root / 'email_scores.parquet', calibration)
    import numpy as np

    from dw1.eval.llm_calibration import _summary
    emails['p_ai'] = [float(np.interp(r.criterion,
                           config['languages'][r.language]['isotonic_x'],
                           config['languages'][r.language]['isotonic_y']))
                      for r in emails.itertuples()]
    results['emails_transfer'] = _summary(emails.label, emails.p_ai)
    emails.to_parquet(root / 'email_predictions.parquet', index=False)
    assert len(store.rows) == len(emails)
    results['n_common'] = len(common)
    results['n_emails'] = len(emails)
    (root / 'metrics.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    rows = ['# Sprint 3: Fast-DetectGPT', '',
            f'Общих текстов: {len(common)}. Писем: {len(emails)}.', '',
            '| Детектор | Язык | test | ECE до/после | F1 после | AUC после |',
            '|---|---|---:|---:|---:|---:|']
    for name in frames:
        for lang in ['en', 'ru']:
            m = results[name][lang]
            rows.append(f"| {name} | {lang} | {m['n_test']} | "
                        f"{m['before']['ece']:.4f}/{m['after']['ece']:.4f} | "
                        f"{m['after']['f1']:.4f} | {m['after']['roc_auc']:.4f} |")
    rows += ['', 'Platt: train; isotonic: validation; показатели: test на общих хэшах.',
             'Для писем шкала перенесена с HC3/CoAT без обучения на письмах.',
             'Это смена домена; калибровка на почте не гарантируется.',
             'Все письма — фишинг; качество определения intent здесь не измеряется.',
             'Метки human у Nazario — допущение исторического корпуса, не экспертиза авторства.',
             '', 'Метрики писем (перенос):', '```json',
             json.dumps(results['emails_transfer'], indent=2), '```']
    (root / 'report.md').write_text('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
