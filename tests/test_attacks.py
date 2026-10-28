from __future__ import annotations

import random

import numpy as np

from scripts.attacks import (
    ATTACK_NAMES,
    ATTACK_REGISTRY,
    apply_attack_to_eml_text,
    attack_back_translation,
    attack_burstiness_split_merge,
    attack_homoglyphs_zerowidth,
    attack_paraphrase,
    attack_sentence_reordering,
    attack_synonym_replacement,
    attack_typo_injection,
    calculate_tpr_at_fpr,
)


def test_registry_contains_all_seven_attacks():
    assert len(ATTACK_NAMES) == 7
    for name in ATTACK_NAMES:
        assert name in ATTACK_REGISTRY
        fn = ATTACK_REGISTRY[name]
        assert callable(fn)


def test_attack_paraphrase():
    text = (
        "Пожалуйста, обратите внимание, что доступ будет заблокирован в связи с тем, "
        "что вы не подтвердили данные."
    )
    res = attack_paraphrase(text, strength=1.0, rng=random.Random(42))
    assert res != text
    assert "просим принять к сведению" in res or "так как" in res

    # Проверка на пустую строку
    assert attack_paraphrase("") == ""


def test_attack_back_translation():
    text = "Добрый день, уважаемый пользователь! Пройдите авторизацию в целях безопасности."
    res = attack_back_translation(text, strength=1.0, rng=random.Random(42))
    assert res != text
    assert "здравствуйте" in res or "клиент" in res or "войдите в систему" in res

    assert attack_back_translation("") == ""


def test_attack_synonym_replacement():
    text = "Срочно подтвердите доступ к корпоративной почте и оплатите счёт."
    res = attack_synonym_replacement(text, strength=1.0, rng=random.Random(42))
    assert res != text
    # Одно из ключевых слов должно быть заменено
    expected_syns = ["верифицируйте", "безотлагательно", "погасите", "инвойс", "права доступа"]
    assert any(syn in res for syn in expected_syns)

    # Проверка регистра
    caps_text = "СРОЧНО"
    caps_res = attack_synonym_replacement(caps_text, strength=1.0, rng=random.Random(42))
    assert caps_res.isupper()

    assert attack_synonym_replacement("") == ""


def test_attack_sentence_reordering():
    text = "Предложение 1. Предложение 2. Предложение 3. Предложение 4. Предложение 5."
    res = attack_sentence_reordering(text, strength=1.0, rng=random.Random(42))
    assert res.startswith("Предложение 1.")
    assert res.endswith("Предложение 5.")

    # Короткий текст не ломается
    short_text = "Одно предложение."
    assert attack_sentence_reordering(short_text, strength=1.0) == short_text
    assert attack_sentence_reordering("") == ""


def test_attack_typo_injection():
    text = "Пожалуйста подтвердите ваши учетные данные немедленно."
    res = attack_typo_injection(text, strength=1.0, rng=random.Random(42))
    assert res != text
    assert len(res) > 0

    assert attack_typo_injection("") == ""


def test_attack_homoglyphs_zerowidth():
    text = "Срочно подтвердите доступ"
    res = attack_homoglyphs_zerowidth(text, strength=1.0, rng=random.Random(42))
    assert res != text
    # Проверяем наличие латинских символов-гомоглифов или zero-width space
    has_lat = any("a" <= ch <= "z" or "A" <= ch <= "Z" for ch in res)
    has_zw = "\u200b" in res
    assert has_lat or has_zw

    assert attack_homoglyphs_zerowidth("") == ""


def test_attack_burstiness_split_merge():
    # Длинное предложение должно быть разбито
    long_sent = (
        "Это очень длинное предложение которое содержит огромное количество "
        "слов и информации для проверки алгоритма разбиения на части."
    )
    res_split = attack_burstiness_split_merge(long_sent, strength=1.0, rng=random.Random(42))
    assert "." in res_split.rstrip(".")

    # Короткие предложения должны быть объединены
    short_sents = "Привет. Как дела."
    res_merge = attack_burstiness_split_merge(short_sents, strength=1.0, rng=random.Random(42))
    assert " и " in res_merge or " and " in res_merge

    assert attack_burstiness_split_merge("") == ""


def test_apply_attack_to_eml_text():
    eml = (
        "From: security@bank.example\n"
        "To: user@example.org\n"
        "Subject: Срочно подтвердите доступ\n"
        "MIME-Version: 1.0\n"
        "Content-Type: text/plain; charset=utf-8\n\n"
        "Пожалуйста, обратите внимание, что доступ будет заблокирован в связи с тем, "
        "что вы не ввели пароль.\n"
    )
    res_eml = apply_attack_to_eml_text(eml, "paraphrase", strength=1.0)
    assert "From: security@bank.example" in res_eml
    assert "To: user@example.org" in res_eml
    assert len(res_eml) > 0


def test_calculate_tpr_at_fpr():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.85, 0.8, 0.75, 0.2, 0.15, 0.1, 0.05])
    tpr = calculate_tpr_at_fpr(y_true, y_score, target_fpr=0.01)
    # При 0 FP достигается TPR
    assert tpr >= 0.0

    # Граничные случаи
    assert calculate_tpr_at_fpr(np.array([]), np.array([])) == 0.0
    assert calculate_tpr_at_fpr(np.array([1, 1]), np.array([0.9, 0.8])) == 0.0
    assert calculate_tpr_at_fpr(np.array([0, 0]), np.array([0.9, 0.8])) == 0.0


def test_run_benchmark_and_main(tmp_path):
    import csv

    from scripts.attacks import main, run_benchmark

    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    out_dir = tmp_path / "out"

    # Создаём минимальный labels.csv и 3 eml файла
    labels = [
        {
            "file": "email_ai.eml",
            "provenance": "ai_generated",
            "intent": "phishing",
            "source": "llmgen",
        },
        {"file": "email_spam.eml", "provenance": "human", "intent": "spam", "source": "seven"},
        {"file": "email_ham.eml", "provenance": "human", "intent": "legitimate", "source": "enron"},
    ]
    with open(bench_dir / "labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "provenance", "intent", "source"])
        writer.writeheader()
        writer.writerows(labels)

    eml_tmpl = (
        "From: test@example.com\nTo: user@example.org\nSubject: Test Email\nMIME-Version: 1.0\n"
        "Content-Type: text/plain; charset=utf-8\n\nЭто текст письма для тестирования.\n"
    )
    for row in labels:
        (bench_dir / row["file"]).write_text(eml_tmpl, encoding="utf-8")

    res = run_benchmark(bench_dir=bench_dir, out_dir=out_dir, strengths=[0.0, 1.0], limit=2)
    assert "baseline" in res
    assert "summary" in res
    assert (out_dir / "attack_summary.csv").exists()
    assert (out_dir / "degradation_curves.json").exists()

    # Проверка main()
    cmd_args = ["--benchmark", str(bench_dir), "--out", str(out_dir / "cli_out"), "--limit", "2"]
    exit_code = main(cmd_args)
    assert exit_code == 0
