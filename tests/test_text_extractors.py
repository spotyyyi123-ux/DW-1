from __future__ import annotations

from dw1.evidence import Evidence
from dw1.extractors.body_text import BodyTextExtractor
from dw1.extractors.cta import CtaExtractor
from dw1.extractors.subject import SubjectExtractor
from dw1.extractors.text_utils import (
    analyze_punctuation,
    compute_burstiness,
    compute_mtld,
    compute_ttr,
    detect_mixed_script,
    extract_emojis,
    find_typos_and_anomalies,
    split_sentences,
    tokenize_words,
)

# =====================================================================
# Тесты утилит (text_utils)
# =====================================================================


def test_split_sentences_and_tokenize():
    text = "Уважаемый коллега! Ваш доступ истекает через 24 часа. Срочно подтвердите данные."
    sents = split_sentences(text)
    assert len(sents) == 3
    tokens = tokenize_words(text)
    assert "коллега" in tokens
    assert "24" in tokens


def test_burstiness_metric():
    # Однородные по длине предложения (типично для LLM) -> низкий / отрицательный burstiness
    uniform_lens = [10, 10, 10, 10, 10]
    res_uniform = compute_burstiness(uniform_lens)
    assert res_uniform["std"] == 0.0
    assert res_uniform["burstiness"] == -1.0 or res_uniform["cv"] == 0.0

    # Разнородные по длине предложения (человеческий текст) -> высокий burstiness
    bursty_lens = [2, 35, 3, 40, 1]
    res_bursty = compute_burstiness(bursty_lens)
    assert res_bursty["cv"] > 0.8
    assert res_bursty["burstiness"] > 0.0

    # Граничные случаи
    empty_res = compute_burstiness([])
    assert empty_res["burstiness"] == 0.0
    one_res = compute_burstiness([5])
    assert one_res["std"] == 0.0


def test_lexical_diversity_ttr_mtld():
    tokens = ["access", "email", "security", "access", "password", "email", "access"]
    ttr = compute_ttr(tokens)
    assert 0.0 < ttr < 1.0

    mtld = compute_mtld(tokens)
    assert mtld > 0.0

    assert compute_ttr([]) == 0.0
    assert compute_mtld([]) == 0.0


def test_punctuation_analysis():
    text = "СРОЧНО!!! Оплатите счёт немедленно?! Сумма: 100 руб."
    punct = analyze_punctuation(text)
    assert punct["multiple_punct_count"] >= 1
    assert punct["caps_words_count"] >= 1
    assert "СРОЧНО" in punct["caps_words"]
    assert punct["expressive_ratio"] > 0.0


def test_mixed_script_and_typos():
    assert detect_mixed_script("sеcure")  # 'е' - кириллица внутри латиницы
    assert not detect_mixed_script("secure")
    assert not detect_mixed_script("доступ")

    # Повторы символов и опечатки
    tokens = ["срооочно", "нормальное", "слово", "pleeease", "sеcure"]
    count, anomalies = find_typos_and_anomalies(tokens)
    assert count >= 3
    assert any("срооочно" in a for a in anomalies)
    assert any("pleeease" in a for a in anomalies)


def test_emoji_extraction():
    text = "🚨 Срочно: подтвердите пароль ⚠️"
    emojis = extract_emojis(text)
    assert len(emojis) == 2
    assert "🚨" in emojis


# =====================================================================
# Тесты BodyTextExtractor
# =====================================================================


def test_body_text_extractor_empty():
    ext = BodyTextExtractor()
    res = ext.extract({"body_text": {"text": ""}})
    assert res["ai_provenance"]["label"] == "unknown"
    assert res["intent"]["label"] == "unknown"


def test_body_text_extractor_short():
    ext = BodyTextExtractor()
    res = ext.extract({"body_text": {"text": "Привет как дела"}})
    assert res["ai_provenance"]["label"] == "unknown"


def test_body_text_extractor_phishing_and_ai_signals():
    ext = BodyTextExtractor()
    text = (
        "Уважаемый коллега, ваш доступ к корпоративной почте истекает через 24 часа. "
        "Пожалуйста, подтвердите учётные данные и ваш пароль по ссылке ниже. "
        "В противном случае доступ к почтовому ящику будет заблокирован."
    )
    res = ext.extract({"body_text": {"text": text}})
    assert res["intent"]["label"] == "phishing"
    assert res["intent"]["confidence"] >= 0.70
    assert len(res["ai_provenance"]["evidence"]) >= 5

    # Валидация контрактом Evidence
    ev = Evidence(signals={"body_text": res})
    ev.validate()


def test_body_text_extractor_human_spam_with_typos_and_caps():
    ext = BodyTextExtractor()
    text = (
        "СРОЧНО!!! Коллега пооочта заблокирована!! "
        "Срооочно введи пароль иначе все сломается прямо сейчас!!!"
    )
    res = ext.extract({"body_text": {"text": text}})
    assert res["ai_provenance"]["label"] == "human"
    assert res["ai_provenance"]["confidence"] >= 0.70


# =====================================================================
# Тесты SubjectExtractor
# =====================================================================


def test_subject_extractor_empty():
    ext = SubjectExtractor()
    res = ext.extract({"subject": {"text": ""}})
    assert res["ai_provenance"]["label"] == "unknown"
    assert res["intent"]["label"] == "unknown"


def test_subject_extractor_urgency_and_name_ru():
    ext = SubjectExtractor()
    parts = {
        "subject": {"text": "Иван, срочно: подтвердите доступ сегодня до 18:00"},
        "headers": {"to": "ivan@example.com"},
    }
    res = ext.extract(parts)
    assert res["intent"]["label"] == "phishing"
    assert res["intent"]["confidence"] >= 0.75

    # Проверка извлечённых доказательств
    evidence_dict = {item["signal"]: item for item in res["intent"]["evidence"]}
    assert evidence_dict["personal_address"]["value"] is True
    assert evidence_dict["personal_address"]["name"] == "Иван"
    assert evidence_dict["urgency_cues"]["count"] >= 1


def test_subject_extractor_fake_re_and_emoji():
    ext = SubjectExtractor()
    parts = {
        "subject": {"text": "🚨 Re: СРОЧНО оплатите счёт ⚠️"},
        "headers": {},  # Нет in-reply-to -> fake thread
    }
    res = ext.extract(parts)
    assert res["ai_provenance"]["label"] == "human"
    evidence_dict = {item["signal"]: item for item in res["ai_provenance"]["evidence"]}
    assert evidence_dict["has_emoji"]["value"] is True
    assert evidence_dict["fake_thread_prefix"]["value"] is True


def test_subject_extractor_en_urgency():
    ext = SubjectExtractor()
    parts = {"subject": {"text": "URGENT: Action required within 24 hours"}}
    res = ext.extract(parts)
    assert res["intent"]["label"] in ("phishing", "spam")
    assert res["intent"]["confidence"] >= 0.70


# =====================================================================
# Тесты CtaExtractor
# =====================================================================


def test_cta_extractor_empty():
    ext = CtaExtractor()
    parts = {"body_text": {"text": ""}, "links": []}
    res = ext.extract(parts)
    assert res["ai_provenance"]["label"] == "unknown"
    assert res["intent"]["label"] == "unknown"


def test_cta_extractor_ru_extracts_deadlines_and_imperatives():
    ext = CtaExtractor()
    parts = {
        "body_text": {
            "text": (
                "Уважаемый пользователь, ваш аккаунт будет отключен. "
                "Пожалуйста, подтвердите доступ по ссылке ниже в течение 24 часов. "
                "Перейдите по ссылке и оплатите счёт."
            )
        },
        "links": [
            {"anchor_text": "Подтвердить доступ", "url": "https://secure-login.test/verify"}
        ],
        "cta": [],
    }
    res = ext.extract(parts)
    assert res["intent"]["label"] == "phishing"
    assert res["intent"]["confidence"] >= 0.80

    # Проверяем, что parts['cta'] наполнился
    assert len(parts["cta"]) >= 1
    assert any(c["imperative"] == "confirm" for c in parts["cta"])

    ev = Evidence(signals={"cta": res}, parts=parts)
    ev.validate()


def test_cta_extractor_en():
    ext = CtaExtractor()
    parts = {
        "body_text": {
            "text": "Please verify your credentials immediately or access will be revoked."
        },
        "links": [{"anchor_text": "Click here to verify", "url": "https://auth.example.com"}],
    }
    res = ext.extract(parts)
    assert res["intent"]["label"] == "phishing"
    evidence_dict = {item["signal"]: item for item in res["intent"]["evidence"]}
    assert evidence_dict["cta_count"]["value"] >= 1
    assert "confirm" in evidence_dict["high_risk_actions"]["value"]
