# -*- coding: utf-8 -*-
from dw1.campaigns.cluster import cluster_emails, normalize_text, shingles


def test_normalize_text_strips_punctuation_and_case():
    assert normalize_text("Срочно!!! Подтвердите  доступ.") == "срочно подтвердите доступ"


def test_shingles_size():
    text = "one two three four five six"
    s = shingles(text, k=3)
    assert len(s) == 4
    assert "one two three" in s


def test_cluster_groups_near_duplicates():
    base = (
        "ваш доступ к корпоративной почте истекает через 24 часа пожалуйста подтвердите "
        "учётные данные перейдите по ссылке ниже и введите логин пароль после чего доступ "
        "будет продлён автоматически без вашего участия"
    )
    docs = [
        ("a", base),
        ("b", base + " спасибо за понимание"),
        ("c", "приглашаем на вебинар по машинному обучению во вторник в 18 00 регистрация по ссылке"),
        ("d", base + " с уважением служба безопасности"),
    ]
    clusters = cluster_emails(docs)
    assert len(clusters) == 1
    assert clusters[0]["size"] == 3  # a, b, d
    ids = {m["id"] for m in clusters[0]["members"]}
    assert ids == {"a", "b", "d"}


def test_cluster_skips_singletons():
    docs = [("a", "текст один"), ("b", "совсем другой текст письма")]
    assert cluster_emails(docs) == []
