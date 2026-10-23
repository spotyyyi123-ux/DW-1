from __future__ import annotations

import math
import re
from typing import Any

# Регулярные выражения для предложений и слов
SENTENCE_TERMINATORS_RE = re.compile(r"([.!?]+(?:\s+|$))")
ABBREVIATIONS = {
    "г.", "руб.", "коп.", "ул.", "д.", "кв.", "т.д.", "т.п.", "и.т.д.", "и.т.п.",
    "e.g.", "i.e.", "mr.", "mrs.", "ms.", "dr.", "vs.",
}
WORD_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9_-]+", re.UNICODE)
ALPHA_WORD_RE = re.compile(r"^[a-zA-Zа-яА-ЯёЁ]+$", re.UNICODE)
CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")
LATIN_RE = re.compile(r"[a-zA-Z]")
EMOJI_RE = re.compile(
    r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\u2300-\u23ff\u2b50-\u2b55]",
    re.UNICODE,
)

# Словарь высокочастотных лексем RU/EN для эвристики опечаток
COMMON_WORDS_RU = {
    # Местоимения, союзы, предлоги, частицы
    "и", "в", "не", "на", "я", "быть", "он", "с", "что", "а", "по", "это", "она", "этот", "к",
    "но", "они", "мы", "как", "из", "у", "который", "то", "за", "свой", "еще", "ещё", "для",
    "от", "о", "же", "вы", "все", "всё", "его", "только", "себя", "один", "бы", "так", "также",
    "тоже", "до", "время", "если", "сам", "когда", "другой", "вот", "говорить", "наш", "мой", "ваш",
    "их", "кто", "чем", "где", "куда", "откуда", "почему", "зачем", "ли", "или", "после", "перед",
    "через", "около", "между", "под", "над", "при", "без", "даже", "нет", "да", "уже", "очень",
    # Деловая и почтовая лексика
    "уважаемый", "уважаемая", "уважаемые", "коллега", "коллеги", "день", "добрый", "утро",
    "вечер", "здравствуйте", "привет", "приветствую", "спасибо", "пожалуйста", "письмо",
    "сообщение", "почта", "доступ", "аккаунт", "учетная", "учётная", "запись", "данные",
    "пароль", "логин", "ссылка", "ссылке", "ссылку", "подтвердите", "подтвердить",
    "подтверждение", "проверить", "проверка", "перейдите", "перейти", "нажмите", "кликните",
    "войти", "вход", "авторизация", "авторизуйтесь", "обновите", "обновить", "обновление",
    "система", "сервис", "служба", "безопасность", "безопасности", "поддержка", "поддержки",
    "счет", "счёт", "оплата", "оплатить", "оплатите", "сумма", "рублей", "руб", "договор",
    "документ", "документы", "вложение", "файл", "информация", "уведомление", "внимание",
    "срочно", "срочный", "часа", "часов", "дней", "минут", "сегодня", "завтра", "срок",
    "истекает", "истечет", "истечёт", "заблокирован", "блокировка", "действие", "действия",
    "требуется", "просим", "просьба", "необходимо", "важно", "клиент", "сотрудник", "отдел",
    "компания", "компании", "руководитель", "директор", "менеджер", "работа", "вопрос", "ответ",
    "детали", "инструкция", "уважением", "службы", "техническая", "корпоративный",
    "корпоративной", "почте", "почты", "продолжить", "восстановить", "восстановление",
    "заявка", "статус", "портал", "кабинет", "личный", "получить", "получите", "отправить",
    "отправьте", "сохранить", "настройки", "политика", "правила",
}

COMMON_WORDS_EN = {
    # Pronouns, conjunctions, prepositions, articles, auxiliary verbs
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on",
    "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we",
    "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their",
    "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make",
    "can", "like", "time", "no", "just", "him", "know", "take", "people", "into", "year", "your",
    "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only",
    "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work",
    "first", "well", "way", "even", "new", "want", "because", "any", "these", "give", "day",
    "most", "us", "is", "are", "was", "were", "has", "had", "been", "am", "please", "thank",
    "thanks", "regards", "sincerely",
    # Business, IT & email vocabulary
    "dear", "colleague", "hello", "hi", "email", "mail", "message", "account", "access",
    "password", "login", "security", "alert", "notice", "notification", "urgent", "urgently",
    "action", "required", "verify", "verification", "confirm", "confirmation", "update", "updated",
    "click", "link", "here", "support", "service", "team", "customer", "invoice", "payment", "pay",
    "due", "amount", "bank", "information", "details", "expires", "expired", "expire", "hours",
    "days", "minutes", "immediately", "immediate", "suspended", "suspension", "terminated",
    "terminate", "failure", "system", "admin", "administrator", "server", "portal", "document",
    "file", "attached", "attachment", "review", "complete", "follow", "instructions", "below",
    "request", "help", "desk", "privacy", "policy", "terms", "corporate", "company", "employee",
    "staff", "management", "department", "office",
}


def split_sentences(text: str) -> list[str]:
    """Разбивает текст на предложения с учётом сокращений."""
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    # Временная маскировка точек в известных сокращениях
    masked = cleaned_text
    for abbr in ABBREVIATIONS:
        pattern = re.compile(re.escape(abbr), re.IGNORECASE)
        masked = pattern.sub(abbr.replace(".", "§§"), masked)

    # Делим по знакам конца предложения
    tokens = re.split(r"[.!?]+(?:\s+|$)", masked)
    sentences: list[str] = []
    for s in tokens:
        # Восстанавливаем точки
        restored = s.replace("§§", ".").strip()
        if restored:
            sentences.append(restored)

    return sentences


def tokenize_words(text: str) -> list[str]:
    """Извлекает токены слов (без знаков препинания)."""
    return WORD_RE.findall(text)


def compute_burstiness(sentence_lengths: list[int]) -> dict[str, float]:
    """Считает burstiness (разброс длин предложений).

    Используется нормализованная метрика Goh & Barabasi:
        B = (sigma - mu) / (sigma + mu) in [-1, 1],
    а также коэффициент вариации CV = sigma / mu.

    LLM-генерации свойственны однородные предложения (B -> -1, низкий CV),
    человеческому тексту — чередование коротких и длинных (B > 0, высокий CV).
    """
    if not sentence_lengths:
        return {"burstiness": 0.0, "cv": 0.0, "mean": 0.0, "std": 0.0}

    n = len(sentence_lengths)
    mean_val = sum(sentence_lengths) / n
    if n < 2 or mean_val == 0.0:
        return {"burstiness": 0.0, "cv": 0.0, "mean": float(mean_val), "std": 0.0}

    var_val = sum((x - mean_val) ** 2 for x in sentence_lengths) / (n - 1)
    std_val = math.sqrt(var_val)
    cv = std_val / mean_val
    denom = std_val + mean_val
    burstiness = (std_val - mean_val) / denom if denom > 0 else 0.0

    return {
        "burstiness": float(burstiness),
        "cv": float(cv),
        "mean": float(mean_val),
        "std": float(std_val),
    }


def compute_ttr(tokens: list[str]) -> float:
    """Type-Token Ratio: отношение уникальных слов к общему числу токенов."""
    if not tokens:
        return 0.0
    lowered = [t.lower() for t in tokens if ALPHA_WORD_RE.match(t)]
    if not lowered:
        return 0.0
    return len(set(lowered)) / len(lowered)


def compute_mtld(tokens: list[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (MTLD) по McCarthy & Jarvis (2010).

    Устойчива к длине текста. Проводится прямой и обратный проход факторов TTR.
    """
    lowered = [t.lower() for t in tokens if ALPHA_WORD_RE.match(t)]
    if not lowered:
        return 0.0
    if len(lowered) < 10:
        # Для слишком коротких текстов возвращаем длину, масштабированную TTR
        return float(len(set(lowered)))

    def _factors(seq: list[str]) -> float:
        factors = 0.0
        seen: set[str] = set()
        count = 0
        for token in seq:
            count += 1
            seen.add(token)
            ttr = len(seen) / count
            if ttr <= threshold:
                factors += 1.0
                seen.clear()
                count = 0
        if count > 0:
            final_ttr = len(seen) / count
            if final_ttr < 1.0:
                factors += (1.0 - final_ttr) / (1.0 - threshold)
        return factors

    f_fwd = _factors(lowered)
    f_bwd = _factors(list(reversed(lowered)))
    total_factors = f_fwd + f_bwd
    if total_factors <= 0.0:
        return float(len(lowered))
    return (2.0 * len(lowered)) / total_factors


def analyze_punctuation(text: str) -> dict[str, Any]:
    """Профиль пунктуации: распределение знаков, восклицания, КАПС."""
    counts = {
        "comma": text.count(","),
        "period": text.count("."),
        "exclamation": text.count("!"),
        "question": text.count("?"),
        "colon": text.count(":"),
        "semicolon": text.count(";"),
        "dash": text.count("-") + text.count("—") + text.count("–"),
        "quotes": text.count('"') + text.count("'") + text.count("«") + text.count("»"),
    }
    total_punct = sum(counts.values())

    # Поиск серийных знаков: !!!, ???, !?
    multiple_punct = len(re.findall(r"[!?]{2,}", text))

    # ALL CAPS слова (длиной >= 2 букв)
    words = TOKEN_WORDS = WORD_RE.findall(text)
    caps_words = [w for w in TOKEN_WORDS if len(w) >= 2 and w.isupper() and ALPHA_WORD_RE.match(w)]

    words_count = max(1, len(words))
    expressive_marks = counts["exclamation"] + counts["question"]
    calm_marks = counts["comma"] + counts["period"]

    expressive_ratio = expressive_marks / max(1, expressive_marks + calm_marks)
    punct_density = total_punct / words_count

    return {
        "counts": counts,
        "total_punct": total_punct,
        "multiple_punct_count": multiple_punct,
        "caps_words_count": len(caps_words),
        "caps_words": caps_words[:5],
        "expressive_ratio": float(expressive_ratio),
        "punct_density": float(punct_density),
    }


def detect_mixed_script(word: str) -> bool:
    """Определяет слово со смешанным алфавитом (кириллица + латиница) — приём обфускации."""
    if len(word) < 3:
        return False
    has_cyrillic = bool(CYRILLIC_RE.search(word))
    has_latin = bool(LATIN_RE.search(word))
    return has_cyrillic and has_latin


def find_typos_and_anomalies(tokens: list[str]) -> tuple[int, list[str]]:
    """Находит потенциальные опечатки, спам-шум и смешение алфавитов."""
    anomalies: list[str] = []
    for raw_token in tokens:
        token = raw_token.strip("_-")
        if not token or not ALPHA_WORD_RE.match(token):
            continue

        # 1. Смешанный алфавит (обфускация фишеров)
        if detect_mixed_script(token):
            anomalies.append(token)
            continue

        lowered = token.lower()

        # Если слово есть в частотных словарях RU или EN — не опечатка
        if lowered in COMMON_WORDS_RU or lowered in COMMON_WORDS_EN:
            continue

        # 2. Повтор одного символа 3+ раз подряд (напр. "срооочно", "pleeease")
        if re.search(r"(.)\1{2,}", lowered):
            anomalies.append(token)
            continue

        # 3. Странный регистр внутри слова (напр. "пОчТа" или "paYment")
        if (
            len(token) > 2
            and not token.isupper()
            and not token.istitle()
            and re.search(r"[a-zа-яё][A-ZА-ЯЁ]", token)
        ):
            anomalies.append(token)
            continue

        # 4. Неестественные скопления согласных (>= 5 согласных подряд)
        if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", lowered) or re.search(
            r"[бвгджзйклмнпрстфхцчшщ]{5,}", lowered
        ):
            anomalies.append(token)
            continue

    return len(anomalies), anomalies


def extract_emojis(text: str) -> list[str]:
    """Извлекает список эмодзи из текста."""
    return EMOJI_RE.findall(text)
