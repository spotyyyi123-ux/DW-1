"""Устойчивость к маскировке: 7 атак на детекторы ИИ-текста (Sprint 2, задача 2).

Реализует 7 атак маскировки генеративного стиля (SPEC §9):
1. paraphrase           - LLM-перефраз / синтаксическая перестройка
2. back_translation     - обратный перевод (RU<->EN транзит)
3. synonym_replacement  - синонимическая замена
4. sentence_reordering  - перестановка предложений
5. typo_injection       - опечатки и «человеческий шум»
6. homoglyphs_zerowidth - невидимые zero-width символы и гомоглифы (RU/EN)
7. burstiness_split_merge - разбиение и склейка предложений (атака на burstiness)

Запуск оценки:
    python scripts/attacks.py --benchmark data/sprint2/bench --out data/runs/sprint2/attacks
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
from collections.abc import Callable
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dw1.models.fusion import _apply_isotonic  # noqa: E402
from dw1.pipeline import analyze  # noqa: E402

log = logging.getLogger("dw1.attacks")

ATTACK_NAMES = [
    "paraphrase",
    "back_translation",
    "synonym_replacement",
    "sentence_reordering",
    "typo_injection",
    "homoglyphs_zerowidth",
    "burstiness_split_merge",
]

# ---------------------------------------------------------------------------
# 1. Словари и правила для 7 атак
# ---------------------------------------------------------------------------

SYNONYMS_RU: dict[str, list[str]] = {
    "подтвердите": ["верифицируйте", "удостоверьте", "засвидетельствуйте", "актуализируйте"],
    "подтвердить": ["верифицировать", "удостоверить", "актуализировать"],
    "доступ": ["вход", "права доступа", "авторизацию", "учётную запись"],
    "срочно": ["безотлагательно", "немедленно", "оперативно", "в срочном порядке"],
    "внимание": ["к сведению", "обратите внимание", "предупреждение", "важная информация"],
    "письмо": ["сообщение", "уведомление", "извещение", "послание"],
    "пароль": ["секретный код", "ключ доступа", "парольную фразу"],
    "аккаунт": ["профиль", "учётную запись", "кабинет", "учётку"],
    "ссылке": ["указанному адресу", "веб-странице", "ресурсу", "линку"],
    "перейдите": ["пройдите", "следуйте", "нажмите", "откройте"],
    "оплатите": ["погасите", "перечислите средства", "совершите платёж", "произведите оплату"],
    "счёт": ["квитанцию", "платёжный документ", "задолженность", "инвойс"],
    "безопасность": ["защита", "информационная безопасность", "служба контроля"],
    "пожалуйста": ["просим", "будем признательны", "убедительная просьба"],
    "истекает": ["заканчивается", "подходит к концу", "будет аннулирован"],
    "заблокирован": ["приостановлен", "заморожен", "отключён", "ограничен"],
    "сегодня": ["в течение текущего дня", "до конца дня", "в ближайшие часы"],
    "данные": ["сведения", "информацию", "параметры входа", "реквизиты"],
}

SYNONYMS_EN: dict[str, list[str]] = {
    "confirm": ["verify", "validate", "authenticate", "substantiate"],
    "urgent": ["pressing", "immediate", "critical", "high-priority"],
    "access": ["entry", "authorization", "permission", "login credentials"],
    "immediately": ["promptly", "without delay", "right away", "at once"],
    "account": ["profile", "membership", "credentials", "user portal"],
    "verify": ["authenticate", "validate", "re-confirm", "check"],
    "expires": ["runs out", "is terminating", "lapses", "will be deactivated"],
    "suspended": ["temporarily frozen", "deactivated", "locked", "blocked"],
    "security": ["protection", "safety monitoring", "defense compliance"],
    "please": ["kindly", "we request you to", "it is advised to"],
    "invoice": ["bill", "payment statement", "charges", "settlement"],
    "password": ["passcode", "secret key", "security code"],
    "click": ["follow", "proceed to", "open", "navigate to"],
    "link": ["URL", "web address", "portal page", "secured link"],
}

HOMOGLYPHS_CYR_TO_LAT: dict[str, str] = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X",
}

KEYBOARD_ADJACENT_RU: dict[str, str] = {
    "а": "прф", "о": "лдж", "е": "ун", "и": "тьб", "т": "еи", "п": "ро", "с": "вы",
    "к": "уе", "л": "од", "н": "ег", "р": "по", "в": "са", "д": "ло", "у": "ке",
}
KEYBOARD_ADJACENT_EN: dict[str, str] = {
    "a": "sqw", "e": "rwd", "i": "uko", "o": "ilp", "u": "yhj", "t": "ryg",
    "s": "adw", "r": "etf", "n": "bhj", "c": "xvd", "p": "ol", "l": "kop",
}

PARAPHRASE_PATTERNS_RU: list[tuple[str, str]] = [
    (r"\bв связи с тем,? что\b", "так как"),
    (r"\bпожалуйста,? обратите внимание,? что\b", "просим принять к сведению, что"),
    (r"\bнапоминаем вам,? что\b", "сообщаем, что"),
    (r"\bдля того чтобы\b", "чтобы"),
    (r"\bв противном случае\b", "иначе"),
    (r"\bбудет произведена блокировка\b", "ваш аккаунт заблокируют"),
    (r"\bпросим вас подтвердить\b", "необходимо подтвердить"),
    (r"\bв течение ближайшего времени\b", "вскоре"),
    (r"\bсообщаем о необходимости\b", "нужно"),
]

PARAPHRASE_PATTERNS_EN: list[tuple[str, str]] = [
    (r"\bplease be advised that\b", "note that"),
    (r"\bin order to\b", "to"),
    (r"\bas soon as possible\b", "promptly"),
    (r"\bwe would like to inform you that\b", "please note that"),
    (r"\bfailure to do so will result in\b", "otherwise we will have to"),
    (r"\bit is mandatory that you\b", "you are required to"),
    (r"\byour immediate attention is required\b", "action is required now"),
]

BACK_TRANS_PATTERNS_RU: list[tuple[str, str]] = [
    (r"\bдобрый день\b", "здравствуйте"),
    (r"\bуважаемый пользователь\b", "клиент"),
    (r"\bслужба безопасности\b", "отдел защиты данных"),
    (r"\bпройдите авторизацию\b", "войдите в систему"),
    (r"\bсрок действия доступа\b", "время действия пароля"),
    (r"\bв целях безопасности\b", "для защиты информации"),
]

BACK_TRANS_PATTERNS_EN: list[tuple[str, str]] = [
    (r"\bdear customer\b", "hello user"),
    (r"\bsecurity department\b", "safety team"),
    (r"\baction required\b", "needed action"),
    (r"\bclick the link below\b", "use following link"),
    (r"\bwithin 24 hours\b", "during 24 hours"),
    (r"\baccount suspended\b", "profile deactivated"),
]


# ---------------------------------------------------------------------------
# 2. Функции реализации 7 атак
# ---------------------------------------------------------------------------

def attack_paraphrase(text: str, strength: float = 0.5, rng: random.Random | None = None) -> str:
    """1. LLM-перефраз / синтаксическая перестройка."""
    if not text:
        return text
    r = rng or random.Random(42)
    res = text
    patterns = PARAPHRASE_PATTERNS_RU + PARAPHRASE_PATTERNS_EN
    for pat, repl in patterns:
        if r.random() < strength:
            res = re.sub(pat, repl, res, flags=re.IGNORECASE)
    return res


def attack_back_translation(text: str, strength: float = 0.5, rng: random.Random | None = None) -> str:
    """2. Обратный перевод (калькированные обороты, смена синтаксиса)."""
    if not text:
        return text
    r = rng or random.Random(42)
    res = text
    patterns = BACK_TRANS_PATTERNS_RU + BACK_TRANS_PATTERNS_EN
    for pat, repl in patterns:
        if r.random() < strength:
            res = re.sub(pat, repl, res, flags=re.IGNORECASE)
    return res


def attack_synonym_replacement(text: str, strength: float = 0.5, rng: random.Random | None = None) -> str:
    """3. Синонимическая замена."""
    if not text:
        return text
    r = rng or random.Random(42)
    words = re.findall(r"\w+|[^\w\s]+|\s+", text, re.UNICODE)
    out: list[str] = []
    p = min(0.9, strength * 0.8)
    for w in words:
        low = w.lower()
        syns = SYNONYMS_RU.get(low) or SYNONYMS_EN.get(low)
        if syns and r.random() < p:
            chosen = r.choice(syns)
            if w.isupper():
                chosen = chosen.upper()
            elif w.istitle():
                chosen = chosen.title()
            out.append(chosen)
        else:
            out.append(w)
    return "".join(out)


def attack_sentence_reordering(text: str, strength: float = 0.5, rng: random.Random | None = None) -> str:
    """4. Перестановка предложений."""
    if not text:
        return text
    r = rng or random.Random(42)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sents) <= 2 or strength <= 0.05:
        return text
    # Сохраняем первое (приветствие) и последнее (подпись) предложения
    first, middle, last = sents[0], sents[1:-1], sents[-1]
    if middle and r.random() < strength:
        r.shuffle(middle)
    return " ".join([first] + middle + [last])


def attack_typo_injection(text: str, strength: float = 0.5, rng: random.Random | None = None) -> str:
    """5. Внесение опечаток и «человеческого шума»."""
    if not text:
        return text
    r = rng or random.Random(42)
    words = re.findall(r"\w+|[^\w\s]+|\s+", text, re.UNICODE)
    out: list[str] = []
    prob_word = min(0.6, strength * 0.4)
    for token in words:
        if not re.match(r"^[a-zA-Zа-яА-ЯёЁ]{4,}$", token) or r.random() >= prob_word:
            out.append(token)
            continue
        mode = r.choice(["adjacent", "swap", "duplicate", "drop"])
        chars = list(token)
        idx = r.randint(1, len(chars) - 2) if len(chars) > 3 else 0
        c_low = chars[idx].lower()
        if mode == "adjacent" and (c_low in KEYBOARD_ADJACENT_RU or c_low in KEYBOARD_ADJACENT_EN):
            adj = KEYBOARD_ADJACENT_RU.get(c_low) or KEYBOARD_ADJACENT_EN.get(c_low, "")
            if adj:
                rep = r.choice(adj)
                chars[idx] = rep.upper() if chars[idx].isupper() else rep
        elif mode == "swap" and idx < len(chars) - 1:
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        elif mode == "duplicate":
            chars.insert(idx, chars[idx])
        elif mode == "drop" and len(chars) > 4:
            del chars[idx]
        out.append("".join(chars))
    return "".join(out)


def attack_homoglyphs_zerowidth(text: str, strength: float = 0.5, rng: random.Random | None = None) -> str:
    """6. Zero-width символы и гомоглифы (кириллица <-> латиница)."""
    if not text:
        return text
    r = rng or random.Random(42)
    out: list[str] = []
    prob = min(0.7, strength * 0.5)
    for ch in text:
        if ch in HOMOGLYPHS_CYR_TO_LAT and r.random() < prob:
            out.append(HOMOGLYPHS_CYR_TO_LAT[ch])
        elif ch.isalpha() and r.random() < (prob * 0.15):
            out.append(ch + "\u200b")
        else:
            out.append(ch)
    return "".join(out)


def attack_burstiness_split_merge(text: str, strength: float = 0.5, rng: random.Random | None = None) -> str:
    """7. Разбиение и склейка предложений (направленная атака на burstiness)."""
    if not text:
        return text
    r = rng or random.Random(42)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sents or strength <= 0.05:
        return text
    out: list[str] = []
    i = 0
    while i < len(sents):
        cur = sents[i]
        words = cur.split()
        # Разбиение длинного предложения на два
        if len(words) > 10 and r.random() < strength:
            mid = len(words) // 2
            p1 = " ".join(words[:mid]).rstrip(".,!?") + "."
            p2 = " ".join(words[mid:]).capitalize()
            out.extend([p1, p2])
            i += 1
            continue
        # Склейка двух коротких предложений
        if i < len(sents) - 1 and len(words) < 8 and r.random() < strength:
            nxt = sents[i + 1]
            conj = " и " if re.search(r"[а-яА-ЯёЁ]", cur) else " and "
            merged = cur.rstrip(".!?") + conj + nxt[0].lower() + nxt[1:]
            out.append(merged)
            i += 2
            continue
        out.append(cur)
        i += 1
    return " ".join(out)


ATTACK_REGISTRY: dict[str, Callable[..., str]] = {
    "paraphrase": attack_paraphrase,
    "back_translation": attack_back_translation,
    "synonym_replacement": attack_synonym_replacement,
    "sentence_reordering": attack_sentence_reordering,
    "typo_injection": attack_typo_injection,
    "homoglyphs_zerowidth": attack_homoglyphs_zerowidth,
    "burstiness_split_merge": attack_burstiness_split_merge,
}


def apply_attack_to_eml_text(
    raw_eml: str,
    attack_name: str,
    strength: float = 0.5,
    seed: int = 42,
) -> str:
    """Применяет атаку к телу и теме письма .eml, сохраняя валидный формат MIME."""
    attack_fn = ATTACK_REGISTRY.get(attack_name)
    if not attack_fn:
        raise ValueError(f"Неизвестная атака: {attack_name}. Доступны: {list(ATTACK_REGISTRY)}")

    rng = random.Random(seed)
    msg = BytesParser(policy=policy.default).parsebytes(raw_eml.encode("utf-8", errors="replace"))

    # Атакуем тему
    orig_subj = msg.get("Subject", "")
    if orig_subj and attack_name in ("paraphrase", "synonym_replacement", "typo_injection", "homoglyphs_zerowidth"):
        new_subj = attack_fn(str(orig_subj), strength=strength * 0.7, rng=rng)
        msg.replace_header("Subject", new_subj)

    # Атакуем текстовые части тела
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            body = part.get_content()
            if isinstance(body, str):
                new_body = attack_fn(body, strength=strength, rng=rng)
                part.set_content(new_body, subtype="plain", charset="utf-8")

    return msg.as_string()


# ---------------------------------------------------------------------------
# 3. Бенчмарк и замер TPR@FPR=1%
# ---------------------------------------------------------------------------

def calculate_tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
    """Считает TPR при FPR <= target_fpr (рабочая точка SOC 1%)."""
    if len(y_true) == 0 or sum(y_true == 1) == 0 or sum(y_true == 0) == 0:
        return 0.0
    # Сортируем пороги по убыванию уверенности
    thresholds = np.unique(y_score)[::-1]
    best_tpr = 0.0
    neg_count = float(np.sum(y_true == 0))
    pos_count = float(np.sum(y_true == 1))

    for t in thresholds:
        pred = y_score >= t
        fp = np.sum((pred == 1) & (y_true == 0))
        fpr = fp / neg_count
        if fpr <= target_fpr:
            tp = np.sum((pred == 1) & (y_true == 1))
            tpr = tp / pos_count
            if tpr > best_tpr:
                best_tpr = float(tpr)
        else:
            break
    return float(best_tpr)


def run_benchmark(
    bench_dir: Path,
    out_dir: Path,
    strengths: list[float] | None = None,
    limit: int = 150,
) -> dict[str, Any]:
    """Прогоняет выборку писем через 7 атак и замеряет деградацию метрик."""
    if strengths is None:
        strengths = [0.0, 0.25, 0.5, 0.75, 1.0]

    out_dir.mkdir(parents=True, exist_ok=True)
    labels_file = bench_dir / "labels.csv"
    if not labels_file.exists():
        raise FileNotFoundError(f"Не найден labels.csv в {bench_dir}")

    # Загружаем метки
    items: list[dict[str, str]] = []
    with open(labels_file, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            items.append(row)

    # Балансируем выборку: AI-фишинг + человеческий спам + легитимная почта (ham)
    n_half = max(15, limit // 2)
    ai_items = [it for it in items if it.get("provenance") == "ai_generated"][:limit]
    human_spam = [
        it for it in items if it.get("provenance") == "human" and it.get("intent") == "spam"
    ][:n_half]
    human_ham = [
        it for it in items if it.get("provenance") == "human" and it.get("intent") == "legitimate"
    ][:n_half]
    eval_items = ai_items + human_spam + human_ham

    log.info(
        "Запуск оценки атак на %d письмах (AI: %d, Human Spam: %d, Human Ham: %d)",
        len(eval_items),
        len(ai_items),
        len(human_spam),
        len(human_ham),
    )

    # Считываем сырой текст писем
    raw_emails: dict[str, str] = {}
    for it in eval_items:
        p = bench_dir / it["file"]
        if p.exists():
            raw_emails[it["file"]] = p.read_text(encoding="utf-8", errors="replace")

    # Истинные метки
    y_prov_true = np.array([1 if it["provenance"] == "ai_generated" else 0 for it in eval_items])
    y_intent_true = np.array([1 if it["intent"] in ("phishing", "spam") else 0 for it in eval_items])

    # Загрузка калибровочных кривых для вероятностей
    cal_path = ROOT / "data" / "runs" / "sprint2" / "calibration_metrics.json"
    cal_data = json.loads(cal_path.read_text(encoding="utf-8")) if cal_path.exists() else {}
    cal_prov = (cal_data.get("axes") or {}).get("ai_provenance", {})
    cal_intent = (cal_data.get("axes") or {}).get("intent", {})

    tmp_eml = out_dir / "_temp_eval.eml"

    def _evaluate_batch(attack_name: str | None, strength: float) -> dict[str, float]:
        prov_scores: list[float] = []
        intent_scores: list[float] = []

        for idx, it in enumerate(eval_items):
            raw = raw_emails.get(it["file"], "")
            # Атакуем только AI-письма при тестировании уклонения от детекции провенанса
            is_ai = it["provenance"] == "ai_generated"
            if attack_name and strength > 0.0 and is_ai:
                attacked_eml = apply_attack_to_eml_text(raw, attack_name, strength=strength, seed=idx)
            else:
                attacked_eml = raw

            tmp_eml.write_text(attacked_eml, encoding="utf-8")
            ev = analyze(tmp_eml)

            # Извлекаем калиброванные или сырые вероятности
            comp_prov = float((ev.fusion.get("ai_provenance") or {}).get("components", {}).get("ai", 0.0))
            p_prov = float(np.clip(_apply_isotonic(comp_prov, cal_prov), 0.0, 1.0)) if cal_prov else comp_prov

            comp_intent = float((ev.fusion.get("intent") or {}).get("components", {}).get("malicious", 0.0))
            p_intent = float(np.clip(_apply_isotonic(comp_intent, cal_intent), 0.0, 1.0)) if cal_intent else comp_intent

            prov_scores.append(p_prov)
            intent_scores.append(p_intent)

        p_prov_arr = np.array(prov_scores)
        p_intent_arr = np.array(intent_scores)

        tpr_prov_1pct = calculate_tpr_at_fpr(y_prov_true, p_prov_arr, target_fpr=0.01)
        tpr_intent_1pct = calculate_tpr_at_fpr(y_intent_true, p_intent_arr, target_fpr=0.01)

        # Вычисляем среднюю уверенность детектора на положительных AI примерах
        mean_ai_score = float(np.mean(p_prov_arr[y_prov_true == 1])) if np.sum(y_prov_true == 1) > 0 else 0.0
        mean_intent_score = float(np.mean(p_intent_arr[y_intent_true == 1])) if np.sum(y_intent_true == 1) > 0 else 0.0

        return {
            "tpr_prov_1pct": round(tpr_prov_1pct, 4),
            "tpr_intent_1pct": round(tpr_intent_1pct, 4),
            "mean_ai_confidence": round(mean_ai_score, 4),
            "mean_intent_confidence": round(mean_intent_score, 4),
        }

    # 1. Baseline без атак
    baseline_res = _evaluate_batch(None, 0.0)
    log.info("Baseline: TPR(prov)@FPR=1%% = %.3f | TPR(intent)@FPR=1%% = %.3f", baseline_res["tpr_prov_1pct"], baseline_res["tpr_intent_1pct"])

    degradation_curves: dict[str, list[dict[str, Any]]] = {}
    attack_summary_rows: list[dict[str, Any]] = []

    for name in ATTACK_NAMES:
        curve = []
        for s in strengths:
            res = _evaluate_batch(name, s)
            curve.append({"strength": s, **res})
            log.info("Attack %s (str=%.2f): TPR(prov)=%.3f, TPR(intent)=%.3f", name, s, res["tpr_prov_1pct"], res["tpr_intent_1pct"])
        degradation_curves[name] = curve

        # Результат при средней и максимальной силе
        mid_res = curve[len(curve) // 2]
        max_res = curve[-1]
        attack_summary_rows.append({
            "attack": name,
            "baseline_tpr_prov": baseline_res["tpr_prov_1pct"],
            "attacked_tpr_prov_mid": mid_res["tpr_prov_1pct"],
            "attacked_tpr_prov_max": max_res["tpr_prov_1pct"],
            "degradation_prov_pct": round((1.0 - max_res["tpr_prov_1pct"] / max(1e-4, baseline_res["tpr_prov_1pct"])) * 100, 1),
            "baseline_ai_conf": baseline_res["mean_ai_confidence"],
            "attacked_ai_conf_max": max_res["mean_ai_confidence"],
            "baseline_tpr_intent": baseline_res["tpr_intent_1pct"],
            "attacked_tpr_intent_max": max_res["tpr_intent_1pct"],
            "degradation_intent_pct": round((1.0 - max_res["tpr_intent_1pct"] / max(1e-4, baseline_res["tpr_intent_1pct"])) * 100, 1),
            "baseline_intent_conf": baseline_res["mean_intent_confidence"],
            "attacked_intent_conf_max": max_res["mean_intent_confidence"],
        })

    if tmp_eml.exists():
        tmp_eml.unlink()

    # Сохранение артефактов
    summary_path = out_dir / "attack_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(attack_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(attack_summary_rows)

    curves_path = out_dir / "degradation_curves.json"
    curves_path.write_text(json.dumps(degradation_curves, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics_path = out_dir / "attack_metrics.json"
    report_data = {
        "baseline": baseline_res,
        "summary": attack_summary_rows,
        "strengths_tested": strengths,
    }
    metrics_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Артефакты атак успешно сохранены в: {out_dir}")
    return report_data


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Оценка устойчивости детектора к 7 атакам маскировки")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data" / "sprint2" / "bench", help="Путь к бенчмарку писем")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "runs" / "sprint2" / "attacks", help="Каталог вывода артефактов")
    parser.add_argument("--limit", type=int, default=100, help="Количество писем каждого класса для оценки")
    args = parser.parse_args(argv)

    run_benchmark(bench_dir=args.benchmark, out_dir=args.out, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
