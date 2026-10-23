"""Готовит небольшую выборку: разделяет происхождение и назначение, проверяет утечки данных."""

from __future__ import annotations

import argparse
import hashlib
import json
import mailbox
import re
from pathlib import Path

import pandas as pd
from datasketch import MinHash, MinHashLSH

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(text):
    normalized = re.sub(r"\W+", " ", str(text).casefold()).strip()
    normalized = re.sub(r"\d+", "NUM", normalized)
    return hashlib.sha256(normalized.encode()).hexdigest()


def group_split(group):
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


def load_hc3(path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            item = json.loads(line)
            group = "hc3-" + fingerprint(item["question"])
            for key, label in [("human_answers", 0), ("chatgpt_answers", 1)]:
                for j, text in enumerate(item[key]):
                    rows.append(
                        dict(
                            message_id=f"hc3-{i}-{label}-{j}",
                            text=text,
                            label=label,
                            source="hc3-en",
                            language="en",
                            group_id=group,
                            split=group_split(group),
                            intent="unknown",
                        )
                    )
    return pd.DataFrame(rows)


def clean_exact(frame):
    frame = frame.copy()
    frame = frame[frame.text.map(lambda x: isinstance(x, str) and len(x.strip()) >= 30)]
    frame["fingerprint"] = frame.text.map(fingerprint)
    conflicts = frame.groupby("fingerprint").label.nunique()
    frame = frame[~frame.fingerprint.isin(conflicts[conflicts > 1].index)]
    return frame.drop_duplicates("fingerprint").copy()


def remove_near_leaks(frame):
    """Удаляет из проверочных частей близкие повторы текстов из предыдущих частей.

    Поиск LSH использует последовательности из пяти слов. Для найденных кандидатов
    точно проверяется сходство Жаккара >= 0.8. Приближённый поиск может пропустить повторы.
    """
    lsh = MinHashLSH(threshold=0.7, num_perm=64)
    indexed = {}
    kept = []
    dropped = 0
    for split in ["train", "validation", "test"]:
        pending = []
        for idx, row in frame[frame.split == split].iterrows():
            words = re.findall(r"\w+", row.text.casefold())
            words = [re.sub(r"\d+", "NUM", w) for w in words]
            grams = {" ".join(words[i : i + 5]) for i in range(len(words) - 4)}
            if not grams:
                kept.append(idx)
                continue
            mh = MinHash(num_perm=64)
            mh.update_batch([s.encode() for s in sorted(grams)])
            leaked = any(
                len(grams & indexed[k]) / len(grams | indexed[k]) >= 0.8 for k in lsh.query(mh)
            )
            if leaked:
                dropped += 1
                continue
            kept.append(idx)
            pending.append((str(idx), mh, grams))
        # Повторы внутри одной части допустимы; здесь исключаем повторы между разными частями.
        for key, mh, grams in pending:
            lsh.insert(key, mh)
            indexed[key] = grams
    return frame.loc[kept].copy(), dropped


def prepare(cap=2000):
    raw = ROOT / "data/raw"
    frames = []
    for split in ["train", "validation"]:
        part = pd.read_parquet(raw / "coat/binary" / f"{split}-00000-of-00001.parquet")
        if not set(part.label.unique()) <= {0, 1}:
            raise ValueError("CoAT labels must be known binary labels")
        part["message_id"] = "coat-" + part.id.astype(str)
        part["source"], part["language"], part["split"] = "coat", "ru", split
        part["group_id"] = part.message_id
        # В исходном test метки скрыты (-1); используем исходную validation как отложенную проверку.
        if split == "validation":
            part["split"] = "test"
        else:
            part["split"] = part.text.map(
                lambda t: "validation" if int(fingerprint(t)[:8], 16) % 10 == 0 else "train"
            )
        part["intent"] = "unknown"
        frames.append(part.drop(columns="id"))
    frames.append(load_hc3(raw / "hc3-en/all.jsonl"))
    data = pd.concat(frames, ignore_index=True)
    before = len(data)
    data = clean_exact(data)
    exact_removed = before - len(data)
    # Берём одинаковый максимум для каждого источника и класса, сохраняя разделение данных.
    samples = []
    for (_, split, _), group in data.groupby(["source", "split", "label"]):
        n = cap if split == "train" else max(100, cap // 4)
        samples.append(group.sample(min(len(group), n), random_state=42))
    data = pd.concat(samples).reset_index(drop=True)
    # Ответы HC3 на один вопрос остаются в одной части. Исходные пары текстов CoAT неизвестны.
    data, near_removed = remove_near_leaks(data)
    data["ai_provenance"] = data.label.map({0: "human", 1: "ai"})
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    data.to_parquet(out / "w1_texts.parquet", index=False)

    emails = []
    for filename, language in [
        ("GPT_Phishing_Email_dataset.csv", "en"),
        ("CN_Phishing_Email_dataset.csv", "zh"),
    ]:
        table = pd.read_csv(raw / "llmgen-phishing" / filename)
        if not {"content", "label"} <= set(table.columns):
            raise ValueError(f"Unexpected LLMGen schema: {table.columns.tolist()}")
        for i, row in table.iterrows():
            emails.append(
                dict(
                    message_id=f"llmgen-{language}-{i}",
                    text=str(row["content"]),
                    source="llmgen-phishing",
                    language=language,
                    ai_provenance="ai",
                    intent="phishing" if int(row["label"]) else "legit",
                    split="external",
                )
            )
    for year in ["2023", "2024", "2025"]:
        for i, msg in enumerate(mailbox.mbox(str(raw / "nazario" / f"phishing-{year}"))):
            chunks = []
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        try:
                            chunks.append(
                                payload.decode(
                                    part.get_content_charset() or "utf-8", errors="replace"
                                )
                            )
                        except LookupError:
                            chunks.append(payload.decode("utf-8", errors="replace"))
            emails.append(
                dict(
                    message_id=f"nazario-{year}-{i}",
                    text="\n".join(chunks),
                    subject=str(msg.get("Subject", "")),
                    source="nazario",
                    language="unknown",
                    ai_provenance="unknown",
                    intent="phishing",
                    split="external",
                )
            )
    pd.DataFrame(emails).to_parquet(out / "w1_emails.parquet", index=False)
    audit = dict(
        seed=42,
        cap_per_source_class_train=cap,
        empty_short_duplicate_conflict_removed=exact_removed,
        cross_split_near_duplicates_removed=near_removed,
        rows=len(data),
        email_rows=len(emails),
        counts=data.groupby(["source", "split", "ai_provenance"])
        .size()
        .rename("count")
        .reset_index()
        .to_dict("records"),
        limitations=[
            "CoAT is general Russian text, HC3 English QA: not an email benchmark.",
            "CoAT official test labels are -1: excluded; official validation is pilot test.",
            "CoAT original source/variant groups are not exposed in these files.",
            "Near-duplicate search is approximate, not proof of zero leakage.",
            "LLMGen is AI-only: report AI detection rate, not binary accuracy/F1.",
            "Nazario 2023-2025 provenance unknown; never label as human.",
            "seven-phishing blocked pending original-license review; CIFAKE pending access.",
        ],
    )
    (out / "preparation.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=2000)
    args = ap.parse_args()
    prepare(args.cap)
