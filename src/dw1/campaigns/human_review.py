"""Prepare a private Sprint 4 packet for 2–3 independent human reviewers."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import zipfile
from pathlib import Path

from dw1.campaigns.human_review_page import render_human_review
from dw1.eval.campaign_review import verify_package

PROTOCOL = "dw1-human-independent-v1"
REQUIRED = ("manifest.json", "campaigns.json", "documents.jsonl")


def read_package(path: Path) -> dict:
    if path.is_dir():
        content = {name: (path / name).read_bytes() for name in REQUIRED}
        source_sha256 = None
    else:
        source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        with zipfile.ZipFile(path) as archive:
            content = {}
            for name in REQUIRED:
                choices = [n for n in archive.namelist() if n == name or n == "review/" + name]
                if len(choices) != 1:
                    raise ValueError(f"Missing or ambiguous archive member: {name}")
                info = archive.getinfo(choices[0])
                if info.file_size > 30_000_000:
                    raise ValueError("Oversized review package member")
                content[name] = archive.read(info)
    manifest = json.loads(content["manifest.json"].decode("utf-8-sig"))
    campaigns = json.loads(content["campaigns.json"].decode("utf-8-sig"))
    docs = [
        json.loads(line)
        for line in content["documents.jsonl"].decode("utf-8-sig").split("\n")
        if line.strip()
    ]
    verify_package(manifest, campaigns, docs)
    if manifest.get("n_documents") != len(docs) or len(docs) != 600:
        raise ValueError("Sprint 4 requires the frozen 600-message population")
    if len({d["id"] for d in docs}) != len(docs):
        raise ValueError("Duplicate document IDs")
    clusters = campaigns["clusters"]
    members = [m for c in clusters for m in c["members"]]
    if len(clusters) != 50 or len(set(members)) != len(members):
        raise ValueError("Expected 50 disjoint clusters")
    if set(members) != {d["id"] for d in docs} or any(
        c["size"] != len(c["members"]) for c in clusters
    ):
        raise ValueError("Cluster membership does not match source texts")
    return {
        "manifest": manifest,
        "campaigns": campaigns,
        "documents": docs,
        "source_sha256": source_sha256,
    }


def payload_for(package: dict, slot: str) -> dict:
    manifest = package["manifest"]
    by_id = {d["id"]: d["text"] for d in package["documents"]}
    clusters = package["campaigns"]["clusters"]
    # Rebuild blank rows from source texts; never reuse proxy labels or another human's work.
    rows = [
        {
            "id": member,
            "text": by_id[member],
            "predicted_cluster": c["cluster_id"],
            "text_sha256": manifest["document_hashes"][member],
            "gold_group": "",
            "review": "",
        }
        for c in clusters
        for member in c["members"]
    ]
    return {
        "protocol": PROTOCOL,
        "package_id": manifest["package_id"],
        "dataset": manifest["provenance"]["dataset"],
        "reviewer_slot": slot,
        "n_documents": len(rows),
        "rows": rows,
        "clusters": [{"cluster_id": c["cluster_id"], "size": c["size"]} for c in clusters],
    }


def instructions(n_reviewers: int) -> str:
    return f"""DW-1 — Sprint 4, задача 3: независимая ручная разметка

КАК НАЧАТЬ
1. Полностью распакуйте архив. Откройте START.html обычным браузером.
2. Выберите страницу A, B или C. Каждая страница предназначена отдельному человеку.
3. Введите собственное имя/ID. Оно должно отличаться у разных людей.

ЧТО ПРОВЕРЯТЬ
Каждый из {n_reviewers} проверяющих читает ВСЕ 600 писем из 50 кластеров.
Нельзя разделить 600 писем между людьми: тогда не получится сравнить независимые мнения.
Не открывайте чужие ответы и автоматическую прокси-разметку до сдачи собственного файла.

Одна истинная группа — один общий текстовый шаблон. Имена, суммы, бренды и URL могут
меняться. Только похожая тема, подпись или общий сюжет ещё не делают письма near-dup.
Если один шаблон попался в другом кластере, используйте прежнее имя группы.
Для другого шаблона создайте новую группу; для одиночки используйте уникальную группу.
Сомнительные случаи помечайте «нужно обсудить», затем вернитесь к ним перед завершением.
Названия групп у разных людей могут отличаться — при сравнении важен состав групп.

КАК СОХРАНЯТЬ
«Сохранить черновик» позволяет прерваться. Файл JSONL появится в Загрузках браузера.
Для продолжения загрузите этот файл в ту же страницу A/B/C через «Продолжить свой черновик».
Страница не сохраняет ответы на сервер; сохраняйте черновик перед закрытием.
«Сохранить готовую разметку» доступно после проверки всех писем и подтверждения независимости.
Передайте координатору human-A-<имя>.jsonl (или B/C).
Переименовывать группы под чужие ответы не нужно.
Экспорт содержит ID, хеши и метки; исходные тексты писем в JSONL не включены.

КООРДИНАТОРУ
Нужны 2 или 3 полных независимых файла от разных людей.
Сначала сравнить состав групп, затем обсудить разногласия и зафиксировать решения.
Подтверждение в интерфейсе документирует слова проверяющего; оно не доказывает личность
или фактическую независимость работы. Это ответственность участников и координатора.

ПУБЛИКАЦИЯ
Этот архив и HTML-страницы содержат сырые письма: не загружать их в публичный GitHub.
Для публикации предназначены обезличенные по тексту JSONL, хеши, эмбеддинги и итоговые метрики.
Итоговый человеческий BCubed F1 будет получен только после разметки и согласования.
"""


def write_human_packet(package: dict, out: Path, n_reviewers: int = 3) -> dict:
    if n_reviewers not in (2, 3):
        raise ValueError("Need 2 or 3 reviewers")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output must be empty; do not overwrite reviewer work")
    out.mkdir(parents=True, exist_ok=True)
    files = []
    links = []
    for slot in "ABC"[:n_reviewers]:
        name = f"reviewer-{slot}.html"
        (out / name).write_text(render_human_review(payload_for(package, slot)), encoding="utf-8")
        files.append(name)
        links.append(f'<li><a href="{name}">Страница проверяющего {slot}</a> — все 600 писем</li>')
    manifest = package["manifest"]
    (out / "START.html").write_text(
        '<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>DW-1 — начать ручную разметку</title>"
        "<style>body{font:19px/1.7 system-ui;max-width:850px;margin:50px auto;padding:20px}"
        "a{color:#285fbe}li{padding:12px}strong{color:#194b8e}</style>"
        "<h1>Ручная разметка писем</h1><p>600 писем · 50 кластеров · Sprint 4</p>"
        f"<p>{html.escape(manifest['provenance']['dataset'])}</p>"
        "<p><strong>Каждую страницу заполняет отдельный человек самостоятельно.</strong>"
        " Все страницы содержат один набор писем с пустыми ответами.</p><ul>"
        + "".join(links)
        + '</ul><p><a href="INSTRUCTIONS.txt">Короткая инструкция</a></p>'
        "<p>Сохраняйте черновики. Готовый JSONL передайте координатору."
        " Архив и эти страницы содержат письма и передаются только участникам разметки.</p>"
        "</html>",
        encoding="utf-8",
    )
    (out / "INSTRUCTIONS.txt").write_text(instructions(n_reviewers), encoding="utf-8-sig")
    protocol = {
        "protocol": PROTOCOL,
        "package_id": manifest["package_id"],
        "n_documents": 600,
        "n_source_clusters": 50,
        "n_reviewers": n_reviewers,
        "review_coverage": "all messages independently per reviewer",
        "initial_answers": "blank; never seeded from proxy or predicted clusters",
        "export_contains_raw_text": False,
        "identity_verification": "self-attestation; not authentication",
        "source_archive_sha256": package["source_sha256"],
        "source_partition_policy": {
            "policy": "greedy disjointing by descending source-cluster size",
            "evidence": "reported by data owner; frozen partition checked for overlaps",
            "original_overlap_recomputed": False,
        },
        "pages": {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in files},
    }
    (out / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    status = {
        "task": "Sprint 4 task 3",
        "status": "awaiting_independent_human_annotations",
        "package_id": manifest["package_id"],
        "n_documents": 600,
        "n_source_clusters": 50,
        "human_bcubed_f1": None,
        "received_human_annotations": 0,
    }
    (out / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with zipfile.ZipFile(out / "human-review.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("START.html", "INSTRUCTIONS.txt", "protocol.json", "status.json", *files):
            archive.write(out / name, name)
    return protocol


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--package", type=Path, required=True, help="Private review directory or review.zip"
    )
    ap.add_argument("--out", type=Path, default=Path("data/runs/sprint4/human-review"))
    ap.add_argument("--reviewers", type=int, choices=(2, 3), default=3)
    args = ap.parse_args()
    result = write_human_packet(read_package(args.package), args.out, args.reviewers)
    print(
        json.dumps(
            {
                "package_id": result["package_id"],
                "documents": 600,
                "clusters": 50,
                "reviewers": args.reviewers,
                "private_archive": str(args.out / "human-review.zip"),
                "human_bcubed_f1": None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
