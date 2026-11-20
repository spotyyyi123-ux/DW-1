"""Protocol tests use authored fixtures only, never real human labels or model-quality claims."""

from __future__ import annotations

import copy
import json
import sys

import numpy as np
import pytest

from dw1.campaigns.documents import document_hash
from dw1.campaigns.review import prepare, write_package
from dw1.eval import campaign_review
from dw1.eval.human_gold import (
    PROTOCOL,
    file_sha256,
    partition_changes,
    public_annotations,
    validate_adjudication,
    validate_human_rows,
)


def declared_rows(author="fixture-reviewer-a", independent=True):
    return [
        {
            "id": f"fixture-{i}",
            "predicted_cluster": "fixture-cluster",
            "package_id": "fixture-package",
            "text_sha256": document_hash("fixture email"),
            "text": "fixture email",
            "notes": "PRIVATE FIXTURE NOTES",
            "gold_group": f"g{i // 2}",
            "review": "near-dup",
            "annotator": author,
            "annotation_source": "human",
            "protocol": PROTOCOL,
            "human_confirmed": True,
            "stage": "independent" if independent else "adjudicated",
            "independent_confirmed": independent,
            "test_fixture": True,
        }
        for i in range(4)
    ]


def write_rows(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )


def adjudication_for(package_id, reviewers, final, author="fixture-adjudicator"):
    return {
        "package_id": package_id,
        "reviewer_annotations_sha256": [file_sha256(p) for p in reviewers],
        "final_annotations_sha256": file_sha256(final),
        "adjudicated_by": author,
        "human_confirmed": True,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("annotation_source", "proxy"),
        ("annotation_source", "unknown"),
        ("annotation_source", None),
        ("protocol", "legacy"),
        ("human_confirmed", False),
        ("human_confirmed", "true"),
        ("independent_confirmed", False),
        ("stage", "adjudicated"),
        ("gold_group", ""),
        ("gold_group", " g"),
        ("annotator", ""),
        ("review", "unchecked"),
    ],
)
def test_independent_human_protocol_rejects_missing_or_false_declarations(field, value):
    rows = declared_rows()
    rows[0][field] = value
    with pytest.raises(ValueError):
        validate_human_rows(rows)


def test_human_file_has_single_author_package_unique_ids_and_correct_stage():
    rows = declared_rows()
    assert validate_human_rows(rows) == "fixture-reviewer-a"
    assert validate_human_rows(declared_rows(independent=False), independent=False)
    with pytest.raises(ValueError, match="stage"):
        validate_human_rows(rows, independent=False)
    for field, value in (
        ("annotator", "fixture-reviewer-b"),
        ("package_id", "other-package"),
        ("id", rows[1]["id"]),
    ):
        broken = copy.deepcopy(rows)
        broken[0][field] = value
        with pytest.raises(ValueError):
            validate_human_rows(broken)
    with pytest.raises(ValueError):
        validate_human_rows([])


def test_partition_comparison_is_independent_of_label_names():
    a, b = declared_rows(), declared_rows("fixture-reviewer-b")
    for row in b:
        row["gold_group"] = "other-prefix-" + row["gold_group"]
    assert partition_changes(a, b) == []
    assert campaign_review.compare_reviewers([a, b])["n_disagreements"] == 0
    b[0]["gold_group"] = "singleton"
    assert partition_changes(a, b) == ["fixture-0", "fixture-1"]
    with pytest.raises(ValueError, match="populations"):
        partition_changes(a, b[:-1])


def test_adjudication_binds_exact_files_not_just_an_adjudicator_name(tmp_path):
    paths = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    final = tmp_path / "final.jsonl"
    write_rows(paths[0], declared_rows())
    write_rows(paths[1], declared_rows("fixture-reviewer-b"))
    write_rows(final, declared_rows("fixture-adjudicator", independent=False))
    record = adjudication_for("fixture-package", paths, final)
    checked = validate_adjudication(record, "fixture-package", list(reversed(paths)), final)
    assert checked["adjudicated_by"] == "fixture-adjudicator"
    assert checked["protocol"] == PROTOCOL
    for key, value in (
        ("package_id", "other"),
        ("human_confirmed", False),
        ("reviewer_annotations_sha256", []),
        ("final_annotations_sha256", "0" * 64),
        ("adjudicated_by", ""),
    ):
        with pytest.raises(ValueError):
            validate_adjudication({**record, key: value}, "fixture-package", paths, final)
    with pytest.raises(ValueError, match="CLI adjudicator"):
        validate_adjudication(record, "fixture-package", paths, final, "another-person")
    paths[0].write_text(paths[0].read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reviewer file hashes"):
        validate_adjudication(record, "fixture-package", paths, final)


def test_public_annotations_omit_text_notes_and_unexpected_fields():
    rows = declared_rows()
    rows[0]["unexpected_private_data"] = "secret"
    exported = public_annotations(rows)
    assert len(exported) == 4
    assert all(
        "text" not in row
        and "notes" not in row
        and "test_fixture" not in row
        and "unexpected_private_data" not in row
        for row in exported
    )
    assert exported[0]["text_sha256"] == rows[0]["text_sha256"]
    assert exported[0]["protocol"] == PROTOCOL


@pytest.mark.parametrize(
    "decisions",
    [
        [],
        [{"ids": ["fixture-0"], "reason": "fixture decision"}],
        [{"ids": ["unknown-id"], "reason": "fixture decision"}],
        [{"ids": ["fixture-0", "fixture-0"], "reason": "fixture decision"}],
        [{"ids": ["fixture-0", "fixture-1"], "reason": " "}],
        [
            {"ids": ["fixture-0"], "reason": "one"},
            {"ids": ["fixture-0", "fixture-1"], "reason": "two"},
        ],
    ],
)
def test_adjudication_requires_complete_nonduplicated_decisions(tmp_path, decisions):
    paths = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    final = tmp_path / "final.jsonl"
    write_rows(paths[0], declared_rows())
    write_rows(paths[1], declared_rows("fixture-reviewer-b"))
    write_rows(final, declared_rows("fixture-adjudicator", independent=False))
    record = adjudication_for("fixture-package", paths, final)
    record["decisions"] = decisions
    with pytest.raises(ValueError, match="decision"):
        validate_adjudication(
            record, "fixture-package", paths, final, required_ids=["fixture-0", "fixture-1"]
        )


def test_without_human_flag_declarations_are_not_promoted_to_human_gold():
    assert (
        campaign_review.annotation_source(
            declared_rows(independent=False), [declared_rows()], False
        )
        == "unverified"
    )


def test_proxy_generator_cannot_reuse_a_human_templates_attestations():
    rows = declared_rows("proxy-cosine-90")
    with pytest.raises(ValueError, match="proxy reviewer"):
        validate_human_rows(rows)


def test_jsonl_uses_physical_newlines_for_unicode_separators(tmp_path):
    path = tmp_path / "unicode.jsonl"
    rows = [{"text": "a\u0085b\u2028c\u2029d"}, {"text": "second record"}]
    write_rows(path, rows)
    assert campaign_review.read_rows(path) == rows


@pytest.fixture
def cli_fixture(tmp_path):
    # All data, reviewer identities, vectors and labels below are generated TEST FIXTURES.
    docs = [
        {"id": f"{i:03}-{j}", "text": f"PRIVATE FIXTURE EMAIL {i}"}
        for i in range(50)
        for j in range(2)
    ]
    clusters = [
        {"cluster_id": f"fixture-{i}", "members": [f"{i:03}-0", f"{i:03}-1"], "size": 2}
        for i in range(50)
    ]
    provenance = {
        "dataset": "AUTOMATED PROTOCOL TEST FIXTURE",
        "source": "test code",
        "license": "test-only",
        "approved_by": "test fixture",
        "external_allowed": True,
    }
    package = prepare({"model_version": "test-fixture", "clusters": clusters}, docs, provenance)
    package_path = tmp_path / "package"
    write_package(package, package_path)

    def labels(author, independent=True):
        return [
            {
                **row,
                "gold_group": "gold-" + row["predicted_cluster"],
                "annotator": author,
                "review": "near-dup",
                "annotation_source": "human",
                "protocol": PROTOCOL,
                "human_confirmed": True,
                "stage": "independent" if independent else "adjudicated",
                "independent_confirmed": independent,
                "notes": "PRIVATE FIXTURE NOTE",
                "test_fixture": True,
            }
            for row in package["annotations"]
        ]

    paths = [tmp_path / "reviewer-a.jsonl", tmp_path / "reviewer-b.jsonl"]
    reviewer_rows = [labels("fixture-reviewer-a"), labels("fixture-reviewer-b")]
    for path, rows in zip(paths, reviewer_rows, strict=True):
        write_rows(path, rows)
    final = tmp_path / "final.jsonl"
    final_rows = labels("fixture-adjudicator", False)
    write_rows(final, final_rows)
    embeddings = tmp_path / "vectors.npz"
    np.savez(
        embeddings,
        ids=[d["id"] for d in package["documents"]],
        vectors=np.repeat(np.eye(50), 2, axis=0),
        text_sha256=[document_hash(d["text"]) for d in package["documents"]],
        metadata=np.asarray(
            json.dumps(
                {
                    "model": "TEST FIXTURE VECTORS",
                    "revision": "0" * 40,
                    "package_id": package["manifest"]["package_id"],
                }
            )
        ),
    )
    return {
        "package": package,
        "package_path": package_path,
        "reviewers": paths,
        "reviewer_rows": reviewer_rows,
        "final": final,
        "final_rows": final_rows,
        "embeddings": embeddings,
        "out": tmp_path / "eval",
        "report": tmp_path / "human.md",
    }


def run_cli(monkeypatch, fixture, extra=(), human=True):
    args = [
        "campaign_review",
        "--package",
        str(fixture["package_path"]),
        "--embeddings",
        str(fixture["embeddings"]),
        "--annotations",
        str(fixture["final"]),
        "--reviewers",
        *map(str, fixture["reviewers"]),
        "--out",
        str(fixture["out"]),
        "--report",
        str(fixture["report"]),
        *extra,
    ]
    if human:
        args.append("--human-gold")
    monkeypatch.setattr(sys, "argv", args)

    def fixed_fixture_partition(docs, vectors):
        assert len(docs) == len(vectors) == 100
        return {
            "model_version": "fixed-test-fixture",
            "parameters": {},
            "assignments": {d["id"]: "fixture-pred-" + d["id"].split("-")[0] for d in docs},
        }

    monkeypatch.setattr(campaign_review, "cluster_v2", fixed_fixture_partition)
    campaign_review.main()


def test_human_eval_accepts_redacted_rows_and_exports_only_sanitized_annotations(
    cli_fixture,
    monkeypatch,
):
    fixture = cli_fixture
    for path, rows in [
        *zip(fixture["reviewers"], fixture["reviewer_rows"], strict=True),
        (fixture["final"], fixture["final_rows"]),
    ]:
        write_rows(path, [{k: v for k, v in row.items() if k != "text"} for row in rows])
    run_cli(monkeypatch, fixture)
    metrics = json.loads((fixture["out"] / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["gold_source"] == "human"
    assert metrics["human_protocol_validated"] is True
    assert metrics["adjudication_required"] is False
    assert metrics["bcubed"]["n"] == 100
    assert "человеческом gold" in fixture["report"].read_text(encoding="utf-8")
    public_files = list((fixture["out"] / "public").glob("*.jsonl"))
    assert len(public_files) == 3
    for path in public_files:
        text = path.read_text(encoding="utf-8")
        assert "PRIVATE FIXTURE" not in text
        rows = campaign_review.read_rows(path)
        assert all("text" not in row and "notes" not in row for row in rows)


@pytest.mark.parametrize("broken", ["proxy", "partial", "duplicate", "hash", "stage", "author"])
def test_human_eval_rejects_bad_inputs_before_any_artifacts(cli_fixture, monkeypatch, broken):
    fixture = cli_fixture
    rows = copy.deepcopy(fixture["reviewer_rows"][0])
    if broken == "proxy":
        rows[0]["annotation_source"] = "proxy"
    elif broken == "partial":
        rows.pop()
    elif broken == "duplicate":
        rows[0] = copy.deepcopy(rows[1])
    elif broken == "hash":
        rows[0].pop("text")
        rows[0]["text_sha256"] = "0" * 64
    elif broken == "stage":
        rows[0]["stage"] = "adjudicated"
    else:
        for row in rows:
            row["annotator"] = "fixture-reviewer-b"
    write_rows(fixture["reviewers"][0], rows)
    with pytest.raises(ValueError):
        run_cli(monkeypatch, fixture)
    assert not fixture["out"].exists()
    assert not fixture["report"].exists()


@pytest.mark.parametrize("change_final", [False, True])
def test_human_disagreement_needs_hash_bound_record_not_just_name(
    cli_fixture,
    monkeypatch,
    change_final,
):
    fixture = cli_fixture
    if change_final:
        fixture["final_rows"][0]["gold_group"] = "fixture-split-singleton"
        write_rows(fixture["final"], fixture["final_rows"])
    else:
        fixture["reviewer_rows"][1][0]["gold_group"] = "fixture-split-singleton"
        write_rows(fixture["reviewers"][1], fixture["reviewer_rows"][1])
    with pytest.raises(ValueError, match="adjudication-record"):
        run_cli(monkeypatch, fixture, ["--adjudicated-by", "fixture-adjudicator"])
    assert not fixture["out"].exists()
    record = adjudication_for(
        fixture["package"]["manifest"]["package_id"], fixture["reviewers"], fixture["final"]
    )
    record["private_notes"] = "PRIVATE ADJUDICATION NOTE"
    record["decisions"] = [
        {"ids": ["000-0", "000-1"], "reason": "PRIVATE FIXTURE ADJUDICATION REASON"}
    ]
    record_path = fixture["final"].parent / "adjudication.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    run_cli(monkeypatch, fixture, ["--adjudication-record", str(record_path)])
    metrics = json.loads((fixture["out"] / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["adjudicated_by"] == "fixture-adjudicator"
    assert metrics["adjudication_record_sha256"] == file_sha256(record_path)
    public_record = (fixture["out"] / "public/adjudication-record.json").read_text(encoding="utf-8")
    assert "private_notes" not in public_record and "PRIVATE" not in public_record
    assert "reason_sha256" in public_record


def test_legacy_proxy_path_remains_usable_and_never_claims_human_gold(cli_fixture, monkeypatch):
    fixture = cli_fixture
    for path, rows, author in [
        (fixture["reviewers"][0], fixture["reviewer_rows"][0], "proxy-cosine-90"),
        (fixture["reviewers"][1], fixture["reviewer_rows"][1], "proxy-cosine-75"),
        (fixture["final"], fixture["final_rows"], "proxy-cosine-85-gold"),
    ]:
        for row in rows:
            for field in (
                "annotation_source",
                "protocol",
                "human_confirmed",
                "stage",
                "independent_confirmed",
            ):
                row.pop(field, None)
            row["annotator"] = author
        write_rows(path, rows)
    run_cli(monkeypatch, fixture, human=False)
    metrics = json.loads((fixture["out"] / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["gold_source"] == "proxy"
    assert metrics["human_protocol_validated"] is False
    report = fixture["report"].read_text(encoding="utf-8")
    assert "автоматическая прокси-разметка" in report
    assert "Человеческий gold этим прогоном не подтверждён" in report
    assert not (fixture["out"] / "public").exists()


@pytest.mark.parametrize("rename_labels", [False, True])
def test_unanimous_independent_export_is_accepted_without_manual_stage_edits(
    cli_fixture,
    monkeypatch,
    rename_labels,
):
    fixture = cli_fixture
    rows = copy.deepcopy(fixture["reviewer_rows"][0])
    if rename_labels:
        for row in rows:
            row["gold_group"] = "equivalent-renaming-" + row["gold_group"]
    write_rows(fixture["final"], rows)
    run_cli(monkeypatch, fixture)
    metrics = json.loads((fixture["out"] / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["gold_source"] == "human"
    assert metrics["final_stage"] == "independent"
    assert metrics["final_annotator"] == "fixture-reviewer-a"
    assert metrics["unanimous_independent_final"] is True
    assert metrics["adjudication_required"] is False
    assert metrics["adjudication"] is None
    public_final = campaign_review.read_rows(fixture["out"] / "public/annotations-final.jsonl")
    assert all(row["stage"] == "independent" for row in public_final)


@pytest.mark.parametrize("changed_final", [False, True])
@pytest.mark.parametrize("with_record", [False, True])
def test_independent_final_is_rejected_on_any_real_disagreement_even_with_record(
    cli_fixture,
    monkeypatch,
    changed_final,
    with_record,
):
    fixture = cli_fixture
    rows = copy.deepcopy(fixture["reviewer_rows"][0])
    if changed_final:
        rows[0]["gold_group"] = "fixture-new-singleton"
    else:
        fixture["reviewer_rows"][1][0]["gold_group"] = "fixture-new-singleton"
        write_rows(fixture["reviewers"][1], fixture["reviewer_rows"][1])
    write_rows(fixture["final"], rows)
    extra = []
    if with_record:
        record = adjudication_for(
            fixture["package"]["manifest"]["package_id"], fixture["reviewers"], fixture["final"]
        )
        record["decisions"] = [{"ids": ["000-0", "000-1"], "reason": "Fixture decision"}]
        record_path = fixture["final"].parent / "adjudication.json"
        record_path.write_text(json.dumps(record), encoding="utf-8")
        extra = ["--adjudication-record", str(record_path)]
    with pytest.raises(ValueError, match="stage='adjudicated'"):
        run_cli(monkeypatch, fixture, extra)
    assert not fixture["out"].exists()
    assert not fixture["report"].exists()


def test_mixed_final_stages_are_rejected_even_for_unanimous_partition(cli_fixture, monkeypatch):
    fixture = cli_fixture
    rows = copy.deepcopy(fixture["reviewer_rows"][0])
    rows[0]["stage"] = "adjudicated"
    write_rows(fixture["final"], rows)
    with pytest.raises(ValueError, match="stage"):
        run_cli(monkeypatch, fixture)
    assert not fixture["out"].exists()
