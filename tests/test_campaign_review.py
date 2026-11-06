import copy
import json
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

from dw1.campaigns.documents import document_hash
from dw1.campaigns.review import normalize_campaigns, prepare, write_package
from dw1.campaigns.review_page import render_review
from dw1.eval.campaign_review import compare_reviewers, validate_rows, verify_package


def inputs(n=50):
    docs = [
        {"id": f"{i:03}-{j}", "text": f"Campaign {i} unique item content"}
        for i in range(n)
        for j in range(2)
    ]
    clusters = [
        {"cluster_id": f"c-{i:03}", "size": 2, "members": [f"{i:03}-{j}" for j in range(2)]}
        for i in range(n)
    ]
    provenance = {
        "dataset": "AUTOMATED TEST FIXTURE",
        "source": "test code",
        "license": "test-only",
        "approved_by": "test-fixture",
        "external_allowed": True,
    }
    return {"model_version": "test-fixture", "clusters": clusters}, docs, provenance


def labeled(package, author, prefix="g"):
    return [
        {
            **r,
            "annotator": author,
            "gold_group": prefix + r["predicted_cluster"],
            "review": "near-dup",
        }
        for r in package["annotations"]
    ]


def test_freeze_deterministic_legacy_and_full_population(tmp_path):
    raw, docs, provenance = inputs(51)
    raw["clusters"][-1]["members"].append("extra")
    raw["clusters"][-1]["size"] = 3
    docs.append({"id": "extra", "text": "last member"})
    package = prepare(raw, docs, provenance)
    assert package["campaigns"]["clusters"][0]["cluster_id"] == "c-050"
    assert len(package["documents"]) == 101
    assert package == prepare(
        {**raw, "clusters": list(reversed(raw["clusters"]))}, list(reversed(docs)), provenance
    )
    write_package(package, tmp_path)
    with zipfile.ZipFile(tmp_path / "external_bundle.zip") as bundle:
        assert "documents.jsonl" in bundle.namelist()
        assert "src/dw1/campaigns/documents.py" in bundle.namelist()
        assert not any("torch" in n for n in bundle.namelist())
    legacy = [{"members": [{"id": "a"}, {"id": "b"}], "size": 2}]
    assert normalize_campaigns(legacy)["clusters"][0]["members"] == ["a", "b"]


def test_prepare_rejects_ambiguous_partition_missing_texts_or_approval():
    raw, docs, provenance = inputs()
    with pytest.raises(ValueError, match="approved"):
        prepare(raw, docs, {**provenance, "external_allowed": False})
    with pytest.raises(ValueError, match="license"):
        prepare(raw, docs, {**provenance, "license": "other"})
    with pytest.raises(ValueError, match="Missing"):
        prepare(raw, docs[:-1], provenance)
    raw["clusters"][1]["members"][0] = raw["clusters"][0]["members"][0]
    with pytest.raises(ValueError, match="Overlapping"):
        prepare(raw, docs, provenance)


def test_review_consensus_is_invariant_to_label_names():
    package = prepare(*inputs())
    a, b = labeled(package, "reviewer-a"), labeled(package, "reviewer-b", "different-label-")
    assert compare_reviewers([a, b])["n_disagreements"] == 0
    b[0]["gold_group"] = "split-singleton"
    result = compare_reviewers([a, b])
    assert result["disagreement_ids"] == ["000-0", "000-1"]
    with pytest.raises(ValueError, match="distinct"):
        compare_reviewers([a, a])


def test_reviews_bound_to_frozen_text_and_partition():
    package = prepare(*inputs())
    validate_rows(labeled(package, "human"), package["manifest"], package["campaigns"])
    broken = labeled(package, "human")
    broken[0]["text"] = "edited"
    with pytest.raises(ValueError, match="hash"):
        validate_rows(broken, package["manifest"], package["campaigns"])
    manifest = copy.deepcopy(package["manifest"])
    manifest["top_n"] = 49
    with pytest.raises(ValueError, match="Manifest"):
        verify_package(manifest, package["campaigns"], package["documents"])


def test_page_escapes_hostile_email():
    raw, docs, provenance = inputs()
    docs[0]["text"] = '</script><script>alert("unsafe")</script>'
    page = render_review(prepare(raw, docs, provenance))
    assert "</script><script>alert" not in page
    assert r"\u003c/script>" in page
    assert "text.textContent=row.text" in page


def test_external_export_help_without_torch():
    output = subprocess.run(
        [sys.executable, "scripts/export_embeddings.py", "--help"], capture_output=True, text=True
    )
    assert output.returncode == 0
    assert "--manifest" in output.stdout


def test_full_review_eval_artifacts_without_external_model(tmp_path):
    # Test vectors and labels check plumbing only. No model-quality claim.
    package = prepare(*inputs())
    root = tmp_path / "review"
    write_package(package, root)
    paths = []
    for name in ("reviewer-a", "reviewer-b"):
        path = tmp_path / (name + ".jsonl")
        path.write_text("\n".join(json.dumps(r) for r in labeled(package, name)), encoding="utf-8")
        paths.append(path)
    docs = package["documents"]
    vectors = tmp_path / "vectors.npz"
    np.savez(
        vectors,
        ids=[d["id"] for d in docs],
        vectors=np.repeat(np.eye(50), 2, axis=0),
        text_sha256=[document_hash(d["text"]) for d in docs],
        metadata=np.asarray(
            json.dumps(
                {
                    "model": "TEST VECTORS",
                    "revision": "0" * 40,
                    "package_id": package["manifest"]["package_id"],
                }
            )
        ),
    )
    out, report = tmp_path / "eval", tmp_path / "report.md"
    cmd = [
        sys.executable,
        "-m",
        "dw1.eval.campaign_review",
        "--package",
        str(root),
        "--embeddings",
        str(vectors),
        "--annotations",
        str(paths[0]),
        "--reviewers",
        *map(str, paths),
        "--out",
        str(out),
        "--report",
        str(report),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["bcubed"]["n"] == len(pd.read_parquet(out / "predictions.parquet")) == 100
    assert (out / "inputs/embeddings.npz").exists()
    assert (out / "inputs/reviewer-2.jsonl").exists()
    assert f"{metrics['bcubed']['f1']:.6f}" in report.read_text(encoding="utf-8")
    again = subprocess.run(cmd, capture_output=True, text=True)
    assert again.returncode != 0  # Never overwrite an existing measured run.
