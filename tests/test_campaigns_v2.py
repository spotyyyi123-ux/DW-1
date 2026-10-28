import json

import numpy as np
import pytest

from dw1.campaigns.template import skeleton
from dw1.campaigns.v2 import cluster_v2, document_hash, load_embeddings
from dw1.eval.campaigns import bcubed, evaluate


def test_bcubed_merge_split_and_perfect():
    assert bcubed(["a", "a", "b", "b"], ["x", "x", "y", "y"])["f1"] == 1
    merged = bcubed(["a", "a", "b", "b"], ["x"] * 4)
    assert merged["precision"] == 0.5 and merged["recall"] == 1
    split = bcubed(["a"] * 4, ["x", "x", "y", "y"])
    assert split["precision"] == 1 and split["recall"] == 0.5
    with pytest.raises(ValueError):
        bcubed([], [])


def test_skeleton_typed_slots():
    result = skeleton(
        [
            "Здравствуйте, Иван! Оплатите 100 RUB для бренда Alpha https://a.invalid/pay",
            "Здравствуйте, Анна! Оплатите 200 RUB для бренда Beta https://b.invalid/pay",
        ]
    )
    assert all("{" + k + "}" in result["text"] for k in ("NAME", "AMOUNT", "BRAND", "LINK"))
    assert result["slots"]["NAME"] == ["Анна", "Иван"]


def test_v2_disjoint_deterministic_and_noise():
    docs = [
        {"id": str(i), "text": text}
        for i, text in enumerate(
            [
                "one two three four five",
                "one two three four five",
                "other words entirely",
                "more independent words",
                "a final message",
            ]
        )
    ]
    vectors = np.eye(5)
    result = cluster_v2(docs, vectors)
    members = [m for c in result["clusters"] for m in c["members"]]
    assert len(members) == len(set(members)) == len(docs)
    assert result["assignments"]["0"] == result["assignments"]["1"]
    assert result == cluster_v2(docs, vectors)
    with pytest.raises(ValueError):
        cluster_v2(docs, np.zeros((5, 3)))


def test_embeddings_order_hash_and_nan(tmp_path):
    docs = [{"id": "a", "text": "hello"}, {"id": "b", "text": "world"}]
    path = tmp_path / "vectors.npz"
    np.savez(
        path,
        ids=["b", "a"],
        vectors=[[0.0, 1.0], [1.0, 0.0]],
        text_sha256=[document_hash("world"), document_hash("hello")],
        metadata=np.asarray(json.dumps({"model": "fixture"})),
    )
    vectors, _ = load_embeddings(path, docs)
    np.testing.assert_equal(vectors, np.eye(2))
    with pytest.raises(ValueError, match="Stale"):
        load_embeddings(path, [{"id": "a", "text": "changed"}, docs[1]])


def test_evaluation_refuses_missing_or_partial_manual_labels():
    result = {
        "model_version": "test",
        "clusters": [{"cluster_id": "c", "size": 2, "members": ["a", "b"]}],
    }
    with pytest.raises(ValueError, match="50"):
        evaluate(result, [])
    rows = [
        {
            "id": x,
            "predicted_cluster": "c",
            "gold_group": "g",
            "review": "near-dup",
            "annotator": "test-fixture",
        }
        for x in ["a", "b"]
    ]
    assert evaluate(result, rows, required_clusters=1)[0]["bcubed"]["f1"] == 1
    rows[0]["gold_group"] = ""
    with pytest.raises(ValueError, match="Manual"):
        evaluate(result, rows, required_clusters=1)


def test_hdbscan_merges_semantic_neighbors_outside_lsh():
    docs = [{"id": str(i), "text": f"unique{i} distinct{i} token{i}"} for i in range(6)]
    vectors = np.asarray([[1, 0, 0], [1, .01, 0], [0, 1, 0], [.01, 1, 0],
                          [0, 0, 1], [0, .01, 1]], dtype=float)
    result = cluster_v2(docs, vectors)
    assert result["lsh_components"] == 6
    assignments = result["assignments"]
    assert assignments["0"] == assignments["1"]
    assert assignments["2"] == assignments["3"]
    assert assignments["4"] == assignments["5"]
    assert len(set(assignments.values())) == 3
