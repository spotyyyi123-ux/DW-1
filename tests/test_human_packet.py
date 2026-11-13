import json
import re
import zipfile

import pytest

from dw1.campaigns.human_review import payload_for, read_package, write_human_packet
from dw1.campaigns.review import prepare


def source_package(tmp_path):
    docs = [
        {"id": f"{i:02}-{j:02}", "text": f"PRIVATE SOURCE {i} {j} <script>noop</script>"}
        for i in range(50)
        for j in range(12)
    ]
    clusters = [
        {
            "cluster_id": f"campaign-{i}",
            "size": 12,
            "members": [f"{i:02}-{j:02}" for j in range(12)],
        }
        for i in range(50)
    ]
    package = prepare(
        {"model_version": "TEST FIXTURE", "clusters": clusters},
        docs,
        {
            "dataset": "TEST FIXTURE",
            "source": "unit-test",
            "license": "test-only",
            "approved_by": "fixture",
            "external_allowed": True,
        },
    )
    root = tmp_path / "source"
    root.mkdir()
    for key, name in (("manifest", "manifest.json"), ("campaigns", "campaigns.json")):
        (root / name).write_text(json.dumps(package[key], ensure_ascii=False), encoding="utf-8")
    (root / "documents.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in docs), encoding="utf-8"
    )
    # Deliberate old proxy answers must NEVER populate independent human packets.
    (root / "annotation.jsonl").write_text(
        json.dumps({"gold_group": "proxy-prefilled", "annotator": "proxy-cosine"}), encoding="utf-8"
    )
    return root


def payload_from_page(path):
    page = path.read_text(encoding="utf-8")
    match = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S)
    return page, json.loads(match.group(1))


def test_private_packet_preserves_all_ids_but_never_seeds_proxy(tmp_path):
    source = read_package(source_package(tmp_path))
    out = tmp_path / "human"
    protocol = write_human_packet(source, out)
    expected_ids = {d["id"] for d in source["documents"]}
    for slot in "ABC":
        page, payload = payload_from_page(out / f"reviewer-{slot}.html")
        assert {r["id"] for r in payload["rows"]} == expected_ids
        assert payload["reviewer_slot"] == slot
        assert all(r["gold_group"] == r["review"] == "" for r in payload["rows"])
        assert "proxy-prefilled" not in page and "proxy-cosine" not in page
        assert "<script>noop</script>" not in page
    assert protocol["export_contains_raw_text"] is False
    assert json.loads((out / "status.json").read_text())["human_bcubed_f1"] is None
    with zipfile.ZipFile(out / "human-review.zip") as archive:
        assert set(archive.namelist()) == {
            "START.html",
            "INSTRUCTIONS.txt",
            "protocol.json",
            "status.json",
            "reviewer-A.html",
            "reviewer-B.html",
            "reviewer-C.html",
        }


def test_same_package_identity_for_two_reviewers(tmp_path):
    source = read_package(source_package(tmp_path))
    out = tmp_path / "two"
    write_human_packet(source, out, n_reviewers=2)
    assert not (out / "reviewer-C.html").exists()
    a, b = payload_for(source, "A"), payload_for(source, "B")
    assert a["package_id"] == b["package_id"] == source["manifest"]["package_id"]
    assert a["rows"] == b["rows"]
    with pytest.raises(ValueError, match="overwrite"):
        write_human_packet(source, out)


def test_source_zip_root_or_review_prefix_and_no_arbitrary_extraction(tmp_path):
    root = source_package(tmp_path)
    for prefix in ("", "review/"):
        path = tmp_path / ("prefixed.zip" if prefix else "root.zip")
        with zipfile.ZipFile(path, "w") as archive:
            for name in ("manifest.json", "campaigns.json", "documents.jsonl"):
                archive.write(root / name, prefix + name)
            archive.writestr("../../untrusted.py", "must never be extracted")
        package = read_package(path)
        assert len(package["documents"]) == 600
        assert len(package["source_sha256"]) == 64
        assert not (tmp_path / "untrusted.py").exists()


def test_source_tampering_is_rejected_before_packet_written(tmp_path):
    root = source_package(tmp_path)
    path = root / "documents.jsonl"
    path.write_text(
        path.read_text(encoding="utf-8").replace("PRIVATE SOURCE 0 0", "EDITED"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="changed"):
        read_package(root)


def test_incomplete_or_ambiguous_archive_rejected(tmp_path):
    root = source_package(tmp_path)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.write(root / "manifest.json", "manifest.json")
        archive.write(root / "manifest.json", "review/manifest.json")
    with pytest.raises(ValueError, match="ambiguous"):
        read_package(bad)
