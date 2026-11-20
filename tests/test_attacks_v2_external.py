"""No model downloads or GPU: exercise strict external-runner boundaries."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/external/run_attacks_v2.py"
spec = importlib.util.spec_from_file_location("attacks_v2_external", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def row(variant_id="a", text="Please send 15 USD to https://example.org/pay today."):
    return dict(
        variant_id=variant_id,
        message_id="m",
        attack="llm_paraphrase",
        strength=0.5,
        text=text,
        text_sha256=runner.sha_text(text),
        source_sha256="1" * 64,
        y_provenance=1,
        y_intent=1,
        split="test",
        source="llmgen",
        changed=False,
    )


def package(tmp_path):
    variants = [dict(row("clean"), attack="clean", strength=0)]
    requests = []
    for attack in (*runner.RULE_ATTACKS, "llm_paraphrase"):
        for strength in runner.STRENGTHS:
            item = dict(row(f"{attack}-{strength}"), attack=attack, strength=strength)
            (requests if attack == "llm_paraphrase" else variants).append(item)
    runner.write_jsonl(tmp_path / "variants.jsonl", variants)
    runner.write_jsonl(tmp_path / "requests.jsonl", requests)
    provenance = dict(
        purpose="sprint4-attacks-v2",
        external_allowed=True,
        internal_customer_data=False,
        approved_by="test-owner",
        approved_at="2026-11-13",
        sources={
            "llmgen": dict(
                public_dataset=True,
                external_allowed=True,
                license="Apache-2.0",
                evidence="test fixture",
            )
        },
    )
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    manifest = dict(
        schema_version=1,
        purpose="sprint4-attacks-v2",
        external_allowed=True,
        attack_scope="ai_generated_only",
        n_messages=1,
        n_requests=len(requests),
        n_variants=len(variants) + len(requests),
        strengths=list(runner.STRENGTHS),
        files={
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in ["variants.jsonl", "requests.jsonl", "provenance.json"]
        },
    )
    manifest["package_id"] = runner.canonical_hash(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_import_has_no_local_llm_dependencies():
    # The module can be imported in the repository's no-torch environment.
    assert callable(runner.main)
    assert "transformers" not in sys.modules


def test_package_permission_and_tamper_fail_closed(tmp_path):
    manifest = package(tmp_path)
    assert runner.load_package(tmp_path)[0]["package_id"] == manifest["package_id"]
    manifest["purpose"] = "bcubed"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="purpose"):
        runner.load_package(tmp_path)
    package(tmp_path)
    (tmp_path / "requests.jsonl").write_text("[]")
    with pytest.raises(ValueError, match="hash"):
        runner.load_package(tmp_path)


def refresh_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"] = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in manifest["files"]
    }
    manifest["package_id"] = runner.canonical_hash(
        {key: value for key, value in manifest.items() if key != "package_id"}
    )
    path.write_text(json.dumps(manifest))


def test_valid_hash_does_not_replace_public_data_license_check(tmp_path):
    package(tmp_path)
    path = tmp_path / "provenance.json"
    provenance = json.loads(path.read_text())
    provenance["sources"]["llmgen"]["license"] = "pending"
    path.write_text(json.dumps(provenance))
    refresh_manifest(tmp_path)
    with pytest.raises(ValueError, match="license"):
        runner.load_package(tmp_path)


def test_grid_incomplete_even_when_manifest_hash_is_valid(tmp_path):
    package(tmp_path)
    path = tmp_path / "requests.jsonl"
    requests = runner.read_jsonl(path)
    runner.write_jsonl(path, requests[:-1])
    refresh_manifest(tmp_path)
    with pytest.raises(ValueError, match="counts"):
        runner.load_package(tmp_path)


def test_protected_spans_must_survive_without_added_numbers():
    text = "Send 15 USD to a@example.org at https://example.org/pay today."
    masked, spans = runner.protect(text)
    assert runner.restore(masked.replace("Send", "Please transfer"), spans).startswith(
        "Please transfer 15"
    )
    with pytest.raises(ValueError, match="placeholders"):
        runner.restore(masked.replace("<DW1_KEEP_0000>", ""), spans)
    with pytest.raises(ValueError, match="added"):
        runner.restore(masked + " plus 200 USD", spans)
    with pytest.raises(ValueError, match="reserved"):
        runner.protect("<DW1_KEEP_0000>")


def test_generation_checkpoint_does_not_accept_changed_metadata(tmp_path):
    request = row()
    calls = []

    def generate(messages, number):
        calls.append(number)
        return (
            messages[-1]["content"]
            .split("<EMAIL_DATA>\n")[1]
            .split("\n</EMAIL_DATA>")[0]
            .replace("Please send", "Kindly transfer")
        )

    path = tmp_path / "partial.jsonl"
    rows = runner.paraphrase_requests([request], generate, path)
    assert rows[0]["changed"] is True
    assert calls == [0]
    runner.paraphrase_requests([request], generate, path)
    assert calls == [0]
    rows[0]["source_sha256"] = "0" * 64
    runner.write_jsonl(path, rows)
    with pytest.raises(ValueError, match="metadata"):
        runner.paraphrase_requests([request], generate, path)


def test_unchanged_rewrite_not_an_llm_attack():
    request = row()
    with pytest.raises(ValueError, match="unchanged"):
        runner.validate_rewrite(request, request["text"])


def test_bad_generation_has_bounded_deterministic_retries(tmp_path):
    request = row()
    calls = []

    def generate(messages, number):
        calls.append(number)
        text = messages[-1]["content"].split("<EMAIL_DATA>\n")[1].split("\n</EMAIL_DATA>")[0]
        return text if number == 0 else text.replace("Please send", "Kindly transfer")

    rows = runner.paraphrase_requests([request], generate, tmp_path / "retry.jsonl")
    assert calls == [0, 1]
    assert rows[0]["text"].startswith("Kindly transfer")
    with pytest.raises(ValueError, match="3 validated attempts"):
        runner.paraphrase_requests(
            [request], lambda messages, number: "", tmp_path / "failed.jsonl"
        )


def test_exact_variant_grid_deduplicates_inference_and_flags_incomplete(tmp_path):
    variants = [row("one"), row("two"), row("three", "A different full text.")]
    calls = []

    def score(text):
        calls.append(text)
        if text.startswith("A different"):
            raise ValueError("less_than_32_target_tokens")
        return dict(criterion=0.75, n_tokens=50, truncated=False)

    scores, skipped = runner.score_variants(variants, score, "model@sha", tmp_path / "fd.jsonl")
    assert [s["variant_id"] for s in scores] == ["one", "two"]
    assert len(calls) == 2
    assert skipped == [{"variant_id": "three", "reason": "less_than_32_target_tokens"}]
    with pytest.raises(ValueError, match="checkpoint"):
        runner.score_variants(variants, score, "different-model", tmp_path / "fd.jsonl")


def test_review_starts_without_automatic_human_approval(tmp_path):
    request = row()
    rewrite = runner.validate_rewrite(
        request, request["text"].replace("Please send", "Kindly transfer")
    )
    runner.write_review(tmp_path, [request], [rewrite], "a" * 64)
    review = runner.read_jsonl(tmp_path / "semantic_review.jsonl")
    assert review[0]["approved"] is None
    assert review[0]["reviewer"] == ""
    assert set(review[0]) == {
        "variant_id",
        "package_id",
        "original_text_sha256",
        "text_sha256",
        "approved",
        "reviewer",
        "meaning_preserved",
        "language_preserved",
    }
    assert "Kindly transfer" in (tmp_path / "semantic_review.html").read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("node") is None, reason="Optional Node DOM-stub export check")
def test_review_download_exports_real_jsonl_with_bound_identity(tmp_path):
    requests = [row("one"), row("two")]
    rewrites = [
        runner.validate_rewrite(r, r["text"].replace("Please send", "Kindly transfer"))
        for r in requests
    ]
    runner.write_review(tmp_path, requests, rewrites, "a" * 64)
    page = (tmp_path / "semantic_review.html").read_text(encoding="utf-8")
    script = re.search(r"<script>(.*?)</script>", page, re.DOTALL).group(1)
    prefix = """
const elements = {};
globalThis.document = {
  getElementById: id => elements[id] ||= {value:'', checked:false, textContent:''},
  createElement: () => ({click(){}})
};
globalThis.localStorage = {getItem(){return null},setItem(){}};
URL.createObjectURL = blob => {globalThis.downloadBlob=blob;return 'blob:fake'};
globalThis.setTimeout = () => 0;
"""
    suffix = """
document.getElementById('who').value='Reviewer';
elements.meaning.checked=true;elements.language.checked=true;
elements.download.onclick();
downloadBlob.text().then(text=>console.log(JSON.stringify(text.split(String.fromCharCode(10)))));
"""
    js = tmp_path / "download.cjs"
    js.write_text(prefix + script + suffix, encoding="utf-8")
    result = subprocess.run(
        [shutil.which("node"), str(js)], capture_output=True, text=True, check=True
    )
    lines = json.loads(result.stdout)
    assert len(lines) == 3 and lines[-1] == ""
    first, second = map(json.loads, lines[:-1])
    assert first["approved"] is True and first["reviewer"] == "Reviewer"
    assert second["approved"] is None
    assert first["package_id"] == "a" * 64
    assert first["original_text_sha256"] == requests[0]["text_sha256"]


def test_no_mutable_model_revision_allowed(tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        runner.main(
            [
                "--package",
                str(tmp_path),
                "--out",
                str(tmp_path),
                "--paraphrase-revision",
                "main",
                "--detector-revision",
                "main",
            ]
        )
