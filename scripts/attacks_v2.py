"""Prepare and score Sprint 4 attacks without importing local torch/transformers.

Private body texts stay in the prepared package; public predictions contain hashes only.
The independent external runner is copied into that package, never fetched from main.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from email import policy
from email.parser import BytesParser
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.attacks import ATTACK_NAMES, ATTACK_REGISTRY  # noqa: E402

STRENGTHS = [0.25, 0.5, 0.75, 1.0]
PURPOSE = "sprint4-attacks-v2"
TRAINING_HASH_POLICY = "exact-utf8+body-lower-whitespace-v1"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_hash(text: str) -> str:
    return digest(text.encode("utf-8"))


def normalized_hash(text: str) -> str:
    return text_hash(" ".join(text.lower().split()))


def canonical(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def body_text(path: Path) -> str:
    """Same plain-body selection as MIME ingest, without image/attachment processing."""
    msg = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    texts = []
    for part in msg.walk():
        if part.get_content_type() == "text/plain" and part.get_content_disposition() != "attachment":
            value = part.get_content()
            if not isinstance(value, str):
                raise ValueError(f"Non-text plain body: {path.name}")
            if value:
                texts.append(value)
    result = "\n\n".join(texts)
    if not result.strip():
        raise ValueError(f"No plain body: {path.name}; HTML-only messages need an explicit protocol")
    return result


def load_benchmark(benchmark: Path, expected_messages: int = 600) -> list[dict]:
    with (benchmark / "labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        labels = list(csv.DictReader(handle))
    if len(labels) != expected_messages:
        raise ValueError(f"Expected {expected_messages} real messages, received {len(labels)}")
    rows: list[dict] = []
    names: set[str] = set()
    hashes: set[str] = set()
    for label in labels:
        name = label.get("file", "")
        if not name or Path(name).name != name or "/" in name or "\\" in name or not name.endswith(".eml"):
            raise ValueError("Expected unique .eml basenames, no paths")
        if name in names:
            raise ValueError(f"Duplicate filename: {name}")
        names.add(name)
        prov, intent, source = label.get("provenance"), label.get("intent"), label.get("source")
        if prov not in {"human", "ai_generated"} or intent not in {"legitimate", "spam", "phishing"}:
            raise ValueError(f"Unsupported labels: {name}")
        if not source:
            raise ValueError(f"Missing source/license key: {name}")
        path = benchmark / name
        if path.is_symlink() or path.resolve().parent != benchmark.resolve():
            raise ValueError("Benchmark files must stay inside the benchmark directory")
        text = body_text(path)
        key = text_hash(text)
        norm_key = normalized_hash(text)
        if norm_key in hashes:
            raise ValueError("Duplicate bodies must be resolved before freezing splits")
        hashes.add(norm_key)
        rows.append(dict(message_id=name, source_sha256=digest(path.read_bytes()),
                         text_sha256=key, text=text, source=source,
                         y_provenance=int(prov == "ai_generated"),
                         y_intent=int(intent != "legitimate"), split=label.get("split", "")))
    # Stable stratification is independent of filenames, label order and global RNG state.
    splits = {row["split"] for row in rows}
    if splits == {""}:
        groups = defaultdict(list)
        for row in rows:
            groups[(row["y_provenance"], row["y_intent"])].append(row)
        for group in groups.values():
            group.sort(key=lambda row: text_hash("dw1-attacks-v2-split:" + normalized_hash(row["text"])))
            count = max(1, round(len(group) * 0.2))
            for i, row in enumerate(group):
                row["split"] = "validation" if i < count else "test"
    elif not splits <= {"validation", "test"}:
        raise ValueError("Supply all splits validation/test, or omit all splits")
    for split in ("validation", "test"):
        for axis in ("y_provenance", "y_intent"):
            if {row[axis] for row in rows if row["split"] == split} != {0, 1}:
                raise ValueError(f"Both classes required in clean {split}/{axis}")
    return sorted(rows, key=lambda row: row["message_id"])


def validate_export(provenance: dict, sources: set[str]) -> None:
    if provenance.get("purpose") != PURPOSE or provenance.get("external_allowed") is not True:
        raise ValueError("Need export permission specifically for sprint4-attacks-v2")
    if provenance.get("internal_customer_data") is not False:
        raise ValueError("Internal customer mail must not be exported")
    if not provenance.get("approved_by") or not provenance.get("approved_at"):
        raise ValueError("Missing export approver/date")
    configs = provenance.get("sources", {})
    for source in sources:
        item = configs.get(source, {})
        if (item.get("external_allowed") is not True or not item.get("license")
                or not item.get("evidence") or item.get("public_dataset") is not True):
            raise ValueError(f"Missing documented public source/license/export permission: {source}")
        if str(item["license"]).strip().lower() in {"other", "unknown", "pending", "на проверке"}:
            raise ValueError(f"Unresolved license: {source}")


def variant_id(message_id: str, attack: str, strength: float) -> str:
    return text_hash(f"{message_id}\n{attack}\n{strength:.2f}")


def prepare(benchmark: Path, provenance_path: Path, out: Path, *, expected_messages=600) -> dict:
    rows = load_benchmark(benchmark, expected_messages)
    provenance = read_json(provenance_path)
    validate_export(provenance, {row["source"] for row in rows})
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output must be empty; keep frozen packages immutable")
    variants, requests = [], []
    for source in rows:
        for attack, strength in [("clean", 0.0)] + [
            (attack, strength) for attack in [*ATTACK_NAMES, "llm_paraphrase"] for strength in STRENGTHS
        ]:
            row = dict(source, attack=attack, strength=strength,
                       variant_id=variant_id(source["message_id"], attack, strength), changed=False)
            if attack == "llm_paraphrase" and row["y_provenance"]:
                requests.append(row)
                continue
            if attack != "clean" and row["y_provenance"]:
                seed = int(text_hash(row["variant_id"])[:16], 16)
                row["text"] = ATTACK_REGISTRY[attack](row["text"], strength, random.Random(seed))
                row["text_sha256"] = text_hash(row["text"])
                row["changed"] = row["text_sha256"] != source["text_sha256"]
            variants.append(row)
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / "variants.jsonl", variants)
    write_rows(out / "requests.jsonl", requests)
    write_json(out / "provenance.json", provenance)
    for name in ("run_attacks_v2.py", "attacks_v2_colab.ipynb", "FAST-DETECT-GPT-LICENSE.txt"):
        shutil.copyfile(ROOT / "scripts" / "external" / name, out / name)
    manifest = dict(schema_version=1, purpose=PURPOSE, external_allowed=True,
                    attack_scope="ai_generated_only", strengths=STRENGTHS,
                    n_messages=len(rows), n_requests=len(requests),
                    n_variants=len(variants) + len(requests),
                    population_counts=dict(Counter(f"prov={r['y_provenance']},intent={r['y_intent']}" for r in rows)),
                    split_counts=dict(Counter(r["split"] for r in rows)),
                    split_policy=("owner_supplied_validation_test" if "split" in
                                  (benchmark / "labels.csv").read_text(encoding="utf-8-sig").split("\n")[0].split(",")
                                  else "normalized_body_hash_stratified_80_20_v1"),
                    benchmark_labels_sha256=digest((benchmark / "labels.csv").read_bytes()),
                    files={p.name: digest(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()})
    manifest["package_id"] = digest(canonical(manifest))
    write_json(out / "manifest.json", manifest)
    bundle = out.parent / (out.name + ".zip")
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.iterdir()):
            archive.write(path, path.name)
    return dict(status="prepared_not_measured", package_id=manifest["package_id"],
                n_messages=len(rows), n_requests=len(requests), bundle=str(bundle.resolve()))


def verify_package(package: Path) -> dict:
    manifest = read_json(package / "manifest.json")
    unsigned = {key: value for key, value in manifest.items() if key != "package_id"}
    if manifest.get("schema_version") != 1 or manifest.get("package_id") != digest(canonical(unsigned)):
        raise ValueError("Invalid package identity/schema")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Invalid package path")
        if digest((package / name).read_bytes()) != expected:
            raise ValueError(f"Package changed: {name}")
    provenance = read_json(package / "provenance.json")
    rows = read_rows(package / "variants.jsonl") + read_rows(package / "requests.jsonl")
    validate_export(provenance, {row["source"] for row in rows})
    if manifest.get("purpose") != PURPOSE or manifest.get("external_allowed") is not True:
        raise ValueError("Export scope mismatch")
    for row in rows:
        if row["text_sha256"] != text_hash(row["text"]):
            raise ValueError("Text hash mismatch")
    return manifest


def verify_models(model_manifest: Path, variants: list[dict]) -> tuple[dict, dict]:
    config = read_json(model_manifest)
    if config.get("schema_version") != 1 or config.get("training_hashes_complete") is not True:
        raise ValueError("Need complete training body hashes for both frozen models")
    if (config.get("training_hash_policy") != TRAINING_HASH_POLICY
            or not config.get("attested_by") or not config.get("attested_at")):
        raise ValueError("Need owner-attested training provenance and shared hash policy")
    models = {}
    import joblib

    for name in ("provenance", "intent"):
        item = config["models"][name]
        path = model_manifest.parent / item["path"]
        if not item.get("version") or digest(path.read_bytes()) != item.get("sha256"):
            raise ValueError(f"Model hash/version mismatch: {name}")
        training = item.get("training_text_sha256", [])
        normalized = item.get("training_body_normalized_sha256", [])
        if (not item.get("training_provenance") or not training or not normalized
                or any(not re.fullmatch(r"[0-9a-f]{64}", key) for key in training + normalized)):
            raise ValueError(f"Missing complete training hash list: {name}")
        if (set(training) & {row["text_sha256"] for row in variants}
                or set(normalized) & {normalized_hash(row["text"]) for row in variants}):
            raise ValueError(f"Training/evaluation text leakage: {name}")
        # Only load trusted team-owned joblib artifacts, after exact file hash validation.
        models[name] = joblib.load(path)
        if list(getattr(models[name], "classes_", [])) != [0, 1]:
            raise ValueError(f"Binary class order [0,1] required: {name}")
    return config, models


def score(package: Path, external: Path, model_manifest: Path, semantic_review: Path, out: Path) -> dict:
    manifest = verify_package(package)
    result = read_json(external / "external_result.json")
    if (result.get("schema_version") != 1 or result.get("package_id") != manifest["package_id"]
            or result.get("status") != "completed"):
        raise ValueError("External run must be complete and match the frozen package")
    if result.get("runner_sha256") != manifest["files"]["run_attacks_v2.py"]:
        raise ValueError("External runner differs from the frozen package")
    if (any(not re.fullmatch(r"[a-f0-9]{40}", str(result.get(field, ""))) for field in
            ("paraphrase_revision", "detector_revision"))
            or not result.get("detector_model") or not result.get("paraphrase_model")):
        raise ValueError("Pinned external detector and paraphraser revisions are required")
    if (result.get("n_scored") != manifest["n_variants"]
            or result.get("n_variants") != manifest["n_variants"] or result.get("skipped") != []):
        raise ValueError("External result population/coverage mismatch")
    for name in ("external_variants.jsonl", "fd_scores.parquet"):
        if result.get("files", {}).get(name) != digest((external / name).read_bytes()):
            raise ValueError(f"External result hash mismatch: {name}")
    variants = read_rows(package / "variants.jsonl")
    requested = {row["variant_id"]: row for row in read_rows(package / "requests.jsonl")}
    generated = read_rows(external / "external_variants.jsonl")
    if len(generated) != len(requested) or {r["variant_id"] for r in generated} != set(requested):
        raise ValueError("Missing/duplicate/unexpected LLM variants")
    reviews = read_rows(semantic_review)
    if len(reviews) != len(requested) or {r["variant_id"] for r in reviews} != set(requested):
        raise ValueError("Semantic review must cover every LLM rewrite once")
    reviews_by_id = {row["variant_id"]: row for row in reviews}
    # Bind output to the exact input labels/splits; external copies cannot change ground truth.
    for rewrite in generated:
        origin = requested[rewrite["variant_id"]]
        review = reviews_by_id[rewrite["variant_id"]]
        key = text_hash(rewrite["text"])
        if rewrite.get("text_sha256") != key or key == origin["text_sha256"]:
            raise ValueError("Invalid or unchanged LLM rewrite")
        from scripts.external.run_attacks_v2 import validate_rewrite

        validate_rewrite(origin, rewrite["text"])
        if (review.get("text_sha256") != key or not str(review.get("reviewer", "")).strip()
                or review.get("package_id") != manifest["package_id"]
                or review.get("original_text_sha256") != origin["text_sha256"]
                or any(review.get(field) is not True for field in
                       ("approved", "meaning_preserved", "language_preserved"))):
            raise ValueError("All rewrites need explicit human semantic/language approval")
        variants.append(dict(origin, text=rewrite["text"], text_sha256=key, changed=True))
    if len(variants) != manifest["n_variants"]:
        raise ValueError("Variant population mismatch")
    scores = pd.read_parquet(external / "fd_scores.parquet")
    required = {"variant_id", "text_sha256", "criterion", "model_version", "n_tokens", "truncated"}
    if not required <= set(scores) or scores[list(required)].isna().any().any():
        raise ValueError("Incomplete Fast-DetectGPT score schema")
    if {"text", "body", "prompt"} & set(scores):
        raise ValueError("Score tables must not contain raw texts")
    if len(scores) != len(variants) or scores.variant_id.duplicated().any():
        raise ValueError("Incomplete or duplicate Fast-DetectGPT scores")
    if set(scores.variant_id) != {row["variant_id"] for row in variants}:
        raise ValueError("Unexpected Fast-DetectGPT score keys")
    if scores.model_version.nunique() != 1 or not str(scores.model_version.iloc[0]).strip():
        raise ValueError("One immutable Fast-DetectGPT model version required")
    if str(scores.model_version.iloc[0]) != result.get("model_version"):
        raise ValueError("Fast-DetectGPT version differs from external result")
    if not np.isfinite(scores.criterion.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite Fast-DetectGPT score")
    if (scores.n_tokens < 32).any() or not scores.truncated.isin([False, True]).all():
        raise ValueError("Invalid Fast-DetectGPT token/truncation metadata")
    if scores.truncated.any():
        raise ValueError("Full-body attack evaluation does not accept truncated detector scores")
    fd = scores.set_index("variant_id")
    for row in variants:
        if fd.loc[row["variant_id"], "text_sha256"] != row["text_sha256"]:
            raise ValueError("Fast-DetectGPT scores belong to different text")
    config, models = verify_models(model_manifest, variants)
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    texts = [row["text"] for row in variants]
    with threadpool_limits(limits=2):
        raw = {name: np.asarray(model.predict_proba(texts))[:, 1] for name, model in models.items()}
    raw["fast_detectgpt"] = np.array([float(fd.loc[row["variant_id"], "criterion"]) for row in variants])
    clean_validation = np.array([r["attack"] == "clean" and r["split"] == "validation" for r in variants])
    predictions = pd.DataFrame([{key: value for key, value in row.items() if key != "text"} for row in variants])
    calibration = {}
    for name, values in raw.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite local model output: {name}")
        axis = "y_intent" if name == "intent" else "y_provenance"
        clf = LogisticRegression(random_state=42).fit(values[clean_validation, None], predictions.loc[clean_validation, axis])
        column = "p_intent" if name == "intent" else f"p_{'tfidf_provenance' if name == 'provenance' else name}"
        predictions[column] = clf.predict_proba(values[:, None])[:, 1]
        predictions[f"raw_{name}"] = values
        calibration[name] = dict(coef=clf.coef_.tolist(), intercept=clf.intercept_.tolist(),
                                 fitted_on="clean_validation_only", n=int(clean_validation.sum()))
    predictions["p_ensemble_provenance"] = 0.5 * (predictions.p_tfidf_provenance + predictions.p_fast_detectgpt)
    predictions["tfidf_model_version"] = config["models"]["provenance"]["version"]
    predictions["intent_model_version"] = config["models"]["intent"]["version"]
    predictions["fd_model_version"] = str(scores.model_version.iloc[0])
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output must be empty; do not overwrite a scored run")
    # Validate population/grid consistency before creating any apparently usable result.
    from dw1.eval.attacks_v2 import validate_predictions

    validate_predictions(predictions, expected_messages=manifest["n_messages"])
    out.mkdir(parents=True, exist_ok=True)
    prediction_path = out / "predictions.parquet"
    predictions.to_parquet(prediction_path, index=False)
    write_json(out / "calibration.json", calibration)
    receipt = dict(status="scored_not_evaluated", package_id=manifest["package_id"],
                   n_messages=manifest["n_messages"], predictions_sha256=digest(prediction_path.read_bytes()),
                   model_manifest_sha256=digest(model_manifest.read_bytes()),
                   external_result_sha256=digest((external / "external_result.json").read_bytes()),
                   semantic_review_sha256=digest(semantic_review.read_bytes()),
                   local_scorer_sha256=digest(Path(__file__).read_bytes()),
                   calibration_sha256=digest((out / "calibration.json").read_bytes()),
                   scoring_scope="body_text_predictors_not_full_multielement_fusion",
                   split_policy=manifest["split_policy"],
                   truncated_variants=int(scores.truncated.astype(bool).sum()),
                   training_hashes="owner_attested_complete_lists_checked_exact_and_normalized",
                   training_hash_policy=config["training_hash_policy"],
                   training_attested_by=config["attested_by"])
    write_json(out / "scoring_manifest.json", receipt)
    return receipt


def plan(out: Path, benchmark: Path, model_manifest: Path) -> dict:
    status = dict(schema_version=1, task="sprint4-task2", status="prepared_not_measured",
                  benchmark_present=(benchmark / "labels.csv").is_file(),
                  model_manifest_present=model_manifest.is_file(),
                  required_population=600, measured_messages=0, llm_paraphrase_measured=False,
                  intent_robustness_conclusion=None,
                  missing=["600 real labeled benchmark emails", "trusted frozen TF-IDF and intent weights with complete training hashes",
                           "attack-specific public-data export provenance", "external LLM rewrites and Fast-DetectGPT scores",
                           "human semantic approval of rewrites"],
                  historical_metrics_reused=False)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "preparation.json", status)
    return status


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "plan"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--benchmark", type=Path, default=Path("data/sprint2/bench"))
        cmd.add_argument("--out", type=Path, default=Path("data/runs/sprint4/attacks") / ("package" if name == "prepare" else "status"))
        if name == "prepare":
            cmd.add_argument("--provenance", type=Path, required=True)
        else:
            cmd.add_argument("--models", type=Path, default=Path("data/runs/sprint4/attacks/models.json"))
    cmd = sub.add_parser("score")
    for name in ("package", "external", "models", "semantic-review", "out"):
        cmd.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.benchmark, args.provenance, args.out)
    elif args.command == "score":
        result = score(args.package, args.external, args.models, args.semantic_review, args.out)
    else:
        result = plan(args.out, args.benchmark, args.models)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
