"""Freeze a real top-50 partition for external embeddings and human review."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from dw1.campaigns.documents import document_hash, load_documents


def canonical_hash(value) -> str:
    return document_hash(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    )


def normalize_campaigns(raw) -> dict:
    source = raw if isinstance(raw, list) else raw["clusters"]
    seen: set[str] = set()
    cluster_ids: set[str] = set()
    clusters = []
    for i, cluster in enumerate(source):
        cid = str(cluster.get("cluster_id", f"campaign-{i}"))
        members = [str(m["id"]) if isinstance(m, dict) else str(m) for m in cluster["members"]]
        if not cid or cid in cluster_ids or not members:
            raise ValueError("Empty/duplicate cluster ID or empty members")
        if len(set(members)) != len(members) or seen.intersection(members):
            raise ValueError("Overlapping cluster membership: regenerate a disjoint partition")
        if cluster.get("size", len(members)) != len(members):
            raise ValueError("Cluster size differs from its members")
        cluster_ids.add(cid)
        seen.update(members)
        clusters.append(
            {**cluster, "cluster_id": cid, "size": len(members), "members": sorted(members)}
        )
    return {
        "model_version": raw.get("model_version", "imported-legacy-lsh")
        if isinstance(raw, dict)
        else "imported-legacy-lsh",
        "clusters": clusters,
    }


def prepare(raw, documents: list[dict], provenance: dict, top_n: int = 50) -> dict:
    if top_n < 1:
        raise ValueError("top_n must be positive")
    if any(
        not str(provenance.get(k, "")).strip()
        for k in ("dataset", "source", "license", "approved_by")
    ):
        raise ValueError("Dataset source, license and external approval are required")
    if provenance.get("external_allowed") is not True:
        raise ValueError("External export is not approved for this corpus")
    if provenance["license"].lower() in ("unknown", "other", "pending", "на проверке"):
        raise ValueError("Unresolved dataset license")
    result = normalize_campaigns(raw)
    candidates = sorted(
        (c for c in result["clusters"] if c["size"] > 1),
        key=lambda c: (-c["size"], c["cluster_id"]),
    )
    if len(candidates) < top_n:
        raise ValueError(f"Need {top_n} nonsingleton clusters; got {len(candidates)}")
    selected = candidates[:top_n]
    by_id = {r["id"]: r["text"] for r in documents}
    if len(by_id) != len(documents):
        raise ValueError("Duplicate document IDs")
    ids = sorted(m for c in selected for m in c["members"])
    missing = set(ids) - set(by_id)
    if missing:
        raise ValueError(f"Missing source texts for {len(missing)} members")
    docs = [{"id": i, "text": by_id[i]} for i in ids]
    hashes = {r["id"]: document_hash(r["text"]) for r in docs}
    manifest = {
        "schema": "campaign-review-1",
        "selection": "size-desc-id-asc",
        "top_n": top_n,
        "n_documents": len(ids),
        "provenance": provenance,
        "document_hashes": hashes,
        "partition_sha256": canonical_hash(selected),
    }
    manifest["package_id"] = canonical_hash(manifest)
    frozen = {
        "model_version": result["model_version"],
        "clusters": selected,
        "review_package_id": manifest["package_id"],
        "document_hashes": hashes,
    }
    annotations = [
        {
            "id": m,
            "predicted_cluster": c["cluster_id"],
            "text": by_id[m],
            "text_sha256": hashes[m],
            "package_id": manifest["package_id"],
            "gold_group": "",
            "review": "",
            "annotator": "",
            "notes": "",
        }
        for c in selected
        for m in c["members"]
    ]
    return {
        "manifest": manifest,
        "campaigns": frozen,
        "documents": docs,
        "annotations": annotations,
    }


def write_package(package: dict, out: Path) -> None:
    from dw1.campaigns.review_page import render_review

    out.mkdir(parents=True, exist_ok=True)
    for key, name in (("manifest", "manifest.json"), ("campaigns", "campaigns.json")):
        (out / name).write_text(
            json.dumps(package[key], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    for key, name in (("documents", "documents.jsonl"), ("annotations", "annotation.jsonl")):
        (out / name).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in package[key]) + "\n",
            encoding="utf-8",
        )
    (out / "review.html").write_text(render_review(package), encoding="utf-8")
    root = Path(__file__).resolve().parents[3]
    with zipfile.ZipFile(out / "external_bundle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in ("documents.jsonl", "manifest.json"):
            bundle.write(out / name, name)
        bundle.write(root / "scripts/export_embeddings.py", "export_embeddings.py")
        for rel in ("dw1/__init__.py", "dw1/campaigns/__init__.py", "dw1/campaigns/documents.py"):
            bundle.write(root / "src" / rel, "src/" + rel)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaigns", type=Path, required=True)
    ap.add_argument("--documents", type=Path, required=True)
    ap.add_argument("--provenance", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/runs/sprint3/campaigns/review"))
    args = ap.parse_args()
    package = prepare(
        json.loads(args.campaigns.read_text(encoding="utf-8-sig")),
        load_documents(args.documents),
        json.loads(args.provenance.read_text(encoding="utf-8-sig")),
    )
    write_package(package, args.out)
    print(f"Frozen 50 clusters / {len(package['documents'])} messages -> {args.out}")


if __name__ == "__main__":
    main()
