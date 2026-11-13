"""Auditable declarations for human gold; this module never invents human labels.

The protocol records participants' declarations. Passing these checks does not prove
that people actually annotated independently; that remains a human responsibility.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

PROTOCOL = "dw1-human-independent-v1"
PUBLIC_FIELDS = (
    "id",
    "predicted_cluster",
    "package_id",
    "text_sha256",
    "gold_group",
    "review",
    "annotator",
    "annotation_source",
    "protocol",
    "human_confirmed",
    "stage",
    "independent_confirmed",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_human_rows(rows: list[dict], independent: bool = True) -> str:
    """Validate a declared independent review or adjudicated final; return its author.

    Population coverage and agreement with the frozen manifest are checked separately
    by campaign_review.validate_rows. Human provenance is deliberately explicit: legacy
    files, proxy rows and missing attestations are never upgraded automatically.
    """
    if not rows:
        raise ValueError("Human annotations must contain all required messages")
    authors: set[str] = set()
    ids: set[str] = set()
    packages: set[str] = set()
    expected_stage = "independent" if independent else "adjudicated"
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Human annotation rows must be objects")
        if row.get("annotation_source") != "human":
            raise ValueError(
                "Human gold requires annotation_source='human'; proxy/unknown rejected"
            )
        if row.get("protocol") != PROTOCOL:
            raise ValueError("Human annotation protocol is missing or unsupported")
        if row.get("human_confirmed") is not True:
            raise ValueError("Human confirmation must be explicitly true")
        if row.get("stage") != expected_stage:
            raise ValueError(f"Expected human annotation stage '{expected_stage}'")
        if independent and row.get("independent_confirmed") is not True:
            raise ValueError("Independent reviewers must explicitly confirm independent work")
        author = row.get("annotator")
        if not isinstance(author, str) or not author.strip() or author != author.strip():
            raise ValueError("Human annotator must be a nonempty trimmed string")
        if author.casefold().startswith("proxy-"):
            raise ValueError("Known proxy reviewer IDs cannot be used as human gold")
        authors.add(author)
        doc_id = row.get("id")
        if not isinstance(doc_id, str) or not doc_id or doc_id in ids:
            raise ValueError("Human annotation IDs must be nonempty and unique")
        ids.add(doc_id)
        package_id = row.get("package_id")
        if not isinstance(package_id, str) or not package_id:
            raise ValueError("Human annotation package_id is missing")
        packages.add(package_id)
        group = row.get("gold_group")
        if not isinstance(group, str) or not group.strip() or group != group.strip():
            raise ValueError("Human gold_group must be a nonempty trimmed string")
        if row.get("review") not in ("near-dup", "different"):
            raise ValueError("Human review must be near-dup or different")
    if len(authors) != 1:
        raise ValueError("Each human annotation file must contain one annotator")
    if len(packages) != 1:
        raise ValueError("Human annotations mix different package IDs")
    return next(iter(authors))


def partition_changes(reference: list[dict], candidate: list[dict]) -> list[str]:
    """Return affected IDs, comparing group membership rather than arbitrary names."""

    def partition(rows):
        groups: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            groups[row["gold_group"]].add(row["id"])
        return {row["id"]: frozenset(groups[row["gold_group"]]) for row in rows}

    if len({r["id"] for r in reference}) != len(reference):
        raise ValueError("Reference annotations contain duplicate IDs")
    if len({r["id"] for r in candidate}) != len(candidate):
        raise ValueError("Candidate annotations contain duplicate IDs")
    a, b = partition(reference), partition(candidate)
    if set(a) != set(b):
        raise ValueError("Final and reviewer populations differ")
    return sorted(i for i in a if a[i] != b[i])


def validate_adjudication(
    record: dict,
    package_id: str,
    reviewer_paths: list[Path],
    final_path: Path,
    adjudicated_by: str = "",
    required_ids: list[str] | None = None,
) -> dict:
    """Bind a human adjudication declaration to the exact review/final files.

    Required JSON fields: package_id, reviewer_annotations_sha256 (2–3 hashes),
    final_annotations_sha256, adjudicated_by and human_confirmed=true. A supplied
    protocol must match PROTOCOL. When required_ids is supplied, decisions must cover
    every affected ID with a nonempty human reason. Each decision is {ids: [...], reason:
    "..."}. Paths are not stored, so records remain portable.
    """
    if not isinstance(record, dict) or record.get("human_confirmed") is not True:
        raise ValueError("Adjudication needs an explicit human confirmation")
    if record.get("package_id") != package_id:
        raise ValueError("Adjudication belongs to another package")
    if record.get("protocol", PROTOCOL) != PROTOCOL:
        raise ValueError("Unsupported adjudication protocol")
    actual = [file_sha256(Path(path)) for path in reviewer_paths]
    supplied = record.get("reviewer_annotations_sha256")
    if (
        not 2 <= len(actual) <= 3
        or not isinstance(supplied, list)
        or not all(isinstance(value, str) for value in supplied)
        or sorted(supplied) != sorted(actual)
    ):
        raise ValueError("Adjudication reviewer file hashes do not match")
    final_hash = file_sha256(Path(final_path))
    if record.get("final_annotations_sha256") != final_hash:
        raise ValueError("Adjudication final annotation hash does not match")
    author = record.get("adjudicated_by")
    if not isinstance(author, str) or not author.strip() or author != author.strip():
        raise ValueError("Adjudication must name a human adjudicator")
    if adjudicated_by and author != adjudicated_by.strip():
        raise ValueError("CLI adjudicator differs from the adjudication record")
    decisions = record.get("decisions", [])
    if not isinstance(decisions, list):
        raise ValueError("Adjudication decisions must be a list")
    covered: set[str] = set()
    normalized = []
    if decisions or required_ids:
        final_rows = [
            json.loads(line)
            for line in final_path.read_text(encoding="utf-8-sig").split("\n")
            if line.strip()
        ]
        population = {row["id"] for row in final_rows}
        for decision in decisions:
            if not isinstance(decision, dict):
                raise ValueError("Each adjudication decision must be an object")
            ids = decision.get("ids")
            reason = decision.get("reason")
            if (
                not isinstance(ids, list)
                or not ids
                or not all(isinstance(i, str) and i for i in ids)
                or len(set(ids)) != len(ids)
                or covered.intersection(ids)
                or not set(ids).issubset(population)
            ):
                raise ValueError("Adjudication decision IDs must be valid and cannot repeat")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("Each adjudication decision needs a human reason")
            covered.update(ids)
            normalized.append({"ids": sorted(ids), "reason": reason.strip()})
    if required_ids is not None and not set(required_ids).issubset(covered):
        raise ValueError("Adjudication decisions do not cover every disputed/changed ID")
    return {
        "protocol": PROTOCOL,
        "package_id": package_id,
        "reviewer_annotations_sha256": sorted(actual),
        "final_annotations_sha256": final_hash,
        "adjudicated_by": author,
        "human_confirmed": True,
        "decisions": normalized,
    }


def public_annotations(rows: list[dict]) -> list[dict]:
    """Whitelist structured annotation fields; never export email text or notes."""
    return [{field: row[field] for field in PUBLIC_FIELDS if field in row} for row in rows]


def public_adjudication(record: dict | None) -> dict | None:
    """Keep decision evidence without publishing potentially quoted email fragments."""
    if record is None:
        return None
    return {
        **{
            key: record[key]
            for key in (
                "protocol",
                "package_id",
                "reviewer_annotations_sha256",
                "final_annotations_sha256",
                "adjudicated_by",
                "human_confirmed",
            )
        },
        "decisions": [
            {
                "ids": decision["ids"],
                "reason_sha256": hashlib.sha256(decision["reason"].encode("utf-8")).hexdigest(),
            }
            for decision in record["decisions"]
        ],
    }
