# -*- coding: utf-8 -*-
"""Сквозной пайплайн: ingest -> экстракторы -> фузия -> кампании -> объяснения."""
from __future__ import annotations

from pathlib import Path

from dw1.campaigns.cluster import cluster
from dw1.evidence import Evidence
from dw1.explain.explain import explain
from dw1.extractors import get_extractors
from dw1.ingest.mime import parse_eml
from dw1.models.fusion import fuse


def analyze(eml_path: str | Path) -> Evidence:
    """Разбирает письмо и строит Evidence JSON v1 (SPEC §3, §4.1)."""
    ingested = parse_eml(eml_path)
    parts = ingested["parts"]

    signals = {ex.name: ex.extract(parts) for ex in get_extractors()}
    fusion = fuse(signals)
    campaign = cluster(parts)
    explanations = explain(signals, fusion, campaign)

    evidence = Evidence(
        message_id=ingested["raw_sha256"][:16],
        ingest={
            "source": str(eml_path),
            "received_at": "",
            "raw_sha256": ingested["raw_sha256"],
        },
        parts=parts,
        signals=signals,
        fusion=fusion,
        campaign=campaign,
        explanations=explanations,
    )
    evidence.validate()
    return evidence
