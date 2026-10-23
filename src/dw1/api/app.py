"""API DW-1 (SPEC §3). Запуск: uvicorn dw1.api.app:app (extras: pip install -e '.[api]')."""
from __future__ import annotations

import tempfile
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, UploadFile
except ImportError as exc:  # pragma: no cover
    raise SystemExit("API требует extras: pip install -e '.[api]'") from exc

from dw1.pipeline import analyze

app = FastAPI(
    title="DW-1 enrichment API",
    version="0.1.0",
    description="Explainable spam-enrichment (M0 skeleton). Evidence JSON v1 — SPEC §4.1",
)


@app.post("/v1/analyze")
async def v1_analyze(file: UploadFile) -> dict:
    """Принимает .eml, возвращает Evidence JSON v1."""
    if not (file.filename or "").lower().endswith(".eml"):
        raise HTTPException(status_code=400, detail="ожидается файл .eml")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "input.eml"
        path.write_bytes(await file.read())
        return analyze(path).to_dict()


@app.get("/v1/message/{message_id}")
async def v1_message(message_id: str) -> dict:
    return {"message_id": message_id, "status": "M0 stub: хранилище evidence появится в M5"}


@app.get("/v1/campaign/{campaign_id}")
async def v1_campaign(campaign_id: str) -> dict:
    return {"campaign_id": campaign_id, "status": "M0 stub: кластеризация появится в M4"}


@app.get("/v1/explain/{message_id}")
async def v1_explain(message_id: str) -> dict:
    return {"message_id": message_id, "status": "M0 stub: объяснения появятся в M6"}
