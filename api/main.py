"""API REST SANAD — uvicorn api.main:app --reload"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from sanad import __version__
from sanad.checklist_pdf import render_checklist_pdf
from sanad.config import settings
from sanad.graph import load_agent, mermaid
from sanad.guardrails import mask_pii
from sanad.schemas import AskRequest, AskResponse, Checklist

log = logging.getLogger("sanad.api")
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        STATE["agent"] = load_agent()
    except FileNotFoundError as e:
        log.error("%s", e)
    yield


app = FastAPI(title="SANAD API", version=__version__, lifespan=lifespan,
              description="Assistant IA agentique pour les démarches administratives tunisiennes (FR/AR/Derja).")


@app.get("/health")
def health():
    manifest = settings.index_dir / "manifest.json"
    return {"status": "ok" if "agent" in STATE else "index_missing", "version": __version__,
            "llm": settings.llm_model, "embeddings": settings.embed_model,
            "index": json.loads(manifest.read_text()) if manifest.exists() else None}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    agent = STATE.get("agent")
    if agent is None:
        raise HTTPException(503, "Index non construit : lancez `python -m sanad.ingest`.")
    history = [t.model_dump() for t in req.history]
    try:
        res = await run_in_threadpool(agent.ask, req.question, history, req.checklist)
    except Exception as e:  # Ollama indisponible, etc.
        log.exception("ask failed")
        raise HTTPException(502, f"Erreur du moteur LLM : {type(e).__name__}") from e
    log.info("q=%r status=%s latency=%.2fs", mask_pii(req.question)[:80], res.status, res.latency_s)
    return res


@app.post("/checklist/pdf")
def checklist_pdf(checklist: Checklist):
    return Response(render_checklist_pdf(checklist), media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="sanad_checklist.pdf"'})


@app.get("/graph", response_model=str)
def graph():
    return mermaid()
