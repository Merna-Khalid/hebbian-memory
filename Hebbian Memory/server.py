"""
server.py — HTTP API for the Hebbian tutor UI.

Serves exactly the endpoints hebbian-tutor-ui/src/api.ts expects,
under /api. Thin shell over core.tutor_engine.TutorEngine.

Run from the project root:

    uvicorn server:app --port 8000

Modes:
    default          production engine (BGE-M3 + LFM2.5 + vad-bert,
                     downloads ~3GB of models on first run)
    HEBBIAN_STUB=1   stub models — no downloads, still needs Neo4j.
                     Good for checking the UI end-to-end.

Config via env: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD.

The Vite dev server proxies /api → localhost:8000 (see vite.config.ts),
so no CORS is needed in dev; CORS is enabled anyway for direct calls.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.tutor_engine import TutorEngine, build_engine, build_stub_engine
from core.signals_sim import SimulatedSignalSource


# ── App + engine lifecycle ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    stub = os.environ.get("HEBBIAN_STUB") == "1"

    # second-brain's SimulatedSignalSource doubles as both the ambient
    # utterance source /api/second-brain/next steps through, and the
    # physio source PhysioFusedEmotionClassifier reads HR from.
    app.state.sb_sim = SimulatedSignalSource()

    if stub:
        print("HEBBIAN_STUB=1 — starting with stub models")
        app.state.engine    = build_stub_engine()
        app.state.sb_engine = build_stub_engine(
            domain="second_brain", signal_source=app.state.sb_sim,
        )
    else:
        app.state.engine    = build_engine()
        app.state.sb_engine = build_engine(
            domain="second_brain", signal_source=app.state.sb_sim,
        )
    yield
    for attr in ("engine", "sb_engine"):
        engine = getattr(app.state, attr, None)
        if engine is not None:
            engine.close()


app = FastAPI(title="Hebbian Tutor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_engine(request: Request) -> TutorEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(503, "engine not ready")
    return engine


def get_sb_engine(request: Request) -> TutorEngine:
    engine = getattr(request.app.state, "sb_engine", None)
    if engine is None:
        raise HTTPException(503, "second-brain engine not ready")
    return engine


# ── Request bodies ────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message    : str
    session_id : str

class SessionEndRequest(BaseModel):
    session_id : str


# ── Session ───────────────────────────────────────────────────────────

@app.post("/api/session/start")
def session_start(request: Request):
    sid = get_engine(request).start_session(trigger="api")
    return {"session_id": sid}


@app.post("/api/session/end")
def session_end(body: SessionEndRequest, request: Request):
    get_engine(request).end_session(body.session_id)
    return {"ok": True}


# ── Chat ──────────────────────────────────────────────────────────────

@app.post("/api/chat")
def chat(body: ChatRequest, request: Request):
    if not body.message.strip():
        raise HTTPException(400, "empty message")
    return get_engine(request).chat(body.message, body.session_id)


# ── Graph ─────────────────────────────────────────────────────────────

@app.get("/api/graph")
def graph(
    request: Request,
    min_weight: float = Query(0.5, ge=0.0),
    limit:      int   = Query(200, ge=1, le=5000),
    layer:      str   = Query("hippocampal"),
    include_isolated: bool = Query(True),
):
    return get_engine(request).graph_data(
        min_weight=min_weight, limit=limit, layer=layer,
        include_isolated=include_isolated,
    )


@app.get("/api/graph/node/{node_id}")
def graph_node(node_id: str, request: Request):
    node = get_engine(request).node_detail(node_id)
    if node is None:
        raise HTTPException(404, f"no concept {node_id}")
    return node


# ── Stats ─────────────────────────────────────────────────────────────

@app.get("/api/stats/system")
def stats_system(request: Request):
    return get_engine(request).system_stats()


@app.get("/api/stats/arme")
def stats_arme(request: Request, limit: int = Query(50, ge=1, le=500)):
    return get_engine(request).arme_history(limit=limit)


# ── Retrieval practice (forgetting-curve exercises) ───────────────────

@app.get("/api/practice/fading")
def practice_fading(request: Request, limit: int = Query(10, ge=1, le=50)):
    """Concepts the learner is about to forget — the practice queue."""
    return get_engine(request).fading_concepts(limit=limit)


@app.post("/api/practice/next")
def practice_next(body: SessionEndRequest, request: Request):
    """
    Generate one exercise from the most faded concepts.
    Returns {"none": true} when there's nothing to practice yet.
    """
    exercise = get_engine(request).next_practice(body.session_id)
    if exercise is None:
        return {"none": True}
    return exercise


class PracticeAnswerRequest(BaseModel):
    session_id : str
    answer     : str


@app.post("/api/practice/answer")
def practice_answer(body: PracticeAnswerRequest, request: Request):
    if not body.answer.strip():
        raise HTTPException(400, "empty answer")
    result = get_engine(request).submit_practice_answer(
        body.session_id, body.answer.strip(),
    )
    if result is None:
        raise HTTPException(409, "no pending exercise — call /api/practice/next first")
    return result


# ── Second brain (simulated BCI) ────────────────────────────────────────

@app.post("/api/second-brain/session/start")
def sb_session_start(request: Request):
    sid = get_sb_engine(request).start_session(trigger="api")
    return {"session_id": sid}


@app.post("/api/second-brain/session/end")
def sb_session_end(body: SessionEndRequest, request: Request):
    get_sb_engine(request).end_session(body.session_id)
    return {"ok": True}


class SecondBrainNextRequest(BaseModel):
    session_id: str


@app.post("/api/second-brain/next")
def sb_next(body: SecondBrainNextRequest, request: Request):
    """
    Advance the simulated signal source by one utterance and ingest it.
    Returns {"done": true} once the scripted scenario list is exhausted —
    call /api/second-brain/reset to start over.
    """
    sim = request.app.state.sb_sim
    utt = sim.next_utterance()
    if utt is None:
        return {"done": True}

    physio = sim.current_physio()
    result = get_sb_engine(request).ingest_ambient(utt, physio, body.session_id)
    return {
        "done"          : False,
        "scenario_label": sim.current_label,
        **result,
    }


@app.post("/api/second-brain/reset")
def sb_reset(request: Request):
    request.app.state.sb_sim.reset()
    return {"ok": True}
