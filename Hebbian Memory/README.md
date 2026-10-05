# Hebbian Memory

A biologically-inspired memory system, with two coexisting personas built
on top of it: a **Japanese tutor** and a **second brain** for ambient
thought (currently fed by a simulated subvocal-EMG / heart-rate /
attention signal, standing in for a BCI rig still being built). Instead
of plain vector RAG, memories live in a Neo4j graph where associations
strengthen and decay Hebbian-style, concepts you revisit stay strongly
weighted, concepts you abandon slowly fade, and an emotion-modulated
learning rate (ARM-E) decides how deeply each turn gets encoded.

## How it works

```
conversation turn / ambient utterance (decoded EMG — simulated for now)
      │
      ▼
BGE-M3 embedding ──► retrieve top-k concepts (cosine + Hebbian + causal score)
      │                     │
      ▼                     ▼
LFM2.5-1.2B-JP responds*  co-retrieved nodes get pairwise edge reinforcement
      │                   ("retrieved together → wired together")
      ▼
ConceptExtractor pulls concepts from the exchange
      │
      ▼
IngestPipeline: embed → emotion (text valence + HR-fused arousal) →
ARM-E m_t (gated by attention) → hippocampal GNN
(Oja's rule + eligibility gating + anti-Hebbian decay)
      │
      ▼
CorticalGNN consolidates on a schedule (every 20 ingests / session end)
```
\* chat/response generation only happens for the Japanese-tutor persona —
ambient second-brain ingestion has no reply, just extraction + encoding.

- `core/graph_store.py` — the only Neo4j interface (incl. multi-hop
  `spreading_activation` and the `get_fading_concepts` practice queue)
- `core/type_profiles.py` — per-concept-type decay profiles: vocabulary
  fades fast, grammar rules are sticky; drives both staleness scoring and
  a per-edge Oja-λ multiplier
- `core/practice_scheduler.py` — adaptive per-type half-lives learned
  from graded practice outcomes (online half-life regression variant),
  persisted under `./practice_state/`
- `core/ingest_pipeline.py` — ingest path (embed → ARM-E → Hebbian update);
  validation-gated: unverified turns get weak bootstrap edges, capped m_t
- `core/arme_v2.py` — emotion-modulated learning rate, including the
  attention gate (`_compute_signal_confidence`)
- `core/hippocampal_gnn_pyg.py` / `core/cortical_gnn.py` — fast/slow layers
- `core/signals.py` — `SignalSource` interface (`UtteranceEvent`,
  `PhysioSample`) that any ambient input adapter implements
- `core/signals_sim.py` — `SimulatedSignalSource`, scripted scenarios
  standing in for the real EMG/HR/attention rig
- `core/physio.py` — heart-rate → arousal mapping and
  `PhysioFusedEmotionClassifier` (text valence + HR-derived arousal)
- `core/tutor_engine.py` — shared engine behind the CLI and the HTTP API;
  `DOMAIN_PRESETS` holds the `japanese_tutor` / `second_brain` personas,
  `ingest_ambient()` is the second brain's entry point
- `server.py` — FastAPI backend for the UI (both personas)
- `japanese_tutor.py` — terminal chat client (Japanese tutor persona)
- `second_brain_sim.py` — terminal driver for the simulated ambient
  stream (second brain persona)
- `hebbian-tutor-ui/` — React + Vite web UI (chat, memory dashboard,
  and the second-brain stream/physio view)

## Prerequisites

- **Neo4j 5.11+** running locally (vector indexes require ≥ 5.11).
  Neo4j Desktop works; default expected at `bolt://localhost:7687`.
- **Python 3.10+** with:

  ```bash
  pip install torch torch_geometric transformers sentence-transformers \
              neo4j fastapi uvicorn matplotlib networkx
  ```

  On Apple Silicon the models run on MPS automatically.
- **Node 18+** for the frontend.

## Configuration

Environment variables (defaults shown):

```bash
export NEO4J_URI="neo4j://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
```

## Running the web app

Two terminals, both from the project root.

Terminal 1 — backend:

```bash
uvicorn server:app --port 8000
```

First run downloads ~3 GB of models (BGE-M3, LFM2.5-1.2B-JP).

For a quick UI check **without downloading models**:

```bash
HEBBIAN_STUB=1 uvicorn server:app --port 8000
```

Stub mode uses deterministic fake embeddings and template responses —
the memory graph, ARM-E gating, and dashboard all still work. Neo4j is
still required.

Terminal 2 — frontend:

```bash
cd hebbian-tutor-ui
npm install        # first time only
npm run dev        # → http://localhost:5173
```

The Vite dev server proxies `/api` → `localhost:8000`, so just open
http://localhost:5173.

- **学習 Chat** — talk to the tutor; each assistant turn shows its ARM-E
  state (quadrant, m_t, VAD) and the concepts extracted into memory.
- **記憶 Memory** — the memory graph: force / hierarchy / timeline views,
  min-weight filter, isolated-node toggle, ARM-E history sparkline.
- **🧠 Second Brain** — the simulated ambient stream: Step / Play / Reset
  through scripted BCI scenarios, a live physio panel (heart rate +
  attention sparklines), and the memory graph filtered to
  `source_type: "sensor"` concepts. Runs against its own `TutorEngine`
  instance sharing the same Neo4j graph.

## Running the CLI instead

Japanese tutor:

```bash
python japanese_tutor.py
```

Type in English or Japanese; `quit` to exit. Sessions persist in Neo4j —
run again any time and the memory is still there.

Second brain (simulated ambient stream):

```bash
python second_brain_sim.py
```

Steps through the same scripted scenarios as the UI's Second Brain tab,
printing a table of scenario / quadrant / m_t / attention / gated / Δw
per ingest. No real EMG/HR/attention hardware yet — see
`core/signals.py` for the interface the real device will implement.

## Notes

- Always run Python entry points **from the project root** — model paths
  (e.g. `./core/vad_model`) are relative to it.
- Cortical consolidation runs synchronously every 20 ingests and at
  session end; state persists under `./cortical_state/`.
- Standalone demos (no Neo4j needed): `python core/arme_v2.py`,
  `python core/hippocampal_gnn_pyg.py` — both write plots to `outputs/`.
- **Attention gating**: ARM-E's signal-confidence gate
  (`abs(e_t) * r_t * attention`) means emotionally-loaded ambient content
  won't deeply encode if attention was low when it happened — a signal
  the text-only Japanese-tutor path doesn't have (it defaults
  `attention=1.0`, a no-op). See `core/arme_v2.py`'s `ARME.step()`.
- Both personas share one Neo4j graph; second-brain concepts are tagged
  `source_type: "sensor"` so the UI/queries can tell them apart from the
  tutor's `"user"`/`"document"`/etc.-sourced concepts.
- **Retrieval practice**: the 練習 Practice tab (or `/practice` in the CLI)
  generates exercises from concepts the adaptive scheduler predicts you're
  about to forget (`p = 2^(−Δt/h_type)`). Correct answers reinforce the
  underlying Hebbian edges and grow that concept type's half-life; wrong
  answers shrink it. Confusable items are interleaved, not crammed, and
  retrieval scoring includes multi-hop spreading activation through the
  Hebbian graph. See `outputs/hebbian_memory_architecture.docx` §9–10 for
  the research this is based on.
