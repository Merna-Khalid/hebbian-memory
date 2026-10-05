"""
generate_architecture_docx.py — builds the Hebbian Memory architecture document.
Run: python generate_architecture_docx.py  (from project root)
Output: outputs/hebbian_memory_architecture.docx
"""
import os
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

INK   = RGBColor(0x22, 0x22, 0x20)
ACCENT= RGBColor(0x37, 0x8A, 0xDD)
GREY  = RGBColor(0x66, 0x66, 0x60)

doc = Document()

# base style
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(10.5)
style.font.color.rgb = INK

def h1(t):
    p = doc.add_heading(t, level=1)
    for r in p.runs: r.font.color.rgb = ACCENT
    return p

def h2(t):
    p = doc.add_heading(t, level=2)
    for r in p.runs: r.font.color.rgb = INK
    return p

def para(t, italic=False, size=None, color=None):
    p = doc.add_paragraph()
    r = p.add_run(t)
    r.italic = italic
    if size: r.font.size = Pt(size)
    if color: r.font.color.rgb = color
    return p

def bullets(items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")

def mono(t):
    p = doc.add_paragraph()
    r = p.add_run(t)
    r.font.name = "Consolas"
    r.font.size = Pt(9)
    return p

# ── Title ──────────────────────────────────────────────────────────────
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Hebbian Memory")
r.font.size = Pt(28); r.bold = True; r.font.color.rgb = INK

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run("A biologically-inspired memory system for language learning\n"
                "Architecture Documentation")
r.font.size = Pt(12); r.font.color.rgb = GREY

meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = meta.add_run("Version 1.2 — August 2026 (research-informed improvements implemented)")
r.font.size = Pt(9); r.font.color.rgb = GREY

doc.add_paragraph()

# ── 1. Overview ────────────────────────────────────────────────────────
h1("1. Overview")
para("Hebbian Memory replaces static vector RAG with a living associative graph stored "
     "in Neo4j. Memories are Concept nodes connected by ASSOCIATED_WITH edges whose "
     "weights follow Hebbian dynamics: concepts retrieved together are wired together; "
     "concepts that stop co-occurring decay apart. An emotion-modulated learning rate "
     "(ARM-E) decides how deeply each exchange encodes, and a two-tier GNN design "
     "mirrors hippocampal fast learning and cortical slow consolidation.")
para("Two personas share one graph: a Japanese tutor (chat + extraction + practice) and "
     "a second brain fed by a simulated ambient BCI stream (subvocal EMG / heart rate / "
     "attention). The same engine drives the CLI, the FastAPI backend, and the React UI.")
para("The system closes a full learning loop: conversation encodes concepts → Hebbian "
     "weights decay when unused → the fading queue surfaces what is about to be "
     "forgotten → retrieval-practice exercises re-encode it on correct recall.")

# ── 2. Architecture diagram (textual) ─────────────────────────────────
h1("2. Data Flow")
flow = """conversation turn / ambient utterance (decoded EMG — simulated)
      |
      v
BGE-M3 embedding --> retrieve top-k concepts (cosine + Hebbian + causal score)
      |                     |
      v                     v
LFM2.5-1.2B-JP responds*  co-retrieved nodes get pairwise edge reinforcement
      |                   ("retrieved together -> wired together")
      v
ConceptExtractor pulls concepts from the exchange
      |
      v
IngestPipeline: embed -> emotion (text VAD, HR-fused arousal) ->
ARM-E m_t (attention-gated) -> hippocampal GNN
(Oja's rule + eligibility gating + anti-Hebbian decay)
      |
      v
CorticalGNN consolidates on schedule (every 20 ingests / session end)
      |
      v
FadingQueue ranks decaying concepts -> PracticeView generates exercises
-> correct answers reinforce edges -> forgetting timer resets
* chat/response generation only for the tutor persona; ambient ingestion has no reply."""
mono(flow)

# ── 3. Component map ──────────────────────────────────────────────────
h1("3. Components")
comps = [
    ("core/graph_store.py",
     "The only Neo4j interface. Concept CRUD, vector index ANN search, Hebbian edge "
     "upserts, batched weight write-backs, co-retrieval reinforcement, local/global "
     "subgraph pulls, session + ARM-E statistics (Welford online means), and the "
     "fading-concept ranking used by the practice queue."),
    ("core/ingest_pipeline.py",
     "The ingest path: embed -> summarize -> classify emotion -> upsert concept -> "
     "bootstrap edges to similar nodes -> pull local subgraph -> ARM-E step -> GNN "
     "propagation -> modulated Hebbian update -> persist weights/eligibility -> stats."),
    ("core/arme_v2.py",
     "ARM-E: emotion-modulated learning-rate engine. Quadrant-aware emotional signal "
     "e_t (valence x arousal interaction), sparse per-edge eligibility gating, "
     "anti-Hebbian normalization of unreinforced edges, attention gate "
     "(|e_t| * r_t * attention) so unattended content does not deeply encode."),
    ("core/hippocampal_gnn_pyg.py / core/cortical_gnn.py",
     "Fast/slow layers. The hippocampal GNN runs one PyG message-passing conv plus an "
     "Oja-rule weight update outside autograd at every ingest. The cortical GNN "
     "consolidates periodically with backpropagation, updating causal scores."),
    ("core/emotion_classifier.py / core/physio.py",
     "vad-bert text valence/arousal/dominance; PhysioFusedEmotionClassifier fuses "
     "heart-rate-derived arousal with text valence for the ambient path."),
    ("core/signals.py / core/signals_sim.py",
     "SignalSource interface (UtteranceEvent, PhysioSample) any future real BCI rig "
     "implements; SimulatedSignalSource scripts scenarios for development."),
    ("core/agents/concept_extractor.py",
     "LLM-based structured extraction: JSON concepts (typed per persona) + relations, "
     "with retry, validation, and fallback to a raw event node."),
    ("core/tutor_engine.py",
     "Shared engine behind CLI and HTTP API. Sessions, the chat turn (retrieve -> "
     "respond -> extract/ingest -> co-retrieval reinforcement), ambient ingestion, "
     "dashboard read models, domain presets, and the retrieval-practice loop "
     "(next_practice / submit_practice_answer)."),
    ("server.py",
     "FastAPI shell exposing /api/session/*, /api/chat, /api/graph*, /api/stats/*, "
     "/api/practice/*, and /api/second-brain/*. HEBBIAN_STUB=1 swaps deterministic "
     "stub models without downloads."),
    ("hebbian-tutor-ui/",
     "React + Vite frontend: 学習 Chat, 練習 Practice (fading queue sidebar + exercise "
     "cards + grading feedback), 記憶 Memory dashboard (force/hierarchy/timeline graph "
     "views, ARM-E history), and 🧠 Second Brain stream view."),
]
for name, desc in comps:
    h2(name)
    para(desc)

# ── 4. Graph schema ───────────────────────────────────────────────────
h1("4. Graph Schema (Neo4j)")
bullets([
    "(:Concept) — node_id (unique), label, text_raw, text_summary, concept_type, "
    "source_type, source_uri, embedding LIST<FLOAT> (1024-d, bge-m3), activation_count, "
    "lifetime ARM-E means (mean_m_t, mean_dominance, mean_r_t) and a recent_events "
    "circular buffer.",
    "(:Concept)-[:ASSOCIATED_WITH {layer}]->(:Concept) — layer ∈ {hippocampal, cortical}; "
    "properties: hebb_weight, eligibility trace, causal_score, co_activation_count, "
    "last_updated.",
    "(:Session) — trigger, started_at, ended_at, total_activations.",
    "(:SessionStats)-[:TRACKED_IN]->(:Concept), (:Session)-[:AGGREGATES]->(:SessionStats) — "
    "per-(session, concept) running means including delta_w_mean, kept as regression data "
    "for tuning the dominance weight w_dom.",
])
h2("Indexes")
bullets([
    "Uniqueness constraints on Concept.node_id and Session.session_id.",
    "Range indexes on concept_type, source_type, updated_at, started_at.",
    "Vector index concept_embedding (1024 dims, cosine) for ANN search.",
])

# ── 5. Algorithms ─────────────────────────────────────────────────────
h1("5. Core Algorithms")

h2("5.1 Retrieval scoring")
para("Score = α·cos_sim + β·assoc_norm + γ·causal_score  (α=0.6, β=0.3, γ=0.1), "
     "where assoc_norm = clamp(direct/w_ceil + δ·spread, 0, 1) with w_ceil=5.0 and "
     "δ=0.3. 'direct' is the average Hebbian edge weight from session seeds; "
     "'spread' is multi-hop spreading activation through the Hebbian graph from the "
     "same seeds, normalized to [0,1] (each path contributes decay^hops × product of "
     "edge weights). Spreading activation lets indirectly-associated concepts surface "
     "even when their cosine similarity to the query is low — pure vector search "
     "cannot see them.")

h2("5.2 Hebbian update (hippocampal, every ingest)")
para("Co-activation c_ij = h_i · h_j on post-propagation features. Oja baseline with "
     "eligibility gating and anti-Hebbian normalization:")
mono("Δw_ij = m_t·η·c_ij − λ·w_ij·||h_i||²        (eligible edges)\n"
     "g_ij ← g_ij·(1 − 1/τ) + κ·max(c_ij, 0)       (trace update)\n"
     "w_ij ← clamp(w_ij + Δw_ij − λ_anti·w_ij·1[not eligible], floor, ceil)")
para("m_t is global (dopamine-like); eligibility traces are local (synaptic tags). "
     "Traces persist per-edge in Neo4j because subgraph indices are not stable across "
     "ingests.")

h2("5.3 ARM-E m_t")
mono("x_t = [r_t, u_t, δ_t·1[Q1/Q4], e_t, dominance]\n"
     "s_t = w·x_t + b ;  m_t = EMA(clamp(exp(s_t), m_min, m_max))\n"
     "gate: if |e_t|·r_t·attention < θ_gate then m_t = 1.0 (neutral)")
para("Quadrant-aware e_t amplifies positive-valence high-arousal states (curiosity, Q1) "
     "and suppresses negative-valence high-arousal states (confusion/anxiety, Q2), so "
     "wrong associations formed under stress fade instead of encoding.")

h2("5.4 Cortical consolidation")
para("Every 20 ingests and at session end, CorticalGNN trains over the global graph "
     "(edges above min-weight), writes causal scores back, and checkpoints state under "
     "./cortical_state/.")

h2("5.5 Fading queue (adaptive practice scheduling)")
mono("strength   = clamp(avg_hippocampal_weight / w_ceil, 0, 1)\n"
     "staleness  = 1 − exp(−Δt / τ_type)          (type-conditioned)\n"
     "fading     = (1 − strength) · staleness · ln(1 + activation_count)")
para("Candidates are then ranked by predicted recall probability under an adaptive "
     "per-type half-life model, p = 2^(−Δt/h_type) — lowest p practices first — "
     "subject to a spacing guard (no re-serve within 10 minutes) and a "
     "semantic-interference filter (candidates whose embedding cosine with an "
     "already-selected target exceeds 0.82 are deferred to a later exercise, so "
     "confusable items like 食べる/食べた interleave instead of cramming).")
para("The half-lives h_type adapt online from graded outcomes: a correct answer grows "
     "the type's half-life ×1.25, a wrong answer shrinks it ×0.75, clamped to "
     "[0.5, 30] days and persisted to ./practice_state/scheduler.json. This is a "
     "lightweight online variant of half-life regression that personalizes as the "
     "learner practices.")
para("A correct answer also increments activations and reinforces pairwise edges "
     "(m_t = 1.5), refreshing updated_at and moving the concept to the back of the "
     "forgetting curve — retrieval practice feeds directly back into the memory model.")

h2("5.6 Type-conditioned plasticity")
para("Concept types carry decay profiles (type_profiles.py): vocabulary and tasks "
     "erode fast (τ ≈ 1–2 days, λ multiplier 1.5–1.6); grammar rules, cultural "
     "frameworks, and entities are sticky (τ up to 14 days, λ multiplier 0.6–0.8). "
     "The staleness timescale drives practice scheduling; the λ multiplier scales "
     "Oja's decay per edge during ingest-time Hebbian updates, so per-type erosion "
     "happens in the graph itself, not only in the queue.")

h2("5.7 Validation-gated plasticity")
para("When LLM extraction fails schema validation and falls back to a raw event node, "
     "the turn is marked unverified: bootstrap edges start at reduced weight (0.6), "
     "ARM-E m_t is capped at 1.0 (no amplification), and co-retrieval reinforcement "
     "of previously retrieved context is skipped for that turn. Hallucinated or "
     "garbage content can enter memory weakly but must earn strength through "
     "subsequent verified co-activation.")

# ── 6. API surface ────────────────────────────────────────────────────
h1("6. API Surface")
api_rows = [
    ("POST /api/session/start | end", "Session lifecycle"),
    ("POST /api/chat", "Full tutor turn; returns response, ARM-E state, extracted concepts, retrieved context"),
    ("GET  /api/graph", "Graph read model (min_weight, limit, layer, isolated toggle)"),
    ("GET  /api/stats/system | /stats/arme", "Counters and ARM-E history sparkline"),
    ("GET  /api/practice/fading", "Ranked fading concepts (the practice queue)"),
    ("POST /api/practice/next", "Generate one exercise from the most faded concepts"),
    ("POST /api/practice/answer", "Grade answer; correct answers reinforce memory"),
    ("POST /api/second-brain/*", "Simulated ambient stream: session, next, reset"),
]
tbl = doc.add_table(rows=1, cols=2)
tbl.style = "Light Grid Accent 1"
hdr = tbl.rows[0].cells
hdr[0].text = "Endpoint"; hdr[1].text = "Purpose"
for ep, purpose in api_rows:
    row = tbl.add_row().cells
    row[0].text = ep; row[1].text = purpose
for row in tbl.rows:
    for cell in row.cells:
        for p in cell.paragraphs:
            for run in p.runs:
                run.font.size = Pt(9)
                if run.font.name != "Calibri":
                    run.font.name = "Consolas"

# ── 7. Running ────────────────────────────────────────────────────────
h1("7. Running the System")
bullets([
    "Prerequisites: Neo4j 5.11+, Python 3.10+ (torch, torch_geometric, transformers, "
    "sentence-transformers, neo4j, fastapi, uvicorn), Node 18+.",
    "Config: NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD environment variables.",
    "Backend: uvicorn server:app --port 8000 (first run downloads ~3 GB of models; "
    "HEBBIAN_STUB=1 skips downloads with stub embeddings/responses).",
    "Frontend: cd hebbian-tutor-ui && npm install && npm run dev → http://localhost:5173.",
    "CLI: python japanese_tutor.py (chat; /practice command for review rounds) or "
    "python second_brain_sim.py (simulated ambient stream table).",
])

# ── 8. Design decisions & tradeoffs ──────────────────────────────────
h1("8. Key Design Decisions")
bullets([
    "LIST<FLOAT> embeddings + native vector index keep the system on Neo4j Community "
    "Edition (no Enterprise block-format requirement).",
    "Directed edges both ways: bootstrap/relation/co-occurrence inserts create "
    "bidirectional pairs since initialization is symmetric; Hebbian updates diverge "
    "them naturally over time.",
    "ARM-E's EMA only updates on non-gated steps, so weak/noisy turns do not drag the "
    "modulation baseline.",
    "Eligibility traces live in the graph, not in process memory — crash-safe and "
    "consistent across restarts.",
    "Practice grading accepts minor orthography variance (kana variants, width, "
    "punctuation) but judges meaning; partial credit marks incorrect with feedback.",
    "Both personas tag source_type so dashboards can separate tutor ('user') from "
    "sensor ('sensor') concepts on the shared graph.",
])

# ── 9. Roadmap ────────────────────────────────────────────────────────
h1("9. Research-Informed Improvements")
para("The following mechanisms were identified from the 2020–2026 literature and "
     "are now implemented (v1.2).")

roadmap = [
    ("Spreading-activation retrieval — IMPLEMENTED",
     "Multi-hop activation diffusion through the Hebbian graph from session seeds "
     "(GraphStore.spreading_activation), blended into the retrieval score via the "
     "δ·spread term. Modeled on HeLa-Mem's dual-path retrieval, which beat prior "
     "SOTA on LoCoMo with fewer context tokens."),
    ("Validation-gated consolidation — IMPLEMENTED",
     "Unverified turns (extraction fallback) receive weak bootstrap edges, an m_t "
     "cap at 1.0, and no co-retrieval reinforcement, following Kairos's "
     "validation-gated Hebbian learning: hallucinated associations cannot be "
     "reinforced."),
    ("Adaptive half-life scheduling — IMPLEMENTED",
     "Per-concept-type half-lives learned online from graded practice outcomes "
     "(PracticeScheduler, ./practice_state/scheduler.json), replacing the fixed "
     "3-day staleness constant. Online variant of half-life regression; the "
     "SessionStats already collected (m_t, dominance, delta_w_mean) remain "
     "available for a full HLR fit as data accumulates."),
    ("Type-conditioned perishability — IMPLEMENTED",
     "Per-type decay profiles (type_profiles.py) drive both the staleness timescale "
     "and a per-edge Oja-λ multiplier, following ScrubJay-MEM's per-memory, "
     "type-conditioned temporal decay."),
    ("Semantic-interference-aware practice — IMPLEMENTED",
     "Target selection defers candidates whose embedding cosine with an "
     "already-selected target exceeds 0.82, so confusable items interleave across "
     "exercises rather than being drilled together (LECTOR-inspired)."),
    ("Desirable difficulty — PARTIAL (queue signal ready)",
     "Exercise difficulty can scale to current strength × staleness; the fading "
     "queue already exposes both. Not yet wired into exercise generation prompts."),
    ("RL-managed memory policies — FUTURE",
     "Longer-term: learn admission/decay/retrieval policies with RL instead of "
     "hand-set η, λ, θ parameters (Memory-R1 / AgeMem trend, 2025–26)."),
]
for title, desc in roadmap:
    h2(title)
    para(desc)

# ── 10. References ────────────────────────────────────────────────────
h1("10. References")
refs = [
    "[1] Zhu, J., Li, J., Zhang, C., Liu, J., & Yang, M. (2026). HeLa-Mem: Hebbian "
    "Learning and Associative Memory for LLM Agents. Proceedings of ACL 2026, "
    "pp. 13757–13769. arXiv:2604.16839. https://github.com/ReinerBRO/HeLa-Mem",
    "[2] Kairos: Validation-Gated Hebbian Learning for Adaptive Agent Memory. "
    "OpenReview. https://openreview.net/forum?id=EN9VRTnZbK",
    "[3] Zaidi, A., Caines, A., Moore, R., Buttery, P., & Rice, A. (2020). Adaptive "
    "Forgetting Curves for Spaced Repetition Language Learning. AIED 2020, LNCS "
    "12164, pp. 358–363. Springer.",
    "[4] Settles, B., & Meeder, B. (2016). A Trainable Spaced Repetition Model for "
    "Language Learning (Half-Life Regression). ACL 2016.",
    "[5] Zhao, J. (2025). LECTOR: LLM-Enhanced Concept-based Test-Oriented "
    "Repetition for Adaptive Spaced Learning. arXiv:2508.03275.",
    "[6] Bhandari, et al. (2026). Caching for the Future: Scrub Jay Episodic Memory "
    "Principles for Agent Memory Systems. arXiv:2608.04746.",
    "[7] Tfatykhov, et al. awesome-agent-memory: Curated research on memory systems "
    "for LLM agents. https://github.com/tfatykhov/awesome-agent-memory",
    "[8] Kensinger, E. A. (2004). Remembering emotional experiences: The "
    "contribution of valence and arousal. Reviews in the Neurosciences (PMC3530455). "
    "[ARM-E quadrant design]",
    "[9] Oja, E. (1982). Simplified neuron model as a principal component analyzer. "
    "Journal of Mathematical Biology 15, 267–273. [weight decay term]",
    "[10] Moraitis, T., et al. (2022). SoftHebb: Synthetic gradients meet "
    "Hebbian learning. [anti-Hebbian normalization]",
]
for ref in refs:
    p = doc.add_paragraph()
    r = p.add_run(ref)
    r.font.size = Pt(8.5); r.font.color.rgb = GREY

os.makedirs("outputs", exist_ok=True)
out = "outputs/hebbian_memory_architecture.docx"
doc.save(out)
print(f"saved {out}")
