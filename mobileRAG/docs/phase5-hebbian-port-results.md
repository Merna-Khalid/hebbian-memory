# Phase 5 results — Hebbian Memory port

Date: 2026-09-15. Scope: port the entire Python "Hebbian Memory" system (`Hebbian Memory/core/`)
into the app — Hebbian graph store, ARM-E, hippocampal GNN, ingest pipeline, concept extraction,
practice scheduler, both personas (Japanese tutor + second brain), and persona UI. Build:
`:app:assembleDebug` SUCCESSFUL (all new code compiles; existing RAG pipeline unaffected except
the LLM-residency coordination below).

Decisions going in (user-approved): cortical GAT consolidation **deferred** (interface-compatible
stub; lands with the build plan's §6 GNN phase); emotion model **pluggable with stub default**
(no 1.1 GB XLM-R / DistilRoBERTa ONNX in this pass); **full persona UI**.

## 1. What was ported

| Python (`Hebbian Memory/core/`) | Kotlin (`app/.../com/mobilerag/hebbian/`) | Notes |
|---|---|---|
| `type_profiles.py` | `TypeProfiles.kt` (52) | verbatim constants |
| `arme_v2.py` | `Arme.kt` (403) | quadrant e_t, r_t/u_t/δ_t, EMA m_t, confidence gate, Oja + eligibility + anti-Hebbian |
| `hippocampal_gnn_pyg.py` | `HippocampalGnn.kt` (46) | `propagate()` only (production path, as in Python) |
| `graph_store.py` (1177, Neo4j) | `store/HebbianStore.kt` (551) + `store/SqliteHebbianStore.kt` (690) + `store/LadybugHebbianStore.kt` (775) + `store/HebbianStoreFactory.kt` (58) | all 17 query patterns; separate DB files (`hebbian.lbug` / `hebbian_graph.db`) |
| `signals.py` / `signals_sim.py` | `signals/Signals.kt` (33) + `signals/SimulatedSignalSource.kt` (129) | 8 scripted scenarios; Box–Muller noise |
| `emotion_classifier.py` | `emotion/EmotionClassifier.kt` (99) | interface + keyword stub + confidence shrink rule |
| `physio.py` | `emotion/Physio.kt` (64) | HR→arousal tanh + fused classifier |
| `practice_scheduler.py` | `PracticeScheduler.kt` (172) | HLR-lite, JSON state cross-readable with Python |
| `consolidation_scheduler.py` | `ConsolidationScheduler.kt` (85) | **stub** (same surface, no GAT) |
| `ingest_pipeline.py` | `IngestPipeline.kt` (377) | full 10-step ingest flow |
| `agents/concept_extractor.py` | `ConceptExtractor.kt` (464) | prompts diffed verbatim; retry/parse/fallback logic |
| `tutor_engine.py` (948) | `TutorEngine.kt` (778) + `HebbianStubEngine.kt` (182) + `HebbianEngineFactory.kt` (132) | both personas, practice flow, ambient ingest |
| `model_setup.py` | replaced | EmbeddingGemma (768-d) + llama.cpp Qwen3 via `llm/HebbianLlmSession.kt` |
| `server.py` / CLIs / React UI | `hebbian/ui/` (10 files) + 4 new tabs (Tutor / Practice / Brain / Memory) | Compose, `MainActivity` now ScrollableTabRow |
| — | `spikes/HebbianMathSpike.kt`, `HebbianIngestSpike.kt`, `SecondBrainSimSpike.kt` | on-device verification, registered in SpikeRegistry |

Not ported (by design): `train_vad.py`, `raw_hebbian_gnn.py` (legacy), notebooks,
`generate_architecture_docx.py`, FastAPI server, CLIs, cortical GAT (`cortical_gnn.py`).

## 2. Golden-value verification (math parity with Python)

`docs/hebbian_golden.py` runs the real `arme_v2.py` / `hippocampal_gnn_pyg.py` (torch 2.10,
conda env `RAG`) on fixed deterministic inputs; outputs pinned in
`docs/hebbian-golden-values.txt`. The Kotlin port was compared offline (kotlinc + driver):
**60/60 checks within tolerance** (max diff ~3e-8; Python is float32, Kotlin uses Double — spike
asserts use 1e-5). The on-device `hebbian_math` spike replays the same scenario (85 checks)
through the real `Arme`/`HippocampalGnn`/`TypeProfiles`.

## 3. Key adaptations (Python → Android)

- **LLM session semantics.** The vendored InferenceEngine is a *stateful* chat session: system
  prompt only settable right after `loadModel`, KV cache accumulates across turns. Python makes
  stateless per-call calls with per-call system prompts. Mapping (`llm/HebbianLlmSession.kt`):
  one persona system prompt per domain loaded once; memory section + practice/extraction
  instructions go **in-band in the user turn**; automatic conversation reset after 12 KV turns
  (build plan's prefill discipline — tutor memory lives in the graph, not the context).
- **Shared model residency.** `generation/LlmResidency.kt` records which (model, system-prompt)
  is resident in the process-wide engine singleton; `RagPipeline` and the tutor check the key so
  switching tabs (RAG grounding prompt vs tutor persona on the same GGUF) reloads correctly
  instead of generating with the wrong system prompt. `ThinkTagFilter` moved to
  `generation/` and shared.
- **Embeddings 1024-d BGE-M3 → 768-d EmbeddingGemma** (`featureDim` configurable; concept embed =
  document mode, retrieval = query mode). Vector search is brute-force cosine over an in-memory
  cache in both stores (no Ladybug vector index needed at this scale).
- **Graph algorithms in shared kernels.** `get_local_subgraph` is a filtered BFS (matches
  Python's no-APOC fallback, not the APOC path — wrong-layer/low-weight edges can't bridge);
  `spreading_activation` runs the same Kotlin kernel in both stores so SQLite and Ladybug return
  identical results (Neo4j/Ladybug path-enumeration order differs). Timestamps are epoch-millis
  Long (Python float seconds; converted only at `staleness()` and recent-events entries).
- **Emotion** defaults to the keyword stub + confidence shrink-to-neutral; `PhysioFusedEmotionClassifier`
  overrides arousal from HR when a `SignalSource` is present (second-brain persona). ONNX VAD
  model drops in later behind the same interface.
- **Practice state** JSON keys match Python exactly (`halflife_days`, `last_practiced`,
  `saved_at`) — state files are cross-readable.

## 4. On-device verification (2026-09-15, RedMagic 10 Pro, 16 GB)

- `hebbian_math`: **PASS — 81/81** golden values within 1e-5 of the Python torch reference.
- `hebbian_ingest`: **PASS — 6/6 on BOTH store backends** (sqlite-hebbian and ladybugdb-hebbian):
  concepts stored, 30/30 edge weights moved, fading queue populated, practice round trip graded,
  half-life adapted 3.0 → 3.75 days. **Both store-backed spikes call `store.clear()` on the shared
  Hebbian store — they wipe real Hebbian data** (logged).
- `second_brain_sim`: **PASS — 5/5** (8/8 ingests; ARM-E contract gated ⇒ m_t = 1.0 holds).
- `hybrid_eval`: **PASS vs baseline 20260914-040002** — recall@5 1.00 both modes; the port did not
  regress the RAG pipeline.
- Real tutor turns: Qwen3-1.7B (JA explanation + `[fact] tabemasu`/`[fact] taberu` chips) and
  **Qwen3-8B Q4_K_M** (richer polite-form explanation + `polite_form` concept) — full
  retrieve → stream → extract → ingest loop with ARM-E state line on both.

### Two LadybugDB bugs found by the on-device runs (fixed)

1. **Second-instance mmap failure**: `Database(path)` mmaps a ~256 GB region; the entity graph
   store already holds one, so the Hebbian store's init failed (`Buffer manager exception: Mmap
   for size 274877906944 failed`). Fixed by constructing with `SystemConfig` —
   `bufferPoolSize = 256 MB`, `maxDBSize = 8 GB` (`LadybugHebbianStore.open`).
2. **Java API can't decode ARRAY columns**: reading `embedding DOUBLE[]` threw
   `RuntimeException: Type of value is not supported in value_get_value` (lbug 0.20.3).
   Embeddings are now stored as fixed-point CSV `STRING` and parsed in Kotlin.
   Diagnosis was possible because `HebbianStoreFactory` persists its choice + error to
   `files/hebbian/store-choice.txt` (logcat is unreliable on this device).

### Model upgrade

Qwen3-8B Q4_K_M (5.03 GB) pushed to `files/models/`; default selection is now saved-pref →
"8B" → "1.7B" → first (both `RagPipeline.selectedModel` and `HebbianEngineFactory`). First 8B
tutor turn (load + chat + summary + extraction) took ~9.5 min wall — dominated by first mmap load
and three sequential LLM calls per turn; prefill at 8B is the cost the build plan warns about.
Mitigations to consider: shorter extraction/summary max-tokens, skipping summaries for short
concepts, or keeping 1.7B for background extraction.

## 5. Known gaps / follow-ups

- `IngestResult` doesn't carry `r_t` (ArmeHistoryEntry substitutes Δw̄); `HebbianStore` has no
  dedicated per-node edge query (nodeDetail uses a 1-hop subgraph); `system_stats()` not ported.
- `buildStubEngine` takes no `SignalSource` (second-brain physio fusion only via the real
  factory or manual wiring, as the sim spike does).
- Second-brain tab needs the GGUF loaded too (extraction runs through the LLM); first Step pays
  the load. Tutor↔Brain↔Chat tab switches reload the model (different system prompts).
- ARM-E gating is conservative with the stub emotion classifier (neutral keywords → low
  confidence → `gated=true`, m_t=1.0) — expected until a real VAD model drops in.
- Deferred: cortical GAT consolidation (stub is wired; build plan §6 matrix-op GNN → ONNX
  Runtime), real VAD emotion model, Text2Cypher equivalents.
