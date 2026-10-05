# Phase 4 results — GraphRAG polish

Date: 2026-09-14. Device: RedMagic 10 Pro (NX789J), Android 15, ~15.5 GB RAM, USB-connected.
Scope: build plan §5 weeks 13–15 — regression eval harness, community summaries + global mode,
memory-ceiling audit + soak, ML Kit backend auto-selection.

## 1. Regression eval harness

- `eval/EvalRunner.kt` owns the full eval loop (was inline in the spike). Loads the newest
  `files/eval/eval-vN.json`, runs all 27 questions in vector and hybrid modes, computes
  recall@5 + gate accuracy overall and per category, plus mean per-channel ms.
- Every run is persisted to `files/eval/history/<yyyyMMdd-HHmmss>.json` with a config
  fingerprint (eval file, question count, MIN_VEC_SCORE, RRF k, entity/edge/chunk counts).
- Each run compares against the newest previous history file: REGRESSION if recall@5 or gate
  accuracy drops > 0.05, else PASS; BASELINE-SET on first run. A config-change note is printed
  when the fingerprint differs from the baseline's.
- `HybridRetrievalSpike` (id `hybrid_eval`, unchanged registration) is now a thin UI wrapper.

**Baseline (20260914-032900.json):** vector recall@5 1.00 (23/23), hybrid 1.00, gate 0.75 both
(97 entities / 503 edges / 12 docs fingerprint). After the global-mode changes the harness was
re-run: **PASS** on both modes — no retrieval regression.

## 2. Community summaries + global mode

- `GraphStore.allEdges()` added (LadybugDB `MATCH (a)-[r]->(b)` + SQLite select).
- `graph/CommunityDetector.kt`: deterministic label propagation over the undirected weighted
  graph (fixed node order, lowest-label tie-break, ≤20 iterations); communities < 3 members
  merged into the strongest-weight neighbor, isolated ones dropped.
- `rag/CommunitySummarizer.kt`: per community, evidence = member entities (by mentions) +
  top-10 internal edges + ≤2 chunk snippets, hard-capped ~600 tokens; LLM (llama.cpp, loaded
  and unloaded around the run) writes a ≤4-sentence thematic summary; summary embedded with
  EmbeddingGemma document mode and persisted in **rag.db v3** (`communities` table).
- Background: `CommunityIndexWorker` (WorkManager, `requiresCharging` + `requiresDeviceIdle`),
  daily periodic KEEP + a one-shot REPLACE enqueued after imports. Manual "Rebuild communities"
  button on the Graph tab for testing (count + last-run timestamp shown).
- **Measured on device:** 97 entities / 503 edges → **7 communities** in ~2.5 min including
  LLM load. Summaries map 1:1 onto the corpus themes and are fully grounded (Japan trip,
  taxes/Cairo apartment, gym program, this project + Droidcon talk, graph-algorithms book,
  car maintenance, sourdough).

### Global retrieval mode

- `rag/retrieve/CommunityRetriever.kt`: cosine of the query embedding over summary embeddings.
- Chat toggle is now Vector → Hybrid → **Global**; global = hybrid channels + community
  summaries prepended as `[C1]…` evidence lines (prompt explains C-lines vs numbered chunks).
- Gate: gated only if the hybrid gate fires AND no community selected.

**Threshold tuning (data-driven, on-device q4f16 scores — local int8 replicas diverge wildly,
do not tune offline):** a topical query ("…about travel?") scores C1 0.313 with the runner-up
at 0.157 (clean separation); an abstract query ("main themes across my notes?") clusters all
7 communities at 0.18–0.29 with no winner. Fixed absolute threshold (0.30) therefore fails the
abstract case. Shipped rule: top hit must clear **floor 0.25**, then take everything **within
0.05 of the top**, cap **5**. Travel → 1 community; themes → 4 communities.
Raw scores are appended to `files/eval/comm-scores.log` for future tuning (logcat is unreliable
on this device — `adb logcat` returns empty).

**Verified:** "What are the main themes across my notes?" (Global) → synthesized answer listing
gym program / car maintenance / this project, drawn from C-lines. "What do my notes say about
travel?" (Global) → Japan-trip answer citing [C1].

## 3. Memory discipline + soak (Android 17 readiness)

- Per-query stats line now ends with `PSS x MB / heap y MB` (one `Debug.getPss()` per query).
- `onTrimMemory`: on `TRIM_MEMORY_MODERATE`+, if no generation is in flight, the LLM is
  unloaded (`AiChat.cleanUp()`; next query lazily reloads). EmbeddingGemma stays loaded.
  GLiNER verified already closed after indexing (`GraphIndexer` finally-block).
- Reference PSS readings on device: ~5.0 GB with Qwen3-1.7B-Q8_0 resident after a global query
  (heap 26 MB). Well inside budget on a 15.5 GB device; the ceiling discipline matters for the
  ~8 GB phones the doc targets.
- `spikes/SoakSpike.kt` (registered): 30-min mixed workload — eval questions alternating
  vector/hybrid, LLM unload/reload every 10 queries, one mid-run re-import, PSS+heap sampled
  every 10 s, JSON report to `files/eval/soak-<ts>.json`.

**Soak result (`eval/soak-20260914-051715.json`, with the FastNative fix):** 30 min,
**105 queries, 0 errors**, 10 LLM unload/reload cycles, 1 mid-run re-import. PSS 1047 MB →
632 MB (Δ **−414 MB** — the LLM is unloaded at soak end; no growth trend across 86 samples),
**peak PSS 4999 MB** while the 1.7B was resident, Java heap peak 30 MB. No termination;
`ApplicationExitInfo` clean after the run.

### Crash postmortem: SIGABRT during the first soak attempt

The first soak run died ~26 min in (`ApplicationExitInfo`: reason APP CRASH(NATIVE), SIGABRT,
RSS 5.0 GB). Tombstone: `SuspendThreadByPeer timed out: DefaultDispatcher-worker-10`, with the
busy thread inside `ggml_compute_forward_flash_attn_ext`. Root cause, two compounding defects
inherited from the vendored llama.cpp Android example:

1. Every llama JNI method (including `processUserPrompt`, which contains multi-second prefill)
   was annotated `@FastNative`, so the calling thread never transitioned to the native state —
   ART saw it as running Java with no safepoint for seconds at a time.
2. Inference ran on `Dispatchers.IO.limitedParallelism(1)` — a view over kotlinx-coroutines'
   shared worker pool. When the pool retires an idle worker it renames a peer via
   `Thread.setName`, and renaming a peer makes ART suspend that thread first. A rename landing
   mid-prefill meant the suspend request went unanswered past the 4 s timeout → ART aborts the
   process by design. (Same crash class documented in the wild for `getAllStackTraces` and
   coroutine worker renames.)

Fix in `llamacpp/.../InferenceEngineImpl.kt`: removed `@FastNative` from all nine JNI entry
points (thread now enters native state during inference; suspend requests honored at the
boundary), and moved all llama work to a dedicated named thread (`llama-inference`, plain
single-thread executor) outside the coroutine pool so the rename vector is gone entirely.
Not soak-specific — any long chat generation could have hit this.

Note: `adb logcat` returns empty buffers on this device/ROM (all buffers, root cause unknown);
native crash forensics went through `dumpsys dropbox --print data_app_native_crash` and
`dumpsys activity exit-info` instead — both work fine.

## 4. ML Kit backend auto-selection

- Settings key `generation_backend = auto | llamacpp | mlkit` (default auto), header toggle
  cycles Auto → llama.cpp → ML Kit. `auto` picks ML Kit only when `checkStatus()` reports
  AVAILABLE; otherwise llama.cpp. On this device (no AICore) auto → llama.cpp — verified
  graceful degradation. GGUF model picker is ignored on the mlkit path.

## Known limitations carried forward

- C1's summary contains a mid-word truncation ("a machiya r.") — LLM output cap artifact;
  cosmetic, fix by raising the summarizer token cap or regenerating.
- Abstract thematic queries include the top-5 communities even when some are marginal — cheap
  (short summaries) and the strict-grounding prompt handles relevance.
- Soak alternates vector/hybrid only (written before global mode existed).
- From Phase 3: gate misses semantically-close hard negatives; FTS is term-coverage not BM25;
  chat history in-memory.
