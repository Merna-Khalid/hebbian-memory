# Phase 3 — Hybrid Retrieval: results

Date: 2026-09-14. Device: RedMagic 10 Pro (NX789J), Android 15.
Status: **complete; headline finding is an honest tie — see "Default-mode decision".**

## What was built

Three-channel retrieval with RRF fusion, per the plan:

- `rag/retrieve/` — `Retriever` interface + `RetrievalHit`, with channels:
  - `VectorRetriever` ("vec") — existing EmbeddingGemma top-10.
  - `FtsRetriever` ("fts") — FTS4 `chunks_fts` table (rag.db v1→v2 with backfill; FTS5 is
    unavailable on this device's stock SQLite). Ranking: distinct question terms matched desc →
    total occurrences desc → chunk id. Stopword-filtered, ≤6 terms.
  - `GraphRetriever` ("graph") — query n-grams (1–3 words, stopword-filtered) →
    `findEntities` anchors (cap 5 by mentionCount) → tier 1 = anchor chunks, tier 2 = neighbor
    chunks via `traverse` (depth 2 only when ≥2 anchors, else 1; ≤20 neighbors, ≤3 chunks each).
  - `Rrf.fuse` — score = Σ 1/(60+rank), limit 10.
- `RetrievalEngine` — shared by `RagPipeline` and the eval spike: embeds once, runs vector always,
  FTS+graph concurrently in hybrid mode, times each channel.
- Gate: vector mode unchanged (top-1 cosine < 0.30 → "I don't know"). Hybrid mode gates only when
  the fused list is empty OR (weak vector AND zero graph anchors AND best FTS chunk < 2 distinct
  terms).
- UI: "Hybrid"/"Vector" TextButton toggle in the chat header (persisted); stats line gains
  per-channel counts/timings (`vec 5/3ms · fts 7/12ms · graph 4/31ms`).
- `HybridRetrievalSpike` (id `hybrid_eval`, logcat tag `HybridEval`) — retrieval-only eval over
  `files/eval/eval-v1.json` in both modes: recall@5 per category + gate accuracy.

## Corpus and eval set

Corpus grew from 4 to 12 notes (`eval/corpus/`), with deliberate cross-note entity links
(Dana Whitfield: taxes ↔ meeting-notes ↔ gift-ideas; Cairo apartment: taxes ↔ apartment-cairo;
Sara Kim: japan-trip ↔ meeting-notes; LadybugDB/ONNX Runtime: design ↔ project-falcon ↔
conference-talk). `eval/eval-v1.json` = 27 questions: 14 vector-ish (v0), 3 keyword-exact
(part numbers, ISBN, booking reference), 6 multi-hop, 4 unanswerable.

Import + graph build for the 9 changed docs: **78 entities, 371 edges, 3.5 s** (in-chat status).
Graph tab after full corpus: **97 entities · 503 edges** (LadybugDB).

## Measured on-device (hybrid_eval spike, retrieval only, no LLM)

| Category | vector recall@5 | hybrid recall@5 |
|---|---|---|
| vector (14) | 14/14 | 14/14 |
| keyword (3) | 3/3 | 3/3 |
| multihop (6) | 6/6 | 6/6 |
| **overall (23 answerable)** | **1.00 (23/23)** | **1.00 (23/23)** |
| gate accuracy (4 unanswerable) | 0.75 | 0.75 |

End-to-end hybrid smoke test in chat: "Which project is my accountant leading?" → fused context
was exactly the 2-hop chain — [1] meeting-notes.md, [2] project-falcon.md, [3] taxes.md — and the
1.7B model answered "Project Falcon" citing them. (The question even survived an adb-typing typo
that mangled "my accountant" into "mysaccountant".)

## Default-mode decision

The checkpoint criterion was "hybrid ≥ vector overall, strictly better on keyword + multi-hop".
Result: **exact tie** — the 12-note corpus is too small and clean to differentiate; vector-only
already saturates recall@5. Per the plan's tie-break rule, **the default retrieval mode is now
`vector`**; the Hybrid toggle remains in the UI, and the hybrid path stays fully wired as the
architecture going forward (Phase 5's GNN reranker drops in as a fourth channel).

## Known limitations

- Gate misses the hard negative "What is the best price for a flight to Tokyo?" in BOTH modes —
  the japan-trip chunk is semantically close (cosine ≥ 0.30) despite not containing prices. The
  gate is cosine-threshold-only; no fix attempted (raising MIN_SCORE risks recall on real
  questions). Other 3 unanswerable questions gate correctly.
- FTS4 ranking is term-coverage-based, not BM25 (no matchinfo decoding); fine at this scale.
- Multi-hop graph wins are not measurable at this corpus size; the eval harness is ready for a
  larger real corpus (import real notes, rerun `hybrid_eval`).
- Answer-level eval remains manual/eyeball (on-device generation is too slow for 27×2 runs).
