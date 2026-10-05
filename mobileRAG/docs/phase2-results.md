# Phase 2 — Graph Memory: results

Date: 2026-09-14. Device: RedMagic 10 Pro (NX789J), Android 15.
Status: **complete, all checkpoints passed on-device.**

## What was built

On-device entity extraction and graph persistence, layered on the Phase 1 naive RAG
(the query/retrieval path is unchanged — that is Phase 3):

- `extract/Gliner2Extractor.kt` — GLiNER2 (`cuerbot/gliner2-multi-v1` ONNX export of
  `fastino/gliner2-multi-v1`, int8, 376 MB) zero-shot NER. Kotlin port of the
  `elcuervo/gliner` (Ruby) decoding pipeline: regex word splitting with char offsets,
  schema prompt `( [P] entities ( [E] <label>… ) ) [SEP_TEXT]`, per-element DJL
  tokenization with `addSpecialTokens=false` (the tokenizer.json TemplateProcessing
  post-processor would otherwise inject [CLS]/[SEP]), sigmoid over the `logits`
  output `[1, seq, max_width=8, num_labels]`, per-label greedy non-overlap spans.
  The actual export is the **`logits` variant** (7 inputs incl. `words_mask`,
  `text_lengths`, `task_type`, `label_positions`, `label_mask`), not the
  `span_logits` variant the cuerbot README implies — both are handled at runtime.
- `extract/EntityResolver.kt` — normalized exact match, then Jaro-Winkler ≥ 0.92
  (same type) fuzzy merge, else `upsertEntity`. Shared `graph/JaroWinkler.kt`.
- `core/GraphStore.kt` v2 — adds upsertEntity/findEntities/getEntity/addMention/
  chunksForEntity/edgesFor/allEntities/removeGraphForChunks/clear; Entity gains
  mentionCount, Edge gains weight. v1 methods unchanged (spikes unaffected).
- `graph/LadybugGraphStore.kt` — full GraphStore over LadybugDB 0.20.3 Cypher
  (prepared statements, `[:REL*1..n]` traversal, DETACH DELETE cascades).
  `graph/SqliteGraphStore.kt` upgraded to graph.db v2 (normalized_name UNIQUE
  (normalized_name, type), mentions table, edge weight, FTS4 + Jaro-Winkler search)
  as schema-of-record/fallback. `graph/GraphStoreFactory.kt` — process-wide
  singleton, LadybugDB first, SQLite on failure.
- `rag/ingest/GraphIndexer.kt` — per doc: removeGraphForChunks (idempotent), then
  per chunk: extract → resolve → addMention + `MENTIONED_WITH` co-occurrence edges
  for every distinct entity pair, each edge carrying `source_chunk_id` provenance.
  Extractor is always closed after a run (RAM discipline). Missing model files fail
  gracefully into `lastStats`.
- Wiring: `Indexer.index()` returns reindexed docIds; ChatViewModel runs graph build
  sequentially after embedding on import, and cascades `removeGraphForChunks` on
  document delete. RagDatabase gains chunksForDocument/chunkIdsForDocument/
  chunkTextsById.
- Graph tab (Chat | **Graph** | Spikes): counts + store id + last-build stats,
  Rebuild button with progress, entity search, entity detail with edge groups and
  source-chunk previews (the provenance demo).

## Decisions made this phase

- **No typed relations.** Neither public GLiNER2 ONNX export (cuerbot, lion-ai)
  includes the relation-scorer head — both are encoder + span head only, so typed
  relation extraction is unavailable without a custom export. Edges are therefore
  `MENTIONED_WITH` co-occurrence within a chunk (per the plan's sanctioned
  fallback). Options for later: custom ONNX export of the boundary model's relation
  head, or an LLM relation-typing pass over extracted entity pairs (indexing-time
  only).
- GLiNER2 default labels: person, organization, location, product, technology,
  event, date, topic (in `Gliner2Extractor.DEFAULT_LABELS`).

## Measured on-device

- GLiNER2 spike (`gliner2_ner`): model load **1115 ms**, extraction **26 ms avg**
  per sentence-scale input, session PSS ~858 MB total (extractor is unloaded after
  indexing). Quality: correct entities on all 3 probe sentences (Marie Curie/person
  0.99, Sorbonne/organization 0.96, "September 9, 2026"/date 1.00, "RedMagic 10
  Pro"/product 0.94, …).
- Graph build over the 4 test notes: **4 docs, 4 chunks, 32 entities, 132 edges,
  ~2.0 s** end-to-end (excluding model load).
- Rebuild is idempotent: second full rebuild produced identical counts (32/132).
- LadybugDB store ran the entire flow live (schema creation, prepared-statement
  writes, traversal, cascade deletes) with zero errors — the previously unproven
  Cypher surface is now validated on-device.
- Chat regression: "When are my taxes due?" → correct grounded answer citing
  `[1] taxes.md` (April 15, 2026; Form 4868 → October 15, 2026). Naive RAG path
  unaffected; generation speed unchanged.

## Checkpoint review (plan weeks 6–9)

| Checkpoint | Status |
|---|---|
| GLiNER extraction pipeline on-device | ✅ (GLiNER2 int8, entities; relations N/A in ONNX exports → co-occurrence edges) |
| Entity resolution (hash + fuzzy) | ✅ normalized exact + Jaro-Winkler |
| Entity–chunk–document schema in LadybugDB/SQLite | ✅ both stores, factory-selected |
| Incremental writes on document changes | ✅ per-doc remove+re-extract; delete cascade |
| Graph visible in-app | ✅ Graph tab |
| Every edge traceable to a source chunk | ✅ source_chunk_id on every edge; detail view shows chunk text |
| Idle-and-charging batch indexing | ⏭ deferred to Phase 4 (extraction is fast enough at import time: ~0.5 s/chunk-scale doc) |

## Known limitations

- `MENTIONED_WITH` edges are untyped; the graph captures *association*, not
  *relation semantics*, until a relation extractor exists.
- Near-duplicate entities survive resolution when surface forms differ
  structurally ("April 15" vs "April 15, 2026") — Jaro-Winkler only merges
  ≥ 0.92 same-type.
- Zero-shot label noise exists (e.g. "radium" → product 0.79); tunable via
  threshold/labels.
- GLiNER2 session adds ~500 MB while loaded; it is closed after every indexing
  run, so chat-time RAM is unaffected.
- Test corpus is still the 4 synthetic notes; real-note import + eval growth is
  planned for Phase 3.
