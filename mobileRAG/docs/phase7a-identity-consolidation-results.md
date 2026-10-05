# Phase 7a results — concept identity + cortical consolidation on the phone

Date: 2026-09-25. Verified on the dev machine (JVM unit tests, PyTorch golden values, Python
checks). **Not yet run on the device** — no phone was connected; see §6.
Scope: the two prerequisites for phase 7 (docs/phase7-inner-state-proposal.md) — concept
identity in both versions, and a real cortical consolidation on the phone, where it was a
logging stub. Plus PC bugs found on the way, and an opt-in merge for old duplicates.

## 1. Why

The phone behaved like near-cosine retrieval. The main reason: on the PC, consolidation
trains a GAT and **writes its embeddings back into `c.embedding`**, the field vector search
reads, so after a few runs the "cosine" term compares graph-shaped embeddings; the phone never
did this. Separately, both versions minted a new node per mention, so re-mentioning a concept
never strengthened anything.

## 2. What shipped

**Concept identity (PC + phone).** `hebbian/ConceptIdentity.kt` / `core/concept_identity.py`:
- Same normalized label (NFKC, lowercase, whitespace collapsed, wrapping punctuation stripped —
  not `# + 〜`) and same type → reuse.
- Otherwise a same-type vector hit with Neo4j-scale score ≥ 0.975 (raw cos ≥ 0.95) → reuse.
  Deliberately above practice's "confusable" 0.82: 食べる and 食べた stay separate.
- Never for unvalidated content or `event` nodes.

Ingest reuse path: no upsert (the node keeps its consolidated embedding and first
description), no LLM summary call, `incrementActivation`; bootstrap edges only if the node has
none. Edge upserts **never weaken** (weight = max, co-activations + 1, eligibility/causal kept)
— otherwise re-extracting a relation would reset a trained 4.8 edge to 1.5. PC: `label_key`
property + index, backfilled in `setup_schema()`.

**Cortical consolidation (phone).** `hebbian/cortical/`:
- `Tape.kt` — minimal reverse-mode autodiff (ONNX Runtime training is deprecated). Fused ops
  (attention aggregation, ELU+dropout, pair dot/cosine) avoid [edges × 1024] intermediates —
  the composed form needed ~200 MB per epoch.
- `CorticalGnn.kt` — port of `core/cortical_gnn.py` with PyG 2.8 `GATConv` semantics, the
  three-part loss, torch Adam, the same negative sampler. Production config = PC parity
  (4 heads × 256 hidden, 30 epochs, lr 5e-4) at 768-d: 1,578,240 params.
- `CorticalConsolidator.kt` — one per memory space (tutor + second brain share it), background
  single low-priority thread, skip-if-running; persists `weights.bin`, `z_prev.bin`,
  `node_order.json`, `meta.json` under `<space>/cortical_state/`.
- `ConsolidationScheduler.kt` — real now (every 20 ingests, session end, manual), same surface.
- `SleepWorker.kt` — nightly consolidation while idle + charging (phase 7 groundwork).
- Clear Hebbian memory also clears cortical state.

**PC fixes** (`Hebbian Memory/core/`):
- Causal scores were written only to `'cortical'` edges, which nothing creates → always 0.
  Now written to the hippocampal edges consolidation trained on (both versions).
- Consolidation write-back set `updated_at = now` on every node, making everything look freshly
  practiced to the practice queue. It no longer touches `updated_at` (both versions).

**Opt-in duplicate merge.** Same rules on both platforms (`store/ConceptMerge.kt`,
`tools/merge_duplicates.py`): groups by label key + type; survivor = most activations; edges
re-pointed and combined (max weight, summed co-activations); history and ARM-E means merged;
session stats and practice history remapped.
- Phone: Settings → Data → "Merge duplicate concepts" — dry-run count first, then a backup
  (`VACUUM INTO` for SQLite; close/copy/reopen of the same instance for LadybugDB) to
  `<space>/backups/hebbian-<timestamp>/`, then the merge.
- PC: dry run by default; `--apply` runs one Neo4j transaction. Back up with `neo4j-admin` first.

## 3. Verification

| Check | Result |
|---|---|
| Golden values: `docs/cortical_golden.py` runs the real `consolidate()` (torch 2.10, PyG 2.8), two cases (5-node demo; 12-node, 4-head, with z_prev) | forward within 1e-5; every step-1 parameter gradient; all 30 epochs of loss components; final z and weights — **all match** |
| Mutation test: LeakyReLU slope 0.2 → 0.21 | both golden tests fail (tests have teeth) |
| Tape ops: finite-difference gradient checks + fused == composed | 24/24 |
| Identity rules (Kotlin) / mirrored Python cases | 8/8 · 27/27 incl. real `IngestPipeline` over a fake store |
| Stub `TutorEngine` end to end over an in-memory store (JVM twin of the `hebbian_ingest` spike) + merge integration | 2/2; fails as expected with identity disabled |
| Consolidation data prep (cap, alignment, dims) | 7/7 |
| LadybugDB kill/recovery (`LadybugCrashTest`, real store via desktop native) | 2/2; fails with the checkpoint removed |
| Merge rules (Kotlin) / Python `--selftest` | 6/6 · 11/11 |
| `./gradlew :app:assembleDebug` | builds |
| Cost at production size (Mac JVM, N=1000, E=5000) | ~0.8 s/epoch ≈ 24 s per run, heap Δ ≈ 128 MB |

Run: `./gradlew :app:testDebugUnitTest` (add `CORTICAL_BENCH=1` for the cost check);
`/opt/anaconda3/envs/RAG/bin/python tools/check_identity.py` and
`tools/merge_duplicates.py --selftest` from `Hebbian Memory/`.

## 4. Deviations from the Python original

- Phone runs are capped at 1,500 nodes (strongest-connected kept) — memory/time safety.
- Phone consolidation always runs in the background; Python's session-end run is synchronous.
- Weights and z_prev persist in a custom binary format, not `torch.save`.

## 5. Also changed earlier in this pass (2026-09-24)

- LLM residency: tab switches swap the system prompt (`InferenceEngine.resetConversation`)
  instead of reloading the GGUF; one mutex covers load + generation; `onTrimMemory` no longer
  instantiates both tab ViewModels (the "double load"). Saving the learner profile no longer
  unloads the model — the prompt hash is part of the residency key.
- Memorizing: no per-concept LLM summary for short text, `/no_think` on internal calls,
  extraction rules actually sent, JSON template fixed, concept cache write-through.
- Theme: the SharedPreferences listener was only weakly held (GC'd) and the Compose state was
  never read; both fixed. Status/nav bar icons follow the app theme.
- Retrieval parity: Neo4j `(1 + cos) / 2` scale on the phone; spreading activation can add
  candidates vector search missed (both versions).

## 6. On-device results (RedMagic 10 Pro, Android 15, 2026-09-26)

All runs in the "Test" space; the "Personal" space was backed up off-device first and left
untouched.

| Spike | Result |
|---|---|
| `cortical_consolidation` (synthetic, production size) | N=300/E=3000: 610 ms/epoch ≈ 18 s per run, heap Δ ≈ 97 MB · N=1000/E=5000: 988 ms/epoch ≈ **30 s per run**, heap Δ ≈ 204 MB (512 MB max heap) |
| `hebbian_ingest` on LadybugDB | 7/7, incl. "re-mention reuses the concept (1 node, 2 activations)" |
| `second_brain_sim` | 5/5 |
| Settings → Data → Merge duplicates, on a real pre-identity graph (six `User:` copies, fully wired) | 5 merged into 1, backup under `backups/hebbian-<stamp>`, all 30 edges became self loops and were dropped; survives restart |
| `concept_merge` (new): merge + backup on scratch LadybugDB **and** SQLite stores | 6/6 — edges re-pointed (w = max, co-activations summed), self loop dropped, survives reopen, backup opens with the pre-merge graph |
| `sleep_now` (new): the SleepWorker consolidation, run twice on the active space | 5/5 — 6 nodes/30 edges, ≈ 11 s per run; all embeddings rewritten, `updatedAt` untouched, causal on 30/30 edges, warm start from saved weights + z_prev |

Found and fixed on the device:

- **LadybugDB WAL can't replay large values (data loss).** LadybugDB 0.20.3 writes WAL
  records it cannot replay when a value is ≥ 4096 bytes — STRING, `DOUBLE[]`, `FLOAT[768]`
  alike (bisected on the Mac with the lbug jar's desktop native; at 8 KB replay even crashed
  the JVM). Every concept embedding (~9 KB as text) is such a value. Android kills processes
  without closing them, so the next launch failed with "Corrupted wal file. Read out invalid
  WAL record type", and the factory **silently fell back to an empty SQLite store** — the
  user's memory appeared wiped. This is what happened to the Personal space on 2026-09-15
  (its LadybugDB WAL held about an hour of data; it has run on SQLite since).
  Fix (`LadybugHebbianStore`): `CHECKPOINT` after every write that can carry a large value
  (`upsertConcept`, `updateConceptEmbeddings`) and after a merge, so the WAL only holds small,
  replayable records (≈ 20 ms per checkpoint on the Mac). If a kill lands inside that window,
  `open` renames the WAL to `hebbian.lbug.wal.corrupt-<ms>` and opens the last checkpoint;
  `store-choice.txt` records `wal_recovery=…`. The factory's choice is now **sticky**: a space
  that already has a SQLite store stays on it (otherwise the fix would swap Personal's real
  SQLite memories for its near-empty LadybugDB file). The entity-graph store only holds short
  strings and is not affected.
  Verified: `LadybugCrashTest` (JVM, real store) — a kill copy after each of 9 tutor turns
  keeps every concept; an unreplayable WAL is set aside and the checkpoint opens. Removing the
  `upsertConcept` checkpoint fails it. On the phone: the Test space's corrupt WAL was recovered
  on launch; after `hebbian_ingest`, `am kill` + relaunch reopened LadybugDB with no recovery
  and all data.
- SQLite `backupTo` always named the copy `hebbian_graph.db`; it now keeps the store's own
  file name (the app's store already used that name, so real backups were fine).
- Merge message: "into 1 concepts" → "into 1 concept".

Notes: this ROM sets `log.tag=S` (all app logging off); `adb shell setprop
log.tag.CorticalConsolidator V` (non-persistent) turns a tag back on. A forced WorkManager
job (`cmd jobscheduler run -f -n androidx.work.systemjobscheduler com.mobilerag <id>`) does
not run `SleepWorker` — periodic work run before it's due is skipped — hence `sleep_now`.
A 6-node consolidation still takes ≈ 11 s: the cost at small N is the 1.6M-parameter
backward/Adam step (30 epochs × ~0.37 s), not the graph.

## 7. Not yet verified

- `SleepWorker` firing on its own schedule (idle + charging, daily).
- A consolidation triggered by real tutor use (every 20 ingests / session end).
- **PC Cypher**: the new queries are unit-checked through a fake store only. Compile them
  against Neo4j without executing anything:
  `NEO4J_PASSWORD=… python tools/check_identity.py --explain`.
- Existing PC graph: `setup_schema()` backfills `label_key` on first start after the update.
