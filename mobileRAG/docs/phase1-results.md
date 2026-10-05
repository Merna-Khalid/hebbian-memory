# Phase 1 Results — Naive On-Device RAG

Device: RedMagic 10 Pro (NX789J), Android 15, ~15.5 GB visible RAM. Date: 2026-09-13.

## What was built

End-to-end naive RAG: in-app file picker (.txt/.md) → sentence-aware chunker (≤500-token chunks, ~50-token overlap, markdown-header aware) → EmbeddingGemma-300M embeddings → brute-force cosine top-5 → strict-grounding prompt with `[n]` provenance → streaming answers via llama.cpp. Chat UI with citation chips (tap → source chunk bottom sheet) and a per-query stats line. Spike harness kept as a second tab.

New code: `embeddings/EmbeddingGemmaEngine.kt`, `embeddings/GemmaTokenizer.kt`, `rag/RagDatabase.kt`, `rag/VectorStore.kt`, `rag/RagPipeline.kt`, `rag/ingest/{Chunker,DocumentImporter,Indexer}.kt`, `chat/{ChatScreen,ChatViewModel}.kt`, `spikes/TokenizerSpike.kt`, `spikes/EmbeddingGemmaSpike.kt`, `QueryTimer` in `spikes/Instrumentation.kt`. MainActivity is now Chat/Spikes tabs.

## Decisions (and deviations from the build-plan doc)

1. **EmbeddingGemma via ONNX Runtime, not LiteRT** — reuses the proven Phase 0 runtime; one runtime serves all models. Model: `onnx-community/embeddinggemma-300m-ONNX` (ungated), `model_q4f16.onnx` (176 MB total). Prompt prefixes per model card (`task: search result | query:` / `title: … | text:`), mean-pooled, L2-normalized, 768-d.
2. **Tokenizer: DJL `ai.djl.huggingface:tokenizers:0.33.0` + `ai.djl.android:tokenizer-native:0.33.0`** (bundles `libdjl_tokenizer.so` for Android; DJL's runtime-download path does not work on Android — it does `System.loadLibrary("djl_tokenizer")`, so the native AAR is required).
3. **Vector store: brute-force in-memory cosine** over SQLite BLOBs, invalidated by a DB `indexVersion`. 4 docs / 4 chunks scan in 1–4 ms; per the doc this holds to ~100k chunks.
4. **Similarity gate (added, not in the plan):** top-1 cosine < 0.30 short-circuits to "I don't know" without loading the LLM. Needed because Qwen3-0.6B ignored the grounding prompt and answered from parametric knowledge (answered "Paris" citing a Japan trip note). Observed margins: relevant ≥ 0.50, unrelated ≤ 0.15.
5. **Qwen3 handling:** `/no_think` appended to the user prompt (thinking otherwise eats the 512-token output budget and the answer is never produced) + a `ThinkTagFilter` stream filter as belt-and-braces. Output budget raised to 1024.
6. **Qwen3-1.7B Q8_0 is the default** (see benchmark). Note: Qwen's official GGUF repo ships no Q6_K for 1.7B — used Q8_0 (1.83 GB).
7. **Model delivery:** everything large stays out of the APK; pushed via `adb push /data/local/tmp` + `run-as cp` into `files/models/`. adb-pushed files need a media-scan broadcast to appear in the system file picker.

## Measured numbers (on-device)

### Tokenizer (DJL)
| tokenizer load | avg encode |
|---|---|
| 1777 ms | 1 ms |

### EmbeddingGemma-300M q4f16 (ONNX)
| model load | avg embed | retrieval sanity |
|---|---|---|
| 1096 ms | 32 ms | query→relevant doc 0.612; unrelated 0.020 / 0.007 |

### End-to-end queries (RAG pipeline)
| Question | Model | embed | search | load | TTFT | decode | RAM (PSS) | Correct? |
|---|---|---|---|---|---|---|---|---|
| Tax filing deadline | 0.6B | 60 ms | 3 ms | 1604 ms | 667 ms | 68.3 tok/s | 2.5 GB | ✓ "April 15, 2026. [1]" |
| Hotel in Tokyo | 0.6B | 57 ms | 3 ms | 1852 ms | 663 ms | 59.1 tok/s | 2.5 GB | ✓ (identified via "Godzilla head" + citation; didn't name it) |
| Sourdough feeding | 0.6B | 44 ms | 2 ms | — | 1226 ms | 46.2 tok/s | 2.4 GB | ✓ exact ratio, [1] |
| Capital of France (unanswerable) | gated | 45 ms | 4 ms | — | — | — | 0.3 GB | ✓ "I don't know", LLM never loaded |
| Graph store DB | 0.6B | 50 ms | 1 ms | 4943 ms | 718 ms | 63.2 tok/s | 2.5 GB | ✓ "LadybugDB (a Kùzu fork)… [1]" |
| Property tax (post-edit re-embed) | 0.6B | 72 ms | 2 ms | 1953 ms | 768 ms | 59.9 tok/s | 2.5 GB | ✓ new fact found |
| Tax filing deadline | 1.7B | 46 ms | 1 ms | 6027 ms | 1311 ms | 29.2 tok/s | 4.6 GB | ✓ quotes passage verbatim |
| Hotel in Tokyo | 1.7B | 44 ms | 1 ms | — | 1899 ms | 25.1 tok/s | 4.6 GB | ✓ names Hotel Gracery Shinjuku + quote |

### Model comparison → default
| | Qwen3-0.6B Q8_0 | Qwen3-1.7B Q8_0 |
|---|---|---|
| mmap load | 1.6–4.9 s | 6.0 s |
| TTFT | 0.66–0.77 s | 1.3–1.9 s |
| decode | 46–68 tok/s | 25–29 tok/s |
| RAM | ~2.5 GB | ~4.6 GB |
| answer quality | correct but terse; drops inline citation markers | names entities, quotes sources |

Default = 1.7B (doc's "balanced" tier; quality jump is obvious, speed still comfortable). 0.6B remains selectable in the UI dropdown.

## Validation performed (physical device)

- Import of 4 markdown notes via in-app picker; indexing with progress.
- Re-import of all 4 after editing one: unchanged docs skipped (hash), edited doc re-embedded; new fact ("property tax due December 20") immediately answerable.
- 4 answerable questions answered correctly with correct #1 citation; citation chip opens bottom sheet with full source chunk + score.
- 1 unanswerable question correctly gated to "I don't know".
- Stats line present per answer; VectorStore/pipeline timings logged to logcat.

## Known limitations (to carry forward)

- Chat history is in-memory; lost on process death (acceptable for dev build).
- 0.6B sometimes omits the inline `[n]` marker in the answer text even though chips are correct.
- Answers sometimes end with "Therefore, the answer is [1]." boilerplate (0.6B); 1.7B does not.
- `model_quantized.onnx` (int8, 310 MB) downloaded but not benchmarked — q4f16 was fast enough (32 ms/embed); revisit if indexing latency matters.
- Repo is not git-init'd; when it is, ignore `models/`, `third_party/llama.cpp`, `*.gguf`.
- Eval set (`docs/eval-v0.json`, 16 questions) covers the 4 synthetic test notes; replace with the user's real notes corpus and grow to 30–50 by the Phase 3 boundary.

## Phase 1 go/no-go checkpoint: PASS

Answers with citations over imported notes ✓; per-stage latency + TTFT + RAM measured per query ✓.
