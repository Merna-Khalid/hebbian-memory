# Full Hebbian System

A biologically-inspired memory system for LLMs, and the apps built on top of it.

Instead of plain vector RAG, memories live in a **graph** where associations strengthen and
decay Hebbian-style: concepts you revisit stay strongly weighted, concepts you abandon slowly
fade, and an emotion-modulated learning rate (**ARM-E**) decides how deeply each turn gets
encoded. Co-retrieved concepts get pairwise edge reinforcement (*"retrieved together → wired
together"*), and a slow **cortical consolidation** pass replays and compresses the fast
hippocampal layer on a schedule, the way sleep does.

Two personas are built on the memory core:

- **Japanese tutor**: a conversational tutor with a configurable persona, plus adaptive
  spaced-repetition practice driven by the graph's own decay curves.
- **Second brain**: ambient ingestion of a simulated subvocal-EMG / heart-rate / attention
  signal stream, standing in for a BCI rig still being built. No replies, just extraction and
  emotion-gated encoding.

## Repository layout

This repo contains two complete implementations of the same system:

| Folder | Platform | Stack |
|---|---|---|
| [`Hebbian Memory/`](Hebbian%20Memory/) | **PC version** | Python 3.10+, PyTorch / PyG, BGE-M3, LFM2.5-1.2B-JP, Neo4j graph store, FastAPI + React/Vite UI |
| [`mobileRAG/`](mobileRAG/) | **Android version** | Kotlin, Jetpack Compose, llama.cpp (GGUF LLMs), EmbeddingGemma-300M, GLiNER2, LadybugDB embedded graph store |

The Android app is a faithful port: its Hebbian math is verified against the Python original
with a golden-value harness (81 numeric checks, plus a PyTorch parity suite for the cortical
GNN: see `mobileRAG/docs/`). It also adds a second, complementary system: a **knowledge RAG**
for your own documents (chunk → embed → entity-extract → hybrid vector/FTS/graph retrieval
with citations).

The Android version runs **fully on-device**: no network calls, no accounts, no telemetry.
Measured on a RedMagic 10 Pro (Snapdragon 8 Elite, 16 GB): Qwen3-1.7B Q8_0 at ~25–29 tok/s,
with Qwen3-8B Q4_K_M as the quality default.

## How the memory works

```
conversation turn / ambient utterance
      │
      ▼
embedding ──► retrieve top-k concepts (cosine + Hebbian + causal score)
      │                │
      ▼                ▼
LLM responds     co-retrieved nodes get pairwise edge reinforcement
      │
      ▼
concept extraction from the exchange
      │
      ▼
ingest: embed → emotion (text valence + HR-fused arousal) →
ARM-E m_t (gated by attention) → hippocampal GNN
(Oja's rule + eligibility gating + anti-Hebbian decay)
      │
      ▼
cortical consolidation on a schedule (nightly "sleep" on Android)
```

Key properties:

- **Per-concept-type decay**: vocabulary fades fast, grammar rules are sticky.
- **Emotion-modulated encoding (ARM-E)**: an attention gate
  (`abs(e_t) · r_t · attention`) means emotionally-loaded content won't deeply encode if
  attention was low when it happened.
- **Practice that targets forgetting**: exercises are generated from concepts the adaptive
  scheduler predicts you're about to forget (`p = 2^(−Δt/h_type)`); correct answers reinforce
  the underlying edges and grow that type's half-life.

## Quick start

**PC version**: needs Neo4j 5.11+ and Python 3.10+:

```bash
cd "Hebbian Memory"
pip install torch torch_geometric transformers sentence-transformers \
            neo4j fastapi uvicorn matplotlib networkx
uvicorn server:app --port 8000          # backend (HEBBIAN_STUB=1 to skip model downloads)
cd hebbian-tutor-ui && npm install && npm run dev   # frontend → http://localhost:5173
```

Or the terminal clients: `python japanese_tutor.py` / `python second_brain_sim.py`.
Full details: [`Hebbian Memory/README.md`](Hebbian%20Memory/README.md).

**Android version**: Android Studio (AGP 8.11.1, Kotlin 2.2.0, compileSdk 36, minSdk 30,
arm64-v8a):

```bash
cd mobileRAG
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

First launch downloads the models once (~2.4 GB from Hugging Face), then the app is fully
offline. Model files are **not** in this repo (see below). Full details, release signing, and
the diagnostics suite: [`mobileRAG/README.md`](mobileRAG/README.md).

## What's not in this repo

- **Model weights** (GGUF LLMs, EmbeddingGemma ONNX, GLiNER2, the PC version's VAD model):
  several GB each; downloaded at first run or staged locally.
- **Build outputs**, signing keys, and local IDE/config files: see `.gitignore`.
- **Memory data**: all user memory lives on the device / in your local Neo4j and is never
  uploaded anywhere.

## Documentation

- `mobileRAG/On-Device_GraphRAG_Android_Build_Plan.md`: the architecture report the on-device
  stack is built from.
- `mobileRAG/docs/phase0..7-results.md`: the build log: spikes, graph memory, hybrid
  retrieval, the Hebbian port, UI, consolidation.
- `mobileRAG/docs/ui-animus.md`: the app's visual language.
- `Hebbian Memory/outputs/hebbian_memory_architecture.docx`: the research write-up
  (ARM-E, practice scheduling, consolidation).
- `mobileRAG/docs/research-notes.md`: annotated sources and the PC-vs-phone audit.
