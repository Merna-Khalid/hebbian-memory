# mobileRAG Help

Everything in this app runs **on your phone** — documents, memories, embeddings, and models never leave the device.

## What this app is

Two memory systems in one app:

- **Knowledge (Chat tab)** — your imported documents, chunked, embedded, and linked into an entity graph. Ask questions, get grounded answers with citations.
- **Hebbian memory (Tutor / Practice / Brain / Memory tabs)** — a biologically-inspired memory where concepts strengthen when you revisit them and fade when you don't, modulated by an emotion-aware learning rate (ARM-E). Two personas live on it: a Japanese tutor and a second brain for ambient thought.

## Chat (general knowledge)

- Import .txt/.md documents from the Chat tab. They are chunked (~500 tokens), embedded with EmbeddingGemma, and indexed for vector + full-text + graph retrieval.
- Retrieval modes: **Vector** (semantic only), **Hybrid** (vector + keyword + entity graph, fused), **Global** (hybrid + community summaries for thematic questions).
- Answers are strictly grounded in your documents and cite them as [1], [2]… Tap a citation chip to see the source passage.

## Tutor (Japanese persona)

- Chat in English or Japanese. Every assistant turn shows its **ARM-E state** and the **concepts** extracted into memory.
- Set your **learner profile** (Settings → Learner profile) — level N5–N1 and goals are injected into the tutor's persona so explanations match you.

## ARM-E glossary

- **Quadrant (Q1–Q4)** — the emotional state of the turn from valence/arousal: Q1 positive+energized, Q2 negative+energized, Q3 negative+calm, Q4 positive+calm.
- **m_t** — the emotion-modulated learning rate for the turn (0.2–3.0). Higher = deeper encoding.
- **gated** — when confidence (emotion × retrieval coherence × attention) is too low, the turn encodes neutrally (m_t = 1.0) instead of amplifying noise.
- **Δw̄** — mean Hebbian weight change this turn: how much memory actually moved.
- **Eligibility** — edges must be co-activated recently to be strengthened; stale ones slowly decay (anti-Hebbian).

## Practice

- The tutor serves exercises for concepts its scheduler predicts you are about to forget (recall probability 2^(−Δt/half-life) per concept type).
- Correct answers reinforce the underlying memory edges and grow that concept type's half-life; wrong answers shrink it. Confusable items are interleaved.

## Brain (second brain)

- Simulates an ambient stream (a stand-in for a future BCI rig): scripted scenarios with heart-rate and attention signals. Attention gates encoding — content encountered while distracted barely encodes.
- Step through scenarios manually or press Play. The Memory tab shows what stuck.

## Memory

- The Hebbian graph: every concept the tutor/brain extracted, with type, activations, mean learning rate, and recent ARM-E events. Filter edges by minimum weight with the slider.

## Memory spaces

- Spaces are isolated memory profiles (Settings → Memory spaces): each has its own documents, entity graph, Hebbian memory, and practice state. Models, theme, and your learner profile are shared.
- Switching a space unloads the model and reloads all data — the app restarts its UI.

## Models & performance

- **Performance presets** (Settings → Performance): Quality (8B + hybrid), Balanced (1.7B + hybrid), Speed (1.7B + vector). Individual overrides below them mark the preset custom.
- First use of a model pays a load (mmap) of several seconds; the 8B's first load is the slowest. Tab switches between Chat/Tutor/Brain reload the model (different system prompts).
- Installing a new GGUF from your computer:

`adb push model.gguf /data/local/tmp/`

`adb shell "run-as com.mobilerag cp /data/local/tmp/model.gguf files/models/"`

- Running low on memory? Settings → Model & memory → Unload model.

## Diagnostics

- Settings → Diagnostics runs the on-device verification spikes (math golden values, ingest loop, second-brain sim, hybrid retrieval eval, and the Phase 0 hardware spikes).
- Warning: the Hebbian ingest/sim spikes **clear the active space's Hebbian memory** for a deterministic run.

## Troubleshooting

- **Wrong/old answers after switching models**: the model picker writes a shared preference; the next query reloads automatically.
- **Store backend**: the app uses LadybugDB (Cypher) with a SQLite fallback. Which one is active is shown in Settings → About; the reason for a fallback is in `files/spaces/<space>/store-choice.txt` (logcat is unreliable on some devices).
- **Model files are large**: they live in app-private storage and count against the app's data, not the APK.
