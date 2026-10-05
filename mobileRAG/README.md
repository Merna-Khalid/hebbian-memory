# mobileRAG

A fully on-device Android app with two complementary memory systems:

1. **Knowledge RAG** — import .txt/.md notes; they're chunked, embedded (EmbeddingGemma-300M),
   entity-extracted (GLiNER2), and stored in an embedded graph database (LadybugDB, SQLite
   fallback). Questions are answered by hybrid retrieval (vector + FTS + graph traversal, RRF
   fusion) and a local GGUF LLM via llama.cpp, with citations.
2. **Hebbian memory** — a port of the "Hebbian Memory" Python system: concepts live in a graph
   whose edges strengthen on co-retrieval and decay otherwise, driven by ARM-E, an
   emotion-modulated learning rate with an attention gate. Two personas: a **Japanese tutor**
   (with adaptive practice scheduling) and a **second brain** for ambient ingestion (simulated
   EMG/HR/attention signal source standing in for a future BCI rig).

Everything is on-device: no network calls, no accounts, no telemetry.

## App map

- **Home** — dashboard: status (model, backend, stores), memory stats, quick actions.
- **Chat** — ask your documents (vector / hybrid / global modes, model picker, import).
- **Tutor** — Japanese tutor chat; each turn shows its ARM-E state (quadrant, m_t, gating,
  Δw̄) and extracted concepts.
- **Memory** — the Hebbian concept graph (list view, min-weight filter, node detail).
- **Settings** — appearance (dark "sea" / light "white room" + vermilion/gold/cyan accent), performance
  presets (Quality/Balanced/Speed), learner profile (feeds the tutor persona), memory spaces
  (isolated data profiles), model management, data tools, diagnostics (spikes), help, about.
- From Home/Settings: **Practice** (spaced-repetition exercises), **Brain** (simulated ambient
  stream), **Graph** (entity graph), **Diagnostics** (on-device spikes), **Help**.

## Building

Android Studio (AGP 8.11.1, Kotlin 2.2.0, compileSdk 36, minSdk 30, arm64-v8a):

```bash
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

The `:llamacpp` module builds llama.cpp from `third_party/llama.cpp` via CMake/NDK.
LadybugDB's Android native binary is cross-compiled out-of-tree and pinned in
`prebuilt/ladybug/arm64-v8a/` (see docs/phase0-results.md).

## Sharing the app (release APK)

```bash
./gradlew :app:assembleRelease   # → app/build/outputs/apk/release/app-release.apk (~140 MB, arm64)
```

- **Signing:** signed with `keystore/mobilerag-release.jks`; the passwords are in
  `keystore/keystore.properties`. Neither is in version control. **Back both up.** Every update
  to a shared install must be signed with the same key, or Android refuses it (the only way
  out is uninstalling, which wipes that phone's memory).
- **First run:** the recipient's phone opens on **Awakening**, which downloads the models once
  (about 2.4 GB from Hugging Face; Wi-Fi recommended). The download runs as a foreground job
  with a progress notification and resumes after pauses, drops or the app being killed. Files
  are checked against exact sizes (`setup/ModelCatalog.kt`). After that the app is offline.
- **Phone requirements:** Android 11+, 64-bit ARM, ~3 GB free RAM for Qwen3-1.7B.
- **Testing a clean first run on a phone that already has the app:**
  `./gradlew :app:assembleRc` builds `com.mobilerag.rc`, a release copy that installs beside it.

## Installing models by hand (development)

Models live in app-private storage (nothing is bundled in the APK):

```bash
adb push Qwen3-8B-Q4_K_M.gguf /data/local/tmp/
adb shell "run-as com.mobilerag cp /data/local/tmp/Qwen3-8B-Q4_K_M.gguf files/models/"
```

Same pattern for `files/models/embeddinggemma/` (tokenizer.json + model_q4f16.onnx +
model_q4f16.onnx_data) and `files/models/gliner2/`. Staged copies are in `models/` in this
repo. Measured on a RedMagic 10 Pro (16 GB): Qwen3-1.7B Q8_0 ~25–29 tok/s; Qwen3-8B Q4_K_M is
the quality default (prefill-bound per turn — see docs/phase5-hebbian-port-results.md).

## Docs

- `On-Device_GraphRAG_Android_Build_Plan.md` — the architecture/technology report the stack
  is built from (LadybugDB, EmbeddingGemma, GLiNER2, llama.cpp, RRF, GNN outlook).
- `docs/phase0..4-results.md` — spikes, naive RAG, graph memory, hybrid retrieval, polish.
- `docs/phase5-hebbian-port-results.md` — the Hebbian Memory port (what/why/how, golden-value
  verification against the Python original, LadybugDB fixes).
- `docs/phase6-ui-results.md` — the UX pass (theme, navigation, profiles, spaces, help).
- `docs/hebbian_golden.py` + `docs/hebbian-golden-values.txt` — numeric parity harness.
- `docs/research-notes.md` — annotated sources (proactive/sleeping agents, sleep & replay
  neuroscience, 2026 on-device model landscape), tooling facts, and the PC-vs-phone audit.
- `docs/phase7-inner-state-proposal.md` — proposal: inner state, sleep cycle, waking questions.
- `docs/ui-animus.md` — the Animus visual language (frosted glass, the entity on Home), where
  it lives in code, and its measured cost.
- `docs/phase7a-identity-consolidation-results.md` — concept identity (both versions) and the
  on-device cortical consolidation; `docs/cortical_golden.py` + `docs/cortical-golden-values.json`
  are its PyTorch parity harness.

## Verification

On-device spikes (Settings → Diagnostics) cover: embeddings, both graph stores, llama.cpp,
tokenizer, GLiNER, hybrid eval (27-question regression harness with history in
`files/eval/history/`), Hebbian math golden values (81 checks vs Python torch), Hebbian ingest
loop, second-brain sim, cortical consolidation cost, duplicate merge + backup on both stores,
"sleep now" (the nightly consolidation on the active space), and a 30-min soak. Note: the Hebbian
ingest/sim spikes clear the active space's Hebbian memory, and "sleep now" rewrites its
embeddings (as nightly sleep does) — run them in a test space.

JVM unit tests (dev machine, no device): `./gradlew :app:testDebugUnitTest` — cortical GNN vs
PyTorch golden values, autodiff gradient checks, concept identity (incl. the stub engine end to
end), duplicate-merge rules, LadybugDB kill/WAL recovery. `CORTICAL_BENCH=1` adds a production-size cost check.
