# Phase 7 (proposal): inner state, sleep, and waking questions

Status: research + design, not implemented. 2026-09-25. Annotated sources, model landscape
and the codebase audit behind this proposal: [research-notes.md](research-notes.md).

## The idea

An LLM exists only while it is invoked; its working state is the context. The Hebbian memory
already outlives any single conversation, but it is passive: nothing happens to it between
turns. This phase gives the memory an **inner state** that keeps changing without input:

- parts of the graph become **excited** (by what happened today, by unresolved questions, by
  memories that are fading, by links that consolidation discovers);
- a **sleep cycle** (idle + charging) replays and consolidates, and the restructuring itself
  produces new candidate links;
- the system **wakes with a few questions**, and when one is worth it, messages the user first.

The user's answer is ordinary ingest — validated, emotionally weighted — so the loop closes:
the system's own curiosity becomes new, verified memory.

## Why it fits this system specifically

The pieces already exist; they are just not connected into a loop:

| Needed | Already here |
|---|---|
| Excitation that spreads | `spreadingActivation` over Hebbian edges |
| Salience of an event | ARM-E `m_t` (once a real emotion model runs on-device) |
| Something to be curious about | `question` concept type; `getFadingConcepts`; recall probabilities |
| A sleep that restructures | cortical consolidation (GAT writes embeddings back; ported to the phone in phase 7a, with a nightly `SleepWorker`) |
| Background time | `CommunityIndexWorker` pattern (WorkManager, idle + charging) |
| A voice | the persona system prompt; resident LLM |

## Prior work, and what would be new

| Work | What it does | Difference here |
|---|---|---|
| Sleep-time compute (Letta, arXiv 2504.13171, 2025) | Pre-computes answers to *anticipated* queries offline | We generate the system's *own* questions, not answers to the user's |
| Generative Agents (Park et al., UIST 2023) | Reflection fires when summed importance > 150 | Same accumulator idea, but the trigger is graph excitation and the output goes *to the user* |
| Inner Thoughts (Liu et al., CHI 2025) | Covert thought stream; speaks when intrinsic motivation is high | In-conversation only; we run *between* sessions, driven by memory dynamics |
| ChatGPT Pulse (OpenAI, Sept 2025) | Overnight: LLM reads memory/chats → morning briefing cards | Closest product. Cloud, LLM-summarization-driven. Ours: on-device, and **the content comes from structural change in memory**, not from rereading the chat log |

The research claim worth testing: *questions produced by memory dynamics (excitation +
sleep restructuring) are more useful than questions an LLM produces by summarizing recent
history.* Pulse is effectively the baseline for that ablation.

Neuroscience grounding (as design inspiration, not a claim of biological fidelity):
- Sleep restructures memory and more than doubles insight into a hidden rule
  (Wagner et al., *Nature* 2004). → insight detection from embedding shift, below.
- Sharp-wave ripples select which experiences get replayed; sleep replay is lower-fidelity and
  mixes experiences, a candidate mechanism for generalization (Joo & Frank, *Nat Rev
  Neurosci* 2018).
  → prioritized replay.
- Spontaneous thought is thought under relaxed constraints (Christoff et al., *Nat Rev
  Neurosci* 2016); DMN activity during rest helps propagate reactivated memories
  (Joshi, Tompary & Kucyi, *Curr Opin Behav Sci* 2025). → excitation spreading at rest.
- Synaptic homeostasis: sleep downscales what waking potentiated (Tononi & Cirelli). →
  normalization/decay during sleep.

## Design

### 1. Inner state (persisted per memory space)

```
InnerState {
  excitation: Map<nodeId, Double>   // a_i, decays with half-life H (≈ 6 h)
  lastTick:   Long
  mood:       EMA of recent valence/m_t
  openLoops:  List<Loop>            // {kind, nodeIds, excitation, createdAt, askedAt?}
  lastSleep:  SleepReport           // what changed, what was found
}
```

Decay is lazy (applied on read from `lastTick`), so nothing runs while idle except sleep.

### 2. Excitation sources

Each adds to `a_i`, then one step of spreading activation distributes a fraction to neighbours:

| Source | Amount |
|---|---|
| Ingest of node i | ∝ `m_t` (salient events excite more) |
| Retrieval that failed (low `r_t`) | the query's nearest nodes — "I didn't know this well" |
| Unresolved `question` concept | persistent until answered |
| Fading but important | `(1 − recall_p) × importance` from the practice scheduler |
| Sleep insight (below) | the two endpoints of a discovered link |

A homeostatic cap on total excitation prevents runaway loops.

### 3. Sleep cycle (WorkManager: nightly, idle + charging)

1. **Replay** — sample the day's episodes weighted by `m_t × recency × excitation`
   (prioritized, like SWR selection); re-run the hippocampal update on their subgraphs.
2. **Consolidate** — cortical GAT over the graph; snapshot embeddings before (`z₀`) and after (`z₁`).
3. **Insight detection** — pairs (i, j) where `cos(z₁ᵢ, z₁ⱼ) − cos(z₀ᵢ, z₀ⱼ)` is large and there
   is no strong direct edge: consolidation pulled them together through the graph's structure.
   This is the "sleep revealed a hidden relation" signal, computed rather than asked of an LLM.
4. **Homeostasis** — merge near-duplicate nodes, apply time decay, prune floor edges.
5. **Dream** — one LLM call: given the top open loops (insights, open questions, fading
   concepts, with their node summaries), write ≤ 3 short questions in the persona's voice,
   each tied to its node ids. Questions, never assertions: a discovered link may be an artefact,
   and the user's answer is what validates it.
6. **Score** — motivation = excitation × novelty × (not asked recently); keep the top few.

### 4. Waking

- If the best question clears a threshold **and** the budget allows (default ≤ 1/day, quiet
  hours respected), post a notification. Otherwise hold the thoughts for the next app open
  ("while you were away…" card; the tutor opens with it).
- The user's reply is ingested as validated with elevated `m_t`; answering a loop discharges
  its excitation. "Not interested" raises that loop kind's threshold.
- For the tutor, many waking questions are retrieval practice delivered at the moment the
  scheduler predicts forgetting — the testing effect comes for free.

### 5. Continuity across invocations

At session start, a short inner-state summary (open loops, last sleep's findings, mood) goes
into the persona prompt. Now that persona swaps are cheap (`resetConversation`), this is one
prompt prefill. The model starts each conversation from where the memory "was", not from zero.

## Risks and guardrails

- **Spurious insights** — embedding shifts can be artefacts. Always phrase as questions; track
  confirmation rate; raise the threshold if it drops.
- **Notification fatigue** — proactive agents improve efficiency but disrupt (CHI 2025
  programming-assistant study). Hard daily budget, quiet hours, one-tap "less of this".
- **Battery** — sleep runs only while charging and idle; the dream step is one LLM call.
- **Privacy** — everything stays on device; nothing new leaves the phone.
- **Anthropomorphism** — UI copy says what happened ("I noticed X and Y keep coming up
  together"), not feelings the system doesn't have.

## Evaluation

1. **Insight precision** — share of sleep-discovered links the user confirms.
2. **Usefulness** — reply rate and thumbs on waking questions.
3. **Learning (tutor)** — recall on items surfaced by waking questions vs matched controls.
4. **The key ablation** — dynamics-driven questions vs an LLM summarizing the last N sessions
   (Pulse-style), same budget, blind-rated. This is what supports (or refutes) the novelty claim.

## Prerequisites and build order

0. Prerequisites (from the PC/phone gap analysis):
   - ✅ **Concept dedup** — done in phase 7a (both versions).
   - ✅ **Cortical consolidation on the phone** — done in phase 7a. ONNX Runtime training was
     not an option (all ORT training packages deprecated; 1.19.2 was the last
     `onnxruntime-training-android`), so it's a hand-written autodiff for the 2-layer GAT
     (4 heads × hidden 256, ~1.6M params at 768-d, 30 Adam steps), golden-verified against
     PyTorch. Insight detection (step 3 of the sleep cycle) needs the before/after
     embeddings, which the consolidator now persists as `z_prev`.
   - **Emotion model on the phone** — export `core/vad_model` (XLM-R VAD) to int8 ONNX, or take
     affect from the extraction JSON.
1. Inner state + excitation sources + lazy decay (no sleep yet); expose it on the Memory tab.
2. Sleep cycle with replay, consolidation snapshots, insight detection, homeostasis.
3. Dream + scoring + waking (in-app card first, notifications second — needs
   `POST_NOTIFICATIONS`).
4. Evaluation harness, including the Pulse-style ablation.

## Parity fixes already made (2026-09-25)

- Phone `cosineSearch` now returns Neo4j's `(1 + cos) / 2` scale. It returned raw cosine,
  which doubled the weight of similarity against the Hebbian terms and made the ingest
  bootstrap threshold much stricter than on the PC.
- Both versions: spreading activation can now add candidates vector search missed
  (`retrieve_for_llm(spread_extra=…)`, `withSpreadCandidates`); before, it could only rerank.

## Sources

- Sleep-time compute — https://arxiv.org/html/2504.13171v1
- Generative Agents — https://dl.acm.org/doi/fullHtml/10.1145/3586183.3606763
- Proactive Conversational Agents with Inner Thoughts — https://arxiv.org/abs/2501.00383
- ChatGPT Pulse — https://openai.com/index/introducing-chatgpt-pulse/
- Proactive programming support trade-offs (CHI 2025) — https://dl.acm.org/doi/10.1145/3706598.3713357
- Sleep inspires insight — https://www.nature.com/articles/nature02223
- Sharp-wave ripple review — https://pmc.ncbi.nlm.nih.gov/articles/PMC6794196/
- Mind-wandering as spontaneous thought — https://www.nature.com/articles/nrn.2016.113
- DMN: spontaneous thought meets consolidation — https://pmc.ncbi.nlm.nih.gov/articles/PMC13318410/
- ORT training deprecation — https://github.com/microsoft/onnxruntime/releases/tag/v1.20.0
- Neo4j vector index cosine scale — https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/
