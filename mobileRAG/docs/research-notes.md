# Research notes

Collected 2026-09-24/25 while optimizing the phone port and designing phase 7
([phase7-inner-state-proposal.md](phase7-inner-state-proposal.md)). Each entry says what the
source claims, how much to trust it, and what it means for this project.

Source quality: **[P]** primary (paper, official docs/release notes) · **[S]** secondary
(blog, news, aggregator: verify before relying on numbers) · **[lead]** only the title/abstract
was seen; not read yet.

---

## 1. Proactive and "sleeping" agents: closest prior work

### Sleep-time compute: Letta, 2025 [P]
https://arxiv.org/html/2504.13171v1 · code: https://github.com/letta-ai/sleep-time-compute ·
blog: https://www.letta.com/blog/sleep-time-compute/
- Agents "think" offline about a context before queries arrive: they anticipate likely
  questions and pre-compute useful quantities.
- Reported: up to ~5× less test-time compute, ~2.5× lower cost per query, up to 18% higher
  accuracy on stateful GSM-Symbolic / AIME variants.
- Caveat from the paper: gains track how **predictable** the future query is from the context.
- **For us:** it's offline compute aimed at *answering the user better*. Ours is aimed at the
  system forming *its own questions*. The predictability caveat suggests a secondary use:
  pre-compute retrieval/practice for the tutor, whose next session is fairly predictable.

### Generative Agents: Park et al., UIST 2023 [P]
https://dl.acm.org/doi/fullHtml/10.1145/3586183.3606763
- Memory stream retrieved by recency × importance × relevance; **reflection** synthesizes
  higher-level insights when the summed importance of recent events exceeds a threshold
  (150 in their implementation, which fired ~2–3 times per simulated day).
- **For us:** the accumulator-with-threshold is the right shape for "excitation builds until
  the system wakes". Our importance signal is ARM-E `m_t` + graph excitation, not an LLM
  rating each event, which is cheaper on-device.

### Proactive Conversational Agents with Inner Thoughts: Liu et al., CHI 2025 [P]
https://arxiv.org/abs/2501.00383 · https://dl.acm.org/doi/10.1145/3706598.3713760
- Formative study (24 participants). A covert, continuous train of thoughts runs in parallel
  to the conversation; each thought gets an **intrinsic-motivation score** (1–5) based on
  relevance, **information gap** and expected impact, and the agent speaks when motivation
  is high and the moment is right. Implementations: a multi-agent playground and a chatbot
  ("Swimmy").
- **For us:** borrow the motivation scoring (information gap ≈ our open loops) for ranking
  dream questions. Their work is within a conversation; ours runs between sessions.

### ChatGPT Pulse: OpenAI, launched 2025-09-25 [P]
https://openai.com/index/introducing-chatgpt-pulse/ · news:
https://techcrunch.com/2025/09/25/openai-launches-chatgpt-pulse-to-proactively-write-you-morning-briefs [S]
- Nightly synthesis from memory, chat history, feedback and connected apps; delivers
  personalized update "cards" the next day. Pro plan first; wider rollout gated on efficiency.
- **For us:** the closest product precedent, and the **baseline for our key ablation**:
  questions from memory dynamics vs an LLM summarizing recent history. Differences: Pulse is
  cloud-based and summarization-driven; ours is on-device and driven by graph structure.

### Proactive AI programming support: CHI 2025 [P]
https://dl.acm.org/doi/10.1145/3706598.3713357
- "Codellaborator" probe, within-subject N = 18: proactive assistance improved efficiency but
  caused workflow disruption.
- **For us:** evidence that proactivity needs a budget and timing, hence the daily cap, quiet
  hours and "less of this".

### Barge-in agent for older adults: CHI 2025 [P]
https://dl.acm.org/doi/full/10.1145/3706598.3714228
- Voice agent that initiates conversation to break silence and supports interruptions; older
  adults reported more engagement and fluency.
- **For us:** agent-initiated conversation can be welcome when it fits the person's context.

### Leads to read [lead]
- ProactiveEval, evaluation framework for proactive dialogue agents: https://arxiv.org/pdf/2508.20973
  (likely useful for our evaluation section).
- Communication Policy Evolution for Proactive LLM Agents: https://arxiv.org/pdf/2606.14314
- IceBreaker, personalized conversation starters: https://arxiv.org/pdf/2604.18375
- Knowing Isn't Understanding, re-grounding generative proactivity: https://arxiv.org/pdf/2602.15259
- LlamaPIE, proactive in-ear assistants (ACL Findings 2025): https://aclanthology.org/2025.findings-acl.710.pdf

---

## 2. Neuroscience grounding

Used as design inspiration, not as a claim that the system models the brain.

### Sleep inspires insight: Wagner, Gais, Haider, Verleger & Born, *Nature* 427, 2004 [P]
https://www.nature.com/articles/nature02223 · https://pubmed.ncbi.nlm.nih.gov/14737168/
- A task with a **hidden abstract rule**: subjects improve gradually, or abruptly once they gain
  insight into the rule. **More than twice as many** gained insight after sleep as after
  wakefulness, regardless of time of day. **No effect without initial training.** Reaction times
  slowed across sleep before insight appeared.
- **For us:** the core justification for "sleep produces questions". Two design consequences:
  (1) insight comes from *restructuring existing representations* → detect it as embedding
  shift across consolidation; (2) "no effect without training" → only consider nodes with
  real activity, never cold nodes.

### The hippocampal sharp-wave ripple: Joo & Frank, *Nat Rev Neurosci* 2018 [P]
https://www.nature.com/articles/s41583-018-0077-1 · https://pmc.ncbi.nlm.nih.gov/articles/PMC6794196/
- SWRs (synchronous hippocampal bursts) replay past **or possible future** experience and
  support both consolidation and retrieval for decision-making.
- Sleep replay has **lower fidelity** than waking replay, which suggests it mixes elements of
  several experiences and so supports **generalization**; imagination may start as retrieval
  followed by recombination.
- **For us:** replay should *mix* the day's episodes (sample across sessions), not replay them
  verbatim. Low-fidelity recombination is a feature.

### Related SWR findings
- Large SWRs promote hippocampo-cortical reactivation and consolidation, *Neuron* 2025 [lead]:
  https://www.cell.com/neuron/abstract/S0896-6273(25)00756-1. A subset of *large* ripples
  drives hippocampus–prefrontal reactivation. → only the strongest replays should reach
  cortical consolidation (a threshold, not every replay).
- Selection of experience for memory by SWRs [lead]:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10659301/. Supports **prioritized** replay
  (weight by salience), our `m_t × recency × excitation` sampling.
- SWRs influence selective DMN activation [lead]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4791429/

### Mind-wandering as spontaneous thought: Christoff et al., *Nat Rev Neurosci* 2016 [P]
https://www.nature.com/articles/nrn.2016.113
- Mind-wandering belongs to a family with creative thought and dreaming; spontaneous thought =
  mental states under **relaxed constraints** on content and transitions; focuses on the
  *dynamics* of how thoughts unfold.
- **For us:** "excitation spreading at rest" = activation moving along Hebbian edges with
  fewer constraints than during a task. A tunable constraint (spread depth/decay) is the knob
  between focused recall and mind-wandering.

### The default mode network: where spontaneous thought meets memory consolidation: Joshi, Tompary & Kucyi, *Curr Opin Behav Sci*, Dec 2025 [P]
https://pmc.ncbi.nlm.nih.gov/articles/PMC13318410/ · https://www.sciencedirect.com/science/article/pii/S235215462500141X
- DMN activity during spontaneous thought provides a context that **promotes propagation of
  reactivated information**, aiding consolidation; spontaneous thought is linked to better memory.
- **For us:** the closest scientific framing of the whole phase-7 idea: rest-time activity and
  consolidation are one system, not two.

### Synaptic homeostasis hypothesis: Tononi & Cirelli [from background knowledge, not retrieved this session]
- Sleep downscales synaptic strength potentiated during waking, restoring
  signal-to-noise. → homeostasis step in the sleep cycle (decay, normalization, pruning).
  Pull the primary paper before citing formally.

---

## 3. Model landscape for the phone (as of Sept 2026)

Numbers are vendor or press claims unless noted. Nothing here was benchmarked on the device.

| Model | Released | Notes | Source |
|---|---|---|---|
| Qwen3 8B / 1.7B / 0.6B (current) | Apr 2025 | A generation behind; the 8B is prefill-bound on the phone (phase 5 measured ~9.5 min for one full tutor turn incl. load) | project docs |
| **Qwen3.5 small** 0.8B / 2B / 4B / 9B | 2026-03-02 | Apache 2.0, natively multimodal. Hybrid **Gated DeltaNet** (3 linear-attention : 1 full-attention layers) → much smaller KV cache. Claimed 9B beats Qwen3-30B on MMLU-Pro (82.5) | [P] https://www.marktechpost.com/2026/03/02/alibaba-just-released-qwen-3-5-small-models-a-family-of-0-8b-to-9b-parameters-built-for-on-device-applications/ [S], https://artificialanalysis.ai/articles/qwen3-5-small-models [S], https://debuggercafe.com/introduction-to-qwen3-5-overview-vllm-and-llama-cpp/ [S] |
| **Gemma 4** E2B / E4B / 26B MoE / 31B; 12B Unified (added 2026-06-03) | 2026 | E2B: 2.3B active / 5.1B total, <3 GB at 4-bit; built for on-device | [P] https://ai.google.dev/gemma/docs/core/model_card_4, https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/ |
| **LFM2.5-1.2B-JP** (Liquid) | 2026-01-05/06 | Japanese-tuned; Liquid reports it matches or beats Qwen3-1.7B, Llama 3.2-1B and Gemma 3-1B on JMMLU, JA M-IFEval and JA GSM8K. Pretraining extended 10T → 28T tokens. Already used by the PC version | [P] https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai, https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP |
| EmbeddingGemma-300M (current) | — | Still a sound on-device embedding choice; keep | — |

Mobile roundups (Phi-4 mini, SmolLM2, Gemma 3, etc.) were low-quality aggregator content [S]
and aren't used for decisions.

**Implications**
- The vendored llama.cpp (commit `f3f1a8f`, 2026-09-08) already registers `qwen35`,
  `qwen35moe`, `gemma4`, `lfm2`, `lfm2moe`, so trying these is a GGUF swap.
- **Risk:** hybrid/recurrent models (Qwen3.5, LFM2) likely can't take `ai_chat.cpp`'s
  `shift_context()`, which removes a middle range of the KV cache. Needs a
  reset-the-conversation fallback before switching.
- `/no_think` is a Qwen3 convention; confirm how thinking is disabled for Qwen3.5 / Gemma 4.
- One resident model only: the JNI layer holds a single global model, so per-tab models
  would reintroduce reloads.

---

## 4. Tooling facts that changed decisions

- **ONNX Runtime training is deprecated [P].** v1.20.0 release notes: all training packages
  deprecated; **1.19.2 was the last** release of `onnxruntime-training-android` (and the PyPI,
  NuGet and CocoaPods training packages). https://github.com/microsoft/onnxruntime/releases/tag/v1.20.0
  → Cortical GAT training on the phone is a small hand-written reverse-mode autodiff in Kotlin
  (`hebbian/cortical/`), verified against the real PyTorch/PyG code with golden values
  (see phase7a-identity-consolidation-results.md).
- **Neo4j cosine scale [P].** The vector index scores cosine as **(1 + cos) / 2**, in [0, 1].
  https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/
  (A secondary source claimed scores are not normalized; the official manual says otherwise.)
  → The phone port used raw cosine; fixed to match (see §5).
- **llama.cpp grammar support** exists in the vendored tree (`json_schema_to_grammar`) but
  isn't exposed through the JNI bridge → grammar-constrained extraction JSON is an open
  improvement.
- **Qwen3.5 in llama.cpp** needs a build that registers the `qwen35` architecture; older
  builds refuse the file [S].

---

## 5. Codebase findings (audit of `Hebbian Memory/` and `mobileRAG/`)

### PC vs phone parity

| | PC (Python + Neo4j) | Phone (before fixes) | Status |
|---|---|---|---|
| Cortical consolidation | GAT (2 layers, 4 heads × hidden 256, ~2.1M params at 1024-d, 30 epochs) **writes embeddings back** into `c.embedding` | Logging stub | **Ported 2026-09-25** (~1.6M params at 768-d; golden-verified) |
| What vector search compares | Graph-shaped embeddings (after consolidation) | Raw EmbeddingGemma | **Follows from the port** |
| Causal scores | Computed, but written only to `'cortical'` edges, which nothing creates → always 0 | Same | **Fixed in both 2026-09-25**: written to the trained hippocampal edges |
| Consolidation write-back | Reset `updated_at` on every node (practice staleness reset) | n/a | **Fixed in both 2026-09-25** |
| Concept identity | New node per mention | Same | **Fixed in both 2026-09-25** (+ opt-in merge tool for old duplicates) |
| Emotion → ARM-E | Fine-tuned XLM-R VAD (`core/vad_model/`), multilingual | English keyword stub; Japanese never passes the gate → `m_t = 1.0` | To port |
| Cosine scale | (1 + cos) / 2 | Raw cos (2× the spread; bootstrap threshold far stricter) | **Fixed 2026-09-25** |
| Spread-only candidates | Filtered to vector pool (`if nid in cos_scores`) | Same | **Fixed in both, 2026-09-25** (`spread_extra` / `withSpreadCandidates`) |
| Extraction system prompt | Sent as the call's system prompt | Never sent → every concept became `fact` | **Fixed 2026-09-24** |
| Extraction template | `{{ }}` left over from `.format` (bug) | Same | Fixed on the phone; **PC still has it** |

Correction (2026-09-25): an earlier version of this table gave the GAT as hidden 16 / ~130K
params (the class defaults) and said the PC fills `causal_score`. `tutor_engine.py` builds it
with hidden 256, and the causal write never matched an edge.

### Algorithm observations (apply to both versions)

1. **No concept identity** (fixed 2026-09-25, see phase7a). Every ingest minted a new UUID
   (`ingest_pipeline.py:218`, `IngestPipeline.ingest`), so repeats created duplicates;
   activation counts stayed at 1, ARM-E novelty was always 1.0 and the usage term ≈ 0.
   → dedup by normalized label + type, then
   embedding similarity.
2. **Hippocampal update measures similarity, not co-activation.** `coact = h_src · h_dst` on
   propagated embeddings is input-independent; all edges in the 2-hop subgraph update each
   ingest. Fixed point `w* = η·c / (λ·s)` is **independent of `m_t`**, so emotion only changes
   convergence speed. Typical values put `w*` near the 5.0 ceiling. (Consolidation mitigates
   this on the PC by turning weights into embedding geometry.) → an activity-based update:
   `Δw = m·η·xᵢxⱼ − λ·xⱼ²·w`, where x is the node's relevance to the current input.
3. **Forgetting isn't in the weights.** Edges decay only when a nearby ingest touches them;
   abandoned regions freeze. → lazy time decay from `lastUpdated`.
4. **Practice adapts per type, not per item**: one miss on a hard word shortens all
   vocabulary. There are three independent forgetting clocks (TypeProfiles τ, scheduler
   half-lives, Oja λ multipliers). → per-concept stability (FSRS-style / per-item HLR).
5. **The genuinely Hebbian signals are good:** co-retrieval reinforcement (where `m_t` *does*
   move the fixed point, to 2.5·m), co-extraction edges and practice reinforcement.

### Performance findings (phone, fixed 2026-09-24)
- Tab switches reloaded the whole GGUF (different system prompts); lazy `by viewModels()` in
  `onTrimMemory` built both tab ViewModels → both warmed up the model.
- One LLM summary call per extracted concept; no `/no_think` on internal calls; malformed
  JSON template → retries; the KV-budget reset reloaded the 8B every 12 turns; the concept
  cache was dropped on every write.
- Theme: the SharedPreferences listener was held only weakly (GC'd), and the Compose state
  was never read.

---

## 6. Open questions

- Does insight-by-embedding-shift find links a human confirms more often than chance?
  (First experiment once consolidation runs on the phone.)
- Excitation half-life and wake threshold: how often should the system want to talk?
  Start at ≤ 1 message/day and learn from "less of this".
- Should replay mix sessions (Joo & Frank's low-fidelity recombination) or stay within
  them? A/B on insight precision.
- Which small model handles both Japanese chat and strict JSON best: Qwen3.5-4B vs
  Gemma 4 E4B, with grammar-constrained decoding?
