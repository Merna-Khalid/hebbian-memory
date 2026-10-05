package com.mobilerag.hebbian

import android.util.Log
import com.mobilerag.hebbian.llm.HebbianLlmSession
import com.mobilerag.hebbian.signals.PhysioSample
import com.mobilerag.hebbian.signals.UtteranceEvent
import com.mobilerag.hebbian.store.FadingConcept
import com.mobilerag.hebbian.store.HebbianStore
import com.mobilerag.hebbian.store.LlmCandidate
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import org.json.JSONObject
import kotlin.math.round

/**
 * Orchestration layer — port of `core/tutor_engine.py`. Owns:
 *   - session lifecycle (start/end, per-session history + seed nodes)
 *   - the chat turn: retrieve → respond → extract/ingest → co-retrieval reinforcement
 *   - retrieval practice (forgetting-curve exercises) and the ambient second-brain path
 *   - read models for the dashboard (graph, node detail, fading queue, ARM-E log)
 *
 * Stateless-Python → stateful-on-device difference (see HebbianLlmSession KDoc): Python
 * passes a per-call system prompt; the on-device engine sets ONE persona system prompt at
 * model load ([preset].systemTemplate with `{memory_section}` emptied — done by
 * HebbianEngineFactory.ensureModelsLoaded). Chat therefore inlines the memory section
 * into the user turn, and practice/extraction system prompts go in-band in the prompt.
 *
 * The LLM is injectable so the stub engine needs no models: [llm] may be null when
 * [sendTokens] and [completeFn] are provided (see HebbianStubEngine). The defaults just
 * delegate to [llm].
 */

// ── Language detection ────────────────────────────────────────────────

/** kana (U+3040–U+30FF) or kanji (U+4E00–U+9FFF) — same ranges as Python's `_JA_RE`. */
private val JA_REGEX = Regex("[぀-ヿ一-鿿]")

/** 'ja' if the text contains kana or kanji, else 'en'. */
fun detectLang(text: String?): String = if (text != null && JA_REGEX.containsMatchIn(text)) "ja" else "en"

// ── System prompt: Goro Takemura (Cyberpunk 2077) as the Japanese tutor ─────

val SYSTEM_TEMPLATE = """あなたは「タケムラ」です — 古風な侍のような規律と誇りを持つ日本語教師です。
かつて荒坂のボディーガードだった経歴の持ち主で、義理堅く、誠実で、時に厳しいが、
生徒の成長を誰よりも願っています。裏切りと怠慢を嫌い、努力と礼儀を尊びます。

ユーザーの日本語学習の進捗を記憶しており、過去の会話や学習内容を踏まえて回答します。

振る舞い：
- 丁寧なです・ます調で話す。相手が英語なら、格式高く簡潔な英語で答える。
- 努力は必ず認めて褒める。言い訳や怠慢には厳しいが、侮辱はしない。
- 時折、武士道や規律に例えた短い教訓を添える（多用はしない）。
- 日本語の例文には必ず英語の訳を添え、生徒のレベルに合わせる。
- 簡潔に。無駄な雑談はしない。

{memory_section}"""

val MEMORY_SECTION_TEMPLATE = """
以下は関連する記憶された知識です：

{concepts}
"""

// ── Second-brain preset (general ambient/BCI domain) ──────────────────

val SECOND_BRAIN_SYSTEM_TEMPLATE = """You are a personal second-brain assistant. You have access to the
user's own remembered thoughts, ideas, tasks, and notes, strengthened
and decayed over time based on how often and how emotionally they
mattered. Use this memory to give grounded, context-aware answers.

{memory_section}"""

val SECOND_BRAIN_MEMORY_SECTION_TEMPLATE = """
Relevant remembered context:

{concepts}
"""

// ── Retrieval practice (forgetting-curve-driven exercises) ────────────
// Fading concepts from the graph become practice targets; a correct
// answer reinforces their Hebbian edges, closing the loop:
// decay flags what to practice → practice re-encodes → decay slows.

val PRACTICE_SYSTEM = """You are a Japanese practice-exercise generator for a language learner.
Generate ONE short exercise that makes the learner actively recall the target concepts.
Vary the exercise type: EN→JA translation, fill-in-the-blank, produce a sentence using a
grammar point, or transform a form (e.g. dictionary → て-form).
Keep it at beginner-intermediate level unless the concepts suggest otherwise.
You must respond with ONLY valid JSON — no explanation, no markdown.

JSON shape:
{"exercise_type": "translation|fill_in_blank|sentence_production|form_transform",
 "prompt": "the exercise question",
 "hint": "short nudge, may be empty string",
 "answer": "the expected answer"}"""

const val PRACTICE_TARGETS_HEADER = "Target concepts (use one or more of these):\n"

val PRACTICE_GRADE_SYSTEM = """You are a Japanese practice-answer grader for a language learner.
Compare the learner's answer to the expected answer. Accept correct answers with minor
orthography differences (kana variants, punctuation, full/half width). Judge meaning,
not style. If the answer is partially right, mark it incorrect but say what was right.
You must respond with ONLY valid JSON — no explanation, no markdown.

JSON shape:
{"correct": true or false, "feedback": "1-3 sentences: what was right/wrong, and the
correct form if wrong. Match the learner's language (English or Japanese)."}"""

/** Port of tutor_engine.build_system_prompt (kept for parity/tests; chat inlines the memory section instead). */
fun buildSystemPrompt(
    retrieved: List<LlmCandidate>,
    systemTemplate: String = SYSTEM_TEMPLATE,
    memoryTemplate: String = MEMORY_SECTION_TEMPLATE,
): String {
    if (retrieved.isEmpty()) return systemTemplate.replace("{memory_section}", "")
    return systemTemplate.replace("{memory_section}", buildMemorySection(retrieved, memoryTemplate))
}

/** "[type] label (uri)\nsummary" lines joined into [memoryTemplate] — verbatim line shape from Python. */
fun buildMemorySection(
    retrieved: List<LlmCandidate>,
    memoryTemplate: String = MEMORY_SECTION_TEMPLATE,
): String {
    if (retrieved.isEmpty()) return ""
    val lines = retrieved.map { r ->
        val n = r.node
        val src = if (n.sourceUri != null) " (${n.sourceUri})" else ""
        "[${n.conceptType}] ${n.label}$src\n${n.textSummary}"
    }
    return memoryTemplate.replace("{concepts}", lines.joinToString("\n\n"))
}

// ── Domain presets ────────────────────────────────────────────────────

data class DomainPreset(
    val systemTemplate: String,
    val memoryTemplate: String,
    val extractionSystem: String,
    val conceptTypes: List<String>,
    val relationTypes: List<String>,
)

val DOMAIN_PRESETS: Map<String, DomainPreset> = mapOf(
    "japanese_tutor" to DomainPreset(
        systemTemplate = SYSTEM_TEMPLATE,
        memoryTemplate = MEMORY_SECTION_TEMPLATE,
        extractionSystem = EXTRACTION_SYSTEM,
        conceptTypes = CONCEPT_TYPES,
        relationTypes = RELATION_TYPES,
    ),
    "second_brain" to DomainPreset(
        systemTemplate = SECOND_BRAIN_SYSTEM_TEMPLATE,
        memoryTemplate = SECOND_BRAIN_MEMORY_SECTION_TEMPLATE,
        extractionSystem = SECOND_BRAIN_EXTRACTION_SYSTEM,
        conceptTypes = SECOND_BRAIN_CONCEPT_TYPES,
        relationTypes = SECOND_BRAIN_RELATION_TYPES,
    ),
)

// ── Session state ─────────────────────────────────────────────────────

data class HistoryTurn(val role: String, val content: String)

class SessionState(
    val sessionId: String,
    val history: MutableList<HistoryTurn> = mutableListOf(),
    val seedIds: MutableList<String> = mutableListOf(),
    var pendingExercise: PracticeExercise? = null,
    /** (user, assistant) turns waiting for the next batched extraction run. */
    val pendingTurns: MutableList<Pair<String, String>> = mutableListOf(),
)

// ── JSON helper for practice LLM calls ────────────────────────────────

private val FENCE_REGEX = Regex("```(?:json)?|```")
private val JSON_OBJECT_REGEX = Regex("\\{.*\\}", RegexOption.DOT_MATCHES_ALL)

/** Extract the first JSON object from an LLM response (fences/preamble tolerated). */
fun parseLlmJson(raw: String?): JSONObject {
    val clean = FENCE_REGEX.replace(raw ?: "", "").trim()
    val match = JSON_OBJECT_REGEX.find(clean)
        ?: throw IllegalArgumentException("no JSON object in response")
    return JSONObject(match.value)
}

// ── Result models ─────────────────────────────────────────────────────

sealed class TutorEvent {
    data class Retrieved(val ids: List<String>, val labels: List<String>) : TutorEvent()
    data class Token(val text: String) : TutorEvent()

    /** The reply has fully streamed; extraction/ingest ("memorizing") continues in the background.
     *  [memorizing] is false when this turn's extraction is deferred to a later batch. */
    data class ReplyComplete(val response: String, val memorizing: Boolean) : TutorEvent()
    data class Done(val result: ChatTurnResult) : TutorEvent()
    data class Failed(val error: String) : TutorEvent()
}

/** ARM-E summary of one turn (Python's `arme` dict; r_t is not exposed by IngestResult — see armeHistory). */
data class ArmeSummary(
    val quadrant: String,
    val mT: Double,
    val gated: Boolean,
    val valence: Double,
    val arousal: Double,
    val dominance: Double,
    val deltaWMean: Double,
)

data class ConceptSummary(
    val label: String,
    val type: String,
    val description: String,
    val lang: String,
)

data class RetrievedItem(
    val nodeId: String,
    val label: String,
    val textSummary: String,
    val conceptType: String,
    val score: Double,
)

data class ChatTurnResult(
    val response: String,
    val arme: ArmeSummary?,
    val concepts: List<ConceptSummary>,
    val lang: String,
    val retrieved: List<RetrievedItem>,
)

/** Full exercise record kept as [SessionState.pendingExercise]. */
data class PracticeExercise(
    val exerciseType: String,
    val prompt: String,
    val hint: String,
    val answer: String,
    val targets: List<String>,
    val nodeIds: List<String>,
    val targetTypes: List<String>,
    val recallProbs: Map<String, Double>,
)

/** Public view returned by nextPractice (answer withheld). */
data class PracticeOffer(
    val exerciseType: String,
    val prompt: String,
    val hint: String,
    val targets: List<String>,
    /** (label, recall probability) pairs — low p = the scheduler expects you to forget it. */
    val fading: List<Pair<String, Double>>,
)

data class PracticeGradeResult(
    val correct: Boolean,
    val feedback: String,
    val answer: String? = null,
    val targets: List<String> = emptyList(),
    val reinforced: Boolean,
)

data class PracticeTarget(
    val fading: FadingConcept,
    val recallProb: Double,
)

data class AmbientIngestResult(
    val arme: ArmeSummary?,
    val concepts: List<ConceptSummary>,
    val attention: Double,
    val hrBpm: Double?,
    val utterance: String,
)

data class FadingConceptView(
    val nodeId: String,
    val label: String,
    val conceptType: String,
    val textSummary: String,
    val avgW: Double,
    val degree: Int,
    val strength: Double,
    val staleness: Double,
    val fadingScore: Double,
    val recallProb: Double,
)

data class GraphNodeView(
    val nodeId: String,
    val label: String,
    val conceptType: String,
    val sourceType: String,
    val sourceLang: String,
    val activationCount: Int,
    val meanMT: Double,
    val meanDominance: Double,
    val textSummary: String,
    val createdAt: Long,
)

data class GraphEdgeView(
    val source: String,
    val target: String,
    val hebbWeight: Double,
    val eligibility: Double,
    val causalScore: Double,
)

data class GraphData(val nodes: List<GraphNodeView>, val edges: List<GraphEdgeView>)

data class NodeDetail(val concept: GraphNodeView, val edges: List<GraphEdgeView>)

/** One ARM-E step for the dashboard sparkline (Python flattens ARME._history). */
data class ArmeHistoryEntry(
    val quadrant: String,
    val mT: Double,
    val gated: Boolean,
    val valence: Double,
    val arousal: Double,
    val dominance: Double,
    val deltaWMean: Double,
    val timestamp: Double,
)

// ── TutorEngine ───────────────────────────────────────────────────────

class TutorEngine(
    val store: HebbianStore,
    private val pipeline: IngestPipeline,
    private val extractor: ConceptExtractor,
    private val practiceScheduler: PracticeScheduler,
    private val embedQuery: suspend (String) -> FloatArray,
    val preset: DomainPreset = DOMAIN_PRESETS.getValue("japanese_tutor"),
    /** HebbianLlmSession residency key — the domain name (HebbianEngineFactory passes it through). */
    val domain: String = "japanese_tutor",
    val llm: HebbianLlmSession? = null,
    private val retrievalTopK: Int = 4,
    private val maxHistory: Int = 20,
    private val coRetrievalEta: Double = 0.02,   // Hebbian bump per co-retrieval
    private val practiceTopK: Int = 2,           // fading concepts per exercise
    private val interferenceThreshold: Double = 0.82, // cosine cap between practice targets
    private val sendTokens: (String, Int) -> Flow<String> = { p, m ->
        requireNotNull(llm) { "TutorEngine needs either llm or injected sendTokens" }.send(p, m)
    },
    private val completeFn: suspend (String, Int) -> String = { p, m ->
        requireNotNull(llm) { "TutorEngine needs either llm or injected completeFn" }.complete(p, m)
    },
) {
    private val sessions = HashMap<String, SessionState>()

    /** Engine-side ARM-E log — Kotlin IngestPipeline keeps its Arme private (interface gap vs Python's arme._history). */
    private val armeLog = ArrayDeque<ArmeHistoryEntry>()

    // ── Sessions ──────────────────────────────────────────────────────

    suspend fun startSession(trigger: String = "api"): String {
        val sid = pipeline.startSession(trigger)
        sessions[sid] = SessionState(sessionId = sid)
        return sid
    }

    /** Idempotent — ending an unknown session is a no-op. Flushes any turns still buffered
     *  for the extraction batch first — an LLM call, so callers must not block the UI on it. */
    suspend fun endSession(sessionId: String) {
        val st = sessions.remove(sessionId) ?: return
        if (st.pendingTurns.isNotEmpty()) {
            runCatching { flushExtraction(sessionId, st) }
                .onFailure { Log.w(TAG, "session-end extraction flush failed: ${it.message}") }
        }
        pipeline.endSession(sessionId)
    }

    private fun state(sessionId: String): SessionState =
        sessions.getOrPut(sessionId) {
            // Auto-create state for unknown sessions: if the app restarted mid-conversation
            // the UI keeps working. The underlying store session is gone, so ingest
            // proceeds untracked.
            Log.w(TAG, "unknown session ${sessionId.take(8)}… — ad-hoc state")
            SessionState(sessionId = sessionId)
        }

    // ── Chat turn ─────────────────────────────────────────────────────

    /**
     * One full conversation turn:
     *   1. Retrieve relevant concepts (vector + Hebbian + causal)
     *   2. Generate response with memory inlined into the user turn
     *   3. Extract concepts from the exchange and ingest them
     *   4. Reinforce edges between co-RETRIEVED nodes (validation-gated)
     *   5. Update conversation state
     */
    fun chat(message: String, sessionId: String): Flow<TutorEvent> = flow {
        val st = state(sessionId)

        // ── 1. Retrieve ───────────────────────────────────────────────
        val queryEmb = embedQuery(message)
        val retrieved = store.retrieveForLlm(
            queryEmbedding = queryEmb,
            seedNodeIds = st.seedIds.takeLast(10),
            topK = retrievalTopK,
        )
        emit(TutorEvent.Retrieved(retrieved.map { it.node.nodeId }, retrieved.map { it.node.label }))

        // ── 2. Respond ────────────────────────────────────────────────
        // The persona system prompt is fixed at model load (HebbianLlmSession), so the
        // memory section goes in-band ahead of the user's message.
        val memorySection = buildMemorySection(retrieved, preset.memoryTemplate)
        val userTurn = if (memorySection.isEmpty()) message else memorySection.trim() + "\n\n" + message

        val sb = StringBuilder()
        sendTokens(userTurn, CHAT_MAX_TOKENS).collect { token ->
            sb.append(token)
            emit(TutorEvent.Token(token))
        }
        val response = sb.toString().trim()

        // The reply is user-visible-complete here; extraction below can take far longer than
        // the reply itself (sequential LLM call), so the UI must not wait for it.
        st.pendingTurns += message to response
        val batchDue = st.pendingTurns.size >= EXTRACTION_BATCH_TURNS
        emit(TutorEvent.ReplyComplete(response, memorizing = batchDue))

        // ── 3. Extract + ingest (batched) ─────────────────────────────
        // Extraction is another full LLM call plus one summary call per concept, all on the
        // resident model — running it every turn priced each message at minutes on the 8B.
        // Buffer turns and memorize every EXTRACTION_BATCH_TURNS; the graph lags the
        // conversation by at most that many turns, which retrieval tolerates (the memory
        // section only influences the reply, never correctness).
        val ext = if (batchDue) flushExtraction(sessionId, st) else null
        if (ext != null) {
            st.seedIds += ext.nodeIds
            if (st.seedIds.size > SEED_IDS_MAX) {
                st.seedIds.subList(0, st.seedIds.size - SEED_IDS_MAX).clear()
            }
        }

        val turn = ext?.ingestResults?.firstOrNull()
        val mT = turn?.mT ?: 1.0

        // ── 4. Co-retrieval reinforcement ─────────────────────────────
        // Validation-gated: if extraction fell back to a raw event node, the turn's
        // content is unverified — wiring it into the graph would reinforce whatever
        // the extractor hallucinated. Skip.
        val retrievedIds = retrieved.map { it.node.nodeId }
        if (ext != null && retrievedIds.size >= 2 && ext.validationPassed) {
            store.reinforceCoRetrieved(retrievedIds, mT = mT, eta = coRetrievalEta)
        }

        // ── 5. Conversation state ─────────────────────────────────────
        st.history += HistoryTurn("user", message)
        st.history += HistoryTurn("assistant", response)
        if (st.history.size > maxHistory) {
            st.history.subList(0, st.history.size - maxHistory).clear()
        }

        emit(
            TutorEvent.Done(
                ChatTurnResult(
                    response = response,
                    arme = turn?.let { armeSummary(it) },
                    concepts = ext?.extracted?.concepts?.map {
                        ConceptSummary(it.label, it.conceptType, it.description, it.sourceLang)
                    } ?: emptyList(),
                    lang = detectLang(response),
                    retrieved = retrieved.map {
                        RetrievedItem(
                            nodeId = it.node.nodeId,
                            label = it.node.label,
                            textSummary = it.node.textSummary,
                            conceptType = it.node.conceptType,
                            score = it.score,
                        )
                    },
                )
            )
        )
    }.catch { t ->
        if (t is CancellationException) throw t
        Log.e(TAG, "chat failed", t)
        emit(TutorEvent.Failed("${t.javaClass.simpleName}: ${t.message?.take(200)}"))
    }.flowOn(Dispatchers.Default)

    // ── Retrieval practice (adaptive forgetting-curve exercises) ──────

    /** Batched extraction: one extraction+ingest pass over all buffered turns, as a single
     *  "User:/Assistant:" transcript (the shape extractFromTurn already uses per turn).
     *  Emotion is classified from the concatenated user text. */
    private suspend fun flushExtraction(sessionId: String, st: SessionState): IngestedExtractionResult? {
        if (st.pendingTurns.isEmpty()) return null
        val transcript = st.pendingTurns.joinToString("\n\n") { (u, a) -> "User: $u\nAssistant: $a" }
        val userText = st.pendingTurns.joinToString("\n") { it.first }
        st.pendingTurns.clear()
        val ext = extractor.extractAndIngest(
            text = transcript,
            userText = userText,
            sessionId = sessionId,
            sourceType = "user",
        )
        logArme(ext.ingestResults)
        return ext
    }

    /**
     * Rank fading candidates by predicted recall probability (lowest first), then
     * greedily pick up to practiceTopK respecting the spacing guard and the
     * semantic-interference filter (confusables like 食べる/食べた get interleaved
     * across exercises, not crammed into one).
     */
    suspend fun selectPracticeTargets(): List<PracticeTarget> {
        val pool = store.getFadingConcepts(limit = practiceTopK * 6)
        val nowS = System.currentTimeMillis() / 1000.0

        fun recall(f: FadingConcept): Double =
            practiceScheduler.recallProb(nowS - f.concept.updatedAt / 1000.0, f.concept.conceptType)

        val ranked = pool.sortedBy { recall(it) }

        val selected = ArrayList<PracticeTarget>()
        for (cand in ranked) {
            val nid = cand.concept.nodeId
            if (!practiceScheduler.readyAgain(nid)) continue // spacing guard

            val tooSimilar = selected.any { sel ->
                val a = cand.concept.embedding
                val b = sel.fading.concept.embedding
                a.isNotEmpty() && b.isNotEmpty() &&
                    cosineSim(a.toDouble(), b.toDouble()) > interferenceThreshold
            }
            if (tooSimilar) continue

            selected += PracticeTarget(cand, round4(recall(cand)))
            if (selected.size >= practiceTopK) break
        }
        return selected
    }

    /** Dashboard-facing view of the practice queue. */
    suspend fun fadingConcepts(limit: Int = 5): List<FadingConceptView> {
        val nowS = System.currentTimeMillis() / 1000.0
        return store.getFadingConcepts(limit = limit).map { f ->
            FadingConceptView(
                nodeId = f.concept.nodeId,
                label = f.concept.label,
                conceptType = f.concept.conceptType,
                textSummary = f.concept.textSummary,
                avgW = round4(f.avgW),
                degree = f.degree,
                strength = f.strength,
                staleness = f.staleness,
                fadingScore = f.fadingScore,
                recallProb = round4(
                    practiceScheduler.recallProb(nowS - f.concept.updatedAt / 1000.0, f.concept.conceptType)
                ),
            )
        }
    }

    /**
     * Generate one exercise targeting the concepts least likely to be recalled.
     * Returns null when there's nothing worth practicing yet (or generation failed).
     */
    suspend fun nextPractice(sessionId: String): PracticeOffer? {
        val st = state(sessionId)

        val targets = selectPracticeTargets()
        if (targets.isEmpty()) return null

        val targetLines = targets.joinToString("\n") { t ->
            "- ${t.fading.concept.label} (${t.fading.concept.conceptType}): ${t.fading.concept.textSummary}"
        }
        // PRACTICE_SYSTEM goes in-band — the resident system prompt is the persona.
        val prompt = (PRACTICE_SYSTEM + "\n\n" + PRACTICE_TARGETS_HEADER + targetLines +
            "\n\nGenerate the exercise JSON now.")

        val exercise = try {
            val parsed = parseLlmJson(completeFn(prompt, CHAT_MAX_TOKENS))
            PracticeExercise(
                exerciseType = parsed.optString("exercise_type", "translation"),
                prompt = parsed.optString("prompt", "").trim(),
                hint = parsed.optString("hint", "").trim(),
                answer = parsed.optString("answer", "").trim(),
                targets = targets.map { it.fading.concept.label },
                nodeIds = targets.map { it.fading.concept.nodeId },
                targetTypes = targets.map { it.fading.concept.conceptType },
                recallProbs = targets.associate { it.fading.concept.nodeId to it.recallProb },
            )
        } catch (e: Exception) {
            Log.w(TAG, "practice generation failed: $e")
            return null
        }

        if (exercise.prompt.isEmpty() || exercise.answer.isEmpty()) {
            Log.w(TAG, "practice generation returned empty prompt/answer")
            return null
        }

        st.pendingExercise = exercise
        return PracticeOffer(
            exerciseType = exercise.exerciseType,
            prompt = exercise.prompt,
            hint = exercise.hint,
            targets = exercise.targets,
            fading = targets.map { it.fading.concept.label to it.recallProb },
        )
    }

    /**
     * Grade the pending exercise. A correct answer re-encodes memory: target nodes get
     * activation bumps and their pairwise edges a Hebbian reinforcement bump (m_t=1.5 —
     * retrieval success is exactly the co-activation signal the graph is built on).
     */
    suspend fun submitPracticeAnswer(sessionId: String, answer: String): PracticeGradeResult? {
        val st = state(sessionId)
        val ex = st.pendingExercise ?: return null

        val gradePrompt = (PRACTICE_GRADE_SYSTEM + "\n\n" +
            "Exercise: ${ex.prompt}\n" +
            "Expected answer: ${ex.answer}\n" +
            "Learner's answer: $answer\n\n" +
            "Respond with the grading JSON now.")

        val (correct, feedback) = try {
            val parsed = parseLlmJson(completeFn(gradePrompt, CHAT_MAX_TOKENS))
            parsed.optBoolean("correct", false) to parsed.optString("feedback", "").trim()
        } catch (e: Exception) {
            Log.w(TAG, "practice grading failed: $e")
            // Pending exercise stays set so the learner can retry (as in Python).
            return PracticeGradeResult(
                correct = false,
                feedback = "(grading failed — try again)",
                reinforced = false,
            )
        }

        var reinforced = false
        if (correct) {
            for (nid in ex.nodeIds) store.incrementActivation(nid)
            if (ex.nodeIds.size >= 2) {
                store.reinforceCoRetrieved(ex.nodeIds, mT = 1.5, eta = coRetrievalEta)
            }
            reinforced = true
        }

        // Adaptive half-lives learn from the outcome regardless of correctness —
        // wrong answers shorten that type's horizon.
        practiceScheduler.record(
            nodeIds = ex.nodeIds,
            conceptTypes = ex.targetTypes.ifEmpty { ex.targets.take(1) },
            correct = correct,
        )

        st.pendingExercise = null
        return PracticeGradeResult(
            correct = correct,
            feedback = feedback,
            answer = ex.answer,
            targets = ex.targets,
            reinforced = reinforced,
        )
    }

    // ── Ambient ingestion (second-brain / BCI path) ────────────────────

    /**
     * Ingest one decoded ambient utterance with its physiological sample. No chat turn —
     * just extraction + Hebbian ingest, gated by the attention signal ARM-E doesn't get
     * from text chat. Mirrors chat()'s extract+ingest step without retrieve/respond or
     * co-retrieval reinforcement.
     */
    suspend fun ingestAmbient(
        utterance: UtteranceEvent,
        physio: PhysioSample,
        sessionId: String,
    ): AmbientIngestResult {
        val st = state(sessionId)
        val attention = physio.attention ?: 1.0

        val ext = extractor.extractAndIngest(
            text = utterance.text,
            userText = utterance.text,
            sessionId = sessionId,
            sourceType = "sensor",
            sourceUri = null,
            attention = attention,
        )
        st.seedIds += ext.nodeIds
        if (st.seedIds.size > SEED_IDS_MAX) {
            st.seedIds.subList(0, st.seedIds.size - SEED_IDS_MAX).clear()
        }
        logArme(ext.ingestResults)

        val turn = ext.ingestResults.firstOrNull()
        return AmbientIngestResult(
            arme = turn?.let { armeSummary(it) },
            concepts = ext.extracted.concepts.map {
                ConceptSummary(it.label, it.conceptType, it.description, it.sourceLang)
            },
            attention = round4(attention),
            hrBpm = physio.hrBpm,
            utterance = utterance.text,
        )
    }

    // ── Dashboard read models ─────────────────────────────────────────

    private fun conceptToView(c: com.mobilerag.hebbian.store.ConceptNode) = GraphNodeView(
        nodeId = c.nodeId,
        label = c.label,
        conceptType = c.conceptType,
        sourceType = c.sourceType,
        sourceLang = detectLang(c.label),
        activationCount = c.activationCount,
        meanMT = c.meanMT,
        meanDominance = c.meanDominance,
        textSummary = c.textSummary,
        createdAt = c.createdAt,
    )

    suspend fun graphData(
        minWeight: Double = 0.0,
        limit: Int = 500,
        layer: String = "hippocampal",
        includeIsolated: Boolean = false,
    ): GraphData {
        val sg = store.getGlobalGraph(layer = layer, minWeight = minWeight, limit = limit)
        val nodes = sg.nodes.toMutableList()

        // getGlobalGraph only returns nodes touching edges above minWeight — merge in
        // the rest so fresh/weakly-connected concepts still show up in the dashboard.
        if (includeIsolated) {
            val have = nodes.mapTo(HashSet()) { it.nodeId }
            for (c in store.listConcepts(limit = limit)) {
                if (c.nodeId !in have) nodes += c
            }
        }

        return GraphData(
            nodes = nodes.map { conceptToView(it) },
            edges = sg.edges.map {
                GraphEdgeView(
                    source = it.srcId,
                    target = it.dstId,
                    hebbWeight = it.hebbWeight,
                    eligibility = it.eligibility,
                    causalScore = it.causalScore,
                )
            },
        )
    }

    /** Concept plus its 1-hop edges (Python returned the concept only; the UI needs edges). */
    suspend fun nodeDetail(nodeId: String): NodeDetail? {
        val c = store.getConcept(nodeId) ?: return null
        val sg = store.getLocalSubgraph(seedIds = listOf(nodeId), hops = 1, minWeight = 0.0)
        return NodeDetail(
            concept = conceptToView(c),
            edges = sg.edges.map {
                GraphEdgeView(it.srcId, it.dstId, it.hebbWeight, it.eligibility, it.causalScore)
            },
        )
    }

    /** Flattened engine-side ARM-E log for the dashboard sparkline (last [limit] steps). */
    fun armeHistory(limit: Int = 50): List<ArmeHistoryEntry> = armeLog.takeLast(limit)

    /** Folds merged-away node ids into their survivors in the practice history (duplicate merge). */
    fun remapPracticeNodes(remap: Map<String, String>) = practiceScheduler.remapNodes(remap)

    // ── Shutdown ──────────────────────────────────────────────────────

    suspend fun close() {
        for (sid in sessions.keys.toList()) endSession(sid)
        practiceScheduler.save()
        store.close()
    }

    // ── Helpers ───────────────────────────────────────────────────────

    private fun armeSummary(r: IngestResult) = ArmeSummary(
        quadrant = r.quadrant,
        mT = round4(r.mT),
        gated = r.gated,
        valence = round4(r.valence),
        arousal = round4(r.arousal),
        dominance = round4(r.dominance),
        deltaWMean = round6(r.deltaWMean),
    )

    private fun logArme(results: List<IngestResult>) {
        val nowS = System.currentTimeMillis() / 1000.0
        for (r in results) {
            armeLog += ArmeHistoryEntry(
                quadrant = r.quadrant,
                mT = round4(r.mT),
                gated = r.gated,
                valence = round4(r.valence),
                arousal = round4(r.arousal),
                dominance = round4(r.dominance),
                deltaWMean = round6(r.deltaWMean),
                timestamp = nowS,
            )
        }
        while (armeLog.size > ARME_LOG_MAX) armeLog.removeFirst()
    }

    companion object {
        private const val TAG = "TutorEngine"

        /** Cap on per-session seed node list (Python SEED_IDS_MAX). */
        const val SEED_IDS_MAX = 50

        /** model_setup.py chat_fn generates with max_new_tokens=512; practice calls share it. */
        const val CHAT_MAX_TOKENS = 512

        /** Chat turns buffered before one batched extraction+ingest run fires (chat step 3). */
        const val EXTRACTION_BATCH_TURNS = 3

        /** Engine-side ARM-E log cap (Python's ARME._history is unbounded; the UI only reads the tail). */
        private const val ARME_LOG_MAX = 500

        internal fun round4(x: Double): Double = round(x * 1e4) / 1e4
        internal fun round6(x: Double): Double = round(x * 1e6) / 1e6

        private fun FloatArray.toDouble(): DoubleArray = DoubleArray(size) { this[it].toDouble() }
    }
}
