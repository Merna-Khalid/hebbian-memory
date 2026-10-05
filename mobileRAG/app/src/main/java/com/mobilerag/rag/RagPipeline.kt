package com.mobilerag.rag

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.arm.aichat.AiChat
import com.mobilerag.core.GenerationBackend
import com.mobilerag.core.GraphStore
import com.mobilerag.embeddings.EmbeddingGemmaEngine
import com.mobilerag.generation.LlamaCppBackend
import com.mobilerag.generation.LlmResidency
import kotlinx.coroutines.sync.withLock
import com.mobilerag.generation.MlKitGenerationBackend
import com.mobilerag.generation.ThinkTagFilter
import com.mobilerag.rag.retrieve.CommunityRetriever
import com.mobilerag.rag.retrieve.RetrievalEngine
import com.mobilerag.rag.retrieve.RetrievalHit
import com.mobilerag.spikes.QueryTimer
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import java.io.File

sealed class RagEvent {
    data class Retrieved(val chunks: List<RetrievalHit>) : RagEvent()
    data class Token(val text: String) : RagEvent()
    data class Done(val stats: QueryStats) : RagEvent()
    data class Failed(val error: String) : RagEvent()
}

data class QueryStats(
    val embedMs: Long,
    val searchMs: Long,
    val modelLoadMs: Long,
    val ttftMs: Long,
    val tokens: Int,
    val tokPerSec: Double,
    val pssKb: Long,
    val heapMb: Long,
    val channelCounts: Map<String, Int> = emptyMap(),
    val channelMs: Map<String, Long> = emptyMap(),
) {
    fun format(): String = buildString {
        append("embed ${embedMs}ms · search ${searchMs}ms")
        if (modelLoadMs > 0) append(" · load ${modelLoadMs}ms")
        append(" · TTFT ${ttftMs}ms · %.1f tok/s · PSS %d MB / heap %d MB".format(tokPerSec, pssKb / 1024, heapMb))
        for (channel in CHANNEL_ORDER) {
            val count = channelCounts[channel] ?: continue
            append(" · $channel $count/${channelMs[channel] ?: 0}ms")
        }
    }

    private companion object {
        val CHANNEL_ORDER = listOf("vec", "fts", "graph", "comm")
    }
}

/**
 * RAG pipeline: hybrid retrieval (vector + FTS + graph, RRF-fused, via [RetrievalEngine]) →
 * strict-grounding prompt → streamed generation. The GGUF model is loaded lazily once and kept
 * resident across queries (mmap load is ~4 s); a model change triggers cleanUp + loadModel.
 */
class RagPipeline(
    private val context: Context,
    private val engine: EmbeddingGemmaEngine,
    private val retrieval: RetrievalEngine,
    private val communities: CommunityRetriever? = null,
) {
    private val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    private var resolvedAutoBackend: String? = null

    // Default is vector: the Phase 3 eval (docs/phase3-results.md) showed a recall tie between
    // modes at this corpus scale, so the simpler channel stays the default until hybrid proves
    // out on a larger corpus. The UI toggle overrides this.
    fun retrievalMode(): String = prefs.getString(KEY_RETRIEVAL_MODE, MODE_VECTOR) ?: MODE_VECTOR

    fun setRetrievalMode(mode: String) {
        prefs.edit().putString(KEY_RETRIEVAL_MODE, mode).apply()
    }

    fun generationBackend(): String = prefs.getString(KEY_GENERATION_BACKEND, BACKEND_AUTO) ?: BACKEND_AUTO

    fun setGenerationBackend(backend: String) {
        prefs.edit().putString(KEY_GENERATION_BACKEND, backend).apply()
        resolvedAutoBackend = null
    }

    /** `auto` prefers Gemini Nano when ML Kit reports it available; devices without AICore fall
     *  back to llama.cpp. The auto resolution is cached per pipeline instance (checkStatus is an
     *  IPC) and invalidated when the setting changes. */
    private suspend fun resolveBackend(): String = when (val setting = generationBackend()) {
        BACKEND_LLAMACPP, BACKEND_MLKIT -> setting
        else -> resolvedAutoBackend ?: run {
            val available = MlKitGenerationBackend().checkAvailability() is GenerationBackend.Status.Available
            (if (available) BACKEND_MLKIT else BACKEND_LLAMACPP).also {
                resolvedAutoBackend = it
                Log.i(TAG, "auto backend -> $it")
            }
        }
    }

    fun availableModels(): List<File> =
        File(context.filesDir, "models").listFiles()
            ?.filter { it.extension == "gguf" && it.canRead() }
            ?.sortedBy { it.name } ?: emptyList()

    fun selectedModel(): File? {
        val models = availableModels()
        val name = pickModel(models.map { it.name }, prefs.getString(KEY_MODEL, null))
        return models.firstOrNull { it.name == name }
    }

    fun ask(question: String): Flow<RagEvent> = flow {
        val timer = QueryTimer()
        timer.mark("retrieve")
        val mode = retrievalMode()
        val hybrid = mode != MODE_VECTOR // global runs the full hybrid pass underneath
        val result = retrieval.retrieve(question, hybrid)
        var commMs = 0L
        val communityHits = if (mode == MODE_GLOBAL && communities != null) {
            val start = SystemClock.elapsedRealtime()
            communities.retrieve(engine.embedQuery(question)).also {
                commMs = SystemClock.elapsedRealtime() - start
            }
        } else {
            emptyList()
        }
        val retrieveMs = timer.since("retrieve")
        val embedMs = result.channelMs["embed"] ?: 0L
        val searchMs = retrieveMs - embedMs
        val channelCounts =
            if (mode == MODE_GLOBAL) result.channelCounts + ("comm" to communityHits.size) else result.channelCounts
        val channelMs =
            if (commMs > 0) (result.channelMs - "embed") + ("comm" to commMs) else result.channelMs - "embed"
        fun stats(loadMs: Long, ttftMs: Long, tokens: Int, tokPerSec: Double) =
            QueryStats(embedMs, searchMs, loadMs, ttftMs, tokens, tokPerSec, QueryTimer.pssKb(), QueryTimer.heapMb(), channelCounts, channelMs)

        // Similarity/coverage gate (see RetrievalEngine): a weak signal means the notes don't
        // cover the message. It used to end in a canned "I don't know" without running the
        // model at all — so "hello" or "tell me about yourself" hit a wall. Now Khepri answers
        // as itself, conversationally, and its persona prompt makes it say when an answer isn't
        // from the notes; no passages are shown, so the UI never implies a source. In global
        // mode a community match (score ≥ 0.30) still overrides the gate.
        val grounded = !(result.gated && communityHits.isEmpty())
        if (!grounded) Log.i(TAG, "gated (mode=$mode) → conversational reply: $channelCounts")

        // Community summaries lead the evidence, then the highest-ranked fused chunks that fit
        // the remaining context budget.
        val communityLines = mutableListOf<String>()
        var budget = CONTEXT_TOKEN_BUDGET
        communityHits.forEachIndexed { i, hit ->
            val line = "[C${i + 1}] ${hit.summary}"
            val tokens = engine.countTokens(line)
            if (tokens <= budget) {
                communityLines += line
                budget -= tokens
            }
        }
        val chosen = mutableListOf<RetrievalHit>()
        if (grounded) for (hit in result.hits.take(TOP_K)) {
            val tokens = engine.countTokens(hit.text)
            if (tokens <= budget) {
                chosen += hit
                budget -= tokens
            }
        }
        emit(RagEvent.Retrieved(chosen))

        val prompt = if (grounded) buildString {
            communityLines.forEach { append(it).append("\n\n") }
            chosen.forEachIndexed { i, c -> append("[${i + 1}] ${c.text}\n\n") }
            if (communityLines.isNotEmpty()) append(COMMUNITY_LINES_NOTE)
            append("Question: $question\n/no_think") // Qwen3: disable thinking so it doesn't eat the output budget
        } else {
            "$question\n/no_think"
        }
        val promptKey = if (grounded) PROMPT_KEY else CHAT_PROMPT_KEY
        val systemPrompt = if (grounded) SYSTEM_PROMPT else CHAT_SYSTEM_PROMPT

        var ttftMs = -1L
        var tokens = 0
        val thinkFilter = ThinkTagFilter()
        // The empty think block Qwen3 still emits under /no_think leaves blank lines; drop
        // whitespace until the first real character.
        var started = false
        suspend fun stream(tokenFlow: Flow<String>) {
            timer.mark("gen")
            tokenFlow.collect { token ->
                if (ttftMs < 0) ttftMs = timer.since("gen")
                tokens++
                var visible = thinkFilter.feed(token)
                if (!started) {
                    visible = visible.trimStart()
                    if (visible.isNotEmpty()) started = true
                }
                if (visible.isNotEmpty()) emit(RagEvent.Token(visible))
            }
        }

        var loadMs = 0L
        if (resolveBackend() == BACKEND_MLKIT) {
            stream(MlKitGenerationBackend().generate(prompt, MAX_OUTPUT_TOKENS))
        } else {
            val model = selectedModel()
            if (model == null) {
                emit(RagEvent.Failed("No .gguf model under ${File(context.filesDir, "models").absolutePath} — push one via adb"))
                return@flow
            }
            // Hold the shared engine from prompt swap through the last token: a tutor turn
            // queued meanwhile must not swap its persona in under this answer.
            LlmResidency.mutex.withLock {
                loadMs = LlmResidency.ensureLocked(context, model, promptKey, systemPrompt).readyMs
                stream(AiChat.getInferenceEngine(context).sendUserPrompt(prompt, MAX_OUTPUT_TOKENS))
            }
        }
        thinkFilter.flush().let { if (started) it else it.trimStart() }.takeIf { it.isNotEmpty() }?.let { emit(RagEvent.Token(it)) }
        val decodeMs = timer.since("gen") - ttftMs.coerceAtLeast(0)
        val tokPerSec = if (decodeMs > 0 && tokens > 1) (tokens - 1) * 1000.0 / decodeMs else 0.0
        emit(RagEvent.Done(stats(loadMs, ttftMs.coerceAtLeast(0), tokens, tokPerSec)))
    }.catch { t ->
        if (t is CancellationException) throw t
        Log.e(TAG, "ask failed", t)
        emit(RagEvent.Failed("${t.javaClass.simpleName}: ${t.message?.take(200)}"))
    }.flowOn(Dispatchers.Default)

    /** Makes [model] resident with the grounding prompt; caller holds [LlmResidency.mutex].
     *  When the tutor left the same GGUF loaded this is a prompt swap, not a reload. Returns
     *  ready time in ms (0 when already resident for RAG). */
    private suspend fun ensureModelLoadedLocked(model: File): Long =
        LlmResidency.ensureLocked(context, model, PROMPT_KEY, SYSTEM_PROMPT).readyMs

    /** Eager warm-up for screen entry (AI Edge Gallery loads the model when the chat opens, not
     *  on first send). No-op on the MLKit backend or when nothing is selected; returns load ms. */
    suspend fun preloadModel(): Long {
        if (resolveBackend() == BACKEND_MLKIT) return 0
        val model = selectedModel() ?: return 0
        return LlmResidency.mutex.withLock { ensureModelLoadedLocked(model) }
    }

    /** Unloads the resident GGUF model so its pages can be reclaimed (e.g. onTrimMemory); the
     *  next query reloads it lazily. Skips (rather than blocks) when the engine is busy. */
    fun unloadModel() {
        LlmResidency.unloadIfIdle(context)
    }

    /** Phase 4: regenerates community summaries over [graph] via [CommunitySummarizer]. On the
     *  llama.cpp backend the shared engine is held for the whole run (many sequential
     *  generations that must all use the grounding prompt); a model this run loaded itself is
     *  unloaded afterwards so background runs leave nothing resident, while a model the user
     *  already had loaded stays. The last-run timestamp is persisted under
     *  [KEY_COMMUNITY_LAST_RUN]. Returns the number of communities persisted. */
    suspend fun rebuildCommunities(graph: GraphStore, db: RagDatabase): Int {
        suspend fun rebuild(generator: GenerationBackend): Int =
            CommunitySummarizer(graph, db, engine, generator).rebuild().also {
                prefs.edit().putLong(KEY_COMMUNITY_LAST_RUN, System.currentTimeMillis()).apply()
            }

        if (resolveBackend() == BACKEND_MLKIT) return rebuild(MlKitGenerationBackend())
        val model = selectedModel()
            ?: throw IllegalStateException("No .gguf model under ${File(context.filesDir, "models").absolutePath}")
        return LlmResidency.mutex.withLock {
            val wasResident = LlmResidency.loadedModel == model.absolutePath
            ensureModelLoadedLocked(model)
            try {
                rebuild(LlamaCppBackend(context)) // generate() delegates to the now-loaded InferenceEngine singleton
            } finally {
                if (!wasResident) LlmResidency.unloadLocked(context)
            }
        }
    }

    fun communityLastRun(): Long = prefs.getLong(KEY_COMMUNITY_LAST_RUN, 0L)

    companion object {
        private const val TAG = "RagPipeline"

        /**
         * The chat GGUF every consumer uses: saved preference, then 8B, then 1.7B, then the
         * first. Benchmarked on RedMagic 10 Pro (docs/phase1-results.md): 1.7B names entities
         * and quotes sources at ~25–29 tok/s; 8B Q4_K_M is the quality default when present.
         * Tutor, practice, second brain and the chat picker all resolve through here — if any
         * of them picked a different file, a tab switch would be a real multi-GB reload.
         */
        fun pickModel(names: List<String>, saved: String?): String? =
            names.firstOrNull { it == saved }
                ?: names.firstOrNull { "8B" in it }
                ?: names.firstOrNull { "1.7B" in it }
                ?: names.firstOrNull()
        private const val PREFS = "rag"
        const val KEY_MODEL = "selected_model"
        const val KEY_RETRIEVAL_MODE = "retrieval_mode"
        const val KEY_GENERATION_BACKEND = "generation_backend"
        const val BACKEND_AUTO = "auto"
        const val BACKEND_LLAMACPP = "llamacpp"
        const val BACKEND_MLKIT = "mlkit"
        const val MODE_VECTOR = "vector"
        const val MODE_HYBRID = "hybrid"
        const val MODE_GLOBAL = "global"
        const val KEY_COMMUNITY_LAST_RUN = "community_last_run"
        private const val TOP_K = 5
        private const val CONTEXT_TOKEN_BUDGET = 2000
        const val MAX_OUTPUT_TOKENS = 1024 // Qwen3 thinking (if any) shares this budget
        private const val COMMUNITY_LINES_NOTE =
            "The [C1], [C2]… lines above are thematic summaries of the whole note set; the numbered [1], [2]… lines are individual passages. Cite them accordingly.\n"
        /** LlmResidency prompt key for the grounding system prompt. */
        private const val PROMPT_KEY = "rag"
        private const val SYSTEM_PROMPT =
            "You are Khepri, the second brain of the person you are talking to: a private assistant on their phone that knows the notes they import. " +
                "Speak to them directly as \"you\". Answer using ONLY the provided context from their notes. Cite sources as [1], [2]… matching the numbered context passages. " +
                "If the context does not contain the answer, say plainly that your notes don't cover it. Never use knowledge outside the context. Be warm and concise."

        /** LlmResidency prompt key for Khepri's conversational (no matching notes) replies. */
        private const val CHAT_PROMPT_KEY = "khepri-chat"
        private const val CHAT_SYSTEM_PROMPT =
            "You are Khepri, a personal second brain that lives entirely on this phone. " +
                "Example of how you talk: \"I'm Khepri — I live on your phone, read the notes you import, and help you think. Nothing you tell me leaves this device.\" " +
                "Your name comes from the Egyptian scarab god of the rising sun and renewal. " +
                "Always address the person as \"you\"; never say \"the user\" or \"they\". Talk naturally, warmly and briefly — two to five sentences unless asked for more. " +
                "Nothing in their notes matched this message, so answer from general knowledge; for factual questions, mention briefly that the answer isn't from their notes. " +
                "Never invent personal details or claim the notes say something."
    }
}
