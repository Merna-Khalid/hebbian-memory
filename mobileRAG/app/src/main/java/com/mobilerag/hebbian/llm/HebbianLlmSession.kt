package com.mobilerag.hebbian.llm

import android.content.Context
import android.util.Log
import com.arm.aichat.AiChat
import com.mobilerag.generation.LlmResidency
import com.mobilerag.generation.ThinkTagFilter
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.sync.withLock
import java.io.File

/**
 * LLM session owner for the Hebbian personas (tutor chat, extraction, summaries, practice
 * generation/grading) over the shared AiChat InferenceEngine singleton.
 *
 * Why this exists: the InferenceEngine is a stateful conversational session — the system prompt
 * is set once per conversation, and every `sendUserPrompt` extends the KV cache. The Python
 * system instead makes stateless calls with per-call system prompts (chat, extraction and
 * grading each use different ones). On-device we map that to: one persona system prompt per
 * session key, everything else sent as in-band user-turn instructions, and a periodic
 * conversation reset to bound prefill cost over the accumulated KV (the build plan's
 * token-budget discipline — tutor memory lives in the graph, not the context, so a reset loses
 * only chit-chat continuity).
 *
 * Residency goes through [LlmResidency]: the persona is re-asserted under the engine mutex on
 * every send, so if Khepri chat (or the other Hebbian persona) took the engine since our last
 * turn, this turn swaps our system prompt back in rather than generating under theirs. Swaps
 * and KV resets reuse the resident weights — only a different GGUF triggers a reload.
 */
class HebbianLlmSession(private val context: Context) {

    private var model: File? = null
    private var promptKey: String? = null
    private var systemPrompt: String? = null

    /** User turns sent since this persona's conversation last started — chat and extraction
     *  calls both count. */
    var turnsSinceReset = 0
        private set

    /** Makes [model] resident with [systemPrompt] (swapping the prompt in place when the same
     *  GGUF is already loaded for another persona) and remembers the configuration so later
     *  sends can re-assert it. Returns ready time in ms (0 when nothing had to change). The
     *  prompt hash is part of the key, so a changed persona/learner block takes effect on the
     *  next call without an explicit unload. */
    suspend fun ensureLoaded(model: File, sessionKey: String, systemPrompt: String): Long =
        LlmResidency.mutex.withLock {
            this.model = model
            this.promptKey = "hebbian|$sessionKey|${systemPrompt.hashCode()}"
            this.systemPrompt = systemPrompt
            acquireLocked()
        }

    /** Caller holds [LlmResidency.mutex]. */
    private suspend fun acquireLocked(): Long {
        val m = checkNotNull(model) { "HebbianLlmSession used before ensureLoaded" }
        val ready = LlmResidency.ensureLocked(context, m, promptKey!!, systemPrompt!!)
        if (ready.conversationReset) turnsSinceReset = 0
        return ready.readyMs
    }

    /** Streams visible tokens for [prompt] (Qwen3 `<think>` blocks filtered). Counts one KV turn;
     *  resets the conversation first when the turn budget is exhausted. Holds the engine for
     *  the whole generation, so concurrent callers (another tab, a session-end extraction
     *  flush) queue instead of interleaving into one KV cache. */
    fun send(prompt: String, maxTokens: Int): Flow<String> = flow {
        LlmResidency.mutex.withLock {
            acquireLocked()
            val gen = AiChat.getInferenceEngine(context)
            if (turnsSinceReset >= MAX_KV_TURNS) {
                Log.i(TAG, "conversation reset after $turnsSinceReset turns (KV budget)")
                gen.resetConversation(systemPrompt!!) // same weights, fresh KV — no reload
                turnsSinceReset = 0
            }
            turnsSinceReset++
            val filter = ThinkTagFilter()
            // Qwen3 soft switch (same as RagPipeline): hidden reasoning shares maxTokens with
            // the answer — for extraction JSON it truncated the output and forced retries.
            gen.sendUserPrompt("$prompt\n/no_think", maxTokens).collect { token ->
                val visible = filter.feed(token)
                if (visible.isNotEmpty()) emit(visible)
            }
            filter.flush().takeIf { it.isNotEmpty() }?.let { emit(it) }
        }
    }

    /** Collects [send] into a single string — for extraction/summary/grading calls. */
    suspend fun complete(prompt: String, maxTokens: Int): String {
        val sb = StringBuilder()
        send(prompt, maxTokens).collect { sb.append(it) }
        return sb.toString().trim()
    }

    /** Unloads the resident model (e.g. onTrimMemory); skips rather than blocks when the
     *  engine is busy loading or generating. */
    fun unload() {
        LlmResidency.unloadIfIdle(context)
    }

    companion object {
        private const val TAG = "HebbianLlmSession"

        /** Max user turns (chat + extraction + grading) kept in the KV cache before a reset. */
        const val MAX_KV_TURNS = 12
    }
}
