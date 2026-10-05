package com.mobilerag.generation

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.arm.aichat.AiChat
import kotlinx.coroutines.sync.Mutex
import java.io.File

/**
 * Process-wide owner of the AiChat InferenceEngine singleton, shared by RagPipeline (Khepri
 * chat), the Hebbian tutor, practice and the second brain.
 *
 * Residency is tracked in two parts:
 *  - [loadedModel]: which GGUF's weights are in memory (the expensive part — an mmap load of
 *    several GB).
 *  - [promptKey]: which consumer's system prompt currently owns the conversation.
 *
 * Switching tabs on the same GGUF only changes the prompt key, and that is served by
 * InferenceEngine.resetConversation (one system-prompt prefill) instead of cleanUp +
 * loadModel. A full reload happens only when the selected GGUF itself changes.
 *
 * [mutex] guards EVERY use of the engine — load, prompt swap and generation — so one tab can't
 * swap its persona in (or unload the model) while another tab's reply or background extraction
 * is still generating. Unload paths use tryLock and skip rather than block: an in-progress
 * generation always wins over a memory-pressure unload.
 */
object LlmResidency {

    private const val TAG = "LlmResidency"

    /** Absolute path of the resident GGUF, or null when nothing is loaded. */
    @Volatile
    var loadedModel: String? = null
        private set

    /** Key of the system prompt that owns the current conversation, or null. */
    @Volatile
    var promptKey: String? = null
        private set

    val mutex = Mutex()

    /**
     * @param readyMs time spent making the engine ready (weights load and/or prompt prefill; 0
     *   when nothing had to change).
     * @param conversationReset true when the KV cache was cleared — the caller's accumulated
     *   conversation is gone.
     */
    data class Ready(val readyMs: Long, val conversationReset: Boolean)

    /**
     * Makes [model] resident with [systemPrompt] owning the conversation. Caller must hold
     * [mutex], and keep holding it through generation.
     */
    suspend fun ensureLocked(context: Context, model: File, promptKey: String, systemPrompt: String): Ready {
        val path = model.absolutePath
        if (loadedModel == path && this.promptKey == promptKey) return Ready(0, false)
        val gen = AiChat.getInferenceEngine(context)
        val start = SystemClock.elapsedRealtime()

        if (loadedModel == path) {
            Log.i(TAG, "prompt swap ${this.promptKey} -> $promptKey (weights stay resident)")
            this.promptKey = null
            try {
                gen.resetConversation(systemPrompt)
            } catch (t: Throwable) {
                // Leave nothing claimed resident so the next call takes the full reload path.
                loadedModel = null
                throw t
            }
            this.promptKey = promptKey
            return Ready(SystemClock.elapsedRealtime() - start, true)
        }

        runCatching { gen.cleanUp() }
        loadedModel = null
        this.promptKey = null
        Log.i(TAG, "loading ${model.name} for $promptKey")
        gen.loadModel(path)
        gen.setSystemPrompt(systemPrompt) // must be called immediately after loadModel
        loadedModel = path
        this.promptKey = promptKey
        return Ready(SystemClock.elapsedRealtime() - start, true)
    }

    /** Unloads the resident weights. Caller must hold [mutex]. */
    fun unloadLocked(context: Context) {
        runCatching { AiChat.getInferenceEngine(context).cleanUp() }
        loadedModel = null
        promptKey = null
        Log.i(TAG, "model unloaded")
    }

    /** Unloads the resident weights (e.g. onTrimMemory) unless the engine is busy — never
     *  blocks, never interrupts a generation. Returns true when something was unloaded. */
    fun unloadIfIdle(context: Context): Boolean {
        if (loadedModel == null) return false
        if (!mutex.tryLock()) return false
        try {
            if (loadedModel == null) return false
            unloadLocked(context)
            return true
        } finally {
            mutex.unlock()
        }
    }
}
