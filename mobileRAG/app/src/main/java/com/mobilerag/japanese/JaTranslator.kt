package com.mobilerag.japanese

import android.content.Context
import android.util.Log
import android.util.LruCache
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.Translator
import com.google.mlkit.nl.translate.TranslatorOptions
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * On-device JA→EN translation for the tap-to-translate overlay.
 *
 * Process-wide singleton: the ML Kit [Translator] owns a loaded model, so creating one per screen
 * would multiply a ~30MB resident footprint on a device that is already hosting a GGUF and an
 * ONNX session. [close] is wired to the same trim-memory path as the LLM.
 *
 * The model downloads once (needs network), then runs fully offline.
 */
object JaTranslator {

    private const val TAG = "JaTranslator"

    sealed interface ModelState {
        /** Not yet requested. */
        data object Idle : ModelState
        /** First-run download in flight. */
        data object Downloading : ModelState
        /** Model on disk; translation is offline from here on. */
        data object Ready : ModelState
        data class Failed(val message: String) : ModelState
    }

    private val _modelState = MutableStateFlow<ModelState>(ModelState.Idle)
    val modelState: StateFlow<ModelState> = _modelState.asStateFlow()

    private var translator: Translator? = null
    private val lock = Mutex()

    /** Tapped words repeat constantly within a conversation; never pay for the same word twice. */
    private val cache = LruCache<String, String>(512)

    private fun getOrCreate(): Translator = translator ?: Translation.getClient(
        TranslatorOptions.Builder()
            .setSourceLanguage(TranslateLanguage.JAPANESE)
            .setTargetLanguage(TranslateLanguage.ENGLISH)
            .build(),
    ).also { translator = it }

    /**
     * Ensures the JA→EN model is on disk. Safe to call repeatedly; concurrent callers share one
     * download. Returns true when translation is possible.
     */
    suspend fun ensureModel(context: Context): Boolean = lock.withLock {
        if (_modelState.value is ModelState.Ready) return@withLock true
        _modelState.value = ModelState.Downloading
        val client = getOrCreate()
        return@withLock try {
            // No DownloadConditions constraints: this is a small, user-initiated, one-time fetch —
            // gating it on wifi+charging would strand the feature on a phone that is neither.
            withContext(Dispatchers.IO) {
                client.downloadModelIfNeeded(DownloadConditions.Builder().build()).await()
            }
            _modelState.value = ModelState.Ready
            true
        } catch (t: CancellationException) {
            _modelState.value = ModelState.Idle
            throw t
        } catch (t: Throwable) {
            Log.w(TAG, "model download failed", t)
            _modelState.value = ModelState.Failed(t.message?.take(120) ?: "download failed")
            false
        }
    }

    /**
     * Translates [word], downloading the model first if needed.
     * Returns null when the model is unavailable or translation fails.
     */
    suspend fun translate(context: Context, word: String): String? {
        val key = word.trim()
        if (key.isEmpty()) return null
        cache.get(key)?.let { return it }
        if (!ensureModel(context)) return null
        val client = translator ?: return null
        return try {
            val result = withContext(Dispatchers.IO) { client.translate(key).await() }
            val cleaned = result?.trim()?.takeIf { it.isNotEmpty() }
            if (cleaned != null) cache.put(key, cleaned)
            cleaned
        } catch (t: CancellationException) {
            throw t
        } catch (t: Throwable) {
            Log.w(TAG, "translate failed for '$key'", t)
            null
        }
    }

    /** Releases the loaded model. Call from onTrimMemory alongside the LLM unload. */
    fun close() {
        runCatching { translator?.close() }
        translator = null
        if (_modelState.value is ModelState.Ready) _modelState.value = ModelState.Idle
    }
}

/**
 * Bridges a Play-services [com.google.android.gms.tasks.Task] into a coroutine.
 * Kept local so the app doesn't need the kotlinx-coroutines-play-services artifact for two calls.
 */
private suspend fun <T> com.google.android.gms.tasks.Task<T>.await(): T =
    suspendCancellableCoroutine { cont ->
        addOnSuccessListener { result -> if (cont.isActive) cont.resume(result) }
        addOnFailureListener { e -> if (cont.isActive) cont.resumeWithException(e) }
        addOnCanceledListener { cont.cancel() }
    }
