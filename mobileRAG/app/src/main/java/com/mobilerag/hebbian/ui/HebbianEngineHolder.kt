package com.mobilerag.hebbian.ui

import android.content.Context
import com.mobilerag.hebbian.HebbianEngineFactory
import com.mobilerag.hebbian.TutorEngine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * Process-wide holder for the shared tutor-domain engine — the Tutor, Practice, and
 * Memory tabs all drive ONE TutorEngine (same singleton idiom as HebbianStoreFactory /
 * GraphStoreFactory). The second-brain tab builds its own second_brain-domain engine and
 * keeps it privately; both engines share the on-disk store via HebbianStoreFactory.
 */
object HebbianEngineHolder {

    private val buildMutex = Mutex()
    private val modelMutex = Mutex()

    @Volatile
    private var tutorEngine: TutorEngine? = null

    /** Builds (stores + EmbeddingGemma session) on first call; call off the main thread. */
    suspend fun tutor(context: Context): TutorEngine {
        tutorEngine?.let { return it }
        return buildMutex.withLock {
            tutorEngine ?: HebbianEngineFactory.build(context.applicationContext, "japanese_tutor")
                .also { tutorEngine = it }
        }
    }

    /** Idempotent GGUF load; returns load ms (0 when already resident). Serialized so two
     *  tabs can't trigger a double load of the same model. */
    suspend fun ensureModel(context: Context, engine: TutorEngine): Long = modelMutex.withLock {
        HebbianEngineFactory.ensureModelsLoaded(context.applicationContext, engine)
    }

    /** Process-level scope for work that must outlive a screen ViewModel — currently the
     *  session-end extraction flush, which is an LLM call and must never block onCleared. */
    val scope = kotlinx.coroutines.CoroutineScope(
        kotlinx.coroutines.SupervisorJob() + kotlinx.coroutines.Dispatchers.IO
    )

    /** The current engine without building it (null when not built yet). */
    fun peek(): TutorEngine? = tutorEngine

    /** Drops the cached engine (memory-space switch). Unloads its LLM first; the stores are
     *  the factories' responsibility (closeCurrent). The next tutor() call rebuilds against
     *  the newly active space. */
    fun reset() {
        tutorEngine?.let { runCatching { it.llm?.unload() } }
        tutorEngine = null
    }
}
