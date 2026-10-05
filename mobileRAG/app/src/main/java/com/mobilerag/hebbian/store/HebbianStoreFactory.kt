package com.mobilerag.hebbian.store

import android.content.Context
import android.util.Log
import kotlinx.coroutines.runBlocking
import java.io.File

/**
 * Picks the Hebbian store: LadybugDB (Cypher) when the native library loads on this
 * device, SQLite otherwise. The choice is logged to logcat under "HebbianStore".
 *
 * [spaceDir] scopes the store to a memory space (SpaceManager): Ladybug uses
 * `<spaceDir>/hebbian.lbug`, the SQLite fallback `<spaceDir>/hebbian_graph.db`, and the
 * `store-choice.txt` diagnostic is written into `<spaceDir>`. With the default null
 * the legacy global paths are used (`files/hebbian/...`, `hebbian_graph.db`).
 *
 * The choice is sticky per space: once a space has a SQLite store (it only ever gets one
 * from a fallback), it stays on SQLite — reopening LadybugDB later would silently swap
 * the user's memories for whatever the older LadybugDB file holds.
 *
 * A single cached store is shared by all callers, so repeated create() calls return
 * the same instance and no database handle leaks. Use [closeCurrent] for lifecycle
 * cleanup (e.g. process teardown or a space switch); after it, the next create()
 * opens a fresh store.
 */
object HebbianStoreFactory {

    private const val TAG = "HebbianStore"

    @Volatile
    private var current: HebbianStore? = null

    fun create(context: Context, spaceDir: File? = null): HebbianStore {
        current?.let { return it }
        return synchronized(this) {
            current?.let { return@synchronized it }
            val appContext = context.applicationContext
            val ladybugPath = if (spaceDir != null) {
                spaceDir.mkdirs()
                File(spaceDir, "hebbian.lbug").absolutePath
            } else {
                File(appContext.filesDir, "hebbian/hebbian.lbug").absolutePath
            }
            // logcat is unreliable on the RedMagic (see phase4-results) — persist the choice
            val choiceDir = spaceDir ?: File(appContext.filesDir, "hebbian")
            val sqlitePath = spaceDir?.let { File(it, "hebbian_graph.db").absolutePath }
            val sqliteFile = sqlitePath?.let(::File) ?: appContext.getDatabasePath("hebbian_graph.db")
            if (sqliteFile.exists()) {
                Log.i(TAG, "Using the space's existing SQLite Hebbian store at $sqliteFile")
                // Keep the original fallback reason if it's recorded.
                val prior = runCatching { File(choiceDir, "store-choice.txt").readText() }.getOrNull()
                if (prior?.startsWith("chosen=sqlite-hebbian") != true) {
                    writeChoice(choiceDir, "sqlite-hebbian", null, "sticky: this space already has a SQLite store")
                }
                return@synchronized SqliteHebbianStore(appContext, sqlitePath).also { current = it }
            }
            val store = try {
                val ladybug = LadybugHebbianStore()
                // open() is suspend but create() runs during app wiring, before a coroutine
                // scope exists — bridge with runBlocking (open dispatches to Dispatchers.IO).
                runBlocking { ladybug.open(ladybugPath) }
                Log.i(TAG, "Using LadybugDB Hebbian store at $ladybugPath")
                ladybug.walRecovery?.let { Log.w(TAG, "LadybugDB WAL recovery: $it") }
                writeChoice(choiceDir, "ladybugdb-hebbian", null, ladybug.walRecovery?.let { "wal_recovery=$it" })
                ladybug
            } catch (t: Throwable) {
                Log.w(TAG, "LadybugDB unavailable (${t.message}); falling back to SQLite")
                writeChoice(choiceDir, "sqlite-hebbian", t)
                SqliteHebbianStore(appContext, sqlitePath)
            }
            current = store
            store
        }
    }

    /** Closes the shared store (if any); the next [create] opens a fresh one. */
    fun closeCurrent() {
        synchronized(this) {
            val store = current ?: return
            current = null
            try {
                runBlocking { store.close() }
            } catch (t: Throwable) {
                Log.w(TAG, "Failed to close Hebbian store (${t.message})")
            }
        }
    }

    private fun writeChoice(dir: File, chosen: String, error: Throwable?, note: String? = null) {
        runCatching {
            val f = File(dir, "store-choice.txt")
            f.parentFile?.mkdirs()
            f.writeText(buildString {
                append("chosen=").append(chosen).append('\n')
                if (note != null) append(note).append('\n')
                if (error != null) append("error=").append(error.stackTraceToString())
            })
        }
    }
}
