package com.mobilerag.graph

import android.content.Context
import android.util.Log
import com.mobilerag.core.GraphStore
import kotlinx.coroutines.runBlocking
import java.io.File

/**
 * Picks the graph store: LadybugDB (Cypher) when the native library loads on this
 * device, SQLite otherwise. The choice is logged to logcat under "GraphStore".
 *
 * [spaceDir] scopes the store to a memory space (SpaceManager): Ladybug uses
 * `<spaceDir>/graph.lbug`, the SQLite fallback `<spaceDir>/graph.db`. With the default
 * null the legacy global paths are used (`files/graph/ladybug`, `graph.db`).
 *
 * A single cached store is shared by all callers (Chat and Graph screens), so repeated
 * create() calls return the same instance and no database handle leaks. Use
 * [closeCurrent] for lifecycle cleanup (e.g. process teardown or a space switch);
 * after it, the next create() opens a fresh store.
 */
object GraphStoreFactory {

    private const val TAG = "GraphStore"

    @Volatile
    private var current: GraphStore? = null

    fun create(context: Context, spaceDir: File? = null): GraphStore {
        current?.let { return it }
        return synchronized(this) {
            current?.let { return@synchronized it }
            val appContext = context.applicationContext
            val ladybugPath = if (spaceDir != null) {
                spaceDir.mkdirs()
                File(spaceDir, "graph.lbug").absolutePath
            } else {
                File(appContext.filesDir, "graph/ladybug").absolutePath
            }
            val store = try {
                val ladybug = LadybugGraphStore()
                // open() is suspend but create() runs during app wiring, before a coroutine
                // scope exists — bridge with runBlocking (open dispatches to Dispatchers.IO).
                runBlocking { ladybug.open(ladybugPath) }
                Log.i(TAG, "Using LadybugDB graph store at $ladybugPath")
                ladybug
            } catch (t: Throwable) {
                Log.w(TAG, "LadybugDB unavailable (${t.message}); falling back to SQLite")
                SqliteGraphStore(appContext, spaceDir?.let { File(it, "graph.db").absolutePath })
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
                Log.w(TAG, "Failed to close graph store (${t.message})")
            }
        }
    }
}
