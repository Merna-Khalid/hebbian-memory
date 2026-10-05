package com.mobilerag.profile

import android.content.Context
import android.util.Log
import java.io.File
import java.util.UUID

/**
 * Memory spaces: per-profile data isolation. Each space owns its RAG database, entity
 * graph, Hebbian graph and practice state under `files/spaces/<id>/`; models, theme and
 * eval data stay global.
 *
 * Spaces are registered in the shared "app_settings" preferences (keys `space_ids`,
 * `space_name_<id>`, `activeSpaceId`) so the settings UI can list/switch them without
 * touching storage internals. The built-in "personal" space always exists by default and
 * receives all pre-spaces data via [ensureMigration].
 *
 * Store factories take the space dir explicitly (see GraphStoreFactory /
 * HebbianStoreFactory); switching spaces is closeCurrent + activity recreate, handled
 * by the settings UI, not here.
 */
object SpaceManager {

    private const val TAG = "SpaceManager"
    private const val PREFS = "app_settings"
    private const val KEY_SPACE_IDS = "space_ids"
    private const val KEY_SPACE_NAME_PREFIX = "space_name_"
    private const val KEY_ACTIVE = "activeSpaceId"
    private const val DEFAULT_ID = "personal"
    private const val DEFAULT_NAME = "Personal"

    /** Written into files/spaces/ once the one-time data migration has run. */
    private const val MIGRATION_MARKER = ".migrated"

    private val migrationLock = Any()

    data class Space(val id: String, val name: String)

    private fun prefs(context: Context) =
        context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    /** All registered spaces; the default "personal" space when none are persisted yet. */
    fun listSpaces(context: Context): List<Space> {
        val p = prefs(context)
        val ids = p.getString(KEY_SPACE_IDS, null)
            ?.split(',')?.filter { it.isNotBlank() }
            ?: return listOf(Space(DEFAULT_ID, DEFAULT_NAME))
        return ids.map { id -> Space(id, p.getString(KEY_SPACE_NAME_PREFIX + id, null) ?: id) }
            .ifEmpty { listOf(Space(DEFAULT_ID, DEFAULT_NAME)) }
    }

    /** The active space; falls back to the first space if the persisted id is stale. */
    fun activeSpace(context: Context): Space {
        val spaces = listSpaces(context)
        val activeId = prefs(context).getString(KEY_ACTIVE, DEFAULT_ID)
        return spaces.firstOrNull { it.id == activeId } ?: spaces.first()
    }

    /**
     * Data dir of the active space (`files/spaces/<activeId>`, created on demand).
     * Runs [ensureMigration] first, so resolving the active dir before opening any
     * store guarantees legacy data has been moved.
     */
    fun activeDir(context: Context): File {
        ensureMigration(context)
        return spaceDir(context, activeSpace(context).id)
    }

    /** Data dir of the given space (`files/spaces/<id>`, created on demand). */
    fun spaceDir(context: Context, id: String): File =
        File(context.applicationContext.filesDir, "spaces/$id").apply { mkdirs() }

    /** Adds a space; id is the slugified name plus a short unique suffix. */
    fun addSpace(context: Context, name: String): Space {
        val trimmed = name.trim()
        require(trimmed.isNotEmpty()) { "Space name must not be blank" }
        val p = prefs(context)
        val spaces = listSpaces(context)
        val existing = spaces.map { it.id }.toSet()
        val base = slugify(trimmed)
        var id = "$base-${UUID.randomUUID().toString().take(4)}"
        while (id in existing) id = "$base-${UUID.randomUUID().toString().take(4)}"
        p.edit()
            .putString(KEY_SPACE_IDS, (spaces.map { it.id } + id).joinToString(","))
            .putString(KEY_SPACE_NAME_PREFIX + id, trimmed)
            .apply()
        return Space(id, trimmed)
    }

    /** Renames a space; the id (and its data dir) stays stable. */
    fun renameSpace(context: Context, id: String, name: String) {
        val trimmed = name.trim()
        require(trimmed.isNotEmpty()) { "Space name must not be blank" }
        require(listSpaces(context).any { it.id == id }) { "Unknown space: $id" }
        prefs(context).edit().putString(KEY_SPACE_NAME_PREFIX + id, trimmed).apply()
    }

    /**
     * Deletes a space and its data dir. Refuses to delete the only remaining space.
     * Deleting the active space first switches to "personal" (or the first remaining
     * space when "personal" itself is being deleted).
     */
    fun deleteSpace(context: Context, id: String) {
        val spaces = listSpaces(context)
        require(spaces.size > 1) { "Cannot delete the only space" }
        require(spaces.any { it.id == id }) { "Unknown space: $id" }
        val p = prefs(context)
        if (activeSpace(context).id == id) {
            val fallback = spaces.firstOrNull { it.id == DEFAULT_ID && it.id != id }
                ?: spaces.first { it.id != id }
            p.edit().putString(KEY_ACTIVE, fallback.id).commit()
        }
        p.edit()
            .putString(KEY_SPACE_IDS, spaces.filter { it.id != id }.joinToString(",") { it.id })
            .remove(KEY_SPACE_NAME_PREFIX + id)
            .apply()
        val dir = File(context.applicationContext.filesDir, "spaces/$id")
        if (dir.exists() && !dir.deleteRecursively()) {
            Log.w(TAG, "Failed to fully delete space dir ${dir.absolutePath}")
        }
    }

    /** Switches the active space. Callers must close stores and recreate UI afterwards. */
    fun setActive(context: Context, id: String) {
        require(listSpaces(context).any { it.id == id }) { "Unknown space: $id" }
        // commit (not apply): the following activity recreate reads this synchronously
        prefs(context).edit().putString(KEY_ACTIVE, id).commit()
    }

    /**
     * One-time move of pre-spaces data into the "personal" space: the Ladybug dirs
     * (`files/graph/ladybug` → `graph.lbug`, `files/hebbian/hebbian.lbug` → `hebbian.lbug`,
     * plus the `store-choice.txt` diagnostic), `files/practice_state/`, and the SQLite
     * databases rag.db / graph.db / hebbian_graph.db with their -journal/-wal/-shm
     * siblings. Idempotent (marker file) and lock-guarded; safe to call on every start.
     */
    fun ensureMigration(context: Context) {
        val appContext = context.applicationContext
        val spacesRoot = File(appContext.filesDir, "spaces")
        if (File(spacesRoot, MIGRATION_MARKER).exists()) return
        synchronized(migrationLock) {
            if (File(spacesRoot, MIGRATION_MARKER).exists()) return
            val personal = File(spacesRoot, DEFAULT_ID).apply { mkdirs() }

            // Ladybug stores map onto the new flat layout (<space>/<store>.lbug); the .wal
            // holds everything not yet checkpointed, so it must move with the main file.
            moveIfPresent(File(appContext.filesDir, "graph/ladybug"), File(personal, "graph.lbug"))
            moveIfPresent(File(appContext.filesDir, "graph/ladybug.wal"), File(personal, "graph.lbug.wal"))
            moveIfPresent(File(appContext.filesDir, "hebbian/hebbian.lbug"), File(personal, "hebbian.lbug"))
            moveIfPresent(File(appContext.filesDir, "hebbian/hebbian.lbug.wal"), File(personal, "hebbian.lbug.wal"))
            moveIfPresent(File(appContext.filesDir, "hebbian/store-choice.txt"), File(personal, "store-choice.txt"))
            moveIfPresent(File(appContext.filesDir, "practice_state"), File(personal, "practice_state"))

            // Preserve any unexpected leftovers of the legacy dirs alongside the new files
            moveIfPresent(File(appContext.filesDir, "graph"), File(personal, "graph"))
            moveIfPresent(File(appContext.filesDir, "hebbian"), File(personal, "hebbian"))

            for (name in listOf("rag.db", "graph.db", "hebbian_graph.db")) {
                val db = appContext.getDatabasePath(name)
                moveIfPresent(db, File(personal, name))
                for (suffix in listOf("-journal", "-wal", "-shm")) {
                    moveIfPresent(File(db.path + suffix), File(personal, name + suffix))
                }
            }

            val p = prefs(appContext)
            if (p.getString(KEY_SPACE_IDS, null) == null) {
                p.edit()
                    .putString(KEY_SPACE_IDS, DEFAULT_ID)
                    .putString(KEY_SPACE_NAME_PREFIX + DEFAULT_ID, DEFAULT_NAME)
                    .apply()
            }
            File(spacesRoot, MIGRATION_MARKER).writeText("1")
            Log.i(TAG, "Migrated legacy data into space '$DEFAULT_ID'")
        }
    }

    /** rename first (atomic, same filesystem); copy+delete fallback. If the destination
     *  already exists (crash between copy and delete on a previous run), the destination
     *  is authoritative and the source is dropped. */
    private fun moveIfPresent(src: File, dst: File) {
        if (!src.exists()) return
        try {
            if (dst.exists()) {
                Log.w(TAG, "Migration target ${dst.name} already exists — dropping leftover ${src.path}")
                if (src.isDirectory) src.deleteRecursively() else src.delete()
                return
            }
            dst.parentFile?.mkdirs()
            if (src.renameTo(dst)) return
            src.copyRecursively(dst, overwrite = true)
            if (!src.deleteRecursively()) Log.w(TAG, "Copied but failed to delete ${src.path}")
        } catch (t: Throwable) {
            Log.w(TAG, "Failed to migrate ${src.path} (${t.message})")
        }
    }

    private fun slugify(name: String): String =
        name.lowercase().replace(Regex("[^a-z0-9]+"), "-").trim('-').ifEmpty { "space" }
}
