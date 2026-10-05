package com.mobilerag.graph

import android.content.Context
import android.database.Cursor
import android.database.SQLException
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import com.mobilerag.core.GraphStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * SQLite graph store: entities/edges/mentions tables, multi-hop traversal via
 * recursive CTE, FTS4 over entity names. This is the schema-of-record and the
 * LadybugDB fallback.
 *
 * Schema v2 (Phase 2): entities gain normalized_name + mention_count under a
 * UNIQUE(normalized_name, type) index, edges gain weight, and the mentions table
 * records entity↔chunk provenance. v1 databases are migrated in place by onUpgrade.
 *
 * [dbPath] overrides the default `graph.db` (SQLiteOpenHelper treats names containing
 * '/' as full paths) — GraphStoreFactory passes the active memory space's path.
 */
class SqliteGraphStore(context: Context, dbPath: String? = null) : GraphStore {

    override val id = "sqlite-recursive-cte"

    private val helper = object : SQLiteOpenHelper(context, dbName(dbPath), null, 2) {
        override fun onCreate(db: SQLiteDatabase) {
            db.execSQL("CREATE TABLE entities(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, type TEXT NOT NULL, normalized_name TEXT NOT NULL DEFAULT '', mention_count INTEGER NOT NULL DEFAULT 0)")
            db.execSQL("CREATE UNIQUE INDEX idx_entities_norm ON entities(normalized_name, type)")
            db.execSQL("CREATE TABLE edges(from_id INTEGER NOT NULL, to_id INTEGER NOT NULL, relation TEXT NOT NULL, source_chunk_id INTEGER, weight REAL NOT NULL DEFAULT 1.0)")
            db.execSQL("CREATE INDEX idx_edges_from ON edges(from_id)")
            db.execSQL("CREATE INDEX idx_edges_to ON edges(to_id)")
            db.execSQL("CREATE TABLE mentions(entity_id INTEGER NOT NULL, chunk_id INTEGER NOT NULL, UNIQUE(entity_id, chunk_id))")
            // Stock Android SQLite lacks FTS5 on some devices (confirmed on RedMagic 10 Pro);
            // FTS4 is universally available and sufficient for keyword search
            db.execSQL("CREATE VIRTUAL TABLE entity_fts USING fts4(name, content=entities)")
        }

        override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
            if (oldVersion < 2) {
                db.execSQL("ALTER TABLE entities ADD COLUMN normalized_name TEXT NOT NULL DEFAULT ''")
                db.execSQL("ALTER TABLE entities ADD COLUMN mention_count INTEGER NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE edges ADD COLUMN weight REAL NOT NULL DEFAULT 1.0")
                db.execSQL("CREATE TABLE mentions(entity_id INTEGER NOT NULL, chunk_id INTEGER NOT NULL, UNIQUE(entity_id, chunk_id))")
                db.execSQL("UPDATE entities SET normalized_name = lower(trim(name))")
                db.execSQL("CREATE UNIQUE INDEX idx_entities_norm ON entities(normalized_name, type)")
            }
        }
    }

    private val db get() = helper.writableDatabase

    override suspend fun open(path: String) = Unit // opened lazily by the helper

    override suspend fun close() = helper.close()

    override suspend fun addEntity(name: String, type: String): Long = withContext(Dispatchers.IO) {
        insertEntity(name, type, normalize(name))
    }

    override suspend fun upsertEntity(name: String, type: String, normalizedName: String): Long =
        withContext(Dispatchers.IO) {
            db.rawQuery(
                "SELECT id FROM entities WHERE normalized_name = ? AND type = ?",
                arrayOf(normalizedName, type),
            ).use { c -> if (c.moveToFirst()) return@withContext c.getLong(0) }
            insertEntity(name, type, normalizedName)
        }

    private fun insertEntity(name: String, type: String, normalizedName: String): Long {
        val id = db.compileStatement("INSERT INTO entities(name, type, normalized_name) VALUES (?, ?, ?)").run {
            bindString(1, name)
            bindString(2, type)
            bindString(3, normalizedName)
            executeInsert()
        }
        db.compileStatement("INSERT INTO entity_fts(rowid, name) VALUES (?, ?)").run {
            bindLong(1, id)
            bindString(2, name)
            executeInsert()
        }
        return id
    }

    override suspend fun getEntity(id: Long): GraphStore.Entity? = withContext(Dispatchers.IO) {
        db.rawQuery(
            "SELECT id, name, type, mention_count FROM entities WHERE id = ?",
            arrayOf(id.toString()),
        ).use { c -> if (c.moveToFirst()) readEntity(c) else null }
    }

    override suspend fun addMention(entityId: Long, chunkId: Long) = withContext(Dispatchers.IO) {
        val rowId = db.compileStatement("INSERT OR IGNORE INTO mentions(entity_id, chunk_id) VALUES (?, ?)").run {
            bindLong(1, entityId)
            bindLong(2, chunkId)
            executeInsert()
        }
        if (rowId != -1L) { // -1 = duplicate (entity, chunk) pair, ignored
            db.compileStatement("UPDATE entities SET mention_count = mention_count + 1 WHERE id = ?").run {
                bindLong(1, entityId)
                executeUpdateDelete()
            }
        }
        Unit
    }

    override suspend fun chunksForEntity(entityId: Long): List<Long> = withContext(Dispatchers.IO) {
        db.rawQuery(
            "SELECT chunk_id FROM mentions WHERE entity_id = ? ORDER BY chunk_id",
            arrayOf(entityId.toString()),
        ).use { c ->
            buildList { while (c.moveToNext()) add(c.getLong(0)) }
        }
    }

    override suspend fun edgesFor(entityId: Long): List<GraphStore.Edge> = withContext(Dispatchers.IO) {
        db.rawQuery(
            "SELECT from_id, to_id, relation, source_chunk_id, weight FROM edges WHERE from_id = ? OR to_id = ?",
            arrayOf(entityId.toString(), entityId.toString()),
        ).use { c ->
            buildList {
                while (c.moveToNext()) {
                    add(
                        GraphStore.Edge(
                            fromId = c.getLong(0),
                            toId = c.getLong(1),
                            relation = c.getString(2),
                            sourceChunkId = if (c.isNull(3)) null else c.getLong(3),
                            weight = c.getDouble(4),
                        ),
                    )
                }
            }
        }
    }

    override suspend fun allEdges(): List<GraphStore.Edge> = withContext(Dispatchers.IO) {
        db.rawQuery("SELECT from_id, to_id, relation, source_chunk_id, weight FROM edges", null).use { c ->
            buildList {
                while (c.moveToNext()) {
                    add(
                        GraphStore.Edge(
                            fromId = c.getLong(0),
                            toId = c.getLong(1),
                            relation = c.getString(2),
                            sourceChunkId = if (c.isNull(3)) null else c.getLong(3),
                            weight = c.getDouble(4),
                        ),
                    )
                }
            }
        }
    }

    override suspend fun allEntities(limit: Int, offset: Int): List<GraphStore.Entity> =
        withContext(Dispatchers.IO) {
            db.rawQuery(
                "SELECT id, name, type, mention_count FROM entities " +
                    "ORDER BY mention_count DESC, name ASC LIMIT ? OFFSET ?",
                arrayOf(limit.toString(), offset.toString()),
            ).use { c -> readEntities(c) }
        }

    override suspend fun findEntities(query: String, limit: Int): List<GraphStore.Entity> =
        withContext(Dispatchers.IO) {
            val q = query.trim()
            if (q.isEmpty()) return@withContext emptyList()
            // entity id -> (entity, direct) where direct = exact/prefix/FTS hit
            val candidates = LinkedHashMap<Long, Pair<GraphStore.Entity, Boolean>>()

            db.rawQuery(
                "SELECT id, name, type, mention_count FROM entities WHERE name LIKE ? ESCAPE '\\'",
                arrayOf(escapeLike(q) + "%"),
            ).use { c -> readEntities(c).forEach { candidates.putIfAbsent(it.id, it to true) } }

            val ftsQuery = q.split(Regex("\\s+"))
                .filter { it.isNotEmpty() }
                .joinToString(" ") { "\"${it.replace("\"", "")}\"*" }
            if (ftsQuery.isNotEmpty()) {
                try {
                    db.rawQuery(
                        "SELECT e.id, e.name, e.type, e.mention_count FROM entities e " +
                            "JOIN entity_fts f ON f.rowid = e.id WHERE entity_fts MATCH ?",
                        arrayOf(ftsQuery),
                    ).use { c -> readEntities(c).forEach { candidates.putIfAbsent(it.id, it to true) } }
                } catch (e: SQLException) {
                    // malformed FTS query (operator characters in input) — keep prefix results
                }
            }

            val total = db.compileStatement("SELECT COUNT(*) FROM entities").simpleQueryForLong()
            if (total <= FUZZY_CANDIDATE_CAP) {
                db.rawQuery("SELECT id, name, type, mention_count FROM entities", null).use { c ->
                    readEntities(c).forEach { e ->
                        if (e.id !in candidates &&
                            jaroWinkler(e.name.lowercase(), q.lowercase()) >= FUZZY_THRESHOLD
                        ) {
                            candidates[e.id] = e to false
                        }
                    }
                }
            }

            candidates.values
                .map { (entity, direct) ->
                    val score = if (entity.name.equals(q, ignoreCase = true)) EXACT_SCORE
                        else jaroWinkler(entity.name.lowercase(), q.lowercase())
                    Triple(entity, direct, score)
                }
                .filter { it.second || it.third >= FUZZY_THRESHOLD }
                .sortedByDescending { it.third }
                .take(limit)
                .map { it.first }
        }

    override suspend fun addEdge(fromId: Long, toId: Long, relation: String, sourceChunkId: Long?) =
        withContext(Dispatchers.IO) {
            db.compileStatement("INSERT INTO edges(from_id, to_id, relation, source_chunk_id) VALUES (?, ?, ?, ?)").run {
                bindLong(1, fromId)
                bindLong(2, toId)
                bindString(3, relation)
                if (sourceChunkId != null) bindLong(4, sourceChunkId) else bindNull(4)
                executeInsert()
            }
            Unit
        }

    override suspend fun traverse(startEntityId: Long, maxDepth: Int): List<GraphStore.Entity> =
        withContext(Dispatchers.IO) {
            val sql = """
                WITH RECURSIVE walk(id, depth) AS (
                    SELECT ?, 0
                    UNION
                    SELECT e.to_id, w.depth + 1
                    FROM edges e JOIN walk w ON e.from_id = w.id
                    WHERE w.depth < ?
                )
                SELECT DISTINCT en.id, en.name, en.type
                FROM entities en JOIN walk w ON en.id = w.id
                WHERE w.depth > 0
            """.trimIndent()
            db.rawQuery(sql, arrayOf(startEntityId.toString(), maxDepth.toString())).use { c ->
                buildList {
                    while (c.moveToNext()) {
                        add(GraphStore.Entity(c.getLong(0), c.getString(1), c.getString(2)))
                    }
                }
            }
        }

    override suspend fun removeGraphForChunks(chunkIds: List<Long>) = withContext(Dispatchers.IO) {
        if (chunkIds.isEmpty()) return@withContext
        val placeholders = chunkIds.joinToString(", ") { "?" }
        val args = chunkIds.map { it.toString() }.toTypedArray()
        db.beginTransaction()
        try {
            val mentionCounts = mutableMapOf<Long, Long>()
            db.rawQuery(
                "SELECT entity_id, COUNT(*) FROM mentions WHERE chunk_id IN ($placeholders) GROUP BY entity_id",
                args,
            ).use { c -> while (c.moveToNext()) mentionCounts[c.getLong(0)] = c.getLong(1) }
            for ((entityId, count) in mentionCounts) {
                db.compileStatement("UPDATE entities SET mention_count = MAX(mention_count - ?, 0) WHERE id = ?").run {
                    bindLong(1, count)
                    bindLong(2, entityId)
                    executeUpdateDelete()
                }
            }
            db.execSQL("DELETE FROM mentions WHERE chunk_id IN ($placeholders)", args)
            db.execSQL("DELETE FROM edges WHERE source_chunk_id IN ($placeholders)", args)
            // External-content FTS4 tables can't delete single index rows, so orphaned
            // entities are dropped here and the FTS index is rebuilt below.
            val removed = db.compileStatement(
                "DELETE FROM entities WHERE mention_count <= 0 AND id NOT IN " +
                    "(SELECT from_id FROM edges UNION SELECT to_id FROM edges)",
            ).executeUpdateDelete()
            if (removed > 0) db.execSQL("INSERT INTO entity_fts(entity_fts) VALUES('rebuild')")
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
        Unit
    }

    override suspend fun clear() = withContext(Dispatchers.IO) {
        db.execSQL("DELETE FROM mentions")
        db.execSQL("DELETE FROM edges")
        db.execSQL("DELETE FROM entities")
        db.execSQL("INSERT INTO entity_fts(entity_fts) VALUES('rebuild')")
        Unit
    }

    override suspend fun entityCount(): Int = withContext(Dispatchers.IO) {
        db.compileStatement("SELECT COUNT(*) FROM entities").simpleQueryForLong().toInt()
    }

    override suspend fun edgeCount(): Int = withContext(Dispatchers.IO) {
        db.compileStatement("SELECT COUNT(*) FROM edges").simpleQueryForLong().toInt()
    }

    private fun readEntity(c: Cursor) = GraphStore.Entity(c.getLong(0), c.getString(1), c.getString(2), c.getInt(3))

    private fun readEntities(c: Cursor): List<GraphStore.Entity> = buildList {
        while (c.moveToNext()) add(readEntity(c))
    }

    private fun normalize(name: String) = name.trim().lowercase()

    private fun escapeLike(s: String): String =
        s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    private fun jaroWinkler(s1: String, s2: String): Double = JaroWinkler.similarity(s1, s2)

    private companion object {
        const val FUZZY_THRESHOLD = 0.85
        const val FUZZY_CANDIDATE_CAP = 5000
        const val EXACT_SCORE = 2.0 // above any Jaro-Winkler score, so exact matches rank first

        /** Full paths (containing '/') are used as-is by SQLiteOpenHelper; create the parent dir. */
        fun dbName(dbPath: String?): String {
            if (dbPath == null) return "graph.db"
            File(dbPath).parentFile?.mkdirs()
            return dbPath
        }
    }
}
