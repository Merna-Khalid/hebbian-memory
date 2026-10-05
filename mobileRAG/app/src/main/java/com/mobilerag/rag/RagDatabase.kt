package com.mobilerag.rag

import android.content.Context
import android.database.SQLException
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import java.io.File
import java.nio.ByteBuffer

/**
 * SQLite store for imported documents, their chunks and embedding blobs.
 * `settings.indexVersion` bumps on any chunk mutation to invalidate the VectorStore cache.
 *
 * Schema v3 (Phase 4): the `communities` table persists detected graph communities with
 * their LLM-written summaries and summary embeddings (replaced wholesale on rebuild).
 *
 * [dbPath] overrides the default `rag.db` (SQLiteOpenHelper treats names containing
 * '/' as full paths) — callers pass the active memory space's path (SpaceManager).
 */
class RagDatabase(context: Context, dbPath: String? = null) {

    data class Document(val id: Long, val name: String, val sha256: String, val importedAt: Long, val chunkCount: Int)
    data class NewChunk(val seq: Int, val text: String, val tokenCount: Int, val embedding: FloatArray)
    data class StoredChunk(val id: Long, val docId: Long, val seq: Int, val text: String, val tokenCount: Int, val embedding: FloatArray)

    /** Chunk without its embedding blob — for graph indexing and UI provenance views. */
    data class ChunkInfo(val id: Long, val seq: Int, val text: String, val tokenCount: Int)

    /** A detected graph community pending persistence (Phase 4). */
    data class NewCommunity(val summary: String, val memberIds: List<Long>, val size: Int, val summaryEmbedding: FloatArray)

    /** A persisted community with its summary embedding. */
    data class StoredCommunity(val id: Long, val summary: String, val memberIds: List<Long>, val size: Int, val summaryEmbedding: FloatArray)

    /** Owning document of a chunk — for retrieval hits from the FTS/graph channels. */
    data class ChunkDoc(val docId: Long, val docName: String)

    sealed interface UpsertResult {
        /** Same name + same content hash — nothing to do. */
        data class Unchanged(val docId: Long) : UpsertResult

        /** New document, or hash changed (old chunks already deleted) — re-embed. */
        data class Reindex(val docId: Long) : UpsertResult
    }

    private val helper = object : SQLiteOpenHelper(context, dbName(dbPath), null, 3) {
        override fun onCreate(db: SQLiteDatabase) {
            db.execSQL("CREATE TABLE documents(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, sha256 TEXT NOT NULL, importedAt INTEGER NOT NULL, chunkCount INTEGER NOT NULL DEFAULT 0)")
            db.execSQL("CREATE TABLE chunks(id INTEGER PRIMARY KEY AUTOINCREMENT, docId INTEGER NOT NULL, seq INTEGER NOT NULL, text TEXT NOT NULL, tokenCount INTEGER NOT NULL, embedding BLOB NOT NULL)")
            db.execSQL("CREATE INDEX idx_chunks_doc ON chunks(docId)")
            db.execSQL("CREATE TABLE settings(key TEXT PRIMARY KEY, value TEXT)")
            createChunksFts(db)
            createCommunities(db)
        }

        override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
            if (oldVersion < 2) {
                createChunksFts(db)
                db.execSQL("INSERT INTO chunks_fts(rowid, text) SELECT id, text FROM chunks")
            }
            if (oldVersion < 3) {
                createCommunities(db)
            }
        }

        private fun createChunksFts(db: SQLiteDatabase) {
            // Stock Android SQLite lacks FTS5 on some devices (confirmed on RedMagic 10 Pro in
            // Phase 0); FTS4 is universally available and sufficient for keyword search
            db.execSQL("CREATE VIRTUAL TABLE chunks_fts USING fts4(content=chunks, text)")
        }

        private fun createCommunities(db: SQLiteDatabase) {
            db.execSQL("CREATE TABLE communities(id INTEGER PRIMARY KEY, summary TEXT, member_ids TEXT, size INTEGER, summary_embedding BLOB, created_at INTEGER)")
        }
    }

    private val db get() = helper.writableDatabase

    suspend fun upsertDocument(name: String, sha256: String): UpsertResult = withContext(Dispatchers.IO) {
        val existing = db.rawQuery("SELECT id, sha256 FROM documents WHERE name = ?", arrayOf(name)).use { c ->
            if (c.moveToFirst()) c.getLong(0) to c.getString(1) else null
        }
        when {
            existing == null -> UpsertResult.Reindex(
                db.compileStatement("INSERT INTO documents(name, sha256, importedAt, chunkCount) VALUES (?, ?, ?, 0)").run {
                    bindString(1, name)
                    bindString(2, sha256)
                    bindLong(3, System.currentTimeMillis())
                    executeInsert()
                }
            )

            existing.second == sha256 -> UpsertResult.Unchanged(existing.first)
            else -> {
                deleteChunks(existing.first)
                db.compileStatement("UPDATE documents SET sha256 = ?, importedAt = ?, chunkCount = 0 WHERE id = ?").run {
                    bindString(1, sha256)
                    bindLong(2, System.currentTimeMillis())
                    bindLong(3, existing.first)
                    executeUpdateDelete()
                }
                bumpIndexVersion()
                UpsertResult.Reindex(existing.first)
            }
        }
    }

    suspend fun insertChunks(docId: Long, chunks: List<NewChunk>) = withContext(Dispatchers.IO) {
        db.beginTransaction()
        try {
            val stmt = db.compileStatement("INSERT INTO chunks(docId, seq, text, tokenCount, embedding) VALUES (?, ?, ?, ?, ?)")
            val ftsStmt = db.compileStatement("INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)")
            for (c in chunks) {
                stmt.bindLong(1, docId)
                stmt.bindLong(2, c.seq.toLong())
                stmt.bindString(3, c.text)
                stmt.bindLong(4, c.tokenCount.toLong())
                stmt.bindBlob(5, floatsToBytes(c.embedding))
                val chunkId = stmt.executeInsert()
                ftsStmt.bindLong(1, chunkId)
                ftsStmt.bindString(2, c.text)
                ftsStmt.executeInsert()
            }
            db.compileStatement("UPDATE documents SET chunkCount = ? WHERE id = ?").run {
                bindLong(1, chunks.size.toLong())
                bindLong(2, docId)
                executeUpdateDelete()
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
        bumpIndexVersion()
    }

    suspend fun allChunksWithEmbeddings(): List<StoredChunk> = withContext(Dispatchers.IO) {
        db.rawQuery("SELECT id, docId, seq, text, tokenCount, embedding FROM chunks ORDER BY docId, seq", null).use { c ->
            buildList {
                while (c.moveToNext()) {
                    add(StoredChunk(c.getLong(0), c.getLong(1), c.getInt(2), c.getString(3), c.getInt(4), bytesToFloats(c.getBlob(5))))
                }
            }
        }
    }

    suspend fun chunksForDocument(docId: Long): List<ChunkInfo> = withContext(Dispatchers.IO) {
        db.rawQuery(
            "SELECT id, seq, text, tokenCount FROM chunks WHERE docId = ? ORDER BY seq",
            arrayOf(docId.toString()),
        ).use { c ->
            buildList {
                while (c.moveToNext()) {
                    add(ChunkInfo(c.getLong(0), c.getInt(1), c.getString(2), c.getInt(3)))
                }
            }
        }
    }

    suspend fun chunkIdsForDocument(docId: Long): List<Long> = withContext(Dispatchers.IO) {
        db.rawQuery("SELECT id FROM chunks WHERE docId = ? ORDER BY seq", arrayOf(docId.toString())).use { c ->
            buildList { while (c.moveToNext()) add(c.getLong(0)) }
        }
    }

    /** Chunk texts keyed by chunk id, for the graph provenance view. */
    suspend fun chunkTextsById(ids: List<Long>): Map<Long, String> = withContext(Dispatchers.IO) {
        if (ids.isEmpty()) return@withContext emptyMap()
        val out = LinkedHashMap<Long, String>(ids.size)
        // Batch below SQLite's host-parameter limit (999)
        for (batch in ids.chunked(500)) {
            val placeholders = batch.joinToString(", ") { "?" }
            db.rawQuery(
                "SELECT id, text FROM chunks WHERE id IN ($placeholders)",
                batch.map { it.toString() }.toTypedArray(),
            ).use { c ->
                while (c.moveToNext()) out[c.getLong(0)] = c.getString(1)
            }
        }
        out
    }

    /** Owning document (id + name) keyed by chunk id, for non-vector retrieval channels. */
    suspend fun chunkDocInfo(ids: List<Long>): Map<Long, ChunkDoc> = withContext(Dispatchers.IO) {
        if (ids.isEmpty()) return@withContext emptyMap()
        val out = LinkedHashMap<Long, ChunkDoc>(ids.size)
        for (batch in ids.chunked(500)) {
            val placeholders = batch.joinToString(", ") { "?" }
            db.rawQuery(
                "SELECT c.id, c.docId, d.name FROM chunks c JOIN documents d ON d.id = c.docId WHERE c.id IN ($placeholders)",
                batch.map { it.toString() }.toTypedArray(),
            ).use { c ->
                while (c.moveToNext()) out[c.getLong(0)] = ChunkDoc(c.getLong(1), c.getString(2))
            }
        }
        out
    }

    /** Chunk ids whose text matches a single FTS term; empty on a malformed/unsupported query. */
    suspend fun searchChunkFts(term: String): List<Long> = withContext(Dispatchers.IO) {
        val quoted = "\"" + term.replace("\"", "\"\"") + "\""
        try {
            db.rawQuery("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", arrayOf(quoted)).use { c ->
                buildList { while (c.moveToNext()) add(c.getLong(0)) }
            }
        } catch (e: SQLException) {
            emptyList()
        }
    }

    /** Atomically replaces the communities table (Phase 4 community summaries). */
    suspend fun replaceCommunities(communities: List<NewCommunity>) = withContext(Dispatchers.IO) {
        db.beginTransaction()
        try {
            db.execSQL("DELETE FROM communities")
            val stmt = db.compileStatement(
                "INSERT INTO communities(summary, member_ids, size, summary_embedding, created_at) VALUES (?, ?, ?, ?, ?)",
            )
            val now = System.currentTimeMillis()
            for (c in communities) {
                stmt.bindString(1, c.summary)
                stmt.bindString(2, JSONArray(c.memberIds).toString())
                stmt.bindLong(3, c.size.toLong())
                stmt.bindBlob(4, floatsToBytes(c.summaryEmbedding))
                stmt.bindLong(5, now)
                stmt.executeInsert()
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }

    /** All persisted communities, ordered by size DESC then id ASC. */
    suspend fun allCommunities(): List<StoredCommunity> = withContext(Dispatchers.IO) {
        db.rawQuery(
            "SELECT id, summary, member_ids, size, summary_embedding FROM communities ORDER BY size DESC, id ASC",
            null,
        ).use { c ->
            buildList {
                while (c.moveToNext()) {
                    val ids = JSONArray(c.getString(2))
                    add(
                        StoredCommunity(
                            id = c.getLong(0),
                            summary = c.getString(1),
                            memberIds = List(ids.length()) { ids.getLong(it) },
                            size = c.getInt(3),
                            summaryEmbedding = bytesToFloats(c.getBlob(4)),
                        ),
                    )
                }
            }
        }
    }

    suspend fun deleteDocument(id: Long) = withContext(Dispatchers.IO) {
        deleteChunks(id)
        db.compileStatement("DELETE FROM documents WHERE id = ?").run {
            bindLong(1, id)
            executeUpdateDelete()
        }
        bumpIndexVersion()
    }

    suspend fun listDocuments(): List<Document> = withContext(Dispatchers.IO) {
        db.rawQuery("SELECT id, name, sha256, importedAt, chunkCount FROM documents ORDER BY name", null).use { c ->
            buildList {
                while (c.moveToNext()) {
                    add(Document(c.getLong(0), c.getString(1), c.getString(2), c.getLong(3), c.getInt(4)))
                }
            }
        }
    }

    suspend fun indexVersion(): Long = withContext(Dispatchers.IO) {
        getSetting(KEY_INDEX_VERSION)?.toLongOrNull() ?: 0L
    }

    fun close() = helper.close()

    private fun deleteChunks(docId: Long) {
        db.compileStatement("DELETE FROM chunks WHERE docId = ?").run {
            bindLong(1, docId)
            executeUpdateDelete()
        }
        // External-content FTS4 tables don't support clean row-level delete (stale entries
        // remain) — rebuild the index from the chunks table, same as SqliteGraphStore does.
        db.execSQL("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    }

    private fun bumpIndexVersion() {
        val next = (getSetting(KEY_INDEX_VERSION)?.toLongOrNull() ?: 0L) + 1
        db.compileStatement("INSERT OR REPLACE INTO settings(key, value) VALUES (?, ?)").run {
            bindString(1, KEY_INDEX_VERSION)
            bindString(2, next.toString())
            executeInsert()
        }
    }

    private fun getSetting(key: String): String? =
        db.rawQuery("SELECT value FROM settings WHERE key = ?", arrayOf(key)).use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        }

    companion object {
        private const val KEY_INDEX_VERSION = "indexVersion"

        /** Full paths (containing '/') are used as-is by SQLiteOpenHelper; create the parent dir. */
        private fun dbName(dbPath: String?): String {
            if (dbPath == null) return "rag.db"
            File(dbPath).parentFile?.mkdirs()
            return dbPath
        }

        fun floatsToBytes(v: FloatArray): ByteArray {
            val buf = ByteBuffer.allocate(v.size * Float.SIZE_BYTES)
            buf.asFloatBuffer().put(v)
            return buf.array()
        }

        fun bytesToFloats(b: ByteArray): FloatArray {
            val buf = ByteBuffer.wrap(b).asFloatBuffer()
            return FloatArray(buf.remaining()).also { buf.get(it) }
        }
    }
}
