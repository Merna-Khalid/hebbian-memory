package com.mobilerag.graph

import com.ladybugdb.Connection
import com.ladybugdb.Database
import com.ladybugdb.QueryResult
import com.mobilerag.core.GraphStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * LadybugDB graph store: Entity/Chunk node tables, MENTIONED_IN and REL rel tables,
 * traversed with variable-length Cypher. Mirrors [SqliteGraphStore] semantics;
 * selected by [GraphStoreFactory] when the native library loads on the device.
 *
 * The embedded engine is single-writer: all connection use is serialized on a lock,
 * and INT64 ids are allocated in-process (max id + 1, under a lock) because this app
 * is the only writer to the database file. Queries use the Java API's prepared
 * statements (Connection.prepare/execute) for values; ints that can't be parameters
 * (paging, hop bounds, id lists) are inlined and are injection-safe.
 */
class LadybugGraphStore : GraphStore {

    override val id = "ladybugdb"

    private var database: Database? = null
    private var connection: Connection? = null
    private val connLock = Any()
    private val idLock = Any()
    private var nextEntityId = 0L

    override suspend fun open(path: String) = withContext(Dispatchers.IO) {
        File(path).parentFile?.mkdirs()
        val db = Database(path)
        try {
            connection = Connection(db)
            database = db
            ensureSchema()
            synchronized(idLock) { nextEntityId = maxEntityId() + 1 }
        } catch (t: Throwable) {
            val conn = connection
            connection = null
            database = null
            try { conn?.close() } catch (_: Throwable) {}
            try { db.close() } catch (_: Throwable) {}
            throw t
        }
        Unit
    }

    override suspend fun close() = withContext(Dispatchers.IO) {
        synchronized(connLock) {
            try {
                connection?.close()
            } finally {
                connection = null
                try {
                    database?.close()
                } finally {
                    database = null
                }
            }
        }
        Unit
    }

    override suspend fun addEntity(name: String, type: String): Long = withContext(Dispatchers.IO) {
        createEntity(name, type, normalize(name))
    }

    override suspend fun upsertEntity(name: String, type: String, normalizedName: String): Long =
        withContext(Dispatchers.IO) {
            val existing = execute(
                "MATCH (e:Entity {normalized_name: \$norm, type: \$type}) RETURN e.id",
                mapOf("norm" to normalizedName, "type" to type),
            ).rows()
            if (existing.isNotEmpty()) return@withContext (existing[0][0] as Number).toLong()
            createEntity(name, type, normalizedName)
        }

    private fun createEntity(name: String, type: String, normalizedName: String): Long {
        val id = allocateEntityId()
        execute(
            "CREATE (:Entity {id: \$id, name: \$name, normalized_name: \$norm, type: \$type, mention_count: \$mc})",
            mapOf("id" to id, "name" to name, "norm" to normalizedName, "type" to type, "mc" to 0L),
        ).close()
        return id
    }

    override suspend fun getEntity(id: Long): GraphStore.Entity? = withContext(Dispatchers.IO) {
        execute(
            "MATCH (e:Entity {id: \$id}) RETURN e.id, e.name, e.type, e.mention_count",
            mapOf("id" to id),
        ).rows().firstOrNull()?.let(::rowToEntity)
    }

    override suspend fun addMention(entityId: Long, chunkId: Long) = withContext(Dispatchers.IO) {
        val chunk = execute("MATCH (c:Chunk {id: \$id}) RETURN c.id", mapOf("id" to chunkId)).rows()
        if (chunk.isEmpty()) {
            execute("CREATE (:Chunk {id: \$id})", mapOf("id" to chunkId)).close()
        }
        val existing = execute(
            "MATCH (e:Entity {id: \$entity})-[m:MENTIONED_IN]->(c:Chunk {id: \$chunk}) RETURN count(m)",
            mapOf("entity" to entityId, "chunk" to chunkId),
        ).rows()
        val count = (existing.firstOrNull()?.firstOrNull() as? Number)?.toLong() ?: 0L
        if (count > 0L) return@withContext // duplicate (entity, chunk) pair, ignored
        execute(
            "MATCH (e:Entity {id: \$entity}), (c:Chunk {id: \$chunk}) CREATE (e)-[:MENTIONED_IN]->(c)",
            mapOf("entity" to entityId, "chunk" to chunkId),
        ).close()
        execute(
            "MATCH (e:Entity {id: \$entity}) SET e.mention_count = e.mention_count + \$one",
            mapOf("entity" to entityId, "one" to 1L),
        ).close()
        Unit
    }

    override suspend fun chunksForEntity(entityId: Long): List<Long> = withContext(Dispatchers.IO) {
        execute(
            "MATCH (e:Entity {id: \$id})-[:MENTIONED_IN]->(c:Chunk) RETURN c.id ORDER BY c.id",
            mapOf("id" to entityId),
        ).rows().map { (it[0] as Number).toLong() }
    }

    override suspend fun edgesFor(entityId: Long): List<GraphStore.Edge> = withContext(Dispatchers.IO) {
        execute(
            "MATCH (a:Entity)-[r:REL]->(b:Entity) WHERE a.id = \$id OR b.id = \$id " +
                "RETURN a.id, b.id, r.relation, r.weight, r.source_chunk_id",
            mapOf("id" to entityId),
        ).rows().map { row ->
            GraphStore.Edge(
                fromId = (row[0] as Number).toLong(),
                toId = (row[1] as Number).toLong(),
                relation = row[2] as String,
                sourceChunkId = (row[4] as? Number)?.toLong(),
                weight = (row[3] as? Number)?.toDouble() ?: 1.0,
            )
        }
    }

    override suspend fun allEdges(): List<GraphStore.Edge> = withContext(Dispatchers.IO) {
        execute(
            "MATCH (a:Entity)-[r:REL]->(b:Entity) " +
                "RETURN a.id, b.id, r.relation, r.weight, r.source_chunk_id",
        ).rows().map { row ->
            GraphStore.Edge(
                fromId = (row[0] as Number).toLong(),
                toId = (row[1] as Number).toLong(),
                relation = row[2] as String,
                sourceChunkId = (row[4] as? Number)?.toLong(),
                weight = (row[3] as? Number)?.toDouble() ?: 1.0,
            )
        }
    }

    override suspend fun allEntities(limit: Int, offset: Int): List<GraphStore.Entity> =
        withContext(Dispatchers.IO) {
            // SKIP/LIMIT inlined: Kùzu/LadybugDB doesn't accept parameters there; Ints are safe.
            execute(
                "MATCH (e:Entity) RETURN e.id, e.name, e.type, e.mention_count " +
                    "ORDER BY e.mention_count DESC, e.name ASC SKIP $offset LIMIT $limit",
            ).rows().map(::rowToEntity)
        }

    override suspend fun findEntities(query: String, limit: Int): List<GraphStore.Entity> =
        withContext(Dispatchers.IO) {
            val q = query.trim()
            if (q.isEmpty()) return@withContext emptyList()
            // LadybugDB has no FTS here: prefix match in Cypher, then a Kotlin-side
            // Jaro-Winkler pass over a bounded candidate set — same ranking contract
            // as SqliteGraphStore.findEntities.
            val candidates = LinkedHashMap<Long, Pair<GraphStore.Entity, Boolean>>()
            execute(
                "MATCH (e:Entity) WHERE LOWER(e.name) STARTS WITH LOWER(\$q) " +
                    "RETURN e.id, e.name, e.type, e.mention_count",
                mapOf("q" to q),
            ).rows().forEach { row ->
                val entity = rowToEntity(row)
                candidates.putIfAbsent(entity.id, entity to true)
            }
            if (entityCount() <= FUZZY_CANDIDATE_CAP) {
                execute("MATCH (e:Entity) RETURN e.id, e.name, e.type, e.mention_count").rows().forEach { row ->
                    val entity = rowToEntity(row)
                    if (entity.id !in candidates &&
                        jaroWinkler(entity.name.lowercase(), q.lowercase()) >= FUZZY_THRESHOLD
                    ) {
                        candidates[entity.id] = entity to false
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
            // Two statement shapes avoid passing a null parameter for source_chunk_id
            // (null param type inference is not verified on this API version).
            if (sourceChunkId == null) {
                execute(
                    "MATCH (a:Entity {id: \$from}), (b:Entity {id: \$to}) " +
                        "CREATE (a)-[:REL {relation: \$rel, weight: \$weight}]->(b)",
                    mapOf("from" to fromId, "to" to toId, "rel" to relation, "weight" to 1.0),
                ).close()
            } else {
                execute(
                    "MATCH (a:Entity {id: \$from}), (b:Entity {id: \$to}) " +
                        "CREATE (a)-[:REL {relation: \$rel, weight: \$weight, source_chunk_id: \$chunk}]->(b)",
                    mapOf(
                        "from" to fromId, "to" to toId, "rel" to relation,
                        "weight" to 1.0, "chunk" to sourceChunkId,
                    ),
                ).close()
            }
            Unit
        }

    override suspend fun traverse(startEntityId: Long, maxDepth: Int): List<GraphStore.Entity> =
        withContext(Dispatchers.IO) {
            // Hop bound inlined: variable-length bounds can't be parameters; Int is safe.
            execute(
                "MATCH (s:Entity {id: \$id})-[:REL*1..$maxDepth]->(e:Entity) " +
                    "RETURN DISTINCT e.id, e.name, e.type, e.mention_count",
                mapOf("id" to startEntityId),
            ).rows().map(::rowToEntity)
        }

    override suspend fun removeGraphForChunks(chunkIds: List<Long>) = withContext(Dispatchers.IO) {
        if (chunkIds.isEmpty()) return@withContext
        // Chunk ids are Longs from our own rag.db, inlined because list-typed parameters
        // are not verified on this LadybugDB version; Longs are injection-safe.
        val idList = chunkIds.joinToString(", ")
        val mentionCounts = execute(
            "MATCH (e:Entity)-[m:MENTIONED_IN]->(c:Chunk) WHERE c.id IN [$idList] RETURN e.id, count(m)",
        ).rows()
        for (row in mentionCounts) {
            execute(
                "MATCH (e:Entity {id: \$id}) SET e.mention_count = e.mention_count - \$dec",
                mapOf("id" to (row[0] as Number).toLong(), "dec" to (row[1] as Number).toLong()),
            ).close()
        }
        execute("MATCH (e:Entity) WHERE e.mention_count < 0 SET e.mention_count = 0").close()
        execute("MATCH (e:Entity)-[m:MENTIONED_IN]->(c:Chunk) WHERE c.id IN [$idList] DELETE m").close()
        // Drop Chunk nodes left with no mentions.
        execute(
            "MATCH (c:Chunk) WHERE c.id IN [$idList] OPTIONAL MATCH (c)<-[m:MENTIONED_IN]-() " +
                "WITH c, count(m) AS refs WHERE refs = 0 DELETE c",
        ).close()
        execute("MATCH ()-[r:REL]->() WHERE r.source_chunk_id IN [$idList] DELETE r").close()
        // Drop entities left with no mentions and no remaining edges.
        execute(
            "MATCH (e:Entity) WHERE e.mention_count <= 0 OPTIONAL MATCH (e)-[r:REL]-() " +
                "WITH e, count(r) AS rels WHERE rels = 0 DETACH DELETE e",
        ).close()
        Unit
    }

    override suspend fun clear() = withContext(Dispatchers.IO) {
        execute("MATCH (e:Entity) DETACH DELETE e").close()
        execute("MATCH (c:Chunk) DETACH DELETE c").close()
        synchronized(idLock) { nextEntityId = 0L }
        Unit
    }

    override suspend fun entityCount(): Int = withContext(Dispatchers.IO) {
        ((execute("MATCH (e:Entity) RETURN count(e)").rows().firstOrNull()?.firstOrNull()) as? Number)?.toInt() ?: 0
    }

    override suspend fun edgeCount(): Int = withContext(Dispatchers.IO) {
        ((execute("MATCH ()-[r:REL]->() RETURN count(r)").rows().firstOrNull()?.firstOrNull()) as? Number)?.toInt() ?: 0
    }

    // ---- internals ----

    private fun ensureSchema() {
        ensureTable(
            "CREATE NODE TABLE IF NOT EXISTS Entity(id INT64, name STRING, normalized_name STRING, type STRING, mention_count INT64, PRIMARY KEY(id))",
            "CREATE NODE TABLE Entity(id INT64, name STRING, normalized_name STRING, type STRING, mention_count INT64, PRIMARY KEY(id))",
        )
        ensureTable(
            "CREATE NODE TABLE IF NOT EXISTS Chunk(id INT64, PRIMARY KEY(id))",
            "CREATE NODE TABLE Chunk(id INT64, PRIMARY KEY(id))",
        )
        ensureTable(
            "CREATE REL TABLE IF NOT EXISTS MENTIONED_IN(FROM Entity TO Chunk)",
            "CREATE REL TABLE MENTIONED_IN(FROM Entity TO Chunk)",
        )
        ensureTable(
            "CREATE REL TABLE IF NOT EXISTS REL(FROM Entity TO Entity, relation STRING, weight DOUBLE, source_chunk_id INT64)",
            "CREATE REL TABLE REL(FROM Entity TO Entity, relation STRING, weight DOUBLE, source_chunk_id INT64)",
        )
    }

    private fun ensureTable(withIfNotExists: String, plain: String) {
        val firstError = tryCreate(withIfNotExists) ?: return
        if (isAlreadyExists(firstError)) return
        // This LadybugDB build may not support IF NOT EXISTS — retry the plain form.
        val secondError = tryCreate(plain) ?: return
        if (!isAlreadyExists(secondError)) {
            throw RuntimeException("LadybugDB schema creation failed: $secondError")
        }
    }

    /** Returns null on success, otherwise the error message (never throws). */
    private fun tryCreate(cypher: String): String? {
        val conn = connection ?: throw IllegalStateException("LadybugGraphStore is not open")
        return synchronized(connLock) {
            try {
                val result = conn.query(cypher)
                try {
                    if (result.isSuccess) null else result.errorMessage
                } finally {
                    result.close()
                }
            } catch (t: Throwable) {
                t.message ?: t.javaClass.simpleName
            }
        }
    }

    private fun isAlreadyExists(message: String): Boolean =
        (message.contains("already") && message.contains("exist", ignoreCase = true)) ||
            message.contains("duplicate", ignoreCase = true)

    private fun execute(cypher: String, params: Map<String, Any?> = emptyMap()): QueryResult =
        synchronized(connLock) {
            val conn = connection ?: throw IllegalStateException("LadybugGraphStore is not open")
            val result = if (params.isEmpty()) {
                conn.query(cypher)
            } else {
                val ps = conn.prepare(cypher)
                try {
                    if (!ps.isSuccess) {
                        throw RuntimeException("Cypher prepare failed: ${ps.errorMessage}\n$cypher")
                    }
                    conn.execute(ps, params)
                } finally {
                    ps.close()
                }
            }
            if (!result.isSuccess) {
                val message = result.errorMessage
                result.close()
                throw RuntimeException("Cypher failed: $message\n$cypher")
            }
            result
        }

    /** Drains the result into rows of plain values and closes the result and its tuples. */
    private fun QueryResult.rows(): List<List<Any?>> {
        val out = ArrayList<List<Any?>>()
        try {
            while (hasNext()) {
                val tuple = next
                try {
                    val row = ArrayList<Any?>(numColumns.toInt())
                    for (i in 0L until numColumns) row.add(tuple.getValue(i).getValue<Any?>())
                    out.add(row)
                } finally {
                    tuple.close()
                }
            }
        } finally {
            close()
        }
        return out
    }

    private fun rowToEntity(row: List<Any?>) = GraphStore.Entity(
        id = (row[0] as Number).toLong(),
        name = row[1] as String,
        type = row[2] as String,
        mentionCount = (row[3] as Number).toInt(),
    )

    private fun maxEntityId(): Long =
        ((execute("MATCH (e:Entity) RETURN coalesce(max(e.id), 0)").rows().firstOrNull()?.firstOrNull()) as? Number)
            ?.toLong() ?: 0L

    private fun allocateEntityId(): Long = synchronized(idLock) {
        val allocated = nextEntityId
        nextEntityId += 1
        allocated
    }

    private fun normalize(name: String) = name.trim().lowercase()

    private fun jaroWinkler(s1: String, s2: String): Double = JaroWinkler.similarity(s1, s2)

    private companion object {
        const val FUZZY_THRESHOLD = 0.85
        const val FUZZY_CANDIDATE_CAP = 5000
        const val EXACT_SCORE = 2.0 // above any Jaro-Winkler score, so exact matches rank first
    }
}
