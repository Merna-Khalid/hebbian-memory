package com.mobilerag.hebbian.store

import android.content.Context
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import com.mobilerag.hebbian.ConceptIdentity
import java.io.File
import java.nio.ByteBuffer
import java.util.UUID
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * SQLite Hebbian store — schema-of-record port of graph_store.py, using the same
 * hand-rolled SQLiteOpenHelper pattern as SqliteGraphStore.kt. Database file
 * `hebbian_graph.db`, version 1.
 *
 * All graph algorithms (vector search, BFS subgraph, spreading activation, scoring)
 * run through the shared kernels in HebbianStore.kt; vector search is brute-force
 * cosine over an in-memory concept cache invalidated on any concept write.
 *
 * [dbPath] overrides the default `hebbian_graph.db` (SQLiteOpenHelper treats names
 * containing '/' as full paths) — HebbianStoreFactory passes the active memory space's
 * path.
 */
class SqliteHebbianStore(context: Context, dbPath: String? = null) : HebbianStore {

    override val id = "sqlite-hebbian"

    private val helper = object : SQLiteOpenHelper(context, dbName(dbPath), null, 1) {
        override fun onCreate(db: SQLiteDatabase) {
            db.execSQL(
                "CREATE TABLE concepts(node_id TEXT PRIMARY KEY, label TEXT NOT NULL DEFAULT '', " +
                    "text_raw TEXT NOT NULL DEFAULT '', text_summary TEXT NOT NULL DEFAULT '', " +
                    "concept_type TEXT NOT NULL DEFAULT 'fact', source_type TEXT NOT NULL DEFAULT 'document', " +
                    "source_uri TEXT, created_at INTEGER NOT NULL DEFAULT 0, updated_at INTEGER NOT NULL DEFAULT 0, " +
                    "activation_count INTEGER NOT NULL DEFAULT 0, embedding BLOB, " +
                    "mean_m_t REAL NOT NULL DEFAULT 1.0, mean_dominance REAL NOT NULL DEFAULT 0.5, " +
                    "mean_r_t REAL NOT NULL DEFAULT 0.5, recent_events TEXT NOT NULL DEFAULT '[]')",
            )
            db.execSQL(
                "CREATE TABLE hebb_edges(src_id TEXT NOT NULL, dst_id TEXT NOT NULL, layer TEXT NOT NULL, " +
                    "hebb_weight REAL NOT NULL DEFAULT 1.0, eligibility REAL NOT NULL DEFAULT 0.0, " +
                    "causal_score REAL NOT NULL DEFAULT 0.0, co_activation_count INTEGER NOT NULL DEFAULT 0, " +
                    "last_updated INTEGER NOT NULL DEFAULT 0, UNIQUE(src_id, dst_id, layer))",
            )
            db.execSQL("CREATE INDEX idx_hebb_edges_src ON hebb_edges(src_id)")
            db.execSQL("CREATE INDEX idx_hebb_edges_dst ON hebb_edges(dst_id)")
            // "trigger" is a SQL keyword — quoted everywhere it appears.
            db.execSQL(
                """CREATE TABLE sessions(session_id TEXT PRIMARY KEY, "trigger" TEXT, """ +
                    "started_at INTEGER, ended_at INTEGER, total_activations INTEGER NOT NULL DEFAULT 0)",
            )
            db.execSQL(
                "CREATE TABLE session_stats(session_id TEXT NOT NULL, node_id TEXT NOT NULL, " +
                    "mean_m_t REAL, mean_dominance REAL, mean_r_t REAL, " +
                    "activation_count INTEGER NOT NULL DEFAULT 0, delta_w_mean REAL, " +
                    "UNIQUE(session_id, node_id))",
            )
        }

        override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) = Unit
    }

    private val db get() = helper.writableDatabase

    override suspend fun open(path: String) = Unit // opened lazily by the helper

    override suspend fun close() = helper.close()

    override suspend fun setupSchema() = withContext(Dispatchers.IO) {
        db.rawQuery("SELECT 1 FROM concepts LIMIT 0", null).close() // force helper onCreate
        Unit
    }

    // ── Concept CRUD ──────────────────────────────────────────────────

    override suspend fun upsertConcept(node: ConceptNode): String = withContext(Dispatchers.IO) {
        val now = System.currentTimeMillis()
        val exists = db.rawQuery("SELECT 1 FROM concepts WHERE node_id = ?", arrayOf(node.nodeId))
            .use { it.moveToFirst() }
        if (exists) {
            // ON MATCH semantics: ARM-E stats, activation_count and created_at are untouched.
            db.compileStatement(
                "UPDATE concepts SET label = ?, text_raw = ?, text_summary = ?, concept_type = ?, " +
                    "source_type = ?, source_uri = ?, updated_at = ?, embedding = ? WHERE node_id = ?",
            ).run {
                bindString(1, node.label)
                bindString(2, node.textRaw)
                bindString(3, node.textSummary)
                bindString(4, node.conceptType)
                bindString(5, node.sourceType)
                if (node.sourceUri != null) bindString(6, node.sourceUri) else bindNull(6)
                bindLong(7, now)
                if (node.embedding.isNotEmpty()) bindBlob(8, floatsToBytes(node.embedding)) else bindNull(8)
                bindString(9, node.nodeId)
                executeUpdateDelete()
            }
        } else {
            db.compileStatement(
                "INSERT INTO concepts(node_id, label, text_raw, text_summary, concept_type, source_type, " +
                    "source_uri, created_at, updated_at, activation_count, embedding, " +
                    "mean_m_t, mean_dominance, mean_r_t, recent_events) " +
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 1.0, 0.5, 0.5, '[]')",
            ).run {
                bindString(1, node.nodeId)
                bindString(2, node.label)
                bindString(3, node.textRaw)
                bindString(4, node.textSummary)
                bindString(5, node.conceptType)
                bindString(6, node.sourceType)
                if (node.sourceUri != null) bindString(7, node.sourceUri) else bindNull(7)
                bindLong(8, now)
                bindLong(9, now)
                if (node.embedding.isNotEmpty()) bindBlob(10, floatsToBytes(node.embedding)) else bindNull(10)
                executeInsert()
            }
        }
        refreshCached(node.nodeId)
        node.nodeId
    }

    override suspend fun getConcept(nodeId: String): ConceptNode? = withContext(Dispatchers.IO) {
        getConceptSync(nodeId)
    }

    override suspend fun incrementActivation(nodeId: String) = withContext(Dispatchers.IO) {
        db.compileStatement(
            "UPDATE concepts SET activation_count = activation_count + 1, updated_at = ? WHERE node_id = ?",
        ).run {
            bindLong(1, System.currentTimeMillis())
            bindString(2, nodeId)
            executeUpdateDelete()
        }
        refreshCached(nodeId)
        Unit
    }

    override suspend fun findByLabelKey(labelKey: String): List<ConceptNode> = withContext(Dispatchers.IO) {
        if (labelKey.isEmpty()) return@withContext emptyList()
        cachedConcepts().filter { ConceptIdentity.labelKey(it.label) == labelKey }
    }

    override suspend fun updateConceptEmbeddings(embeddings: Map<String, FloatArray>) =
        withContext(Dispatchers.IO) {
            if (embeddings.isEmpty()) return@withContext
            db.beginTransaction()
            try {
                val stmt = db.compileStatement("UPDATE concepts SET embedding = ? WHERE node_id = ?")
                for ((nodeId, emb) in embeddings) {
                    stmt.bindBlob(1, floatsToBytes(emb))
                    stmt.bindString(2, nodeId)
                    stmt.executeUpdateDelete()
                }
                db.setTransactionSuccessful()
            } finally {
                db.endTransaction()
            }
            invalidateCache()
            Unit
        }

    // ── Vector search ─────────────────────────────────────────────────

    override suspend fun vectorSearch(
        embedding: FloatArray,
        topK: Int,
        conceptType: String?,
    ): List<Pair<ConceptNode, Double>> = withContext(Dispatchers.IO) {
        cosineSearch(cachedConcepts(), embedding, topK, conceptType)
    }

    // ── Hebbian edge CRUD ─────────────────────────────────────────────

    override suspend fun upsertEdge(edge: HebbianEdge) = withContext(Dispatchers.IO) {
        // Never weaken on re-assertion (see HebbianStore.upsertEdge); UNIQUE(src_id, dst_id,
        // layer) is the conflict target. SQLite UPSERT needs 3.24+ (API 30 ships 3.28).
        db.compileStatement(
            "INSERT INTO hebb_edges(src_id, dst_id, layer, hebb_weight, eligibility, " +
                "causal_score, co_activation_count, last_updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?) " +
                "ON CONFLICT(src_id, dst_id, layer) DO UPDATE SET " +
                "hebb_weight = MAX(hebb_weight, excluded.hebb_weight), " +
                "co_activation_count = co_activation_count + 1, " +
                "last_updated = excluded.last_updated",
        ).run {
            bindString(1, edge.srcId)
            bindString(2, edge.dstId)
            bindString(3, edge.layer)
            bindDouble(4, edge.hebbWeight)
            bindDouble(5, edge.eligibility)
            bindDouble(6, edge.causalScore)
            bindLong(7, edge.coActivationCount.toLong())
            bindLong(8, System.currentTimeMillis())
            executeInsert()
        }
        Unit
    }

    override suspend fun updateHebbWeights(updates: List<HebbWeightUpdate>) = withContext(Dispatchers.IO) {
        if (updates.isEmpty()) return@withContext
        val now = System.currentTimeMillis()
        db.beginTransaction()
        try {
            val stmt = db.compileStatement(
                "UPDATE hebb_edges SET hebb_weight = ?, eligibility = ?, last_updated = ?, " +
                    "co_activation_count = co_activation_count + 1 " +
                    "WHERE src_id = ? AND dst_id = ? AND layer = ?",
            )
            for (u in updates) {
                stmt.bindDouble(1, u.weight)
                stmt.bindDouble(2, u.eligibility)
                stmt.bindLong(3, now)
                stmt.bindString(4, u.srcId)
                stmt.bindString(5, u.dstId)
                stmt.bindString(6, u.layer)
                stmt.executeUpdateDelete()
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
        Unit
    }

    override suspend fun updateCausalScores(updates: List<CausalScoreUpdate>, layer: String) =
        withContext(Dispatchers.IO) {
            if (updates.isEmpty()) return@withContext
            db.beginTransaction()
            try {
                val stmt = db.compileStatement(
                    "UPDATE hebb_edges SET causal_score = ? WHERE src_id = ? AND dst_id = ? AND layer = ?",
                )
                for (u in updates) {
                    stmt.bindDouble(1, u.score)
                    stmt.bindString(2, u.srcId)
                    stmt.bindString(3, u.dstId)
                    stmt.bindString(4, layer)
                    stmt.executeUpdateDelete()
                }
                db.setTransactionSuccessful()
            } finally {
                db.endTransaction()
            }
            Unit
        }

    override suspend fun reinforceCoRetrieved(
        nodeIds: List<String>,
        mT: Double,
        eta: Double,
        lam: Double,
        wFloor: Double,
        wCeil: Double,
        layer: String,
    ) = withContext(Dispatchers.IO) {
        if (nodeIds.size < 2) return@withContext
        val now = System.currentTimeMillis()
        db.beginTransaction()
        try {
            for (src in nodeIds) {
                for (dst in nodeIds) {
                    if (src == dst) continue
                    val current = db.rawQuery(
                        "SELECT hebb_weight FROM hebb_edges WHERE src_id = ? AND dst_id = ? AND layer = ?",
                        arrayOf(src, dst, layer),
                    ).use { c -> if (c.moveToFirst()) c.getDouble(0) else null }
                    if (current != null) {
                        db.compileStatement(
                            "UPDATE hebb_edges SET hebb_weight = ?, last_updated = ?, " +
                                "co_activation_count = co_activation_count + 1 " +
                                "WHERE src_id = ? AND dst_id = ? AND layer = ?",
                        ).run {
                            bindDouble(1, reinforcedWeight(current, mT, eta, lam, wFloor, wCeil))
                            bindLong(2, now)
                            bindString(3, src)
                            bindString(4, dst)
                            bindString(5, layer)
                            executeUpdateDelete()
                        }
                    } else {
                        db.compileStatement(
                            "INSERT OR IGNORE INTO hebb_edges(src_id, dst_id, layer, hebb_weight, " +
                                "eligibility, causal_score, co_activation_count, last_updated) " +
                                "VALUES (?, ?, ?, 1.0, 0.0, 0.0, 1, ?)",
                        ).run {
                            bindString(1, src)
                            bindString(2, dst)
                            bindString(3, layer)
                            bindLong(4, now)
                            executeInsert()
                        }
                    }
                }
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
        Unit
    }

    // ── Subgraph retrieval for GNN ────────────────────────────────────

    override suspend fun getLocalSubgraph(
        seedIds: List<String>,
        hops: Int,
        maxNodes: Int,
        layer: String,
        minWeight: Double,
    ): Subgraph = withContext(Dispatchers.IO) {
        if (seedIds.isEmpty()) return@withContext Subgraph(emptyList(), emptyList())
        localSubgraphKernel(
            seedIds, hops, maxNodes, layer, minWeight,
            nodeById = ::getConceptSync,
            outEdges = ::outEdgesSync,
        )
    }

    override suspend fun getGlobalGraph(
        layer: String,
        minWeight: Double,
        limit: Int,
    ): Subgraph = withContext(Dispatchers.IO) {
        val edges = db.rawQuery(
            "SELECT src_id, dst_id, hebb_weight, eligibility, causal_score, co_activation_count, " +
                "last_updated, layer FROM hebb_edges WHERE layer = ? AND hebb_weight >= ? " +
                "ORDER BY hebb_weight DESC LIMIT ?",
            arrayOf(layer, minWeight.toString(), limit.toString()),
        ).use { c ->
            buildList {
                while (c.moveToNext()) add(readEdge(c))
            }
        }
        if (edges.isEmpty()) return@withContext Subgraph(emptyList(), emptyList())
        val nodeIds = LinkedHashSet<String>()
        for (e in edges) {
            nodeIds += e.srcId
            nodeIds += e.dstId
        }
        val nodes = nodeIds.chunked(500).flatMap { chunk ->
            val placeholders = chunk.joinToString(", ") { "?" }
            db.rawQuery(
                "SELECT $CONCEPT_COLUMNS FROM concepts WHERE node_id IN ($placeholders)",
                chunk.toTypedArray(),
            ).use { c ->
                buildList {
                    while (c.moveToNext()) add(readConcept(c))
                }
            }
        }
        Subgraph(nodes, edges)
    }

    override suspend fun listConcepts(limit: Int): List<ConceptNode> = withContext(Dispatchers.IO) {
        db.rawQuery(
            "SELECT $CONCEPT_COLUMNS FROM concepts ORDER BY updated_at DESC LIMIT ?",
            arrayOf(limit.toString()),
        ).use { c ->
            buildList {
                while (c.moveToNext()) add(readConcept(c))
            }
        }
    }

    // ── ARM-E stats updates ───────────────────────────────────────────

    override suspend fun updateConceptArmeStats(nodeId: String, mT: Double, dominance: Double, rT: Double) =
        withContext(Dispatchers.IO) {
            val current = getConceptSync(nodeId) ?: return@withContext
            val updated = armeStatsUpdate(
                current.meanMT, current.meanDominance, current.meanRT,
                current.activationCount, current.recentEvents,
                mT, dominance, rT, System.currentTimeMillis(),
            )
            db.compileStatement(
                "UPDATE concepts SET mean_m_t = ?, mean_dominance = ?, mean_r_t = ?, recent_events = ? " +
                    "WHERE node_id = ?",
            ).run {
                bindDouble(1, updated.meanMT)
                bindDouble(2, updated.meanDominance)
                bindDouble(3, updated.meanRT)
                bindString(4, encodeRecentEvents(updated.recentEvents))
                bindString(5, nodeId)
                executeUpdateDelete()
            }
            refreshCached(nodeId)
            Unit
        }

    // ── Session management ────────────────────────────────────────────

    override suspend fun startSession(trigger: String): String = withContext(Dispatchers.IO) {
        val sessionId = UUID.randomUUID().toString()
        db.compileStatement(
            """INSERT INTO sessions(session_id, "trigger", started_at, ended_at, total_activations) """ +
                "VALUES (?, ?, ?, NULL, 0)",
        ).run {
            bindString(1, sessionId)
            bindString(2, trigger)
            bindLong(3, System.currentTimeMillis())
            executeInsert()
        }
        sessionId
    }

    override suspend fun endSession(sessionId: String) = withContext(Dispatchers.IO) {
        db.compileStatement("UPDATE sessions SET ended_at = ? WHERE session_id = ?").run {
            bindLong(1, System.currentTimeMillis())
            bindString(2, sessionId)
            executeUpdateDelete()
        }
        Unit
    }

    override suspend fun incrementSessionActivations(sessionId: String, count: Int) = withContext(Dispatchers.IO) {
        db.compileStatement(
            "UPDATE sessions SET total_activations = total_activations + ? WHERE session_id = ?",
        ).run {
            bindLong(1, count.toLong())
            bindString(2, sessionId)
            executeUpdateDelete()
        }
        Unit
    }

    override suspend fun upsertSessionStats(
        sessionId: String,
        nodeId: String,
        mT: Double,
        dominance: Double,
        rT: Double,
        deltaW: Double,
    ) = withContext(Dispatchers.IO) {
        db.beginTransaction()
        try {
            val existing = db.rawQuery(
                "SELECT mean_m_t, mean_dominance, mean_r_t, activation_count, delta_w_mean " +
                    "FROM session_stats WHERE session_id = ? AND node_id = ?",
                arrayOf(sessionId, nodeId),
            ).use { c ->
                if (c.moveToFirst()) {
                    SessionStatsRow(c.getDouble(0), c.getDouble(1), c.getDouble(2), c.getInt(3), c.getDouble(4))
                } else {
                    null
                }
            }
            if (existing == null) {
                db.compileStatement(
                    "INSERT INTO session_stats(session_id, node_id, mean_m_t, mean_dominance, mean_r_t, " +
                        "activation_count, delta_w_mean) VALUES (?, ?, ?, ?, ?, 1, ?)",
                ).run {
                    bindString(1, sessionId)
                    bindString(2, nodeId)
                    bindDouble(3, mT)
                    bindDouble(4, dominance)
                    bindDouble(5, rT)
                    bindDouble(6, deltaW)
                    executeInsert()
                }
            } else {
                // Welford online mean with n = activation_count + 1 (graph_store.py:717-726).
                val n = existing.activationCount + 1
                db.compileStatement(
                    "UPDATE session_stats SET mean_m_t = ?, mean_dominance = ?, mean_r_t = ?, " +
                        "delta_w_mean = ?, activation_count = ? WHERE session_id = ? AND node_id = ?",
                ).run {
                    bindDouble(1, existing.meanMT + (mT - existing.meanMT) / n)
                    bindDouble(2, existing.meanDominance + (dominance - existing.meanDominance) / n)
                    bindDouble(3, existing.meanRT + (rT - existing.meanRT) / n)
                    bindDouble(4, existing.deltaWMean + (deltaW - existing.deltaWMean) / n)
                    bindLong(5, n.toLong())
                    bindString(6, sessionId)
                    bindString(7, nodeId)
                    executeUpdateDelete()
                }
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
        Unit
    }

    // ── LLM retrieval ─────────────────────────────────────────────────

    override suspend fun retrieveForLlm(
        queryEmbedding: FloatArray,
        seedNodeIds: List<String>,
        topK: Int,
        alpha: Double,
        beta: Double,
        gamma: Double,
        wCeil: Double,
        useSpreading: Boolean,
        deltaSpread: Double,
    ): List<LlmCandidate> = withContext(Dispatchers.IO) {
        val rawSpread = if (useSpreading && seedNodeIds.isNotEmpty()) {
            spreadingActivationKernel(
                seedNodeIds, layer = "hippocampal", depth = 2, decay = 0.45,
                minWeight = 0.3, maxPaths = 2000, outEdges = ::outEdgesSync,
            )
        } else {
            emptyMap()
        }
        val hits = withSpreadCandidates(
            cosineSearch(cachedConcepts(), queryEmbedding, topK * 3, conceptType = null),
            rawSpread, queryEmbedding, extra = topK, minSpread = SPREAD_CANDIDATE_MIN,
            nodeById = ::getConceptSync,
        )
        if (hits.isEmpty()) return@withContext emptyList()
        val candidateIds = hits.map { it.first.nodeId }

        val assocScores: Map<String, Double>
        val causalScores: Map<String, Double>
        if (seedNodeIds.isNotEmpty()) {
            // Directed edges FROM seeds TO candidates, any layer (graph_store.py:788-795).
            val seedPh = seedNodeIds.joinToString(", ") { "?" }
            val candPh = candidateIds.joinToString(", ") { "?" }
            val assoc = HashMap<String, Double>()
            val causal = HashMap<String, Double>()
            db.rawQuery(
                "SELECT dst_id, AVG(hebb_weight), AVG(causal_score) FROM hebb_edges " +
                    "WHERE src_id IN ($seedPh) AND dst_id IN ($candPh) GROUP BY dst_id",
                (seedNodeIds + candidateIds).toTypedArray(),
            ).use { c ->
                while (c.moveToNext()) {
                    assoc[c.getString(0)] = c.getDouble(1)
                    causal[c.getString(0)] = c.getDouble(2)
                }
            }
            assocScores = assoc
            causalScores = causal
        } else {
            assocScores = emptyMap()
            causalScores = emptyMap()
        }

        val candidateSet = candidateIds.toHashSet()
        val spreadScores = rawSpread.filterKeys { it in candidateSet }

        combineLlmScores(
            hits, assocScores, causalScores, spreadScores,
            topK, alpha, beta, gamma, wCeil, deltaSpread,
        )
    }

    // ── Practice: fading concepts ─────────────────────────────────────

    override suspend fun getFadingConcepts(
        limit: Int,
        minActivations: Int,
        wCeil: Double,
        candidateMult: Int,
    ): List<FadingConcept> = withContext(Dispatchers.IO) {
        // Hippocampal edges in both directions (undirected in Python — graph_store.py:883).
        val raw = db.rawQuery(
            "SELECT $CONCEPT_COLUMNS_C, AVG(e.hebb_weight), COUNT(e.src_id) FROM concepts c " +
                "LEFT JOIN hebb_edges e ON e.layer = 'hippocampal' " +
                "AND (e.src_id = c.node_id OR e.dst_id = c.node_id) " +
                "WHERE c.activation_count >= ? GROUP BY c.node_id " +
                "ORDER BY c.updated_at ASC LIMIT ?",
            arrayOf(minActivations.toString(), (limit * candidateMult).toString()),
        ).use { c ->
            buildList {
                while (c.moveToNext()) {
                    add(
                        FadingRaw(
                            node = readConcept(c),
                            avgW = if (c.isNull(CONCEPT_COLUMN_COUNT)) 0.0 else c.getDouble(CONCEPT_COLUMN_COUNT),
                            degree = c.getInt(CONCEPT_COLUMN_COUNT + 1),
                        ),
                    )
                }
            }
        }
        rankFadingConcepts(raw, System.currentTimeMillis(), limit, wCeil)
    }

    override suspend fun spreadingActivation(
        seedNodeIds: List<String>,
        layer: String,
        depth: Int,
        decay: Double,
        minWeight: Double,
        maxPaths: Int,
    ): Map<String, Double> = withContext(Dispatchers.IO) {
        if (seedNodeIds.isEmpty()) return@withContext emptyMap()
        spreadingActivationKernel(seedNodeIds, layer, depth, decay, minWeight, maxPaths, ::outEdgesSync)
    }

    // ── Phase 4: w_d regression data ─────────────────────────────────

    override suspend fun getDominanceRegressionData(minActivations: Int): List<DominanceRegressionRow> =
        withContext(Dispatchers.IO) {
            db.rawQuery(
                "SELECT s.session_id, c.concept_type, ss.mean_m_t, ss.mean_dominance, " +
                    "ss.delta_w_mean, ss.activation_count FROM session_stats ss " +
                    "JOIN sessions s ON s.session_id = ss.session_id " +
                    "JOIN concepts c ON c.node_id = ss.node_id " +
                    "WHERE ss.activation_count >= ? ORDER BY s.started_at DESC",
                arrayOf(minActivations.toString()),
            ).use { c ->
                buildList {
                    while (c.moveToNext()) {
                        add(
                            DominanceRegressionRow(
                                sessionId = c.getString(0),
                                conceptType = c.getString(1),
                                mT = c.getDouble(2),
                                dominance = c.getDouble(3),
                                deltaW = c.getDouble(4),
                                activationCount = c.getInt(5),
                            ),
                        )
                    }
                }
            }
        }

    override suspend fun clear() = withContext(Dispatchers.IO) {
        db.execSQL("DELETE FROM session_stats")
        db.execSQL("DELETE FROM sessions")
        db.execSQL("DELETE FROM hebb_edges")
        db.execSQL("DELETE FROM concepts")
        invalidateCache()
        Unit
    }

    // ---- internals ----

    private class SessionStatsRow(
        val meanMT: Double,
        val meanDominance: Double,
        val meanRT: Double,
        val activationCount: Int,
        val deltaWMean: Double,
    )

    override suspend fun backupTo(dir: File) = withContext(Dispatchers.IO) {
        dir.mkdirs()
        val target = File(dir, File(helper.databaseName).name) // same file name as the live db
        target.delete() // VACUUM INTO refuses an existing file
        // SQLite ≥ 3.27 (API 30 ships 3.28): a consistent snapshot while the db stays open.
        db.execSQL("VACUUM INTO '" + target.absolutePath.replace("'", "''") + "'")
    }

    override suspend fun applyMerge(plan: MergePlan) = withContext(Dispatchers.IO) {
        if (plan.groups.isEmpty()) return@withContext
        val ids = plan.touchedIds.toList()
        db.beginTransaction()
        try {
            // Edges: every edge touching a merged node, re-pointed and combined.
            val touched = LinkedHashMap<Triple<String, String, String>, HebbianEdge>()
            for (chunk in ids.chunked(400)) {
                val ph = chunk.joinToString(", ") { "?" }
                db.rawQuery(
                    "SELECT src_id, dst_id, hebb_weight, eligibility, causal_score, co_activation_count, " +
                        "last_updated, layer FROM hebb_edges WHERE src_id IN ($ph) OR dst_id IN ($ph)",
                    (chunk + chunk).toTypedArray(),
                ).use { c -> while (c.moveToNext()) readEdge(c).let { touched[Triple(it.srcId, it.dstId, it.layer)] = it } }
                db.execSQL("DELETE FROM hebb_edges WHERE src_id IN ($ph) OR dst_id IN ($ph)", (chunk + chunk).toTypedArray())
            }
            val insertEdge = db.compileStatement(
                "INSERT INTO hebb_edges(src_id, dst_id, layer, hebb_weight, eligibility, causal_score, " +
                    "co_activation_count, last_updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            )
            for (e in ConceptMerge.mergeEdges(touched.values.toList(), plan.remap)) {
                insertEdge.bindString(1, e.srcId)
                insertEdge.bindString(2, e.dstId)
                insertEdge.bindString(3, e.layer)
                insertEdge.bindDouble(4, e.hebbWeight)
                insertEdge.bindDouble(5, e.eligibility)
                insertEdge.bindDouble(6, e.causalScore)
                insertEdge.bindLong(7, e.coActivationCount.toLong())
                insertEdge.bindLong(8, e.lastUpdated)
                insertEdge.executeInsert()
            }

            // Session stats.
            val rows = ArrayList<SessionStatRow>()
            for (chunk in ids.chunked(800)) {
                val ph = chunk.joinToString(", ") { "?" }
                db.rawQuery(
                    "SELECT session_id, node_id, mean_m_t, mean_dominance, mean_r_t, activation_count, " +
                        "delta_w_mean FROM session_stats WHERE node_id IN ($ph)",
                    chunk.toTypedArray(),
                ).use { c ->
                    while (c.moveToNext()) {
                        rows += SessionStatRow(
                            statId = null, sessionId = c.getString(0), nodeId = c.getString(1),
                            meanMT = c.getDouble(2), meanDominance = c.getDouble(3), meanRT = c.getDouble(4),
                            activationCount = c.getInt(5), deltaWMean = c.getDouble(6),
                        )
                    }
                }
                db.execSQL("DELETE FROM session_stats WHERE node_id IN ($ph)", chunk.toTypedArray())
            }
            val insertStat = db.compileStatement(
                "INSERT INTO session_stats(session_id, node_id, mean_m_t, mean_dominance, mean_r_t, " +
                    "activation_count, delta_w_mean) VALUES (?, ?, ?, ?, ?, ?, ?)",
            )
            for (r in ConceptMerge.mergeSessionStats(rows, plan.remap)) {
                insertStat.bindString(1, r.sessionId)
                insertStat.bindString(2, r.nodeId)
                insertStat.bindDouble(3, r.meanMT)
                insertStat.bindDouble(4, r.meanDominance)
                insertStat.bindDouble(5, r.meanRT)
                insertStat.bindLong(6, r.activationCount.toLong())
                insertStat.bindDouble(7, r.deltaWMean)
                insertStat.executeInsert()
            }

            // Concepts: survivor takes the merged history, the rest go.
            for (g in plan.groups) {
                val m = ConceptMerge.mergedNode(g)
                db.compileStatement(
                    "UPDATE concepts SET activation_count = ?, created_at = ?, updated_at = ?, mean_m_t = ?, " +
                        "mean_dominance = ?, mean_r_t = ?, recent_events = ? WHERE node_id = ?",
                ).run {
                    bindLong(1, m.activationCount.toLong())
                    bindLong(2, m.createdAt)
                    bindLong(3, m.updatedAt)
                    bindDouble(4, m.meanMT)
                    bindDouble(5, m.meanDominance)
                    bindDouble(6, m.meanRT)
                    bindString(7, encodeRecentEvents(m.recentEvents))
                    bindString(8, m.nodeId)
                    executeUpdateDelete()
                }
            }
            for (chunk in plan.remap.keys.chunked(800)) {
                db.execSQL(
                    "DELETE FROM concepts WHERE node_id IN (${chunk.joinToString(", ") { "?" }})",
                    chunk.toTypedArray(),
                )
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
        invalidateCache()
    }

    private val cacheLock = Any()

    @Volatile
    private var conceptCache: List<ConceptNode>? = null

    private fun invalidateCache() = synchronized(cacheLock) { conceptCache = null }

    /** Write-through for a single-concept write: re-reads just that row into the cache.
     *  Ingest writes each concept three times (upsert, activation, ARM-E stats) and
     *  vector-searches in between — invalidating instead forced a full reload of every
     *  concept and its 768-d embedding several times per extracted concept. */
    private fun refreshCached(nodeId: String) = synchronized(cacheLock) {
        val cache = conceptCache ?: return@synchronized
        val fresh = getConceptSync(nodeId)
        val i = cache.indexOfFirst { it.nodeId == nodeId }
        conceptCache = when {
            fresh == null -> if (i >= 0) cache.filterIndexed { j, _ -> j != i } else cache
            i >= 0 -> cache.toMutableList().also { it[i] = fresh }
            else -> cache + fresh
        }
    }

    /** All concepts, cached for brute-force vector search; invalidated on any concept write. */
    private fun cachedConcepts(): List<ConceptNode> = synchronized(cacheLock) {
        conceptCache?.let { return it }
        val loaded = db.rawQuery("SELECT $CONCEPT_COLUMNS FROM concepts", null).use { c ->
            buildList {
                while (c.moveToNext()) add(readConcept(c))
            }
        }
        conceptCache = loaded
        loaded
    }

    private fun getConceptSync(nodeId: String): ConceptNode? =
        db.rawQuery(
            "SELECT $CONCEPT_COLUMNS FROM concepts WHERE node_id = ?",
            arrayOf(nodeId),
        ).use { c -> if (c.moveToFirst()) readConcept(c) else null }

    /** Outgoing edges of one node, deterministically ordered for the shared kernels. */
    private fun outEdgesSync(nodeId: String): List<HebbianEdge> =
        db.rawQuery(
            "SELECT src_id, dst_id, hebb_weight, eligibility, causal_score, co_activation_count, " +
                "last_updated, layer FROM hebb_edges WHERE src_id = ? ORDER BY dst_id, layer",
            arrayOf(nodeId),
        ).use { c ->
            buildList {
                while (c.moveToNext()) add(readEdge(c))
            }
        }

    private fun readConcept(c: Cursor) = ConceptNode(
        nodeId = c.getString(0),
        label = c.getString(1) ?: "",
        textRaw = c.getString(2) ?: "",
        textSummary = c.getString(3) ?: "",
        conceptType = c.getString(4) ?: "fact",
        sourceType = c.getString(5) ?: "document",
        sourceUri = if (c.isNull(6)) null else c.getString(6),
        createdAt = c.getLong(7),
        updatedAt = c.getLong(8),
        activationCount = c.getInt(9),
        embedding = if (c.isNull(10)) FloatArray(0) else bytesToFloats(c.getBlob(10)),
        meanMT = c.getDouble(11),
        meanDominance = c.getDouble(12),
        meanRT = c.getDouble(13),
        recentEvents = decodeRecentEvents(c.getString(14)),
    )

    private fun readEdge(c: Cursor) = HebbianEdge(
        srcId = c.getString(0),
        dstId = c.getString(1),
        hebbWeight = c.getDouble(2),
        eligibility = c.getDouble(3),
        causalScore = c.getDouble(4),
        coActivationCount = c.getInt(5),
        lastUpdated = c.getLong(6),
        layer = c.getString(7),
    )

    private companion object {
        const val CONCEPT_COLUMNS =
            "node_id, label, text_raw, text_summary, concept_type, source_type, source_uri, " +
                "created_at, updated_at, activation_count, embedding, mean_m_t, mean_dominance, mean_r_t, recent_events"
        const val CONCEPT_COLUMNS_C =
            "c.node_id, c.label, c.text_raw, c.text_summary, c.concept_type, c.source_type, c.source_uri, " +
                "c.created_at, c.updated_at, c.activation_count, c.embedding, " +
                "c.mean_m_t, c.mean_dominance, c.mean_r_t, c.recent_events"
        const val CONCEPT_COLUMN_COUNT = 15

        /** Full paths (containing '/') are used as-is by SQLiteOpenHelper; create the parent dir. */
        fun dbName(dbPath: String?): String {
            if (dbPath == null) return "hebbian_graph.db"
            File(dbPath).parentFile?.mkdirs()
            return dbPath
        }

        // Same big-endian float-blob encoding as rag/RagDatabase.kt.
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
