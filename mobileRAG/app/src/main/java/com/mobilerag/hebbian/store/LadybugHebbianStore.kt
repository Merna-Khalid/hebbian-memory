package com.mobilerag.hebbian.store

import com.ladybugdb.Connection
import com.ladybugdb.Database
import com.ladybugdb.QueryResult
import com.ladybugdb.SystemConfig
import com.mobilerag.hebbian.ConceptIdentity
import java.io.File
import java.util.Locale
import java.util.UUID
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * LadybugDB Hebbian store — the graph_store.py schema as LadybugDB node/rel tables,
 * queried with Cypher. Mirrors [SqliteHebbianStore] semantics exactly: all graph
 * algorithms run through the shared kernels in HebbianStore.kt, and vector search is
 * the same brute-force cosine over an in-memory cache (no LadybugDB vector index).
 *
 * Follows LadybugGraphStore.kt's idioms: the embedded engine is single-writer, so all
 * connection use is serialized on [connLock]; multi-statement operations (MERGE
 * emulation, Welford updates) hold the lock across their statements — synchronized is
 * reentrant, so [execute] can be called inside. Values go through prepared-statement
 * parameters where the Java API supports them; values that can't be parameters are
 * inlined only when injection-safe (ints via LIMIT, embeddings rendered as fixed-point
 * CSV string literals via [cypherString], id lists escaped the same way). `source_uri` is stored
 * as "" for null (null parameter type inference is unverified on this API version,
 * same caveat as LadybugGraphStore) and mapped back to null on read.
 *
 * WAL durability (LadybugDB 0.20.3): a write-ahead-log record holding any value ≥ 4096
 * bytes cannot be replayed — reopening after a process kill fails with "Corrupted wal
 * file. Read out invalid WAL record type" (STRING and DOUBLE[]/FLOAT[] alike; see
 * docs/phase7a-identity-consolidation-results.md). Every concept embedding (~9 KB) is such a
 * value, and Android kills processes without closing them, so every write that can carry
 * one ([upsertConcept], [updateConceptEmbeddings]) is followed by a CHECKPOINT: the WAL
 * then only ever holds small, replayable records. If a kill lands inside that window,
 * [open] sets the unreplayable WAL aside ([walRecovery]) and opens the last checkpoint.
 */
class LadybugHebbianStore : HebbianStore {

    override val id = "ladybugdb-hebbian"

    private var database: Database? = null
    private var connection: Connection? = null
    private val connLock = Any()

    /** Database path from the last successful [open] — [backupTo] reopens it. */
    private var path: String? = null

    /** Set when [open] had to set an unreplayable WAL aside: where it went, and why. */
    @Volatile
    var walRecovery: String? = null
        private set

    override suspend fun open(path: String) = withContext(Dispatchers.IO) { openSync(path) }

    private fun openSync(path: String) {
        try {
            openOnce(path)
        } catch (t: Throwable) {
            val wal = File("$path.wal")
            if (!wal.isFile || t.message?.contains("wal", ignoreCase = true) != true) throw t
            // Unreplayable WAL (see the class doc): keep it for inspection, open the last
            // checkpoint. Only the writes since that checkpoint are lost.
            val aside = File("$path.wal.corrupt-${System.currentTimeMillis()}")
            if (!wal.renameTo(aside)) throw t
            walRecovery = "set aside ${aside.name} (${aside.length()} B): ${t.message}"
            openOnce(path)
        }
    }

    private fun openOnce(path: String) {
        File(path).parentFile?.mkdirs()
        // The default config mmaps a 256 GB region per Database — the entity graph store
        // already holds one, and a second fails (Buffer manager exception: Mmap for size
        // 274877906944 failed). Cap this instance so both stores coexist.
        val config = SystemConfig().apply {
            bufferPoolSize = 256L * 1024 * 1024
            maxDBSize = 8L * 1024 * 1024 * 1024
        }
        val db = Database(path, config)
        try {
            connection = Connection(db)
            database = db
            ensureSchema()
            this.path = path
        } catch (t: Throwable) {
            val conn = connection
            connection = null
            database = null
            try { conn?.close() } catch (_: Throwable) {}
            try { db.close() } catch (_: Throwable) {}
            throw t
        }
    }

    override suspend fun close() = withContext(Dispatchers.IO) {
        synchronized(connLock) { closeSync() }
        Unit
    }

    private fun closeSync() {
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

    override suspend fun setupSchema() = withContext(Dispatchers.IO) {
        ensureSchema()
        Unit
    }

    // ── Concept CRUD ──────────────────────────────────────────────────

    override suspend fun upsertConcept(node: ConceptNode): String = withContext(Dispatchers.IO) {
        val now = System.currentTimeMillis()
        val embedding = doubleArrayLiteral(node.embedding)
        val sourceUri = node.sourceUri ?: ""
        synchronized(connLock) {
            val exists = execute(
                "MATCH (c:Concept {node_id: \$nid}) RETURN c.node_id",
                mapOf("nid" to node.nodeId),
            ).rows().isNotEmpty()
            if (exists) {
                // ON MATCH semantics: ARM-E stats, activation_count and created_at untouched.
                execute(
                    "MATCH (c:Concept {node_id: \$nid}) SET c.label = \$label, c.text_raw = \$raw, " +
                        "c.text_summary = \$summary, c.concept_type = \$ctype, c.source_type = \$stype, " +
                        "c.source_uri = \$uri, c.updated_at = \$now, c.embedding = $embedding",
                    mapOf(
                        "nid" to node.nodeId, "label" to node.label, "raw" to node.textRaw,
                        "summary" to node.textSummary, "ctype" to node.conceptType,
                        "stype" to node.sourceType, "uri" to sourceUri, "now" to now,
                    ),
                ).close()
            } else {
                execute(
                    "CREATE (:Concept {node_id: \$nid, label: \$label, text_raw: \$raw, " +
                        "text_summary: \$summary, concept_type: \$ctype, source_type: \$stype, " +
                        "source_uri: \$uri, created_at: \$now, updated_at: \$now, activation_count: 0, " +
                        "embedding: $embedding, mean_m_t: 1.0, mean_dominance: 0.5, mean_r_t: 0.5, " +
                        "recent_events: '[]'})",
                    mapOf(
                        "nid" to node.nodeId, "label" to node.label, "raw" to node.textRaw,
                        "summary" to node.textSummary, "ctype" to node.conceptType,
                        "stype" to node.sourceType, "uri" to sourceUri, "now" to now,
                    ),
                ).close()
            }
            checkpoint()
        }
        refreshCached(node.nodeId)
        node.nodeId
    }

    override suspend fun getConcept(nodeId: String): ConceptNode? = withContext(Dispatchers.IO) {
        getConceptSync(nodeId)
    }

    override suspend fun incrementActivation(nodeId: String) = withContext(Dispatchers.IO) {
        execute(
            "MATCH (c:Concept {node_id: \$nid}) " +
                "SET c.activation_count = c.activation_count + 1, c.updated_at = \$now",
            mapOf("nid" to nodeId, "now" to System.currentTimeMillis()),
        ).close()
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
            synchronized(connLock) {
                for ((nodeId, emb) in embeddings) {
                    // Embedding inlined as a DOUBLE[] literal, as upsertConcept does.
                    execute(
                        "MATCH (c:Concept {node_id: \$nid}) SET c.embedding = ${doubleArrayLiteral(emb)}",
                        mapOf("nid" to nodeId),
                    ).close()
                }
                checkpoint()
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
        val now = System.currentTimeMillis()
        synchronized(connLock) {
            val exists = execute(
                "MATCH (a:Concept {node_id: \$src})-[r:ASSOCIATED_WITH]->(b:Concept {node_id: \$dst}) " +
                    "WHERE r.layer = \$layer RETURN r.hebb_weight",
                mapOf("src" to edge.srcId, "dst" to edge.dstId, "layer" to edge.layer),
            ).rows().isNotEmpty()
            if (exists) {
                // Never weaken on re-assertion (see HebbianStore.upsertEdge): eligibility and
                // causal are kept, weight only rises. max() computed here, not in Cypher, to
                // stay within the LadybugDB functions verified on this version.
                val current = execute(
                    "MATCH (a:Concept {node_id: \$src})-[r:ASSOCIATED_WITH]->(b:Concept {node_id: \$dst}) " +
                        "WHERE r.layer = \$layer RETURN r.hebb_weight",
                    mapOf("src" to edge.srcId, "dst" to edge.dstId, "layer" to edge.layer),
                ).rows().firstOrNull()?.get(0) as? Number
                execute(
                    "MATCH (a:Concept {node_id: \$src})-[r:ASSOCIATED_WITH]->(b:Concept {node_id: \$dst}) " +
                        "WHERE r.layer = \$layer SET r.hebb_weight = \$w, " +
                        "r.co_activation_count = r.co_activation_count + 1, r.last_updated = \$now",
                    mapOf(
                        "src" to edge.srcId, "dst" to edge.dstId, "layer" to edge.layer,
                        "w" to maxOf(current?.toDouble() ?: edge.hebbWeight, edge.hebbWeight),
                        "now" to now,
                    ),
                ).close()
            } else {
                execute(
                    "MATCH (a:Concept {node_id: \$src}), (b:Concept {node_id: \$dst}) " +
                        "CREATE (a)-[:ASSOCIATED_WITH {hebb_weight: \$w, eligibility: \$elig, " +
                        "causal_score: \$causal, co_activation_count: \$count, last_updated: \$now, " +
                        "layer: \$layer}]->(b)",
                    mapOf(
                        "src" to edge.srcId, "dst" to edge.dstId, "w" to edge.hebbWeight,
                        "elig" to edge.eligibility, "causal" to edge.causalScore,
                        "count" to edge.coActivationCount.toLong(), "now" to now, "layer" to edge.layer,
                    ),
                ).close()
            }
        }
        Unit
    }

    override suspend fun updateHebbWeights(updates: List<HebbWeightUpdate>) = withContext(Dispatchers.IO) {
        if (updates.isEmpty()) return@withContext
        val now = System.currentTimeMillis()
        synchronized(connLock) {
            for (u in updates) {
                execute(
                    "MATCH (a:Concept {node_id: \$src})-[r:ASSOCIATED_WITH]->(b:Concept {node_id: \$dst}) " +
                        "WHERE r.layer = \$layer SET r.hebb_weight = \$w, r.eligibility = \$elig, " +
                        "r.last_updated = \$now, r.co_activation_count = r.co_activation_count + 1",
                    mapOf(
                        "src" to u.srcId, "dst" to u.dstId, "layer" to u.layer,
                        "w" to u.weight, "elig" to u.eligibility, "now" to now,
                    ),
                ).close()
            }
        }
        Unit
    }

    override suspend fun updateCausalScores(updates: List<CausalScoreUpdate>, layer: String) =
        withContext(Dispatchers.IO) {
            if (updates.isEmpty()) return@withContext
            synchronized(connLock) {
                for (u in updates) {
                    execute(
                        "MATCH (a:Concept {node_id: \$src})-[r:ASSOCIATED_WITH]->(b:Concept {node_id: \$dst}) " +
                            "WHERE r.layer = \$layer SET r.causal_score = \$score",
                        mapOf("src" to u.srcId, "dst" to u.dstId, "layer" to layer, "score" to u.score),
                    ).close()
                }
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
        synchronized(connLock) {
            for (src in nodeIds) {
                for (dst in nodeIds) {
                    if (src == dst) continue
                    val current = execute(
                        "MATCH (a:Concept {node_id: \$src})-[r:ASSOCIATED_WITH]->(b:Concept {node_id: \$dst}) " +
                            "WHERE r.layer = \$layer RETURN r.hebb_weight",
                        mapOf("src" to src, "dst" to dst, "layer" to layer),
                    ).rows().firstOrNull()?.firstOrNull()?.let { (it as? Number)?.toDouble() }
                    if (current != null) {
                        execute(
                            "MATCH (a:Concept {node_id: \$src})-[r:ASSOCIATED_WITH]->(b:Concept {node_id: \$dst}) " +
                                "WHERE r.layer = \$layer SET r.hebb_weight = \$w, r.last_updated = \$now, " +
                                "r.co_activation_count = r.co_activation_count + 1",
                            mapOf(
                                "src" to src, "dst" to dst, "layer" to layer,
                                "w" to reinforcedWeight(current, mT, eta, lam, wFloor, wCeil),
                                "now" to now,
                            ),
                        ).close()
                    } else {
                        execute(
                            "MATCH (a:Concept {node_id: \$src}), (b:Concept {node_id: \$dst}) " +
                                "CREATE (a)-[:ASSOCIATED_WITH {hebb_weight: 1.0, eligibility: 0.0, " +
                                "causal_score: 0.0, co_activation_count: 1, last_updated: \$now, " +
                                "layer: \$layer}]->(b)",
                            mapOf("src" to src, "dst" to dst, "now" to now, "layer" to layer),
                        ).close()
                    }
                }
            }
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
        // LIMIT inlined: LadybugDB doesn't accept parameters there; Int is safe.
        val edges = execute(
            "MATCH (a:Concept)-[r:ASSOCIATED_WITH]->(b:Concept) " +
                "WHERE r.layer = \$layer AND r.hebb_weight >= \$minW " +
                "RETURN a.node_id, b.node_id, r.hebb_weight, r.eligibility, r.causal_score, " +
                "r.co_activation_count, r.last_updated, r.layer ORDER BY r.hebb_weight DESC LIMIT $limit",
            mapOf("layer" to layer, "minW" to minWeight),
        ).rows().map(::rowToEdge)
        if (edges.isEmpty()) return@withContext Subgraph(emptyList(), emptyList())
        val nodeIds = LinkedHashSet<String>()
        for (e in edges) {
            nodeIds += e.srcId
            nodeIds += e.dstId
        }
        Subgraph(conceptsByIds(nodeIds.toList()), edges)
    }

    override suspend fun listConcepts(limit: Int): List<ConceptNode> = withContext(Dispatchers.IO) {
        // LIMIT inlined: LadybugDB doesn't accept parameters there; Int is safe.
        execute("MATCH (c:Concept) RETURN $CONCEPT_RETURN ORDER BY c.updated_at DESC LIMIT $limit")
            .rows().map(::rowToConcept)
    }

    // ── ARM-E stats updates ───────────────────────────────────────────

    override suspend fun updateConceptArmeStats(nodeId: String, mT: Double, dominance: Double, rT: Double) =
        withContext(Dispatchers.IO) {
            synchronized(connLock) {
                val current = getConceptSync(nodeId) ?: return@synchronized
                val updated = armeStatsUpdate(
                    current.meanMT, current.meanDominance, current.meanRT,
                    current.activationCount, current.recentEvents,
                    mT, dominance, rT, System.currentTimeMillis(),
                )
                execute(
                    "MATCH (c:Concept {node_id: \$nid}) SET c.mean_m_t = \$mm, " +
                        "c.mean_dominance = \$md, c.mean_r_t = \$mr, c.recent_events = \$re",
                    mapOf(
                        "nid" to nodeId, "mm" to updated.meanMT, "md" to updated.meanDominance,
                        "mr" to updated.meanRT, "re" to encodeRecentEvents(updated.recentEvents),
                    ),
                ).close()
            }
            refreshCached(nodeId)
            Unit
        }

    // ── Session management ────────────────────────────────────────────

    override suspend fun startSession(trigger: String): String = withContext(Dispatchers.IO) {
        val sessionId = UUID.randomUUID().toString()
        // ended_at omitted (NULL) until endSession writes it.
        execute(
            "CREATE (:Session {session_id: \$sid, trigger: \$trigger, started_at: \$now, total_activations: 0})",
            mapOf("sid" to sessionId, "trigger" to trigger, "now" to System.currentTimeMillis()),
        ).close()
        sessionId
    }

    override suspend fun endSession(sessionId: String) = withContext(Dispatchers.IO) {
        execute(
            "MATCH (s:Session {session_id: \$sid}) SET s.ended_at = \$now",
            mapOf("sid" to sessionId, "now" to System.currentTimeMillis()),
        ).close()
        Unit
    }

    override suspend fun incrementSessionActivations(sessionId: String, count: Int) = withContext(Dispatchers.IO) {
        execute(
            "MATCH (s:Session {session_id: \$sid}) SET s.total_activations = s.total_activations + \$count",
            mapOf("sid" to sessionId, "count" to count.toLong()),
        ).close()
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
        synchronized(connLock) {
            val existing = execute(
                "MATCH (ss:SessionStats) WHERE ss.session_id = \$sid AND ss.node_id = \$nid " +
                    "RETURN ss.stat_id, ss.mean_m_t, ss.mean_dominance, ss.mean_r_t, " +
                    "ss.activation_count, ss.delta_w_mean",
                mapOf("sid" to sessionId, "nid" to nodeId),
            ).rows().firstOrNull()
            val statId: String
            if (existing == null) {
                statId = UUID.randomUUID().toString()
                execute(
                    "CREATE (:SessionStats {stat_id: \$stat, session_id: \$sid, node_id: \$nid, " +
                        "mean_m_t: \$mt, mean_dominance: \$dom, mean_r_t: \$rt, activation_count: 1, " +
                        "delta_w_mean: \$dw})",
                    mapOf(
                        "stat" to statId, "sid" to sessionId, "nid" to nodeId, "mt" to mT,
                        "dom" to dominance, "rt" to rT, "dw" to deltaW,
                    ),
                ).close()
            } else {
                statId = existing[0] as String
                // Welford online mean with n = activation_count + 1 (graph_store.py:717-726).
                val n = (existing[4] as Number).toInt() + 1
                execute(
                    "MATCH (ss:SessionStats {stat_id: \$stat}) SET ss.mean_m_t = \$mm, " +
                        "ss.mean_dominance = \$md, ss.mean_r_t = \$mr, ss.delta_w_mean = \$dw, " +
                        "ss.activation_count = \$count",
                    mapOf(
                        "stat" to statId,
                        "mm" to (existing[1] as Number).toDouble() + (mT - (existing[1] as Number).toDouble()) / n,
                        "md" to (existing[2] as Number).toDouble() + (dominance - (existing[2] as Number).toDouble()) / n,
                        "mr" to (existing[3] as Number).toDouble() + (rT - (existing[3] as Number).toDouble()) / n,
                        "dw" to (existing[5] as Number).toDouble() + (deltaW - (existing[5] as Number).toDouble()) / n,
                        "count" to n.toLong(),
                    ),
                ).close()
            }
            // MERGE emulation for AGGREGATES / TRACKED_IN (Python uses MERGE — graph_store.py:729-730).
            val aggregates = execute(
                "MATCH (s:Session {session_id: \$sid})-[r:AGGREGATES]->(ss:SessionStats {stat_id: \$stat}) " +
                    "RETURN count(r)",
                mapOf("sid" to sessionId, "stat" to statId),
            ).rows().firstOrNull()?.firstOrNull()?.let { (it as? Number)?.toLong() } ?: 0L
            if (aggregates == 0L) {
                execute(
                    "MATCH (s:Session {session_id: \$sid}), (ss:SessionStats {stat_id: \$stat}) " +
                        "CREATE (s)-[:AGGREGATES]->(ss)",
                    mapOf("sid" to sessionId, "stat" to statId),
                ).close()
            }
            val tracked = execute(
                "MATCH (ss:SessionStats {stat_id: \$stat})-[r:TRACKED_IN]->(c:Concept {node_id: \$nid}) " +
                    "RETURN count(r)",
                mapOf("stat" to statId, "nid" to nodeId),
            ).rows().firstOrNull()?.firstOrNull()?.let { (it as? Number)?.toLong() } ?: 0L
            if (tracked == 0L) {
                execute(
                    "MATCH (ss:SessionStats {stat_id: \$stat}), (c:Concept {node_id: \$nid}) " +
                        "CREATE (ss)-[:TRACKED_IN]->(c)",
                    mapOf("stat" to statId, "nid" to nodeId),
                ).close()
            }
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
        val candidateIds = hits.mapTo(HashSet()) { it.first.nodeId }

        val assocScores = HashMap<String, Double>()
        val causalScores = HashMap<String, Double>()
        if (seedNodeIds.isNotEmpty()) {
            // Seed id list inlined (escaped): list-typed parameters are not verified on
            // this LadybugDB version. Directed edges FROM seeds TO candidates, any layer.
            val seedList = seedNodeIds.joinToString(", ") { cypherString(it) }
            for (cid in candidateIds) {
                val row = execute(
                    "MATCH (seed:Concept)-[r:ASSOCIATED_WITH]->(cand:Concept {node_id: \$cid}) " +
                        "WHERE seed.node_id IN [$seedList] RETURN avg(r.hebb_weight), avg(r.causal_score)",
                    mapOf("cid" to cid),
                ).rows().firstOrNull() ?: continue
                // avg over an empty match set is null — candidate has no seed edges.
                val avgW = (row[0] as? Number)?.toDouble() ?: continue
                assocScores[cid] = avgW
                causalScores[cid] = (row[1] as? Number)?.toDouble() ?: 0.0
            }
        }

        val spreadScores = rawSpread.filterKeys { it in candidateIds }

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
        // LIMIT inlined; Int is safe.
        val candidates = execute(
            "MATCH (c:Concept) WHERE c.activation_count >= \$minAct " +
                "RETURN $CONCEPT_RETURN ORDER BY c.updated_at ASC LIMIT ${limit * candidateMult}",
            mapOf("minAct" to minActivations.toLong()),
        ).rows().map(::rowToConcept)
        val raw = candidates.map { node ->
            // Hippocampal edges in both directions (undirected in Python — graph_store.py:883).
            val row = execute(
                "MATCH (c:Concept {node_id: \$nid})-[r:ASSOCIATED_WITH]-() " +
                    "WHERE r.layer = 'hippocampal' RETURN avg(r.hebb_weight), count(r)",
                mapOf("nid" to node.nodeId),
            ).rows().firstOrNull()
            FadingRaw(
                node = node,
                avgW = (row?.get(0) as? Number)?.toDouble() ?: 0.0,
                degree = (row?.get(1) as? Number)?.toInt() ?: 0,
            )
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
        // Same Kotlin kernel as SQLite (guarantees identical results), instead of the
        // variable-length Cypher form — LadybugDB's *1..2 path enumeration order and
        // LIMIT cut-off differ from Neo4j's, so the kernel is the portable definition.
        spreadingActivationKernel(seedNodeIds, layer, depth, decay, minWeight, maxPaths, ::outEdgesSync)
    }

    // ── Phase 4: w_d regression data ─────────────────────────────────

    override suspend fun getDominanceRegressionData(minActivations: Int): List<DominanceRegressionRow> =
        withContext(Dispatchers.IO) {
            execute(
                "MATCH (s:Session)-[:AGGREGATES]->(ss:SessionStats)-[:TRACKED_IN]->(c:Concept) " +
                    "WHERE ss.activation_count >= \$minAct " +
                    "RETURN s.session_id, c.concept_type, ss.mean_m_t, ss.mean_dominance, " +
                    "ss.delta_w_mean, ss.activation_count ORDER BY s.started_at DESC",
                mapOf("minAct" to minActivations.toLong()),
            ).rows().map { row ->
                DominanceRegressionRow(
                    sessionId = row[0] as String,
                    conceptType = row[1] as? String ?: "fact",
                    mT = (row[2] as? Number)?.toDouble() ?: 0.0,
                    dominance = (row[3] as? Number)?.toDouble() ?: 0.0,
                    deltaW = (row[4] as? Number)?.toDouble() ?: 0.0,
                    activationCount = (row[5] as? Number)?.toInt() ?: 0,
                )
            }
        }

    override suspend fun clear() = withContext(Dispatchers.IO) {
        synchronized(connLock) {
            execute("MATCH (s:Session)-[r:AGGREGATES]->() DELETE r").close()
            execute("MATCH ()-[r:TRACKED_IN]->() DELETE r").close()
            execute("MATCH ()-[r:ASSOCIATED_WITH]->() DELETE r").close()
            execute("MATCH (ss:SessionStats) DELETE ss").close()
            execute("MATCH (s:Session) DELETE s").close()
            execute("MATCH (c:Concept) DELETE c").close()
        }
        invalidateCache()
        Unit
    }

    // ── Maintenance ───────────────────────────────────────────────────

    override suspend fun backupTo(dir: File) = withContext(Dispatchers.IO) {
        synchronized(connLock) {
            val p = path ?: throw IllegalStateException("LadybugHebbianStore is not open")
            val src = File(p)
            dir.mkdirs()
            // Closing flushes the engine's state to disk, so the copy is consistent. The
            // same instance reopens before the lock is released: callers never see it closed.
            closeSync()
            try {
                // The database plus its siblings (write-ahead log etc.), file or directory.
                src.parentFile?.listFiles { f -> f.name.startsWith(src.name) }?.forEach { f ->
                    f.copyRecursively(File(dir, f.name), overwrite = true)
                }
            } finally {
                openSync(p)
            }
        }
        Unit
    }

    override suspend fun applyMerge(plan: MergePlan) = withContext(Dispatchers.IO) {
        if (plan.groups.isEmpty()) return@withContext
        // No multi-statement transaction on this API version: the caller takes backupTo
        // first, and the connection lock is held throughout so no store call interleaves.
        synchronized(connLock) {
            val ids = plan.touchedIds.toList()

            // Edges: every edge touching a merged node, re-pointed and combined.
            val touched = LinkedHashMap<Triple<String, String, String>, HebbianEdge>()
            for (chunk in ids.chunked(500)) {
                val list = chunk.joinToString(", ") { cypherString(it) }
                execute(
                    "MATCH (a:Concept)-[r:ASSOCIATED_WITH]->(b:Concept) " +
                        "WHERE a.node_id IN [$list] OR b.node_id IN [$list] " +
                        "RETURN a.node_id, b.node_id, r.hebb_weight, r.eligibility, r.causal_score, " +
                        "r.co_activation_count, r.last_updated, r.layer",
                ).rows().map(::rowToEdge).forEach { touched[Triple(it.srcId, it.dstId, it.layer)] = it }
            }
            for (chunk in ids.chunked(500)) {
                val list = chunk.joinToString(", ") { cypherString(it) }
                execute(
                    "MATCH (a:Concept)-[r:ASSOCIATED_WITH]->(b:Concept) " +
                        "WHERE a.node_id IN [$list] OR b.node_id IN [$list] DELETE r",
                ).close()
            }
            for (e in ConceptMerge.mergeEdges(touched.values.toList(), plan.remap)) {
                execute(
                    "MATCH (a:Concept {node_id: \$src}), (b:Concept {node_id: \$dst}) " +
                        "CREATE (a)-[:ASSOCIATED_WITH {hebb_weight: \$w, eligibility: \$elig, " +
                        "causal_score: \$causal, co_activation_count: \$count, last_updated: \$ts, " +
                        "layer: \$layer}]->(b)",
                    mapOf(
                        "src" to e.srcId, "dst" to e.dstId, "w" to e.hebbWeight, "elig" to e.eligibility,
                        "causal" to e.causalScore, "count" to e.coActivationCount.toLong(),
                        "ts" to e.lastUpdated, "layer" to e.layer,
                    ),
                ).close()
            }

            // Session stats: drop the losers of each collision, re-point the rest.
            val rows = ArrayList<SessionStatRow>()
            for (chunk in ids.chunked(500)) {
                val list = chunk.joinToString(", ") { cypherString(it) }
                execute(
                    "MATCH (ss:SessionStats) WHERE ss.node_id IN [$list] RETURN ss.stat_id, ss.session_id, " +
                        "ss.node_id, ss.mean_m_t, ss.mean_dominance, ss.mean_r_t, ss.activation_count, ss.delta_w_mean",
                ).rows().forEach { r ->
                    rows += SessionStatRow(
                        statId = r[0] as String, sessionId = r[1] as String, nodeId = r[2] as String,
                        meanMT = (r[3] as? Number)?.toDouble() ?: 1.0,
                        meanDominance = (r[4] as? Number)?.toDouble() ?: 0.5,
                        meanRT = (r[5] as? Number)?.toDouble() ?: 0.5,
                        activationCount = (r[6] as? Number)?.toInt() ?: 0,
                        deltaWMean = (r[7] as? Number)?.toDouble() ?: 0.0,
                    )
                }
            }
            val survivors = ConceptMerge.mergeSessionStats(rows, plan.remap)
            val survivorIds = survivors.mapNotNullTo(HashSet()) { it.statId }
            for (r in rows) {
                if (r.statId in survivorIds) continue
                val sid = mapOf("stat" to r.statId)
                execute("MATCH (s:Session)-[r:AGGREGATES]->(ss:SessionStats {stat_id: \$stat}) DELETE r", sid).close()
                execute("MATCH (ss:SessionStats {stat_id: \$stat})-[r:TRACKED_IN]->(c:Concept) DELETE r", sid).close()
                execute("MATCH (ss:SessionStats {stat_id: \$stat}) DELETE ss", sid).close()
            }
            val originalNode = rows.associate { it.statId to it.nodeId }
            for (r in survivors) {
                if (originalNode[r.statId] == r.nodeId) continue
                val p = mapOf("stat" to r.statId, "nid" to r.nodeId)
                execute("MATCH (ss:SessionStats {stat_id: \$stat})-[r:TRACKED_IN]->(c:Concept) DELETE r", mapOf("stat" to r.statId)).close()
                execute("MATCH (ss:SessionStats {stat_id: \$stat}) SET ss.node_id = \$nid", p).close()
                execute(
                    "MATCH (ss:SessionStats {stat_id: \$stat}), (c:Concept {node_id: \$nid}) CREATE (ss)-[:TRACKED_IN]->(c)",
                    p,
                ).close()
            }

            // Concepts: survivor takes the merged history, the rest go (no relations left).
            for (g in plan.groups) {
                val m = ConceptMerge.mergedNode(g)
                execute(
                    "MATCH (c:Concept {node_id: \$nid}) SET c.activation_count = \$count, c.created_at = \$created, " +
                        "c.updated_at = \$updated, c.mean_m_t = \$mm, c.mean_dominance = \$md, c.mean_r_t = \$mr, " +
                        "c.recent_events = \$re",
                    mapOf(
                        "nid" to m.nodeId, "count" to m.activationCount.toLong(), "created" to m.createdAt,
                        "updated" to m.updatedAt, "mm" to m.meanMT, "md" to m.meanDominance, "mr" to m.meanRT,
                        "re" to encodeRecentEvents(m.recentEvents),
                    ),
                ).close()
            }
            for (chunk in plan.remap.keys.chunked(500)) {
                val list = chunk.joinToString(", ") { cypherString(it) }
                execute("MATCH (c:Concept) WHERE c.node_id IN [$list] DELETE c").close()
            }
            checkpoint()
        }
        invalidateCache()
    }

    // ---- internals ----

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
        val loaded = execute("MATCH (c:Concept) RETURN $CONCEPT_RETURN").rows().map(::rowToConcept)
        conceptCache = loaded
        loaded
    }

    private fun getConceptSync(nodeId: String): ConceptNode? =
        execute(
            "MATCH (c:Concept {node_id: \$nid}) RETURN $CONCEPT_RETURN",
            mapOf("nid" to nodeId),
        ).rows().firstOrNull()?.let(::rowToConcept)

    private fun conceptsByIds(nodeIds: List<String>): List<ConceptNode> =
        nodeIds.chunked(500).flatMap { chunk ->
            // Id list inlined (escaped): list-typed parameters are not verified on this API version.
            val idList = chunk.joinToString(", ") { cypherString(it) }
            execute("MATCH (c:Concept) WHERE c.node_id IN [$idList] RETURN $CONCEPT_RETURN")
                .rows().map(::rowToConcept)
        }

    /** Outgoing edges of one node, deterministically ordered for the shared kernels. */
    private fun outEdgesSync(nodeId: String): List<HebbianEdge> =
        execute(
            "MATCH (a:Concept {node_id: \$nid})-[r:ASSOCIATED_WITH]->(b:Concept) " +
                "RETURN a.node_id, b.node_id, r.hebb_weight, r.eligibility, r.causal_score, " +
                "r.co_activation_count, r.last_updated, r.layer ORDER BY b.node_id, r.layer",
            mapOf("nid" to nodeId),
        ).rows().map(::rowToEdge)

    private fun rowToConcept(row: List<Any?>) = ConceptNode(
        nodeId = row[0] as String,
        label = row[1] as? String ?: "",
        textRaw = row[2] as? String ?: "",
        textSummary = row[3] as? String ?: "",
        conceptType = row[4] as? String ?: "fact",
        sourceType = row[5] as? String ?: "document",
        sourceUri = (row[6] as? String)?.takeIf { it.isNotEmpty() }, // "" stores null
        createdAt = (row[7] as? Number)?.toLong() ?: 0L,
        updatedAt = (row[8] as? Number)?.toLong() ?: 0L,
        activationCount = (row[9] as? Number)?.toInt() ?: 0,
        embedding = toFloatArray(row[10]),
        meanMT = (row[11] as? Number)?.toDouble() ?: 1.0,
        meanDominance = (row[12] as? Number)?.toDouble() ?: 0.5,
        meanRT = (row[13] as? Number)?.toDouble() ?: 0.5,
        recentEvents = decodeRecentEvents(row[14] as? String),
    )

    private fun rowToEdge(row: List<Any?>) = HebbianEdge(
        srcId = row[0] as String,
        dstId = row[1] as String,
        hebbWeight = (row[2] as? Number)?.toDouble() ?: 1.0,
        eligibility = (row[3] as? Number)?.toDouble() ?: 0.0,
        causalScore = (row[4] as? Number)?.toDouble() ?: 0.0,
        coActivationCount = (row[5] as? Number)?.toInt() ?: 0,
        lastUpdated = (row[6] as? Number)?.toLong() ?: 0L,
        layer = row[7] as? String ?: "hippocampal",
    )

    private fun toFloatArray(v: Any?): FloatArray = when (v) {
        null -> FloatArray(0)
        is FloatArray -> v
        is DoubleArray -> FloatArray(v.size) { v[it].toFloat() }
        is List<*> -> FloatArray(v.size) { (v[it] as? Number)?.toFloat() ?: 0f }
        is Array<*> -> FloatArray(v.size) { (v[it] as? Number)?.toFloat() ?: 0f }
        is String -> v.split(',').mapNotNull { it.trim().toFloatOrNull() }.toFloatArray()
        else -> FloatArray(0)
    }

    /** Fixed-point CSV string literal — the LadybugDB Java API (0.20.3) cannot decode
     *  DOUBLE[] ARRAY result values ("Type of value is not supported in value_get_value"),
     *  so embeddings are stored as text. Injection-safe (numbers and commas only), no
     *  exponent notation. */
    private fun doubleArrayLiteral(v: FloatArray): String =
        cypherString(v.joinToString(",") { String.format(Locale.US, "%.9f", it.toDouble()) })

    /** Single-quoted, escaped Cypher string literal for inlined id lists. */
    private fun cypherString(s: String): String =
        "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"

    private fun ensureSchema() {
        ensureTable(
            "CREATE NODE TABLE IF NOT EXISTS Concept(node_id STRING, label STRING, text_raw STRING, " +
                "text_summary STRING, concept_type STRING, source_type STRING, source_uri STRING, " +
                "created_at INT64, updated_at INT64, activation_count INT64, embedding STRING, " +
                "mean_m_t DOUBLE, mean_dominance DOUBLE, mean_r_t DOUBLE, recent_events STRING, " +
                "PRIMARY KEY(node_id))",
            "CREATE NODE TABLE Concept(node_id STRING, label STRING, text_raw STRING, " +
                "text_summary STRING, concept_type STRING, source_type STRING, source_uri STRING, " +
                "created_at INT64, updated_at INT64, activation_count INT64, embedding STRING, " +
                "mean_m_t DOUBLE, mean_dominance DOUBLE, mean_r_t DOUBLE, recent_events STRING, " +
                "PRIMARY KEY(node_id))",
        )
        ensureTable(
            "CREATE NODE TABLE IF NOT EXISTS Session(session_id STRING, trigger STRING, started_at INT64, " +
                "ended_at INT64, total_activations INT64, PRIMARY KEY(session_id))",
            "CREATE NODE TABLE Session(session_id STRING, trigger STRING, started_at INT64, " +
                "ended_at INT64, total_activations INT64, PRIMARY KEY(session_id))",
        )
        ensureTable(
            "CREATE NODE TABLE IF NOT EXISTS SessionStats(stat_id STRING, session_id STRING, node_id STRING, " +
                "mean_m_t DOUBLE, mean_dominance DOUBLE, mean_r_t DOUBLE, activation_count INT64, " +
                "delta_w_mean DOUBLE, PRIMARY KEY(stat_id))",
            "CREATE NODE TABLE SessionStats(stat_id STRING, session_id STRING, node_id STRING, " +
                "mean_m_t DOUBLE, mean_dominance DOUBLE, mean_r_t DOUBLE, activation_count INT64, " +
                "delta_w_mean DOUBLE, PRIMARY KEY(stat_id))",
        )
        ensureTable(
            "CREATE REL TABLE IF NOT EXISTS ASSOCIATED_WITH(FROM Concept TO Concept, hebb_weight DOUBLE, " +
                "eligibility DOUBLE, causal_score DOUBLE, co_activation_count INT64, last_updated INT64, " +
                "layer STRING)",
            "CREATE REL TABLE ASSOCIATED_WITH(FROM Concept TO Concept, hebb_weight DOUBLE, " +
                "eligibility DOUBLE, causal_score DOUBLE, co_activation_count INT64, last_updated INT64, " +
                "layer STRING)",
        )
        ensureTable(
            "CREATE REL TABLE IF NOT EXISTS AGGREGATES(FROM Session TO SessionStats)",
            "CREATE REL TABLE AGGREGATES(FROM Session TO SessionStats)",
        )
        ensureTable(
            "CREATE REL TABLE IF NOT EXISTS TRACKED_IN(FROM SessionStats TO Concept)",
            "CREATE REL TABLE TRACKED_IN(FROM SessionStats TO Concept)",
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
        val conn = connection ?: throw IllegalStateException("LadybugHebbianStore is not open")
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

    /** Folds the WAL into the database file (see the class doc). Caller holds [connLock]. */
    private fun checkpoint() = execute("CHECKPOINT").close()

    private fun isAlreadyExists(message: String): Boolean =
        (message.contains("already") && message.contains("exist", ignoreCase = true)) ||
            message.contains("duplicate", ignoreCase = true)

    private fun execute(cypher: String, params: Map<String, Any?> = emptyMap()): QueryResult =
        synchronized(connLock) {
            val conn = connection ?: throw IllegalStateException("LadybugHebbianStore is not open")
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

    private companion object {
        const val CONCEPT_RETURN =
            "c.node_id, c.label, c.text_raw, c.text_summary, c.concept_type, c.source_type, c.source_uri, " +
                "c.created_at, c.updated_at, c.activation_count, c.embedding, " +
                "c.mean_m_t, c.mean_dominance, c.mean_r_t, c.recent_events"
    }
}
