package com.mobilerag.core

/**
 * A property-graph store. Implementations: LadybugDB (Cypher), SQLite (recursive CTEs).
 *
 * v2 (Phase 2 — graph memory): entities carry a normalized name and a mention count,
 * edges carry a weight, and entity↔chunk mentions are persisted for provenance
 * (which rag.db chunk introduced which entity), so query-time entity anchoring and
 * cascading removal of source chunks are possible. All v1 methods are unchanged.
 */
interface GraphStore {
    val id: String

    data class Entity(val id: Long, val name: String, val type: String, val mentionCount: Int = 0)
    data class Edge(
        val fromId: Long,
        val toId: Long,
        val relation: String,
        val sourceChunkId: Long? = null,
        val weight: Double = 1.0,
    )

    suspend fun open(path: String)
    suspend fun close()

    /** Raw insert with no dedup — kept for spike compatibility; prefer [upsertEntity]. */
    suspend fun addEntity(name: String, type: String): Long
    suspend fun addEdge(fromId: Long, toId: Long, relation: String, sourceChunkId: Long? = null)

    /** Multi-hop traversal from a starting entity, up to [maxDepth] hops. */
    suspend fun traverse(startEntityId: Long, maxDepth: Int = 2): List<Entity>

    suspend fun entityCount(): Int
    suspend fun edgeCount(): Int

    // ---- v2 (Phase 2) ----

    /**
     * Returns the existing entity id when (normalizedName, type) is already present,
     * else creates the entity. Mention counting happens via [addMention], not here.
     */
    suspend fun upsertEntity(name: String, type: String, normalizedName: String): Long

    /**
     * Name search: exact/prefix match plus fuzzy (Jaro-Winkler) matching, ranked by
     * score and capped at [limit]. Used by UI search and query-time entity anchoring.
     */
    suspend fun findEntities(query: String, limit: Int = 20): List<Entity>

    suspend fun getEntity(id: Long): Entity?

    /**
     * Records that [entityId] is mentioned in rag.db chunk [chunkId] and increments
     * mention_count. Duplicate (entity, chunk) pairs are ignored.
     */
    suspend fun addMention(entityId: Long, chunkId: Long)

    suspend fun chunksForEntity(entityId: Long): List<Long>

    /** Edges touching [entityId] in either direction. */
    suspend fun edgesFor(entityId: Long): List<Edge>

    /** All edges in the graph — bulk read for offline analytics (community detection). */
    suspend fun allEdges(): List<Edge>

    /** All entities, ordered by mention_count DESC then name ASC. */
    suspend fun allEntities(limit: Int = 500, offset: Int = 0): List<Entity>

    /**
     * Deletes mentions and edges whose source chunk is in [chunkIds], decrements
     * mention_count accordingly, and deletes entities left with 0 mentions and no edges.
     */
    suspend fun removeGraphForChunks(chunkIds: List<Long>)

    /** Wipes all graph data. */
    suspend fun clear()
}
