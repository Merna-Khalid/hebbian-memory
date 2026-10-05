package com.mobilerag.rag.retrieve

import com.mobilerag.core.GraphStore
import com.mobilerag.rag.RagDatabase

/**
 * Graph channel: anchor the question to entities via n-gram name search, then expand.
 * Tier 1 = chunks that mention an anchor entity; tier 2 = chunks of traversed neighbour
 * entities. Hits are ranked tier-1 first, then by entity rank within the tier.
 */
class GraphRetriever(private val store: GraphStore, private val db: RagDatabase) : Retriever {
    override val channel = "graph"

    /** Number of anchor entities matched by the last [retrieve] call (gate input). */
    @Volatile
    var lastAnchorCount: Int = 0
        private set

    override suspend fun retrieve(question: String, queryEmbedding: FloatArray?): List<RetrievalHit> {
        lastAnchorCount = 0
        val ngrams = QueryTerms.ngrams(question)
        if (ngrams.isEmpty()) return emptyList()

        val anchorsById = LinkedHashMap<Long, GraphStore.Entity>()
        for (ngram in ngrams) {
            for (entity in store.findEntities(ngram, limit = 3)) {
                anchorsById.putIfAbsent(entity.id, entity)
            }
        }
        val anchors = anchorsById.values.sortedByDescending { it.mentionCount }.take(MAX_ANCHORS)
        lastAnchorCount = anchors.size
        if (anchors.isEmpty()) return emptyList()

        val depth = if (anchors.size >= 2) 2 else 1
        val rankedIds = LinkedHashMap<Long, Int>() // chunkId -> best rank, tier 1 before tier 2
        var rank = 0
        for (anchor in anchors) {
            for (chunkId in store.chunksForEntity(anchor.id)) {
                rankedIds.putIfAbsent(chunkId, ++rank)
            }
        }
        val neighbors = anchors.flatMap { store.traverse(it.id, depth) }
            .distinctBy { it.id }
            .filter { it.id !in anchorsById.keys }
            .sortedByDescending { it.mentionCount }
            .take(MAX_NEIGHBORS)
        for (neighbor in neighbors) {
            for (chunkId in store.chunksForEntity(neighbor.id).take(MAX_CHUNKS_PER_NEIGHBOR)) {
                rankedIds.putIfAbsent(chunkId, ++rank)
            }
        }
        if (rankedIds.isEmpty()) return emptyList()

        val ids = rankedIds.entries.sortedBy { it.value }.take(TOP_K).map { it.key }
        val texts = db.chunkTextsById(ids)
        val docs = db.chunkDocInfo(ids)
        return ids.mapIndexed { i, id ->
            val doc = docs[id] ?: RagDatabase.ChunkDoc(0, "?")
            RetrievalHit(id, doc.docId, doc.docName, texts[id] ?: "", channel, i + 1)
        }
    }

    companion object {
        private const val MAX_ANCHORS = 5
        private const val MAX_NEIGHBORS = 20
        private const val MAX_CHUNKS_PER_NEIGHBOR = 3
        private const val TOP_K = 10
    }
}
