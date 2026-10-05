package com.mobilerag.rag.retrieve

import com.mobilerag.rag.RagDatabase

/**
 * Keyword channel: FTS4 search per question term. Chunks are ranked by distinct terms matched
 * (desc), then total term occurrences in the chunk text (desc), then chunk id (asc).
 */
class FtsRetriever(private val db: RagDatabase) : Retriever {
    override val channel = "fts"

    /** Distinct-term coverage of the best chunk from the last [retrieve] call (gate input). */
    @Volatile
    var lastBestDistinctTerms: Int = 0
        private set

    override suspend fun retrieve(question: String, queryEmbedding: FloatArray?): List<RetrievalHit> {
        lastBestDistinctTerms = 0
        val terms = QueryTerms.terms(question, cap = MAX_TERMS)
        if (terms.isEmpty()) return emptyList()

        val matchedBy = mutableMapOf<Long, MutableSet<String>>() // chunkId -> terms matched
        for (term in terms) {
            for (chunkId in db.searchChunkFts(term)) {
                matchedBy.getOrPut(chunkId) { mutableSetOf() } += term
            }
        }
        if (matchedBy.isEmpty()) return emptyList()
        lastBestDistinctTerms = matchedBy.values.maxOf { it.size }

        val ids = matchedBy.keys.toList()
        val texts = db.chunkTextsById(ids)
        val docs = db.chunkDocInfo(ids)

        val ranked = ids.map { id ->
            val text = texts[id] ?: ""
            val occurrences = matchedBy[id]!!.sumOf { term -> countOccurrences(text.lowercase(), term) }
            Triple(id, matchedBy[id]!!.size, occurrences)
        }.sortedWith(
            compareByDescending<Triple<Long, Int, Int>> { it.second }
                .thenByDescending { it.third }
                .thenBy { it.first }
        ).take(TOP_K)

        return ranked.mapIndexed { i, (id, _, _) ->
            val doc = docs[id] ?: RagDatabase.ChunkDoc(0, "?")
            RetrievalHit(id, doc.docId, doc.docName, texts[id] ?: "", channel, i + 1)
        }
    }

    private fun countOccurrences(text: String, term: String): Int {
        var count = 0
        var idx = text.indexOf(term)
        while (idx >= 0) {
            count++
            idx = text.indexOf(term, idx + term.length)
        }
        return count
    }

    companion object {
        private const val MAX_TERMS = 6
        private const val TOP_K = 10
    }
}
