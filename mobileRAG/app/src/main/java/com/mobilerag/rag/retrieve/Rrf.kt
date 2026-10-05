package com.mobilerag.rag.retrieve

/**
 * Reciprocal Rank Fusion: score = Σ 1.0/(k + rank) over every channel a chunk appears in.
 * Ties break by best (lowest) channel rank, then chunk id.
 */
object Rrf {

    const val DEFAULT_K = 60

    fun fuse(channels: List<List<RetrievalHit>>, k: Int = DEFAULT_K, limit: Int = 10): List<RetrievalHit> {
        data class Acc(var score: Double, var bestRank: Int, var hit: RetrievalHit)

        val byChunk = mutableMapOf<Long, Acc>()
        for (channel in channels) {
            for (hit in channel) {
                val acc = byChunk.getOrPut(hit.chunkId) { Acc(0.0, Int.MAX_VALUE, hit) }
                acc.score += 1.0 / (k + hit.channelRank)
                if (hit.channelRank < acc.bestRank) acc.bestRank = hit.channelRank
                // Prefer the instance carrying a raw score (vec cosine) for display/gating
                if (acc.hit.score == null && hit.score != null) acc.hit = hit
            }
        }
        return byChunk.values
            .sortedWith(compareByDescending<Acc> { it.score }.thenBy { it.bestRank }.thenBy { it.hit.chunkId })
            .take(limit)
            .map { it.hit }
    }
}
