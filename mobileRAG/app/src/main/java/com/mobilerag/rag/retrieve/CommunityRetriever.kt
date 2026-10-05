package com.mobilerag.rag.retrieve

import com.mobilerag.rag.RagDatabase
import java.io.File
import kotlin.math.sqrt

/**
 * Global channel (Phase 4): cosine over persisted community summary embeddings. Lives outside
 * RetrievalEngine (used by RagPipeline directly) so the eval harness's vector/hybrid behavior
 * is untouched. When [scoreLogFile] is set, every query's raw scores are appended there
 * (one line per query) for threshold tuning — logcat is unreliable on some devices.
 */
class CommunityRetriever(private val db: RagDatabase, private val scoreLogFile: File? = null) {

    data class CommunityHit(val communityId: Long, val summary: String, val score: Float)

    /**
     * Topical queries produce one clear winner far above the rest (e.g. travel → C1 0.31, next 0.16);
     * abstract "themes of my notes" queries cluster everything just under 0.30 with no winner.
     * So: require an absolute floor for the top hit, then take everything within [GAP] of the
     * top score (capped) — one community for topical queries, several for thematic ones.
     */
    suspend fun retrieve(queryEmbedding: FloatArray): List<CommunityHit> {
        val scored = db.allCommunities()
            .map { CommunityHit(it.id, it.summary, cosine(queryEmbedding, it.summaryEmbedding)) }
            .sortedByDescending { it.score }
        scoreLogFile?.let { f ->
            runCatching {
                f.parentFile?.mkdirs()
                f.appendText(scored.joinToString(" ") { "C${it.communityId}=%.3f".format(it.score) } + "\n")
            }
        }
        val top = scored.firstOrNull()?.takeIf { it.score >= MIN_SCORE } ?: return emptyList()
        return scored.filter { top.score - it.score <= GAP }.take(MAX_RESULTS)
    }

    private fun cosine(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size) return 0f
        var dot = 0f
        var na = 0f
        var nb = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        return if (na > 0f && nb > 0f) dot / (sqrt(na) * sqrt(nb)) else 0f
    }

    companion object {
        const val MIN_SCORE = 0.25f
        private const val GAP = 0.05f
        private const val MAX_RESULTS = 5
    }
}
