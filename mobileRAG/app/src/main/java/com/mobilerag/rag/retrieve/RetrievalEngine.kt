package com.mobilerag.rag.retrieve

import android.os.SystemClock
import com.mobilerag.core.GraphStore
import com.mobilerag.embeddings.EmbeddingGemmaEngine
import com.mobilerag.rag.RagDatabase
import com.mobilerag.rag.VectorStore
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import java.util.concurrent.ConcurrentHashMap

data class RetrievalResult(
    val hits: List<RetrievalHit>,
    val gated: Boolean,
    val channelCounts: Map<String, Int>,
    val channelMs: Map<String, Long>,
)

/**
 * Hybrid retrieval shared by RagPipeline and the eval spike: embeds the question once, runs the
 * vector channel always plus the FTS and graph channels concurrently in hybrid mode, fuses with
 * RRF, and applies the no-match gate.
 *
 * Gate: vector mode → gated when no hits or top-1 cosine < [MIN_VEC_SCORE]. Hybrid mode → gated
 * when the fused list is empty, or when the vector top-1 is weak AND no graph anchors matched
 * AND the best FTS chunk matched fewer than 2 distinct terms.
 */
class RetrievalEngine(
    engine: EmbeddingGemmaEngine,
    vectorStore: VectorStore,
    db: RagDatabase,
    store: GraphStore,
) {
    private val embed = engine
    private val vector = VectorRetriever(vectorStore)
    private val fts = FtsRetriever(db)
    private val graph = GraphRetriever(store, db)

    suspend fun retrieve(question: String, hybrid: Boolean): RetrievalResult {
        val embedStart = SystemClock.elapsedRealtime()
        val queryEmbedding = embed.embedQuery(question)
        val embedMs = SystemClock.elapsedRealtime() - embedStart

        val channels = LinkedHashMap<String, List<RetrievalHit>>()
        val channelMs = ConcurrentHashMap<String, Long>() // written from concurrent async channels
        channelMs["embed"] = embedMs

        suspend fun timed(retriever: Retriever): List<RetrievalHit> {
            val start = SystemClock.elapsedRealtime()
            val hits = retriever.retrieve(question, queryEmbedding)
            channelMs[retriever.channel] = SystemClock.elapsedRealtime() - start
            return hits
        }

        if (hybrid) {
            coroutineScope {
                val vecDeferred = async { timed(vector) }
                val ftsDeferred = async { timed(fts) }
                val graphDeferred = async { timed(graph) }
                channels["vec"] = vecDeferred.await()
                channels["fts"] = ftsDeferred.await()
                channels["graph"] = graphDeferred.await()
            }
        } else {
            channels["vec"] = timed(vector)
        }

        val fused = Rrf.fuse(channels.values.toList(), limit = FINAL_LIMIT)
        val vecTop1 = channels["vec"]?.firstOrNull()?.score
        val gated = if (hybrid) {
            fused.isEmpty() ||
                ((vecTop1 ?: 0f) < MIN_VEC_SCORE && graph.lastAnchorCount == 0 && fts.lastBestDistinctTerms < 2)
        } else {
            fused.isEmpty() || (vecTop1 ?: 0f) < MIN_VEC_SCORE
        }
        return RetrievalResult(
            hits = fused,
            gated = gated,
            channelCounts = channels.mapValues { it.value.size },
            channelMs = channelMs,
        )
    }

    companion object {
        const val MIN_VEC_SCORE = 0.30f
        private const val FINAL_LIMIT = 10
    }
}
