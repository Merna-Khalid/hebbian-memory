package com.mobilerag.rag

import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * In-memory brute-force cosine index over all chunk embeddings in RagDatabase.
 * Loaded on first query; reloaded when the DB index version changes.
 */
class VectorStore(private val db: RagDatabase, private val dimensions: Int) {

    data class ScoredChunk(val chunkId: Long, val docId: Long, val docName: String, val text: String, val score: Float)

    private class Index(
        val version: Long,
        val matrix: FloatArray, // row-major, size = n * dimensions; embeddings are L2-normalized
        val chunkIds: LongArray,
        val docIds: LongArray,
        val docNames: List<String>,
        val texts: List<String>,
    ) {
        val size: Int get() = chunkIds.size
    }

    private val mutex = Mutex()
    private var cached: Index? = null

    suspend fun topK(query: FloatArray, k: Int = 5): List<ScoredChunk> {
        val index = index()
        if (index.size == 0) return emptyList()
        val start = SystemClock.elapsedRealtime()
        val scores = FloatArray(index.size)
        for (row in 0 until index.size) {
            var dot = 0f
            val offset = row * dimensions
            for (d in 0 until dimensions) dot += index.matrix[offset + d] * query[d]
            scores[row] = dot
        }
        val top = scores.indices.sortedByDescending { scores[it] }.take(k.coerceAtMost(index.size))
        Log.i(TAG, "scan ${index.size} chunks in ${SystemClock.elapsedRealtime() - start} ms")
        return top.map { i -> ScoredChunk(index.chunkIds[i], index.docIds[i], index.docNames[i], index.texts[i], scores[i]) }
    }

    private suspend fun index(): Index = mutex.withLock {
        val version = db.indexVersion()
        cached?.takeIf { it.version == version }?.let { return@withLock it }
        val docNames = db.listDocuments().associate { it.id to it.name }
        val chunks = db.allChunksWithEmbeddings()
        val matrix = FloatArray(chunks.size * dimensions)
        val chunkIds = LongArray(chunks.size)
        val docIds = LongArray(chunks.size)
        val names = ArrayList<String>(chunks.size)
        val texts = ArrayList<String>(chunks.size)
        chunks.forEachIndexed { i, c ->
            c.embedding.copyInto(matrix, destinationOffset = i * dimensions)
            chunkIds[i] = c.id
            docIds[i] = c.docId
            names += docNames[c.docId] ?: "?"
            texts += c.text
        }
        Log.i(TAG, "loaded ${chunks.size} chunk embeddings (index v$version)")
        Index(version, matrix, chunkIds, docIds, names, texts).also { cached = it }
    }

    companion object {
        private const val TAG = "VectorStore"
    }
}
