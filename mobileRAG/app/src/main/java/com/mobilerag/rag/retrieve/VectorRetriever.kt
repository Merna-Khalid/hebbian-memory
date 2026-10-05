package com.mobilerag.rag.retrieve

import com.mobilerag.rag.VectorStore

/** Vector channel: brute-force cosine top-k over chunk embeddings. */
class VectorRetriever(private val vectorStore: VectorStore) : Retriever {
    override val channel = "vec"

    override suspend fun retrieve(question: String, queryEmbedding: FloatArray?): List<RetrievalHit> {
        val embedding = queryEmbedding ?: return emptyList()
        return vectorStore.topK(embedding, TOP_K).mapIndexed { i, c ->
            RetrievalHit(c.chunkId, c.docId, c.docName, c.text, channel, i + 1, c.score)
        }
    }

    companion object {
        private const val TOP_K = 10
    }
}
