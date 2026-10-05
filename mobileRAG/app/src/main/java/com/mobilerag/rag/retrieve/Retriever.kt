package com.mobilerag.rag.retrieve

/**
 * One retrieval channel's view of a chunk. [channel] names the source ("vec", "fts", "graph")
 * and [channelRank] is the 1-based position within that channel's own ranking. [score] carries
 * the raw vector cosine when the hit came from the vec channel (used by the similarity gate and
 * citation display).
 */
data class RetrievalHit(
    val chunkId: Long,
    val docId: Long,
    val docName: String,
    val text: String,
    val channel: String,
    val channelRank: Int,
    val score: Float? = null,
)

interface Retriever {
    val channel: String

    /** [queryEmbedding] is pre-computed once per question by the caller; null when unavailable. */
    suspend fun retrieve(question: String, queryEmbedding: FloatArray?): List<RetrievalHit>
}

/** Shared question tokenization: lowercase alphanumeric words, stopwords dropped. */
internal object QueryTerms {
    private val STOPWORDS = setOf(
        "a", "an", "the", "is", "are", "was", "were", "what", "when", "where", "who", "which",
        "how", "do", "does", "did", "i", "my", "me", "in", "on", "at", "to", "of", "for", "and",
        "or", "with", "by", "from",
    )

    /** Distinct content words of [question], lowercased, stopword-free, in order of appearance. */
    fun terms(question: String, cap: Int = Int.MAX_VALUE): List<String> =
        words(question).distinct().take(cap)

    /** All consecutive n-grams (n = 1..3) over the stopword-filtered words of [question]. */
    fun ngrams(question: String): List<String> {
        val words = words(question)
        val out = LinkedHashSet<String>()
        for (n in 1..3) {
            for (i in 0..words.size - n) {
                out += words.subList(i, i + n).joinToString(" ")
            }
        }
        return out.toList()
    }

    private fun words(question: String): List<String> =
        question.lowercase().split(Regex("[^a-z0-9]+"))
            .filter { it.isNotEmpty() && it !in STOPWORDS }
}
