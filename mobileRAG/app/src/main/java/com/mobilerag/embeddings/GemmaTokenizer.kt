package com.mobilerag.embeddings

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import java.nio.file.Path

/** EmbeddingGemma SentencePiece tokenizer via DJL HuggingFace fast tokenizers (tokenizer.json). */
class GemmaTokenizer(tokenizerJson: Path) : AutoCloseable {

    data class Encoding(val inputIds: LongArray, val attentionMask: LongArray)

    private val tokenizer = HuggingFaceTokenizer.newInstance(tokenizerJson)

    fun encode(text: String, maxTokens: Int = MAX_TOKENS): Encoding {
        val enc = tokenizer.encode(text)
        val ids = enc.ids
        val mask = enc.attentionMask
        return if (ids.size <= maxTokens) {
            Encoding(ids, mask)
        } else {
            Encoding(ids.copyOf(maxTokens), mask.copyOf(maxTokens))
        }
    }

    fun countTokens(text: String): Int = tokenizer.encode(text).ids.size

    override fun close() = tokenizer.close()

    companion object {
        /** EmbeddingGemma context is 2048; chunks target 300–500 tokens, so 1024 leaves headroom. */
        const val MAX_TOKENS = 1024
    }
}
