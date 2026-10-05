package com.mobilerag.core

/** A text embedding engine. Implementations: ONNX Runtime (all-MiniLM / EmbeddingGemma), LiteRT. */
interface EmbeddingEngine {
    val id: String
    val dimensions: Int

    /** Returns a normalized embedding vector for [text]. */
    suspend fun embed(text: String): FloatArray
}
