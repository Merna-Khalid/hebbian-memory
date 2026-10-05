package com.mobilerag.core

import kotlinx.coroutines.flow.Flow

/** A text generation backend (LLM). Implementations: llama.cpp, ML Kit Gemini Nano, GenieX/QNN. */
interface GenerationBackend {
    val id: String

    sealed interface Status {
        data object Available : Status
        data class Unavailable(val reason: String) : Status
        data object Unknown : Status
    }

    suspend fun checkAvailability(): Status

    /** Streams generated tokens. Implementations must be cancellable via coroutine cancellation. */
    fun generate(prompt: String, maxTokens: Int = 256): Flow<String>
}
