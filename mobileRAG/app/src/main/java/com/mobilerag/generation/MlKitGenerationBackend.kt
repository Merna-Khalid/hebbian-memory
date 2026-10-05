package com.mobilerag.generation

import com.google.mlkit.genai.common.FeatureStatus
import com.google.mlkit.genai.prompt.Generation
import com.mobilerag.core.GenerationBackend
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/** Gemini Nano via ML Kit GenAI Prompt API. Expected UNAVAILABLE on RedMagic/nubia (no AICore). */
class MlKitGenerationBackend : GenerationBackend {
    override val id = "mlkit-gemini-nano"

    override suspend fun checkAvailability(): GenerationBackend.Status {
        return try {
            val status = Generation.getClient().checkStatus()
            when (status) {
                FeatureStatus.AVAILABLE -> GenerationBackend.Status.Available
                FeatureStatus.DOWNLOADABLE -> GenerationBackend.Status.Unavailable("Gemini Nano downloadable (not yet on device)")
                FeatureStatus.DOWNLOADING -> GenerationBackend.Status.Unavailable("Gemini Nano downloading")
                else -> GenerationBackend.Status.Unavailable("FeatureStatus.UNAVAILABLE (no AICore on this device)")
            }
        } catch (t: Throwable) {
            GenerationBackend.Status.Unavailable("${t.javaClass.simpleName}: ${t.message?.take(150)}")
        }
    }

    override fun generate(prompt: String, maxTokens: Int): Flow<String> = flow {
        val client = Generation.getClient()
        emit(client.generateContent(prompt).candidates.firstOrNull()?.text ?: "")
    }
}
