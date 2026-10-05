package com.mobilerag.generation

import android.content.Context
import com.arm.aichat.AiChat
import com.mobilerag.core.GenerationBackend
import kotlinx.coroutines.flow.Flow
import java.io.File

/** llama.cpp GGUF backend — universal device coverage. Model must be pushed to external files dir. */
class LlamaCppBackend(private val context: Context) : GenerationBackend {
    override val id = "llamacpp-gguf"

    private val modelFile: File?
        get() = File(context.getExternalFilesDir(null), "models")
            .listFiles()?.firstOrNull { it.extension == "gguf" }

    override suspend fun checkAvailability(): GenerationBackend.Status {
        val model = modelFile
            ?: return GenerationBackend.Status.Unavailable("No .gguf model pushed to device (see LlamaCppSpike log)")
        return try {
            val engine = AiChat.getInferenceEngine(context)
            engine.loadModel(model.absolutePath)
            engine.cleanUp()
            GenerationBackend.Status.Available
        } catch (t: Throwable) {
            GenerationBackend.Status.Unavailable("${t.javaClass.simpleName}: ${t.message?.take(150)}")
        }
    }

    override fun generate(prompt: String, maxTokens: Int): Flow<String> {
        val engine = AiChat.getInferenceEngine(context)
        return engine.sendUserPrompt(prompt, maxTokens)
    }
}
