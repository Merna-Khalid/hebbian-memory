package com.mobilerag.spikes

import android.content.Context
import com.mobilerag.core.GenerationBackend
import com.mobilerag.generation.MlKitGenerationBackend

/**
 * Step 6: probe Gemini Nano (ML Kit GenAI) availability.
 * Expectation per the plan doc: UNAVAILABLE on RedMagic/nubia — AICore device list doesn't cover them.
 */
class MlKitProbeSpike : Spike {
    override val id = "mlkit_probe"
    override val title = "Gemini Nano probe"
    override val description = "Checks AICore/Gemini Nano availability via ML Kit GenAI checkStatus()."

    override suspend fun run(context: Context, log: (String) -> Unit): SpikeResult {
        val backend = MlKitGenerationBackend()
        return when (val status = backend.checkAvailability()) {
            is GenerationBackend.Status.Available -> {
                log("Gemini Nano is AVAILABLE on this device")
                SpikeResult(true, "AICore present — ML Kit backend usable", mapOf("status" to "AVAILABLE"))
            }
            is GenerationBackend.Status.Unavailable -> {
                log("Not available: ${status.reason}")
                // Expected outcome on RedMagic — probe succeeded even though the backend is absent
                SpikeResult(true, "Probe OK: ${status.reason}", mapOf("status" to "UNAVAILABLE"))
            }
            else -> SpikeResult(false, "Probe returned unknown state")
        }
    }
}
