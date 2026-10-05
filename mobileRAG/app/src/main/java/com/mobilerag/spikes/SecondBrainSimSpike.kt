package com.mobilerag.spikes

import android.content.Context
import com.mobilerag.hebbian.ConceptExtractor
import com.mobilerag.hebbian.ConsolidationScheduler
import com.mobilerag.hebbian.DOMAIN_PRESETS
import com.mobilerag.hebbian.IngestPipeline
import com.mobilerag.hebbian.PracticeScheduler
import com.mobilerag.hebbian.TutorEngine
import com.mobilerag.hebbian.emotion.PhysioFusedEmotionClassifier
import com.mobilerag.hebbian.emotion.StubEmotionClassifier
import com.mobilerag.hebbian.signals.SimulatedSignalSource
import com.mobilerag.hebbian.store.HebbianStore
import com.mobilerag.hebbian.store.HebbianStoreFactory
import com.mobilerag.hebbian.stubEmbedding
import com.mobilerag.hebbian.stubGenerate
import com.mobilerag.hebbian.stubSummary
import com.mobilerag.profile.SpaceManager
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import java.io.File

/**
 * Second-brain simulated-BCI spike — on-device mirror of `second_brain_sim.py`
 * (HEBBIAN_STUB=1). Runs all 8 scripted [SimulatedSignalSource] scenarios
 * (seed = 7) through `engine.ingestAmbient` on the second_brain preset and
 * prints the same per-ingest table as the Python CLI.
 *
 * Engine wiring replicates [com.mobilerag.hebbian.buildStubEngine] but swaps in
 * [PhysioFusedEmotionClassifier] so heart rate drives arousal, exactly as
 * Python's build_stub_engine(domain="second_brain", signal_source=sim) does —
 * buildStubEngine itself takes no signal source (API gap). The store is the real
 * [HebbianStoreFactory] one (LadybugDB first, SQLite fallback), CLEARED first.
 *
 * NOTE: RNG noise differs from Python — kotlin.random Box–Muller gaussians vs
 * Python's random.gauss — so the quadrant/gating pattern matches the Python sim
 * qualitatively, not bit-exactly.
 */
class SecondBrainSimSpike : Spike {
    override val id = "second_brain_sim"
    override val title = "Second brain (simulated BCI)"
    override val description = "All 8 SimulatedSignalSource scenarios (seed 7) through ingestAmbient on the second_brain preset, HR-fused arousal; prints the second_brain_sim.py table."

    override suspend fun run(context: Context, log: (String) -> Unit): SpikeResult {
        return try {
            val store = HebbianStoreFactory.create(context, SpaceManager.activeDir(context))
            log("Hebbian store implementation: ${store.id}")
            log("WARNING: clearing the shared Hebbian store for a deterministic run")
            store.clear()

            val stateDir = File(context.filesDir, "second_brain_sim_spike")
            stateDir.deleteRecursively()

            val sim = SimulatedSignalSource(seed = 7)
            val engine = buildSecondBrainStubEngine(store, stateDir, sim)
            val sessionId = engine.startSession("simulated_bci")

            log("")
            log("=".repeat(88))
            log("second brain — simulated ambient ingest (HR-fused arousal, stub models)")
            log("RNG differs from Python (Box–Muller vs gauss): quadrant patterns are")
            log("qualitatively comparable, not bit-exact.")
            log("=".repeat(88))
            log(
                "%-24s %-4s %-7s %-7s %-6s %-6s %-9s %s".format(
                    "scenario", "Q", "m_t", "gated", "attn", "hr", "Δw", "concepts",
                ),
            )
            log("-".repeat(88))

            var ingested = 0
            var gatedCount = 0
            var armeMissing = 0
            var invariantViolations = 0
            var emptyConcepts = 0
            var attentionOutOfRange = 0

            while (true) {
                val utt = sim.nextUtterance() ?: break
                val physio = sim.currentPhysio()
                val result = engine.ingestAmbient(utt, physio, sessionId)
                ingested++

                if (physio.attention == null || physio.attention !in 0.0..1.0) attentionOutOfRange++
                if (result.concepts.isEmpty()) emptyConcepts++

                val arme = result.arme
                if (arme == null) {
                    armeMissing++
                    log("%-24s (no concepts extracted)".format(sim.currentLabel))
                    continue
                }
                // ARM-E contract: a gated step must report neutral m_t = 1.0.
                if (arme.gated && arme.mT != 1.0) invariantViolations++
                if (arme.gated) gatedCount++

                log(
                    "%-24s %-4s %-7.3f %-7s %-6.2f %-6.1f %+9.4f %s".format(
                        sim.currentLabel,
                        arme.quadrant,
                        arme.mT,
                        if (arme.gated) "YES" else "no",
                        result.attention,
                        physio.hrBpm ?: Double.NaN,
                        arme.deltaWMean,
                        result.concepts.joinToString { it.label },
                    ),
                )
            }
            engine.endSession(sessionId)
            // NB: engine.close() intentionally NOT called — it would close the
            // shared HebbianStoreFactory store other spikes/app screens may hold.

            val nScenarios = SimulatedSignalSource.DEFAULT_SCENARIOS.size
            val checks = linkedMapOf(
                "all $nScenarios scenarios ingested" to (ingested == nScenarios),
                "ARM-E state on every ingest" to (armeMissing == 0),
                "concepts extracted on every ingest" to (emptyConcepts == 0),
                "gated ⇒ m_t == 1.0 (ARM-E contract)" to (invariantViolations == 0),
                "attention in [0, 1]" to (attentionOutOfRange == 0),
            )
            log("-".repeat(88))
            log("gated: $gatedCount/$ingested ingests (low attention/confidence → neutral m_t)")
            for ((name, ok) in checks) log("%-4s %s".format(if (ok) "PASS" else "FAIL", name))

            val failedChecks = checks.filterValues { !it }.keys
            SpikeResult(
                passed = failedChecks.isEmpty(),
                summary = if (failedChecks.isEmpty())
                    "8 ambient ingests on ${store.id}, $gatedCount gated — ARM-E contract holds"
                else "FAILED: ${failedChecks.joinToString()}",
                metrics = mapOf(
                    "store" to store.id,
                    "ingests" to "$ingested/$nScenarios",
                    "gated" to "$gatedCount",
                    "checks" to "${checks.size - failedChecks.size}/${checks.size}",
                ),
            )
        } catch (t: Throwable) {
            SpikeResult(
                passed = false,
                summary = "Failed: ${t.javaClass.simpleName}: ${t.message?.take(200)}",
                error = t.stackTraceToString().take(2000),
            )
        }
    }

    /**
     * [com.mobilerag.hebbian.buildStubEngine] with the second_brain preset and a
     * [PhysioFusedEmotionClassifier] wrapping the keyword stub, mirroring Python's
     * build_stub_engine(domain="second_brain", signal_source=sim). Duplicated here
     * because buildStubEngine accepts no signal source / emotion classifier.
     */
    private fun buildSecondBrainStubEngine(
        store: HebbianStore,
        stateDir: File,
        sim: SimulatedSignalSource,
    ): TutorEngine {
        val preset = DOMAIN_PRESETS.getValue("second_brain")
        val consolidation = ConsolidationScheduler(store)
        val pipeline = IngestPipeline(
            store = store,
            embedFn = { text -> stubEmbedding(text) },
            summaryFn = { text -> stubSummary(text) },
            emotionClassifier = PhysioFusedEmotionClassifier(StubEmotionClassifier(), sim),
            consolidation = consolidation,
        )
        val extractor = ConceptExtractor(
            pipeline = pipeline,
            generate = ::stubGenerate,
            store = store,
            extractionSystem = preset.extractionSystem,
            conceptTypes = preset.conceptTypes,
            relationTypes = preset.relationTypes,
        )
        val practiceScheduler = PracticeScheduler(File(stateDir, "practice_state"))

        val stubSend: (String, Int) -> Flow<String> = { p, m ->
            flow {
                val full = stubGenerate(p, m)
                var i = 0
                while (i < full.length) {
                    val end = minOf(i + 24, full.length)
                    emit(full.substring(i, end))
                    i = end
                }
            }
        }

        return TutorEngine(
            store = store,
            pipeline = pipeline,
            extractor = extractor,
            practiceScheduler = practiceScheduler,
            embedQuery = { text -> stubEmbedding(text) },
            preset = preset,
            domain = "second_brain",
            llm = null,
            sendTokens = stubSend,
            completeFn = ::stubGenerate,
        )
    }
}
