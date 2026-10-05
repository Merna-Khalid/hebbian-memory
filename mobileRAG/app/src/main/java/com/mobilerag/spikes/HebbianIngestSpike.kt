package com.mobilerag.spikes

import android.content.Context
import com.mobilerag.hebbian.ChatTurnResult
import com.mobilerag.hebbian.ConceptIdentity
import com.mobilerag.hebbian.PracticeScheduler
import com.mobilerag.hebbian.TutorEvent
import com.mobilerag.hebbian.buildStubEngine
import com.mobilerag.hebbian.store.HebbianStoreFactory
import com.mobilerag.profile.SpaceManager
import java.io.File
import kotlin.math.abs

/**
 * Hebbian end-to-end ingest spike — no models. Runs ~6 scripted Japanese-tutor
 * chat turns (食べる/飲む/行く vocabulary) through [buildStubEngine] (SHA-256 hash
 * embeddings, template JSON generator) against the real store from
 * [HebbianStoreFactory] (LadybugDB first, SQLite fallback — the store is CLEARED
 * first for determinism).
 *
 * Asserts: extracted concepts land in the store, at least one hippocampal edge
 * moves off its initial weight, the fading queue is non-empty, and a practice
 * round trip works (nextPractice → submitPracticeAnswer → graded correct → the
 * type's adaptive half-life shifts 3.0 → 3.75 days).
 *
 * Wall-clock-dependent values (m_t, staleness) vary run to run; the assertions
 * are structural, not numeric. Extraction runs every 3 turns; the stub extractor
 * emits one concept per batch, labelled with the batch's first >3-char content word
 * ("Today", "conjugate", then "Today" again). The third batch therefore re-mentions
 * the first concept, which concept identity must resolve to the same node.
 * (Before identity, every turn minted a "User:" copy and the copies wired to each
 * other — the spike's edges came from that duplication.)
 */
class HebbianIngestSpike : Spike {
    override val id = "hebbian_ingest"
    override val title = "Hebbian ingest loop (stub engine)"
    override val description = "9 tutor chat turns via buildStubEngine against the real (cleared) Hebbian store: concepts stored, a re-mention reuses its node, edge weights move, fading queue, practice round trip + half-life shift."

    override suspend fun run(context: Context, log: (String) -> Unit): SpikeResult {
        return try {
            val checks = linkedMapOf<String, Boolean>()
            val store = HebbianStoreFactory.create(context, SpaceManager.activeDir(context))
            log("Hebbian store implementation: ${store.id}")
            log("WARNING: clearing the shared Hebbian store for a deterministic run")
            store.clear()

            val stateDir = File(context.filesDir, "hebbian_ingest_spike")
            stateDir.deleteRecursively()

            val engine = buildStubEngine(store, stateDir)
            val sessionId = engine.startSession("spike")
            log("session: $sessionId")

            val turns = listOf(
                "Today I learned 食べる (taberu), to eat. ご飯を食べます。",
                "What is the te-form of 食べる? I want to say \"please eat\".",
                "Now 飲む (nomu), to drink — 水を飲みます。",
                "How do I conjugate 飲む in the past tense?",
                "行く (iku) means to go, right? 学校に行きます。",
                "Why is the te-form of 行く irregular — 行って?",
                "Today I reviewed 食べる again — 食べました。",
                "Is 食べて the te-form I asked about?",
                "Let me try: 毎日ご飯を食べます。",
            )

            val doneResults = mutableListOf<ChatTurnResult>()
            var chatFailed = false
            for ((i, msg) in turns.withIndex()) {
                var done: ChatTurnResult? = null
                var failed: String? = null
                engine.chat(msg, sessionId).collect { ev ->
                    when (ev) {
                        is TutorEvent.Done -> done = ev.result
                        is TutorEvent.Failed -> failed = ev.error
                        else -> Unit
                    }
                }
                val r = done
                if (r == null) {
                    chatFailed = true
                    log("turn ${i + 1}: FAILED — ${failed ?: "no Done event"}")
                    continue
                }
                doneResults += r
                val arme = r.arme
                log(
                    "turn ${i + 1}: concepts=[${r.concepts.joinToString { it.label }}] " +
                        if (arme != null)
                            "ARM-E ${arme.quadrant} m_t=%.4f gated=${arme.gated} ".format(arme.mT) +
                                "VAD=(%.2f, %.2f, %.2f) Δw=%+.6f".format(
                                    arme.valence, arme.arousal, arme.dominance, arme.deltaWMean,
                                )
                        else "ARM-E (none)",
                )
            }
            checks["${turns.size} chat turns completed"] = doneResults.size == turns.size && !chatFailed

            // ── 1. Extracted concepts appear in the store ───────────────
            val extractedLabels = doneResults.flatMap { r -> r.concepts.map { it.label } }
            val storedLabels = store.listConcepts(limit = 500).map { it.label }.toSet()
            log("stored concepts: ${storedLabels.size} — extracted this run: ${extractedLabels.toSet()}")
            checks["extracted concepts in store"] =
                extractedLabels.isNotEmpty() && storedLabels.containsAll(extractedLabels)

            // ── 1b. Concept identity: batch 3 re-mentions "Today" ──────
            val todays = store.listConcepts(limit = 500).filter { ConceptIdentity.labelKey(it.label) == "today" }
            log("'Today' nodes: ${todays.size}, activations: ${todays.map { it.activationCount }}")
            checks["re-mention reuses the concept (1 node, ≥2 activations)"] =
                todays.size == 1 && todays[0].activationCount >= 2

            // ── 2. A hippocampal edge moved off its initial weight ──────
            // Initial weights: 0.6 (unvalidated bootstrap), 1.0 (bootstrap),
            // 1.2 (co-occurrence), 1.5 (explicit relation) — see IngestPipeline /
            // ConceptExtractor / HebbianStore.reinforceCoRetrieved.
            val graph = engine.graphData(minWeight = 0.0, layer = "hippocampal", includeIsolated = true)
            val initialWs = listOf(0.6, 1.0, 1.2, 1.5)
            val moved = graph.edges.filter { e -> initialWs.all { abs(e.hebbWeight - it) > 1e-6 } }
            log("hippocampal edges: ${graph.edges.size}, moved off initial: ${moved.size}")
            moved.take(5).forEach { e ->
                log("  moved edge ${e.source.take(8)}→${e.target.take(8)} w=%.6f elig=%.4f".format(e.hebbWeight, e.eligibility))
            }
            checks["edge weight moved off initial"] = graph.edges.isNotEmpty() && moved.isNotEmpty()

            // ── 3. Fading queue non-empty ───────────────────────────────
            val fading = engine.fadingConcepts(limit = 5)
            log("fading queue (${fading.size}):")
            fading.forEach { f ->
                log("  [${f.conceptType}] ${f.label}  strength=%.3f staleness=%.4f recall=%.3f".format(f.strength, f.staleness, f.recallProb))
            }
            checks["fading queue non-empty"] = fading.isNotEmpty()

            // ── 4. Practice round trip ──────────────────────────────────
            // The stub generator's expected answer is fixed (HebbianStubEngine.stubGenerate).
            val offer = engine.nextPractice(sessionId)
            if (offer == null) {
                log("practice: nextPractice returned null")
                checks["practice round trip"] = false
                checks["half-life shifted 3.0 → 3.75 days"] = false
            } else {
                log("practice offer: [${offer.exerciseType}] ${offer.prompt} (targets: ${offer.targets})")
                val grade = engine.submitPracticeAnswer(sessionId, "ラーメンを食べたいです。")
                log("practice grade: correct=${grade?.correct} reinforced=${grade?.reinforced} feedback=${grade?.feedback}")
                checks["practice round trip"] =
                    grade != null && grade.correct && grade.reinforced

                // Scheduler persisted the outcome; a fresh instance reads it back.
                // fact prior = 3.0 days × GROWTH_ON_CORRECT 1.25 = 3.75 days.
                val hlDays = PracticeScheduler(File(stateDir, "practice_state")).halflifeS("fact") / 86400.0
                log("adaptive half-life for 'fact': %.4f days (prior 3.0)".format(hlDays))
                checks["half-life shifted 3.0 → 3.75 days"] = abs(hlDays - 3.75) < 1e-3
            }

            engine.endSession(sessionId)
            // NB: engine.close() intentionally NOT called — it would close the
            // shared HebbianStoreFactory store other spikes/app screens may hold.

            val failedChecks = checks.filterValues { !it }.keys
            for ((name, ok) in checks) log("%-4s %s".format(if (ok) "PASS" else "FAIL", name))
            SpikeResult(
                passed = failedChecks.isEmpty(),
                summary = if (failedChecks.isEmpty())
                    "Ingest loop works end-to-end on ${store.id} (${checks.size}/${checks.size} checks)"
                else "FAILED: ${failedChecks.joinToString()}",
                metrics = mapOf(
                    "store" to store.id,
                    "turns" to "${doneResults.size}/${turns.size}",
                    "edges moved" to "${moved.size}/${graph.edges.size}",
                    "fading queue" to "${fading.size}",
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
}
