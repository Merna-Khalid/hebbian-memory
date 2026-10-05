package com.mobilerag.hebbian

import org.json.JSONObject
import java.io.File
import kotlin.math.pow
import kotlin.math.round
import kotlin.math.sqrt

/**
 * Adaptive, outcome-driven practice scheduling. Port of `core/practice_scheduler.py`
 * (stdlib only).
 *
 * Per-concept-type half-lives adapt to observed recall outcomes:
 *
 *     correct answer → that type's half-life ×= 1.25   (memory holds longer)
 *     wrong answer   → that type's half-life ×= 0.75   (forgetting faster than assumed)
 *
 * clamped to [0.5, 30] days and persisted to disk so estimates survive restarts. Recall
 * probability follows the Ebbinghaus-style exponential `p = 2^(−Δt/h)`. A lightweight
 * online variant of half-life regression (Settles & Meeder HLR) — no training corpus
 * needed; bootstraps from priors and personalizes as you practice.
 */

private const val SECONDS_PER_DAY = 24.0 * 3600.0

/** Plain-Kotlin cosine similarity for interference checks. */
fun cosineSim(a: DoubleArray, b: DoubleArray): Double {
    if (a.isEmpty() || b.isEmpty() || a.size != b.size) return 0.0
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    for (i in a.indices) {
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    }
    if (na == 0.0 || nb == 0.0) return 0.0
    return dot / (sqrt(na) * sqrt(nb))
}

/**
 * Owns per-type adaptive half-lives (days), last-practiced timestamps per node
 * (interleave/recently-seen guard), and recall-probability queries used to rank the
 * practice queue. State persists as JSON under [stateDir].
 */
class PracticeScheduler(
    private val stateDir: File,
    private val relearnGapS: Double = 10.0 * 60.0, // don't re-serve an item within 10 min
) {
    private val stateFile = File(stateDir, "scheduler.json")

    private val halflifeDays = mutableMapOf<String, Double>()
    private val lastPracticed = mutableMapOf<String, Double>()

    init {
        load()
    }

    // ── Persistence ───────────────────────────────────────────────────

    private fun load() {
        try {
            val data = JSONObject(stateFile.readText())
            data.optJSONObject("halflife_days")?.let { obj ->
                for (key in obj.keys()) halflifeDays[key] = obj.getDouble(key)
            }
            data.optJSONObject("last_practiced")?.let { obj ->
                for (key in obj.keys()) lastPracticed[key] = obj.getDouble(key)
            }
        } catch (_: Exception) {
            // fresh start — missing or corrupt state file
        }
    }

    /** Atomic write: tmp file + rename, so a crash mid-write can't corrupt state. */
    fun save() {
        stateDir.mkdirs()
        val halflives = JSONObject()
        for ((k, v) in halflifeDays) halflives.put(k, v)
        val practiced = JSONObject()
        for ((k, v) in lastPracticed) practiced.put(k, v)
        val root = JSONObject()
            .put("halflife_days", halflives)
            .put("last_practiced", practiced)
            .put("saved_at", nowSeconds())
        val tmp = File(stateDir, "scheduler.json.tmp")
        tmp.writeText(root.toString())
        if (!tmp.renameTo(stateFile)) {
            // renameTo over an existing dest can fail on some filesystems — fall back
            stateFile.delete()
            tmp.renameTo(stateFile)
        }
    }

    // ── Queries ───────────────────────────────────────────────────────

    /** Half-life in seconds for [conceptType], clamped to [MIN_HALF_LIFE_DAYS, MAX_HALF_LIFE_DAYS]. */
    fun halflifeS(conceptType: String): Double {
        val hDays = halflifeDays[conceptType]
            ?: DEFAULT_HALF_LIFE_DAYS[conceptType]
            ?: NEUTRAL_HALF_LIFE_DAYS
        return hDays.coerceIn(MIN_HALF_LIFE_DAYS, MAX_HALF_LIFE_DAYS) * SECONDS_PER_DAY
    }

    /** Ebbinghaus exponential: p = 2^(−Δt/h). Low p → practice next. */
    fun recallProb(elapsedS: Double, conceptType: String): Double =
        2.0.pow(-maxOf(elapsedS, 0.0) / halflifeS(conceptType))

    /** False while an item was practiced too recently (spacing guard). */
    fun readyAgain(nodeId: String): Boolean {
        val last = lastPracticed[nodeId] ?: return true
        return (nowSeconds() - last) >= relearnGapS
    }

    /** Current adaptive half-lives (days) — for dashboards/debugging. */
    fun halflifeTable(): Map<String, Double> {
        val merged = LinkedHashMap(DEFAULT_HALF_LIFE_DAYS)
        merged.putAll(halflifeDays)
        return merged.mapValues { (_, v) -> round(v * 1000.0) / 1000.0 }
    }

    // ── Updates ───────────────────────────────────────────────────────

    /**
     * Update half-lives from a graded exercise and stamp practice times. All target
     * types shift together, but each distinct type shifts only once per call however
     * many of its concepts were targeted.
     */
    fun record(nodeIds: List<String>, conceptTypes: List<String>, correct: Boolean) {
        val factor = if (correct) GROWTH_ON_CORRECT else SHRINK_ON_WRONG
        for (ctype in conceptTypes.distinct()) {
            val current = halflifeS(ctype) / SECONDS_PER_DAY
            val updated = (current * factor).coerceIn(MIN_HALF_LIFE_DAYS, MAX_HALF_LIFE_DAYS)
            halflifeDays[ctype] = round(updated * 10000.0) / 10000.0
        }

        val now = nowSeconds()
        for (nid in nodeIds) lastPracticed[nid] = now

        // prune stale entries (>90 days idle)
        val cutoff = now - 90.0 * SECONDS_PER_DAY
        lastPracticed.entries.removeAll { it.value < cutoff }
        save()
    }

    /** After a duplicate merge: fold dropped node ids into their survivors (latest practice wins). */
    fun remapNodes(remap: Map<String, String>) {
        var changed = false
        for ((from, to) in remap) {
            val t = lastPracticed.remove(from) ?: continue
            lastPracticed[to] = maxOf(lastPracticed[to] ?: 0.0, t)
            changed = true
        }
        if (changed) save()
    }

    private fun nowSeconds(): Double = System.currentTimeMillis() / 1000.0

    companion object {
        /** Priors per concept type (days) — loosely ordered by consolidation depth. */
        val DEFAULT_HALF_LIFE_DAYS: Map<String, Double> = linkedMapOf(
            "vocabulary" to 2.0,
            "example" to 2.0,
            "question" to 2.0,
            "task" to 1.0,
            "event" to 1.5,
            "fact" to 3.0,
            "idea" to 4.0,
            "plan" to 4.0,
            "procedure" to 6.0,
            "feeling" to 6.0,
            "entity" to 7.0,
            "grammar" to 10.0,
            "cultural" to 14.0,
        )
        const val NEUTRAL_HALF_LIFE_DAYS = 3.0

        const val GROWTH_ON_CORRECT = 1.25
        const val SHRINK_ON_WRONG = 0.75
        const val MIN_HALF_LIFE_DAYS = 0.5
        const val MAX_HALF_LIFE_DAYS = 30.0
    }
}
