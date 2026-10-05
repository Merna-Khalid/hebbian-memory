package com.mobilerag.hebbian.store

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Mirrored by "Hebbian Memory/tools/merge_duplicates.py" (its --selftest). */
class ConceptMergeTest {

    private fun node(
        id: String, label: String, type: String = "vocabulary", activations: Int = 1,
        created: Long = 0, updated: Long = 0, mT: Double = 1.0, events: List<Double> = emptyList(),
    ) = ConceptNode(
        nodeId = id, label = label, textRaw = "$label text", textSummary = label, conceptType = type,
        sourceType = "user", sourceUri = null, createdAt = created, updatedAt = updated,
        activationCount = activations, embedding = floatArrayOf(1f, 0f), meanMT = mT,
        recentEvents = events.map { JSONObject().put("ts", it).put("m_t", 1.0).toString() },
    )

    private fun edge(s: String, d: String, w: Double, elig: Double = 0.0, causal: Double = 0.0, co: Int = 1, ts: Long = 0) =
        HebbianEdge(s, d, w, elig, causal, co, ts, "hippocampal")

    @Test
    fun plansGroupsBySameLabelKeyAndType() {
        val plan = ConceptMerge.plan(listOf(
            node("a", "食べる", activations = 1),
            node("b", "「食べる」", activations = 4),
            node("c", "食べる", type = "grammar"),        // other type: its own concept
            node("d", "飲む"),
            node("e", "User: hi", type = "event"),         // events never merge
            node("f", "User: hi", type = "event"),
        ))
        assertEquals(1, plan.groups.size)
        assertEquals("b", plan.groups[0].keep.nodeId)      // most activated survives
        assertEquals(mapOf("a" to "b"), plan.remap)
    }

    @Test
    fun activationTieKeepsEarliest() {
        val plan = ConceptMerge.plan(listOf(node("new", "x", activations = 2, created = 20), node("old", "x", activations = 2, created = 10)))
        assertEquals("old", plan.groups[0].keep.nodeId)
    }

    @Test
    fun mergedNodeCombinesHistory() {
        val keep = node("k", "x", activations = 3, created = 50, updated = 60, mT = 2.0, events = listOf(5.0, 7.0))
        val drop = node("d", "x", activations = 1, created = 10, updated = 90, mT = 1.0, events = listOf(6.0))
        val m = ConceptMerge.mergedNode(MergeGroup(keep, listOf(drop)))
        assertEquals("k", m.nodeId)
        assertEquals("x text", m.textRaw)                    // survivor's own content
        assertEquals(4, m.activationCount)
        assertEquals(10L, m.createdAt)
        assertEquals(90L, m.updatedAt)
        assertEquals((2.0 * 3 + 1.0 * 1) / 4, m.meanMT, 1e-12) // activation-weighted
        assertEquals(listOf(5.0, 6.0, 7.0), m.recentEvents.map { JSONObject(it).getDouble("ts") })
    }

    @Test
    fun recentEventsKeepTheLatestTwenty() {
        val keep = node("k", "x", activations = 2, events = (1..15).map { it.toDouble() })
        val drop = node("d", "x", events = (16..30).map { it.toDouble() })
        val m = ConceptMerge.mergedNode(MergeGroup(keep, listOf(drop)))
        assertEquals((11..30).map { it.toDouble() }, m.recentEvents.map { JSONObject(it).getDouble("ts") })
    }

    @Test
    fun edgesArePointedAtSurvivorAndCombined() {
        val remap = mapOf("d" to "k")
        val merged = ConceptMerge.mergeEdges(listOf(
            edge("k", "n", 1.0, elig = 0.5, causal = 0.1, co = 2, ts = 5),
            edge("d", "n", 3.0, elig = 0.2, causal = 0.4, co = 3, ts = 9),  // collides with k→n
            edge("d", "k", 2.0),                                            // becomes a self loop → dropped
            edge("m", "d", 1.5),                                            // re-pointed m→k
        ), remap)
        val byKey = merged.associateBy { it.srcId to it.dstId }
        assertEquals(setOf("k" to "n", "m" to "k"), byKey.keys)
        val kn = byKey.getValue("k" to "n")
        assertEquals(3.0, kn.hebbWeight, 0.0)
        assertEquals(0.5, kn.eligibility, 0.0)
        assertEquals(0.4, kn.causalScore, 0.0)
        assertEquals(5, kn.coActivationCount)
        assertEquals(9L, kn.lastUpdated)
        assertEquals(1.5, byKey.getValue("m" to "k").hebbWeight, 0.0)
    }

    @Test
    fun sessionStatsCollisionKeepsTheBusierRow() {
        val rows = listOf(
            SessionStatRow("s1", "sess", "k", 1.0, 0.5, 0.5, activationCount = 2, deltaWMean = 0.0),
            SessionStatRow("s2", "sess", "d", 1.5, 0.5, 0.5, activationCount = 5, deltaWMean = 0.1),
            SessionStatRow("s3", "other", "d", 1.2, 0.5, 0.5, activationCount = 1, deltaWMean = 0.0),
        )
        val out = ConceptMerge.mergeSessionStats(rows, mapOf("d" to "k"))
        assertEquals(2, out.size)
        assertTrue(out.all { it.nodeId == "k" })
        assertEquals("s2", out.single { it.sessionId == "sess" }.statId)
        assertEquals("s3", out.single { it.sessionId == "other" }.statId)
    }
}
