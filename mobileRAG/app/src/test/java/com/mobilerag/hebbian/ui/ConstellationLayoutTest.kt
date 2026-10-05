package com.mobilerag.hebbian.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ConstellationLayoutTest {

    // The Personal space's graph (2026-09-26), both directions as stored.
    private val labels = mapOf(
        "hi" to "Hiragana", "ko" to "Konnichiwa", "ar" to "Arigatou", "on" to "Onega",
        "gr" to "greeting", "qu" to "question_about_learning", "in" to "invitation_to_interact",
    )
    private val edges = listOf(
        Triple("hi", "ko", 1.5), Triple("ko", "hi", 1.5), Triple("hi", "ar", 1.5), Triple("ar", "hi", 1.5),
        Triple("hi", "on", 1.5), Triple("on", "hi", 1.5), Triple("gr", "qu", 1.31), Triple("qu", "gr", 1.3),
        Triple("gr", "in", 1.31), Triple("in", "gr", 1.28), Triple("qu", "in", 1.31), Triple("in", "qu", 1.29),
        Triple("ko", "ar", 1.2), Triple("ko", "on", 1.2), Triple("ar", "on", 1.2), Triple("hi", "gr", 1.03),
    )

    @Test
    fun placesEveryNodeInsideTheUnitSquareDeterministically() {
        val a = ConstellationLayout.build(labels, edges)
        val b = ConstellationLayout.build(labels, edges)
        assertEquals(a, b)
        assertEquals(7, a.nodes.size)
        a.nodes.forEach { n ->
            assertTrue(n.x in 0f..1f && n.y in 0f..1f)
            assertTrue(!n.x.isNaN() && !n.y.isNaN())
        }
    }

    @Test
    fun mergesDirectionsAndPicksTheHubByIncidentWeight() {
        val r = ConstellationLayout.build(labels, edges)
        assertEquals(10, r.links.size) // 16 directed rows → 10 undirected pairs
        assertEquals("Hiragana", r.nodes[r.hub!!].label)
        assertEquals(1f, r.links.maxOf { it.w })
        assertEquals(0f, r.links.minOf { it.w }) // weakest shown link → 0 (range-normalized)
    }

    @Test
    fun capsNodesAndDropsLinksToLeftOutConcepts() {
        val r = ConstellationLayout.build(labels, edges, maxNodes = 3)
        assertEquals(3, r.nodes.size)
        r.links.forEach { assertTrue(it.a < 3 && it.b < 3) }
    }

    @Test
    fun emptyAndIsolatedGraphsDontCrash() {
        assertTrue(ConstellationLayout.build(emptyMap(), emptyList()).isEmpty)
        val lone = ConstellationLayout.build(mapOf("x" to "x"), emptyList())
        assertEquals(1, lone.nodes.size)
        assertEquals(null, lone.hub)
    }
}
