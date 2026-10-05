package com.mobilerag.hebbian.cortical

import com.mobilerag.hebbian.store.ConceptNode
import com.mobilerag.hebbian.store.HebbianEdge
import com.mobilerag.hebbian.store.Subgraph
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import kotlin.math.sqrt

class CorticalPrepTest {

    private fun node(id: String, vararg emb: Float) = ConceptNode(
        nodeId = id, label = id, textRaw = id, textSummary = id, conceptType = "fact", sourceType = "user",
        sourceUri = null, createdAt = 0, updatedAt = 0, activationCount = 1, embedding = emb,
    )

    private fun edge(s: String, d: String, w: Double) = HebbianEdge(
        srcId = s, dstId = d, hebbWeight = w, eligibility = 0.0, causalScore = 0.0,
        coActivationCount = 1, lastUpdated = 0, layer = "hippocampal",
    )

    @Test
    fun normalizesAndIndexesEdges() {
        val sg = Subgraph(
            listOf(node("a", 3f, 4f), node("b", 0f, 2f)),
            listOf(edge("a", "b", 1.5), edge("b", "a", 2.0)),
        )
        val p = CorticalPrep.prepare(sg, maxNodes = 10)!!
        assertEquals(listOf("a", "b"), p.ids)
        assertArrayEquals(doubleArrayOf(0.6, 0.8, 0.0, 1.0), p.x.data, 1e-12)
        assertArrayEquals(intArrayOf(0, 1), p.src)
        assertArrayEquals(intArrayOf(1, 0), p.dst)
        assertArrayEquals(doubleArrayOf(1.5, 2.0), p.w, 0.0)
        assertEquals(0, p.droppedNodes)
    }

    @Test
    fun dropsNodesWithMissingOrOddEmbeddingsAndTheirEdges() {
        val sg = Subgraph(
            listOf(node("a", 1f, 0f), node("b", 0f, 1f), node("c"), node("d", 1f, 1f, 1f)),
            listOf(edge("a", "b", 1.0), edge("a", "c", 1.0), edge("d", "b", 1.0)),
        )
        val p = CorticalPrep.prepare(sg, maxNodes = 10)!!
        assertEquals(listOf("a", "b"), p.ids)
        assertEquals(1, p.src.size)
        assertEquals(2, p.droppedNodes)
    }

    @Test
    fun nodeCapKeepsStrongestConnectedNodes() {
        val sg = Subgraph(
            listOf(node("a", 1f, 0f), node("b", 0f, 1f), node("c", 1f, 1f), node("d", 1f, 2f)),
            listOf(edge("a", "b", 5.0), edge("b", "a", 5.0), edge("c", "d", 0.6), edge("b", "c", 0.6)),
        )
        val p = CorticalPrep.prepare(sg, maxNodes = 2)!!
        assertEquals(listOf("a", "b"), p.ids)
        assertEquals(2, p.src.size) // c/d edges dropped with their nodes
        assertEquals(2, p.droppedNodes)
    }

    @Test
    fun noEmbeddingsAtAllIsNull() {
        assertNull(CorticalPrep.prepare(Subgraph(listOf(node("a")), emptyList()), 10))
    }

    @Test
    fun zPrevAlignsByIdAndAnchorsNewNodesToX() {
        val x = Mat(3, 2, doubleArrayOf(1.0, 0.0, 0.0, 1.0, 0.5, 0.5))
        val prev = Mat(2, 2, doubleArrayOf(9.0, 9.0, 7.0, 7.0))
        val z = CorticalPrep.alignZPrev(listOf("c", "a"), prev, listOf("a", "b", "c"), x)!!
        assertArrayEquals(doubleArrayOf(7.0, 7.0, 0.0, 1.0, 9.0, 9.0), z.data, 0.0)
    }

    @Test
    fun zPrevOfOtherWidthIsIgnored() {
        val x = Mat(1, 2, doubleArrayOf(1.0, 0.0))
        assertNull(CorticalPrep.alignZPrev(listOf("a"), Mat(1, 3), listOf("a"), x))
    }

    @Test
    fun rowCosineMatchesDefinition() {
        val z = Mat(2, 2, doubleArrayOf(1.0, 0.0, 1.0, 1.0))
        assertEquals(1.0 / sqrt(2.0), CorticalPrep.rowCosine(z, 0, 1), 1e-12)
    }
}
