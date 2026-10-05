package com.mobilerag.graph.ui

import com.mobilerag.core.GraphStore
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sqrt

class GraphLayout3DTest {

    private fun e(id: Long, mentions: Int = 1) = GraphStore.Entity(id, "e$id", "thing", mentions)

    // Two triangles joined by one bridge; each triangle is a community.
    private val entities = (1L..6L).map { e(it, it.toInt()) }
    private val edges = listOf(
        GraphStore.Edge(1, 2, "r"), GraphStore.Edge(2, 3, "r"), GraphStore.Edge(3, 1, "r"),
        GraphStore.Edge(4, 5, "r"), GraphStore.Edge(5, 6, "r"), GraphStore.Edge(6, 4, "r"),
        GraphStore.Edge(3, 4, "r"), GraphStore.Edge(4, 3, "r"), GraphStore.Edge(1, 1, "self"),
    )
    private val communities = listOf(Triple(10L, "first", listOf(1L, 2L, 3L)), Triple(11L, "second", listOf(4L, 5L, 6L)))

    @Test
    fun deterministicAndInsideTheUnitSphere() {
        val a = GraphLayout3D.build(entities, edges, communities)
        assertEquals(a, GraphLayout3D.build(entities, edges, communities))
        a.nodes.forEach { n ->
            val r = sqrt(n.x * n.x + n.y * n.y + n.z * n.z)
            assertTrue("inside the sphere: $r", r <= 1.0001f)
            assertTrue(!n.x.isNaN() && !n.y.isNaN() && !n.z.isNaN())
        }
    }

    @Test
    fun mergesDirectionsDropsSelfLoopsAndWeightsParallelEdges() {
        val r = GraphLayout3D.build(entities, edges, communities)
        assertEquals(7, r.links.size) // 3 + 3 + 1 bridge; 1→1 dropped, 3↔4 merged
        val bridge = r.links.single { setOf(r.nodes[it.a].id, r.nodes[it.b].id) == setOf(3L, 4L) }
        assertEquals(1f, bridge.w) // two parallel edges = the most
    }

    @Test
    fun communitiesClusterTogether() {
        val r = GraphLayout3D.build(entities, edges, communities)
        fun d(a: Int, b: Int): Float {
            val p = r.nodes[a]; val q = r.nodes[b]
            return sqrt((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y) + (p.z - q.z) * (p.z - q.z))
        }
        val within = (d(0, 1) + d(4, 5)) / 2
        val across = (d(0, 5) + d(1, 4)) / 2
        assertTrue("within $within < across $across", within < across)
        assertEquals(listOf(0, 0, 0, 1, 1, 1), r.nodes.map { it.community })
    }

    @Test
    fun emptyAndSingleNode() {
        assertTrue(GraphLayout3D.build(emptyList(), emptyList(), emptyList()).isEmpty)
        val one = GraphLayout3D.build(listOf(e(1)), emptyList(), emptyList())
        assertEquals(1, one.nodes.size)
        assertEquals(-1, one.nodes[0].community)
    }
}
