package com.mobilerag.graph.ui

import com.mobilerag.graph.LadybugGraphStore
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import kotlin.math.sqrt

/**
 * Opt-in tuning harness: LAYOUT_DATA=<dir with graph.lbug + communities.txt ("id|[ids]")>.
 * Prints how the 3D layout spreads a real entity graph (median nearest-neighbour distance,
 * community radius vs distance between communities).
 */
class GraphLayoutRealDataTest {
    @Test
    fun spread(): Unit = runBlocking {
        val dir = System.getenv("LAYOUT_DATA")?.let(::File)
        assumeTrue(dir != null && File(dir, "graph.lbug").isFile)
        val work = kotlin.io.path.createTempDirectory("layout").toFile()
        File(dir, "graph.lbug").copyTo(File(work, "graph.lbug"))
        val store = LadybugGraphStore()
        store.open(File(work, "graph.lbug").absolutePath)
        val entities = store.allEntities(limit = 2000)
        val edges = store.allEdges()
        store.close()
        val communities = File(dir!!, "communities.txt").readLines().filter { it.isNotBlank() }.map { line ->
            val (id, ids) = line.split("|", limit = 2)
            val arr = JSONArray(ids)
            Triple(id.toLong(), "", List(arr.length()) { arr.getLong(it) })
        }
        System.getenv("L3D_ATTRACT")?.toDoubleOrNull()?.let { GraphLayout3D.ATTRACT = it }
        System.getenv("L3D_GRAVITY")?.toDoubleOrNull()?.let { GraphLayout3D.GRAVITY = it }
        System.getenv("L3D_COHESION")?.toDoubleOrNull()?.let { GraphLayout3D.COHESION = it }
        val r = GraphLayout3D.build(entities, edges, communities)
        fun d(a: GraphLayout3D.Node, b: GraphLayout3D.Node) =
            sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z))
        val nn = r.nodes.map { a -> r.nodes.filter { it !== a }.minOf { d(a, it) } }.sorted()
        val groups = r.nodes.groupBy { it.community }
        val cents = groups.mapValues { (_, g) -> Triple(g.map { it.x }.average(), g.map { it.y }.average(), g.map { it.z }.average()) }
        val radius = groups.map { (c, g) ->
            val (x, y, z) = cents.getValue(c)
            g.map { sqrt((it.x - x) * (it.x - x) + (it.y - y) * (it.y - y) + (it.z - z) * (it.z - z)) }.average()
        }.average()
        val cl = cents.values.toList()
        var sum = 0.0; var cnt = 0
        for (i in cl.indices) for (j in i + 1 until cl.size) {
            val (a, b, c) = cl[i]; val (e, f, g) = cl[j]
            sum += sqrt((a - e) * (a - e) + (b - f) * (b - f) + (c - g) * (c - g)); cnt++
        }
        val rad = r.nodes.map { sqrt(it.x * it.x + it.y * it.y + it.z * it.z) }.sorted()
        println("LAYOUT n=${r.nodes.size} links=${r.links.size} comms=${r.communities.size} " +
            "nnMedian=%.3f nnP10=%.3f commRadius=%.3f betweenComms=%.3f radiusMedian=%.3f".format(
                nn[nn.size / 2], nn[nn.size / 10], radius, sum / cnt, rad[rad.size / 2]))
    }
}
