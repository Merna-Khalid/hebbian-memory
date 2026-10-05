package com.mobilerag.graph.ui

import com.mobilerag.core.GraphStore
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * The entity graph placed in 3D for the constellation view: a seeded Fruchterman–Reingold
 * layout with an extra pull toward each community's centre, so communities read as clusters.
 * Pure and deterministic (unit-tested) — the same graph always draws the same sky.
 */
object GraphLayout3D {

    // Tuned on a real 97-entity / 477-link / 7-community graph (GraphLayoutRealDataTest):
    // nearest-neighbour median 0.036 → 0.197 and community radius 0.068 → 0.273 of the
    // unit sphere, with communities still ~1.2 apart. Vars so the harness can sweep them.
    internal var ATTRACT = 2.0
    internal var GRAVITY = 1.5
    internal var COHESION = 0.05

    data class Node(
        val id: Long,
        val name: String,
        val type: String,
        val mentions: Int,
        /** Index into [Result.communities], or -1 when the entity is in none. */
        val community: Int,
        val x: Float, val y: Float, val z: Float,
        /** 0..1 by log(mentions) — drives node size and which labels show. */
        val weight: Float,
    )

    /** Undirected link; [w] 0..1 across the range of parallel-edge counts shown. */
    data class Link(val a: Int, val b: Int, val w: Float)

    data class Community(val id: Long, val summary: String, val size: Int)

    data class Result(val nodes: List<Node>, val links: List<Link>, val communities: List<Community>) {
        val isEmpty: Boolean get() = nodes.isEmpty()
        val neighbours: Map<Int, List<Int>> by lazy {
            val m = HashMap<Int, MutableList<Int>>()
            links.forEach { l -> m.getOrPut(l.a) { ArrayList() } += l.b; m.getOrPut(l.b) { ArrayList() } += l.a }
            m
        }
    }

    fun build(
        entities: List<GraphStore.Entity>,
        edges: List<GraphStore.Edge>,
        communities: List<Triple<Long, String, List<Long>>>,
        seed: Long = 7,
        iterations: Int = 240,
    ): Result {
        if (entities.isEmpty()) return Result(emptyList(), emptyList(), emptyList())
        val index = entities.withIndex().associate { (i, e) -> e.id to i }
        val n = entities.size

        // Undirected links, parallel edges counted.
        val count = HashMap<Pair<Int, Int>, Int>()
        for (e in edges) {
            val a = index[e.fromId] ?: continue
            val b = index[e.toId] ?: continue
            if (a == b) continue
            val key = if (a < b) a to b else b to a
            count[key] = (count[key] ?: 0) + 1
        }
        val maxC = count.values.maxOrNull() ?: 1
        val minC = count.values.minOrNull() ?: 1
        val span = (maxC - minC).coerceAtLeast(1)

        val comm = IntArray(n) { -1 }
        val kept = ArrayList<Community>()
        for ((cid, summary, members) in communities) {
            val ci = kept.size
            var any = false
            for (m in members) index[m]?.let { comm[it] = ci; any = true }
            if (any) kept += Community(cid, summary, members.size)
        }

        // Chunk co-occurrence makes every community a dense clique; pulling with the full
        // weight of all those springs crushes it to a point. Each spring is divided by the
        // endpoints' degrees (geometric mean) so dense and sparse regions spread alike.
        val degree = IntArray(n)
        for ((key, _) in count) { degree[key.first]++; degree[key.second]++ }

        val rnd = java.util.Random(seed)
        val px = DoubleArray(n) { rnd.nextDouble() - 0.5 }
        val py = DoubleArray(n) { rnd.nextDouble() - 0.5 }
        val pz = DoubleArray(n) { rnd.nextDouble() - 0.5 }
        val k = 1.0 / Math.cbrt(n.toDouble()) * 1.4
        val dx = DoubleArray(n); val dy = DoubleArray(n); val dz = DoubleArray(n)
        val cx = DoubleArray(kept.size); val cy = DoubleArray(kept.size); val cz = DoubleArray(kept.size)
        val cn = IntArray(kept.size)
        var temp = 0.1
        repeat(iterations) {
            dx.fill(0.0); dy.fill(0.0); dz.fill(0.0)
            for (i in 0 until n) for (j in i + 1 until n) {
                var ex = px[i] - px[j]; var ey = py[i] - py[j]; var ez = pz[i] - pz[j]
                var d = sqrt(ex * ex + ey * ey + ez * ez)
                if (d < 1e-6) { ex = 1e-3 * (i - j); ey = 1e-3; ez = -1e-3; d = sqrt(ex * ex + ey * ey + ez * ez) }
                val f = k * k / d
                dx[i] += ex / d * f; dy[i] += ey / d * f; dz[i] += ez / d * f
                dx[j] -= ex / d * f; dy[j] -= ey / d * f; dz[j] -= ez / d * f
            }
            for ((key, c) in count) {
                val (a, b) = key
                val ex = px[a] - px[b]; val ey = py[a] - py[b]; val ez = pz[a] - pz[b]
                val d = max(sqrt(ex * ex + ey * ey + ez * ez), 1e-6)
                val f = d * d / k * (0.5 + 0.5 * (c - minC).toDouble() / span) /
                    sqrt(max(degree[a], 1).toDouble() * max(degree[b], 1).toDouble()) * ATTRACT
                dx[a] -= ex / d * f; dy[a] -= ey / d * f; dz[a] -= ez / d * f
                dx[b] += ex / d * f; dy[b] += ey / d * f; dz[b] += ez / d * f
            }
            // Community cohesion: pull members toward their community's centroid.
            cx.fill(0.0); cy.fill(0.0); cz.fill(0.0); cn.fill(0)
            for (i in 0 until n) if (comm[i] >= 0) { cx[comm[i]] += px[i]; cy[comm[i]] += py[i]; cz[comm[i]] += pz[i]; cn[comm[i]]++ }
            for (i in 0 until n) {
                val c = comm[i]
                if (c >= 0 && cn[c] > 1) {
                    dx[i] += (cx[c] / cn[c] - px[i]) * COHESION
                    dy[i] += (cy[c] / cn[c] - py[i]) * COHESION
                    dz[i] += (cz[c] / cn[c] - pz[i]) * COHESION
                }
                // Gravity keeps islands (disconnected entities) in view.
                dx[i] -= px[i] * GRAVITY; dy[i] -= py[i] * GRAVITY; dz[i] -= pz[i] * GRAVITY
                val d = max(sqrt(dx[i] * dx[i] + dy[i] * dy[i] + dz[i] * dz[i]), 1e-9)
                val step = min(d, temp)
                px[i] += dx[i] / d * step; py[i] += dy[i] / d * step; pz[i] += dz[i] / d * step
            }
            temp = max(temp * 0.985, 0.002)
        }

        // Centre, scale so 92% of the stars fill the unit sphere, and settle the few outliers
        // on its rim — scaling to the farthest one shrank the whole graph to a knot.
        val mx = px.average(); val my = py.average(); val mz = pz.average()
        val radii = DoubleArray(n) { i -> sqrt((px[i] - mx) * (px[i] - mx) + (py[i] - my) * (py[i] - my) + (pz[i] - mz) * (pz[i] - mz)) }
        val r = max(radii.sorted()[((n - 1) * 0.92).toInt()], 1e-9)
        for (i in 0 until n) {
            val scale = if (radii[i] > r) r / radii[i] else 1.0
            px[i] = mx + (px[i] - mx) * scale; py[i] = my + (py[i] - my) * scale; pz[i] = mz + (pz[i] - mz) * scale
        }
        val maxM = entities.maxOf { it.mentionCount }.coerceAtLeast(1)
        val nodes = entities.mapIndexed { i, e ->
            Node(
                id = e.id, name = e.name, type = e.type, mentions = e.mentionCount, community = comm[i],
                x = ((px[i] - mx) / r).toFloat(), y = ((py[i] - my) / r).toFloat(), z = ((pz[i] - mz) / r).toFloat(),
                weight = (ln(1.0 + e.mentionCount) / ln(1.0 + maxM)).toFloat(),
            )
        }
        val links = count.map { (key, c) -> Link(key.first, key.second, (c - minC).toFloat() / span) }
        return Result(nodes, links, kept)
    }
}
