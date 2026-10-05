package com.mobilerag.hebbian.ui

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.unit.sp
import com.mobilerag.ui.theme.AnimusMono
import com.mobilerag.ui.theme.AppTheme
import com.mobilerag.ui.theme.MemoryLight
import com.mobilerag.ui.theme.drawGlow
import com.mobilerag.ui.theme.LocalAmbientTime
import com.mobilerag.ui.theme.rememberGlowSprite
import kotlin.math.PI
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * The entity, drawn: the strongest part of the Hebbian graph as a constellation.
 *
 * [ConstellationLayout.build] is pure (unit-tested): it merges both edge directions (max
 * weight), keeps the [maxNodes] concepts with the most incident weight, and runs a seeded
 * Fruchterman–Reingold layout, so the same memory always draws the same sky.
 */
object ConstellationLayout {

    /** A placed concept; [x]/[y] in 0..1 of the drawing area, [strength] 0..1 of the strongest. */
    data class Node(val id: String, val label: String, val x: Float, val y: Float, val strength: Float)

    /**
     * An undirected link between node indices. [w] is 0..1 across the range of weights shown
     * (weakest → 0, strongest → 1): young graphs cluster near their initial weights, and a
     * scale anchored at 0 would draw every link equally bright.
     */
    data class Link(val a: Int, val b: Int, val w: Float)

    data class Result(val nodes: List<Node>, val links: List<Link>, val hub: Int?) {
        val isEmpty: Boolean get() = nodes.isEmpty()
    }

    fun build(
        labels: Map<String, String>,
        edges: List<Triple<String, String, Double>>,
        maxNodes: Int = 14,
        seed: Long = 42,
        iterations: Int = 300,
    ): Result {
        // Undirected, max of both directions; self loops dropped.
        val und = HashMap<Pair<String, String>, Double>()
        for ((s, d, w) in edges) {
            if (s == d || s !in labels || d !in labels) continue
            val key = if (s < d) s to d else d to s
            und[key] = max(und[key] ?: 0.0, w)
        }
        val strength = HashMap<String, Double>()
        for ((k, w) in und) {
            strength[k.first] = (strength[k.first] ?: 0.0) + w
            strength[k.second] = (strength[k.second] ?: 0.0) + w
        }
        val chosen = labels.keys
            .sortedWith(compareByDescending<String> { strength[it] ?: 0.0 }.thenBy { it })
            .take(maxNodes)
        if (chosen.isEmpty()) return Result(emptyList(), emptyList(), null)
        val index = chosen.withIndex().associate { (i, id) -> id to i }
        val links = und.mapNotNull { (k, w) ->
            val a = index[k.first] ?: return@mapNotNull null
            val b = index[k.second] ?: return@mapNotNull null
            Triple(a, b, w)
        }
        val maxW = links.maxOfOrNull { it.third }?.takeIf { it > 0 } ?: 1.0
        val minW = links.minOfOrNull { it.third } ?: 0.0
        val spanW = maxW - minW
        val maxS = chosen.maxOf { strength[it] ?: 0.0 }.takeIf { it > 0 } ?: 1.0

        // Fruchterman–Reingold in the unit square; attraction scaled by link weight.
        val n = chosen.size
        val rnd = java.util.Random(seed)
        val px = DoubleArray(n) { rnd.nextDouble() }
        val py = DoubleArray(n) { rnd.nextDouble() }
        val k = sqrt(1.0 / n) * 0.9
        var temp = 0.12
        val dx = DoubleArray(n)
        val dy = DoubleArray(n)
        repeat(iterations) {
            dx.fill(0.0); dy.fill(0.0)
            for (i in 0 until n) for (j in i + 1 until n) {
                var ex = px[i] - px[j]
                var ey = py[i] - py[j]
                var d = sqrt(ex * ex + ey * ey)
                if (d < 1e-6) { ex = 1e-3 * (i - j); ey = 1e-3; d = sqrt(ex * ex + ey * ey) }
                val f = k * k / d
                dx[i] += ex / d * f; dy[i] += ey / d * f
                dx[j] -= ex / d * f; dy[j] -= ey / d * f
            }
            for ((a, b, w) in links) {
                val ex = px[a] - px[b]
                val ey = py[a] - py[b]
                val d = max(sqrt(ex * ex + ey * ey), 1e-6)
                val f = d * d / k * (0.4 + 0.6 * w / maxW)
                dx[a] -= ex / d * f; dy[a] -= ey / d * f
                dx[b] += ex / d * f; dy[b] += ey / d * f
            }
            // Weak pull to the centre keeps disconnected islands on screen.
            for (i in 0 until n) {
                dx[i] += (0.5 - px[i]) * 0.02
                dy[i] += (0.5 - py[i]) * 0.02
                val d = max(sqrt(dx[i] * dx[i] + dy[i] * dy[i]), 1e-9)
                val step = min(d, temp)
                px[i] += dx[i] / d * step
                py[i] += dy[i] / d * step
            }
            temp = max(temp * 0.985, 0.002)
        }
        // Fit to the area.
        fun fit(v: DoubleArray): FloatArray {
            val lo = v.min(); val hi = v.max()
            val span = hi - lo
            return FloatArray(n) { i -> if (span < 1e-9) 0.5f else ((v[i] - lo) / span).toFloat() }
        }
        val fx = fit(px)
        val fy = fit(py)
        val nodes = chosen.mapIndexed { i, id ->
            Node(id, labels.getValue(id), fx[i], fy[i], ((strength[id] ?: 0.0) / maxS).toFloat())
        }
        val hub = nodes.indices.maxByOrNull { nodes[it].strength }?.takeIf { nodes[it].strength > 0f }
        return Result(
            nodes,
            links.map { (a, b, w) -> Link(a, b, if (spanW > 1e-9) ((w - minW) / spanW).toFloat() else 1f) },
            hub,
        )
    }
}

/**
 * Draws a [ConstellationLayout.Result] into the region [left..right] × [top..bottom] (fractions
 * of the canvas): guide rings on the hub, glowing links (strong ones bloom), pulsing nodes,
 * labels that flip to the left near the right edge, and an accent ring on the hub.
 */
@Composable
fun Constellation(
    layout: ConstellationLayout.Result,
    modifier: Modifier = Modifier,
    left: Float = 0.06f,
    right: Float = 0.94f,
    top: Float = 0.1f,
    bottom: Float = 0.92f,
) {
    val glass = AppTheme.glass
    val time by LocalAmbientTime.current
    val sprite = rememberGlowSprite()
    val measurer = rememberTextMeasurer()
    val light = if (glass.dark) MemoryLight else glass.slab
    val core = if (glass.dark) Color(0xFFF2FFFF) else Color(0xFF2A2F34)
    val labelColor = if (glass.dark) Color(0xFFBFE9EC) else Color(0xFF3A4046)
    val glowFilter = remember(light) { ColorFilter.tint(light, BlendMode.Modulate) }
    val labelStyle = TextStyle(fontFamily = AnimusMono, fontSize = 9.5.sp, color = labelColor, letterSpacing = 0.3.sp)
    val description = remember(layout) {
        if (layout.isEmpty) "No concepts yet"
        else "Memory graph: " + layout.nodes.sortedByDescending { it.strength }.take(5).joinToString { it.label }
    }

    Canvas(modifier.semantics { contentDescription = description }) {
        if (layout.isEmpty) return@Canvas
        fun pos(nd: ConstellationLayout.Node) = Offset(
            size.width * (left + (right - left) * nd.x),
            size.height * (top + (bottom - top) * nd.y),
        )
        val pts = layout.nodes.map(::pos)
        val t = time

        layout.hub?.let { h ->
            val c = pts[h]
            val r1 = size.minDimension * 0.30f
            if (glass.dark) {
                drawCircle(
                    Brush.radialGradient(listOf(light.copy(alpha = 0.45f), Color.Transparent), center = c, radius = r1 * 1.3f),
                    radius = r1 * 1.3f, center = c,
                )
            }
            drawCircle(light.copy(alpha = 0.14f), radius = r1, center = c, style = Stroke(1f))
            drawCircle(
                light.copy(alpha = 0.08f), radius = r1 * 1.9f, center = c,
                style = Stroke(1f, pathEffect = PathEffect.dashPathEffect(floatArrayOf(2f * density, 5f * density))),
            )
        }

        for (l in layout.links) {
            val a = pts[l.a]
            val b = pts[l.b]
            val alpha = 0.07f + 0.83f * l.w * l.w * l.w
            if (glass.dark && l.w > 0.75f) drawLine(light.copy(alpha = alpha * 0.35f), a, b, strokeWidth = 4f * density)
            drawLine(light.copy(alpha = alpha), a, b, strokeWidth = 1f * density)
        }

        layout.nodes.forEachIndexed { i, nd ->
            val c = pts[i]
            val pulse = 0.72f + 0.28f * sin((t / 3.2f + i * 0.19f) * 2f * PI.toFloat())
            val r = (2.3f + 1.6f * nd.strength) * density
            drawGlow(sprite, c, r * 7f, (if (glass.dark) 0.9f else 0.35f) * pulse, glowFilter)
            drawCircle(core, radius = r, center = c)
        }

        layout.hub?.let { h -> drawCircle(glass.mark, radius = 13f * density, center = pts[h], style = Stroke(1.2f * density)) }

        val maxLabel = (size.width * 0.42f).toInt().coerceAtLeast(1)
        layout.nodes.forEachIndexed { i, nd ->
            val m = measurer.measure(
                nd.label, labelStyle, overflow = TextOverflow.Ellipsis, maxLines = 1,
                constraints = Constraints(maxWidth = maxLabel),
            )
            val c = pts[i]
            val gap = 9f * density
            val x = if (c.x > size.width * 0.62f) c.x - gap - m.size.width else c.x + gap
            val y = (c.y - m.size.height - 2f * density).coerceAtLeast(0f)
            drawText(m, topLeft = Offset(x.coerceIn(0f, size.width - m.size.width), y))
        }
    }
}
