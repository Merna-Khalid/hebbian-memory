package com.mobilerag.graph.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
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
import com.mobilerag.ui.theme.LocalAmbientTime
import com.mobilerag.ui.theme.MemoryLight
import com.mobilerag.ui.theme.drawGlow
import com.mobilerag.ui.theme.rememberGlowSprite
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.sin

/** Community hues on the sea (luminous) and in the white room (inked). Index = community. */
private val SeaHues = listOf(
    MemoryLight, Color(0xFFF2C46D), Color(0xFFFF7A5C), Color(0xFFB9A3FF),
    Color(0xFF7CF2B0), Color(0xFF8FB8FF), Color(0xFFFF9BD2), Color(0xFFE8F2A0),
)
private val FogHues = listOf(
    Color(0xFF1A7680), Color(0xFF8C6210), Color(0xFFB42A17), Color(0xFF5B45B0),
    Color(0xFF1E7D4F), Color(0xFF2E5AA8), Color(0xFFA0306E), Color(0xFF6E7A1E),
)

fun communityHue(index: Int, dark: Boolean): Color {
    if (index < 0) return if (dark) Color(0xFF9FC3C8) else Color(0xFF6A7278)
    val list = if (dark) SeaHues else FogHues
    return list[index % list.size]
}

/**
 * The entity graph as a 3D constellation: a slow turn on the ambient clock, drag to rotate,
 * pinch to zoom, double-tap to reset, tap a star to select it (tap empty space to clear).
 * Depth is carried by perspective, size and brightness; orbit rings give the space a shape.
 */
@Composable
fun EntityConstellation3D(
    layout: GraphLayout3D.Result,
    selected: Int?,
    onSelect: (Int?) -> Unit,
    modifier: Modifier = Modifier,
    /** Vertical centre of the sphere as a fraction of the height (moves up for the card). */
    centerY: Float = 0.5f,
) {
    val glass = AppTheme.glass
    val time = LocalAmbientTime.current
    val sprite = rememberGlowSprite()
    val measurer = rememberTextMeasurer()
    var yaw by remember { mutableFloatStateOf(0f) }
    var pitch by remember { mutableFloatStateOf(-0.38f) }
    var zoom by remember { mutableFloatStateOf(1f) }
    // The slow turn: accumulated only while nothing is selected, so a chosen star holds still.
    val spin = remember { FloatArray(2) } // [angle, last ambient time]
    // Screen positions from the last frame, for hit-testing taps.
    val screenX = remember(layout) { FloatArray(layout.nodes.size) }
    val screenY = remember(layout) { FloatArray(layout.nodes.size) }
    val filters = remember(glass.dark) {
        (0 until 8).map { ColorFilter.tint(communityHue(it, glass.dark), BlendMode.Modulate) } +
            ColorFilter.tint(communityHue(-1, glass.dark), BlendMode.Modulate)
    }
    // Labels: the most-mentioned entities, plus the selection and its neighbours.
    val labelled = remember(layout, selected) {
        val top = layout.nodes.indices.sortedByDescending { layout.nodes[it].weight }.take(10).toMutableSet()
        selected?.let { s -> top += s; layout.neighbours[s]?.let { top += it } }
        top
    }
    val neighbourSet = remember(layout, selected) { selected?.let { layout.neighbours[it]?.toSet() } ?: emptySet() }
    val labelStyle = TextStyle(fontFamily = AnimusMono, fontSize = 9.5.sp, color = if (glass.dark) Color(0xFFD8F6F7) else Color(0xFF2A2F34))
    // Measured once per label; depth fading is applied as draw alpha, not by re-measuring.
    val labelCache = remember(layout, labelStyle) { HashMap<Int, androidx.compose.ui.text.TextLayoutResult>() }

    Canvas(
        modifier
            .semantics { contentDescription = "Entity graph in 3D: ${layout.nodes.size} entities, ${layout.communities.size} communities. Drag to rotate, pinch to zoom, tap an entity to inspect it." }
            .pointerInput(layout) {
                detectTransformGestures { _, pan, zoomChange, _ ->
                    yaw += pan.x * 0.006f
                    pitch = (pitch + pan.y * 0.006f).coerceIn(-1.35f, 1.35f)
                    zoom = (zoom * zoomChange).coerceIn(0.55f, 3.2f)
                }
            }
            .pointerInput(layout) {
                detectTapGestures(
                    onDoubleTap = { yaw = 0f; pitch = -0.38f; zoom = 1f },
                    onTap = { p ->
                        var best = -1
                        var bestD = 28f * density
                        for (i in screenX.indices) {
                            val d = hypot(screenX[i] - p.x, screenY[i] - p.y)
                            if (d < bestD) { bestD = d; best = i }
                        }
                        onSelect(best.takeIf { it >= 0 })
                    },
                )
            },
    ) {
        if (layout.isEmpty) return@Canvas
        val cx = size.width / 2f
        val cy = size.height * centerY
        val radius = min(size.width, size.height) * 0.40f * zoom
        val t = time.value
        if (selected == null) spin[0] += (t - spin[1]).coerceIn(0f, 0.2f) * 0.07f
        spin[1] = t
        val y0 = yaw + spin[0]
        val cyw = cos(y0); val syw = sin(y0)
        val cp = cos(pitch); val sp = sin(pitch)
        val camera = 3.2f

        val n = layout.nodes.size
        val sx = FloatArray(n); val sy = FloatArray(n); val depth = FloatArray(n); val persp = FloatArray(n)
        fun project(x: Float, y: Float, z: Float, out: (Float, Float, Float, Float) -> Unit) {
            val x1 = x * cyw + z * syw
            val z1 = -x * syw + z * cyw
            val y1 = y * cp - z1 * sp
            val z2 = y * sp + z1 * cp
            val p = camera / (camera - z2)
            out(cx + x1 * radius * p, cy + y1 * radius * p, z2, p)
        }
        layout.nodes.forEachIndexed { i, nd ->
            project(nd.x, nd.y, nd.z) { x, y, z, p -> sx[i] = x; sy[i] = y; depth[i] = z; persp[i] = p }
            screenX[i] = sx[i]; screenY[i] = sy[i]
        }
        fun fog(z: Float) = (0.28f + 0.72f * (z + 1f) / 2f).coerceIn(0.2f, 1f)

        // Orbit rings: the equator and a meridian of the sphere the graph lives in.
        val ringColor = if (glass.dark) MemoryLight else Color(0xFF8C949A)
        for (ring in 0..1) {
            val path = Path()
            for (k in 0..72) {
                val a = k / 72f * 2f * PI.toFloat()
                val (x, y, z) = if (ring == 0) Triple(cos(a) * 1.12f, 0f, sin(a) * 1.12f) else Triple(0f, cos(a) * 1.12f, sin(a) * 1.12f)
                project(x, y, z) { px, py, _, _ -> if (k == 0) path.moveTo(px, py) else path.lineTo(px, py) }
            }
            drawPath(
                path, ringColor.copy(alpha = if (ring == 0) 0.16f else 0.09f),
                style = Stroke(1f, pathEffect = PathEffect.dashPathEffect(floatArrayOf(2f * density, 6f * density))),
            )
        }

        // Links: brightness by weight and depth; the selection's links light up, the rest recede.
        for (l in layout.links) {
            val hot = selected != null && (l.a == selected || l.b == selected)
            val dim = selected != null && !hot
            val base = if (hot) 0.85f else 0.10f + 0.40f * l.w
            val a = base * fog((depth[l.a] + depth[l.b]) / 2f) * (if (dim) 0.35f else 1f)
            val c = if (hot) (if (glass.dark) Color(0xFFEFFFFF) else glass.slab) else communityHue(layout.nodes[l.a].community, glass.dark)
            drawLine(c.copy(alpha = a), Offset(sx[l.a], sy[l.a]), Offset(sx[l.b], sy[l.b]), strokeWidth = (if (hot) 1.6f else 1f) * density)
        }

        // Stars, far to near.
        val order = (0 until n).sortedBy { depth[it] }
        for (i in order) {
            val nd = layout.nodes[i]
            val f = fog(depth[i])
            val dim = selected != null && i != selected && i !in neighbourSet
            val alpha = f * (if (dim) 0.35f else 1f)
            val r = (1.6f + 2.6f * nd.weight) * density * persp[i]
            val filter = filters[if (nd.community in 0..7) nd.community else 8]
            drawGlow(sprite, Offset(sx[i], sy[i]), r * (if (glass.dark) 7.5f else 4f), alpha * (if (glass.dark) 0.9f else 0.35f), filter)
            drawCircle((if (glass.dark) Color(0xFFF2FFFF) else Color(0xFF2A2F34)).copy(alpha = alpha), radius = r, center = Offset(sx[i], sy[i]))
        }
        selected?.let { s ->
            drawCircle(glass.mark, radius = 12f * density * persp[s], center = Offset(sx[s], sy[s]), style = Stroke(1.3f * density))
        }

        // Labels by priority — the selection, its neighbours, then the most-mentioned — each
        // placed only where it doesn't overlap one already drawn, so dense clusters stay legible.
        val maxW = (size.width * 0.4f).toInt().coerceAtLeast(1)
        val placed = ArrayList<androidx.compose.ui.geometry.Rect>()
        val priority = labelled.sortedWith(
            compareByDescending<Int> { it == selected }.thenByDescending { it in neighbourSet }.thenByDescending { layout.nodes[it].weight },
        )
        for (i in priority) {
            val f = fog(depth[i])
            if (f < 0.35f && i != selected) continue
            val m = labelCache.getOrPut(i) {
                measurer.measure(
                    layout.nodes[i].name, labelStyle,
                    maxLines = 1, overflow = TextOverflow.Ellipsis, constraints = Constraints(maxWidth = maxW),
                )
            }
            val gap = 8f * density
            val x = (if (sx[i] > size.width * 0.66f) sx[i] - gap - m.size.width else sx[i] + gap).coerceIn(0f, size.width - m.size.width)
            val y = (sy[i] - m.size.height - 2f * density).coerceAtLeast(0f)
            val rect = androidx.compose.ui.geometry.Rect(x - 3f, y - 1f, x + m.size.width + 3f, y + m.size.height + 1f)
            if (placed.any { it.overlaps(rect) }) continue
            placed += rect
            drawText(m, topLeft = Offset(x, y), alpha = if (i == selected) 1f else f)
        }
    }
}
