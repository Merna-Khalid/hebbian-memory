package com.mobilerag.ui.theme

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.PowerManager
import android.provider.Settings
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.State
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawWithCache
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.ImageShader
import androidx.compose.ui.graphics.ShaderBrush
import androidx.compose.ui.graphics.TileMode
import androidx.compose.ui.graphics.Paint
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.RadialGradientShader
import androidx.compose.ui.graphics.drawscope.CanvasDrawScope
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import dev.chrisbanes.haze.HazeState
import dev.chrisbanes.haze.hazeSource
import dev.chrisbanes.haze.rememberHazeState
import kotlin.math.PI
import kotlin.math.sin

/**
 * The Animus atmosphere every screen floats in, and the blur source for frosted panels.
 *
 * Dark — the **sea**: teal-black with cyan light breaking through, drifting motes of memory
 * light, a perspective grid floor receding into the dark, scanlines, vignette.
 * Light — the **white room**: grey-white fog, faint towers, a floor of cubes, rising motes.
 *
 * Layers that never change are recorded once in their own graphics layer; only the motion
 * layer redraws per frame, and it stops entirely when [rememberAmbientMotion] says so
 * (backgrounded, battery saver, or animations turned off in system settings).
 */
@Composable
fun AnimusBackground(
    modifier: Modifier = Modifier,
    content: @Composable BoxScope.() -> Unit,
) {
    val glass = AppTheme.glass
    val haze = rememberHazeState()
    // One clock for everything ambient (atmosphere, constellation, scan, flicker), so all of it
    // updates in the same frame — separate clocks drift apart and multiply the frame rate.
    val time = rememberAmbientTime(rememberAmbientMotion() && !AmbientGate.scrolling)
    CompositionLocalProvider(LocalAmbientTime provides time) {
        Box(modifier.fillMaxSize()) {
            Box(Modifier.fillMaxSize().hazeSource(haze)) {
                if (glass.dark) SeaAtmosphere() else FogAtmosphere()
            }
            CompositionLocalProvider(LocalHazeState provides haze) {
                Box(Modifier.fillMaxSize(), content = content)
            }
        }
    }
}

/**
 * The shared ambient clock (seconds). Read it inside draw or graphics-layer lambdas so a tick
 * redraws without recomposing. Outside [AnimusBackground] it stands still at 0.
 */
val LocalAmbientTime = staticCompositionLocalOf<State<Float>> { mutableFloatStateOf(0f) }

/**
 * Scrolling holds the ambient clock still: a moving frosted panel must be re-blurred every frame
 * anyway, and an animating backdrop under it doubles that work. Nobody sees motes pause for the
 * half second a list is in motion.
 */
object AmbientGate {
    var scrolling by mutableStateOf(false)
}

/** Declare that this screen's list is scrolling (e.g. `listState.isScrollInProgress`). */
@Composable
fun PauseAmbientWhile(active: Boolean) {
    LaunchedEffect(active) { AmbientGate.scrolling = active }
    DisposableEffect(Unit) { onDispose { AmbientGate.scrolling = false } }
}

/** The atmosphere's blur source; null outside [AnimusBackground] (panels then fall back to a tint). */
val LocalHazeState = staticCompositionLocalOf<HazeState?> { null }

/**
 * Whether ambient animation should run: the screen is resumed, battery saver is off, and the
 * user hasn't turned animations off (Developer options / accessibility "Remove animations").
 */
@Composable
fun rememberAmbientMotion(): Boolean {
    val context = LocalContext.current
    val lifecycleState by LocalLifecycleOwner.current.lifecycle.currentStateFlow.collectAsState()
    val power = remember(context) { context.getSystemService(PowerManager::class.java) }
    var powerSave by remember { mutableStateOf(power?.isPowerSaveMode == true) }
    DisposableEffect(context) {
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(c: Context?, i: Intent?) {
                powerSave = power?.isPowerSaveMode == true
            }
        }
        context.registerReceiver(receiver, IntentFilter(PowerManager.ACTION_POWER_SAVE_MODE_CHANGED))
        onDispose { context.unregisterReceiver(receiver) }
    }
    val resumed = lifecycleState.isAtLeast(Lifecycle.State.RESUMED)
    // Re-read on each resume: the setting can change while we're in the background.
    val animatorScale = remember(resumed) {
        Settings.Global.getFloat(context.contentResolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f)
    }
    return resumed && !powerSave && animatorScale > 0f
}

/**
 * Seconds of ambient time; advances only while [running] (pausing keeps the last frame).
 * Published at most ~30 times a second: the motion is slow drift, and every published tick
 * redraws the atmosphere and re-blurs the frosted panels (≈ 11–13 ms of GPU per frame on the
 * RedMagic at full rate), so halving the rate halves the battery cost with no visible change.
 */
@Composable
fun rememberAmbientTime(running: Boolean): State<Float> {
    val t = remember { mutableFloatStateOf(0f) }
    LaunchedEffect(running) {
        if (!running) return@LaunchedEffect
        var last = -1L
        var pending = 0f
        while (true) {
            withFrameNanos { now ->
                if (last >= 0) pending += ((now - last) / 1e9f).coerceAtMost(0.1f)
                last = now
                if (pending >= AMBIENT_FRAME_S) {
                    t.floatValue += pending
                    pending = 0f
                }
            }
        }
    }
    return t
}

// Slightly under 1/30 s so a 90/120 Hz display publishes every 3rd/4th vsync (≈ 30 fps), not every 4th/5th.
private const val AMBIENT_FRAME_S = 0.028f

// ── Shared: the glow sprite and seeded motes ────────────────────────────────

/** A white radial glow (hot core, soft halo), tinted per draw — one bitmap, no per-frame shaders. */
@Composable
internal fun rememberGlowSprite(): ImageBitmap = remember {
    val px = 96
    val bmp = ImageBitmap(px, px)
    val canvas = androidx.compose.ui.graphics.Canvas(bmp)
    val paint = Paint().apply {
        shader = RadialGradientShader(
            center = Offset(px / 2f, px / 2f),
            radius = px / 2f,
            colors = listOf(Color.White, Color.White.copy(alpha = 0.85f), Color.White.copy(alpha = 0.28f), Color.Transparent),
            colorStops = listOf(0f, 0.14f, 0.36f, 1f),
        )
    }
    canvas.drawCircle(Offset(px / 2f, px / 2f), px / 2f, paint)
    bmp
}

internal fun DrawScope.drawGlow(sprite: ImageBitmap, center: Offset, diameter: Float, alpha: Float, filter: ColorFilter) {
    if (alpha <= 0.003f || diameter < 1f) return
    val d = diameter.toInt().coerceAtLeast(2)
    drawImage(
        image = sprite,
        dstOffset = IntOffset((center.x - d / 2f).toInt(), (center.y - d / 2f).toInt()),
        dstSize = IntSize(d, d),
        alpha = alpha.coerceIn(0f, 1f),
        colorFilter = filter,
    )
}

private class Mote(
    val x: Float, val y: Float,      // 0..1 of the screen
    val sizeDp: Float, val alpha: Float,
    val period: Float, val phase: Float,
    val big: Boolean,
)

private fun motes(seed: Int, n: Int, bigEvery: Int): List<Mote> {
    val r = java.util.Random(seed.toLong())
    return List(n) { i ->
        val big = bigEvery > 0 && i % bigEvery == 0
        Mote(
            x = r.nextFloat(), y = r.nextFloat(),
            sizeDp = if (big) 5f + r.nextFloat() * 6f else 1f + r.nextFloat() * 2.2f,
            alpha = if (big) 0.16f + r.nextFloat() * 0.14f else 0.35f + r.nextFloat() * 0.55f,
            period = 9f + r.nextFloat() * 14f, phase = r.nextFloat(),
            big = big,
        )
    }
}

/** Drift up and a little sideways, fading in and out over one period (the mockup's `drift`). */
private fun DrawScope.drawMotes(
    list: List<Mote>, t: Float, sprite: ImageBitmap, filter: ColorFilter,
    dxDp: Float, dyDp: Float, glowScale: Float,
) {
    val w = size.width
    val h = size.height
    for (m in list) {
        val p = ((t / m.period) + m.phase) % 1f
        val env = when {
            p < 0.18f -> p / 0.18f
            p > 0.82f -> (1f - p) / 0.18f
            else -> 1f
        }
        val c = Offset(m.x * w + dxDp.dp2px(this) * p, m.y * h + dyDp.dp2px(this) * p)
        drawGlow(sprite, c, m.sizeDp.dp2px(this) * glowScale, m.alpha * env, filter)
    }
}

private fun Float.dp2px(d: DrawScope) = this * d.density

// ── The sea ─────────────────────────────────────────────────────────────────

@Composable
private fun SeaAtmosphere() {
    val time by LocalAmbientTime.current
    val sprite = rememberGlowSprite()
    val moteFilter = remember { ColorFilter.tint(Color(0xFFBEFAFA), BlendMode.Modulate) }
    val list = remember { motes(seed = 7, n = 70, bigEvery = 14) }

    // Static, baked once into one bitmap: the water, its light, scanlines (a 1×3 px tile) and
    // the vignette. As separate layers these were five full-screen fills per frame (~5 ms of
    // GPU on the RedMagic); baked they are a single texture draw.
    val scanTile = remember {
        ImageBitmap(1, 3).also { bmp ->
            androidx.compose.ui.graphics.Canvas(bmp).drawRect(0f, 0f, 1f, 1f, Paint().apply { color = Color.White.copy(alpha = 0.028f) })
        }
    }
    Box(
        Modifier.fillMaxSize().baked {
            drawRect(Brush.verticalGradient(listOf(SeaMid, SeaBase, SeaDeep)))
            drawRect(
                Brush.radialGradient(
                    listOf(SeaGlowTeal, SeaGlowTeal.copy(alpha = 0f)),
                    center = Offset(size.width * 0.85f, size.height * 0.08f),
                    radius = size.maxDimension * 0.62f,
                ),
            )
            drawRect(
                Brush.radialGradient(
                    listOf(SeaGlowDeep, SeaGlowDeep.copy(alpha = 0f)),
                    center = Offset(0f, size.height * 0.78f),
                    radius = size.maxDimension * 0.5f,
                ),
            )
            drawRect(ShaderBrush(ImageShader(scanTile, TileMode.Repeated, TileMode.Repeated)))
            drawRect(
                Brush.radialGradient(
                    0.55f to Color.Transparent,
                    1f to Color(0xFF00060A).copy(alpha = 0.55f),
                    center = Offset(size.width / 2f, size.height * 0.45f),
                    radius = size.maxDimension * 0.75f,
                ),
            )
        },
    )

    // Motion: drifting light, motes, and the floor slowly coming toward us.
    Canvas(Modifier.fillMaxSize().graphicsLayer()) {
        val t = time
        val c = Offset(size.width * (0.05f + 0.06f * sin(t / 23f)), size.height * (0.28f + 0.03f * sin(t / 31f)))
        drawCircle(
            Brush.radialGradient(listOf(MemoryLight.copy(alpha = 0.10f), Color.Transparent), center = c, radius = size.width * 0.62f),
            radius = size.width * 0.62f,
            center = c,
        )
        drawGridFloor(t)
        drawMotes(list, t, sprite, moteFilter, dxDp = 14f, dyDp = -70f, glowScale = 5f)
    }
}

/**
 * Paints [block] once into a bitmap the size of the layer (again only on resize); every frame
 * after that is one texture draw. For layers that never animate.
 */
private fun Modifier.baked(block: DrawScope.() -> Unit): Modifier = this.drawWithCache {
    val bmp = ImageBitmap(size.width.toInt().coerceAtLeast(1), size.height.toInt().coerceAtLeast(1))
    CanvasDrawScope().draw(this, layoutDirection, androidx.compose.ui.graphics.Canvas(bmp), size, block)
    onDrawBehind { drawImage(bmp) }
}

/** Perspective grid in the lower third, receding to a horizon, slowly advancing. */
private fun DrawScope.drawGridFloor(t: Float) {
    val w = size.width
    val h = size.height
    val horizon = h * 0.70f
    val cx = w / 2f
    val depth = h - horizon
    val line = MemoryLight
    val fade = Brush.verticalGradient(
        0f to line.copy(alpha = 0f),
        0.30f to line.copy(alpha = 0.20f),
        0.70f to line.copy(alpha = 0.20f),
        1f to line.copy(alpha = 0.04f),
        startY = horizon,
        endY = h,
    )
    // Rows: z from near (1) to far; y = horizon + depth / z. The phase walks z toward us.
    val phase = (t * 0.09f) % 1f
    for (k in 0 until 16) {
        val z = k + 1f - phase
        if (z < 0.6f) continue
        val y = horizon + depth * 1.15f / z
        if (y > h) continue
        drawLine(fade, Offset(0f, y), Offset(w, y), strokeWidth = 1f)
    }
    // Columns converge on the vanishing point.
    val spacing = w * 0.16f
    for (i in -9..9) {
        val bottomX = cx + i * spacing * 2.2f
        val topX = cx + i * spacing * 0.18f
        drawLine(fade, Offset(topX, horizon), Offset(bottomX, h * 1.02f), strokeWidth = 1f)
    }
}

// ── The white room ──────────────────────────────────────────────────────────

@Composable
private fun FogAtmosphere() {
    val time by LocalAmbientTime.current
    val sprite = rememberGlowSprite()
    val moteFilter = remember { ColorFilter.tint(Color.White, BlendMode.Modulate) }
    val list = remember { motes(seed = 5, n = 26, bigEvery = 0) }

    // Static, baked once: sky, towers in the haze, the cube floor, and fog over its far edge.
    Box(
        Modifier.fillMaxSize().baked {
            drawRect(
                Brush.verticalGradient(
                    0f to FogSky, 0.26f to FogMid, 0.50f to FogBase, 0.70f to Color(0xFFEFF1F2), 1f to FogFloor,
                ),
            )
            fun tower(x: Float, top: Float, wFrac: Float, a: Float) {
                drawRect(
                    Brush.verticalGradient(
                        listOf(Color(0xFF78848C).copy(alpha = 0f), Color(0xFF78848C).copy(alpha = a), Color(0xFF78848C).copy(alpha = 0f)),
                        startY = size.height * top, endY = size.height * 0.62f,
                    ),
                    topLeft = Offset(size.width * x, size.height * top),
                    size = Size(size.width * wFrac, size.height * (0.62f - top)),
                )
            }
            tower(0.46f, 0.08f, 0.18f, 0.20f)
            tower(0.70f, 0.14f, 0.10f, 0.15f)
            tower(0.18f, 0.20f, 0.07f, 0.10f)
            for (tile in cubeFloor(size)) {
                drawPath(tile.top, tile.shade)
                tile.front?.let { drawPath(it, tile.shade.copy(red = tile.shade.red * 0.86f, green = tile.shade.green * 0.87f, blue = tile.shade.blue * 0.88f)) }
                drawLine(Color.White.copy(alpha = 0.9f), tile.farEdge.first, tile.farEdge.second, strokeWidth = 1.2f)
            }
            drawRect(
                Brush.verticalGradient(
                    0f to Color(0xFFEFF1F2), 0.5f to Color(0xFFEFF1F2).copy(alpha = 0.85f), 1f to Color(0xFFEFF1F2).copy(alpha = 0f),
                    startY = size.height * 0.60f, endY = size.height * 0.80f,
                ),
            )
        },
    )

    // Motion: haze banks drifting, motes rising.
    Canvas(Modifier.fillMaxSize().graphicsLayer()) {
        val t = time
        fun bank(cx: Float, cy: Float, rx: Float, a: Float) {
            drawCircle(
                Brush.radialGradient(listOf(Color.White.copy(alpha = a), Color.Transparent), center = Offset(cx, cy), radius = rx),
                radius = rx, center = Offset(cx, cy),
            )
        }
        val drift = 26f * density
        bank(size.width * 0.20f + drift * sin(t * 2f * PI.toFloat() / 18f), size.height * 0.16f, size.width * 0.70f, 0.55f)
        bank(size.width * 0.85f + drift * sin(t * 2f * PI.toFloat() / 22f + 1.6f), size.height * 0.40f, size.width * 0.75f, 0.60f)
        drawMotes(list, t, sprite, moteFilter, dxDp = -8f, dyDp = -50f, glowScale = 3f)
    }
}

private class Tile(val top: Path, val front: Path?, val shade: Color, val farEdge: Pair<Offset, Offset>)

/** A floor of square tiles in perspective, some raised a little (drawn with a front face). */
private fun cubeFloor(size: Size): List<Tile> {
    val r = java.util.Random(5)
    val w = size.width
    val h = size.height
    val horizon = h * 0.60f
    val cx = w / 2f
    val f = (h - horizon) * 1.3f     // focal scale
    val tileW = 1f
    fun project(x: Float, z: Float): Offset = Offset(cx + x * f / z * 0.55f, horizon + f / z * 0.55f)
    val out = ArrayList<Tile>()
    // Far rows first so near tiles overdraw them.
    for (row in 14 downTo 0) {
        val z0 = 1.0f + row * 0.55f
        val z1 = z0 + 0.52f
        for (col in -9..8) {
            val x0 = col * tileW + 0.03f
            val x1 = x0 + tileW - 0.06f
            val a = project(x0, z1); val b = project(x1, z1)
            val c = project(x1, z0); val d = project(x0, z0)
            if (c.y < horizon || d.x > w * 1.2f || c.x < -w * 0.2f) continue
            val raised = r.nextFloat() < 0.12f
            val lift = if (raised) (c.y - a.y) * 0.45f else 0f
            val top = Path().apply {
                moveTo(a.x, a.y - lift); lineTo(b.x, b.y - lift); lineTo(c.x, c.y - lift); lineTo(d.x, d.y - lift); close()
            }
            val front = if (raised) Path().apply {
                moveTo(d.x, d.y - lift); lineTo(c.x, c.y - lift); lineTo(c.x, c.y); lineTo(d.x, d.y); close()
            } else null
            val l = (226 + r.nextInt(26)) / 255f
            out += Tile(top, front, Color(l, l + 0.004f, l + 0.008f), Offset(a.x, a.y - lift) to Offset(b.x, b.y - lift))
        }
    }
    return out
}
