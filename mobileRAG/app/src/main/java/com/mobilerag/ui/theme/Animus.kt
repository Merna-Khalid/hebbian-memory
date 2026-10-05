package com.mobilerag.ui.theme

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.height
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.graphics.Shadow
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.chrisbanes.haze.HazeInputScale
import dev.chrisbanes.haze.HazeStyle
import dev.chrisbanes.haze.HazeTint
import dev.chrisbanes.haze.hazeEffect

/*
 * The Animus building blocks: frosted glass, rim light, corner ticks, the chromatic title,
 * arc gauges, the day badge, HUD stats. Screens compose these; Glass.kt's older components
 * are restyled on top of them so every screen speaks the same language.
 */

/**
 * Frosted glass: the atmosphere behind the panel, blurred (API 31+; a denser tint below), with
 * the panel tint over it. Clipped to [shape].
 */
@Composable
fun Modifier.frost(shape: Shape = RectangleShape, tint: Color? = null): Modifier {
    val glass = AppTheme.glass
    val haze = LocalHazeState.current
    val fill = tint ?: glass.panel
    return if (haze == null) {
        this.clip(shape).background(fill)
    } else {
        this.clip(shape).hazeEffect(
            state = haze,
            style = HazeStyle(
                backgroundColor = if (glass.dark) SeaBase else FogBase,
                tints = listOf(HazeTint(fill)),
                blurRadius = 18.dp,
                noiseFactor = if (glass.dark) 0.06f else 0.03f,
                fallbackTint = HazeTint(fill.copy(alpha = (fill.alpha + 0.35f).coerceAtMost(0.92f))),
            ),
        ) {
            // Blur a third-size copy: at an 18dp radius the result is indistinguishable, and it
            // is most of the per-frame GPU cost while the atmosphere animates.
            inputScale = HazeInputScale.Fixed(0.33f)
        }
    }
}

/** 1px rim light plus a bright hairline along the top edge (the glass catching the light). */
@Composable
fun Modifier.rimLight(): Modifier {
    val glass = AppTheme.glass
    return this.drawWithContent {
        drawContent()
        val s = 1f
        drawRect(glass.rimBrush, style = Stroke(s))
        drawLine(glass.gloss, Offset(s, s), Offset(size.width - s, s), strokeWidth = s)
    }
}

/**
 * Registration ticks: L-corners top-left and bottom-right in ink, a short accent bar top-right —
 * the one craft detail every frame carries.
 */
@Composable
fun Modifier.cornerTicks(inset: Dp = 7.dp, arm: Dp = 10.dp, accent: Boolean = true): Modifier {
    val glass = AppTheme.glass
    val ink = if (glass.dark) Frost.copy(alpha = 0.85f) else Color(0xFF9C978C)
    val mark = glass.mark
    return this.drawWithContent {
        drawContent()
        val i = inset.toPx()
        val a = arm.toPx()
        val sw = 1.dp.toPx()
        drawLine(ink, Offset(i, i), Offset(i + a, i), sw)
        drawLine(ink, Offset(i, i), Offset(i, i + a), sw)
        drawLine(ink, Offset(size.width - i, size.height - i), Offset(size.width - i - a, size.height - i), sw)
        drawLine(ink, Offset(size.width - i, size.height - i), Offset(size.width - i, size.height - i - a), sw)
        if (accent) {
            val bar = 2.dp.toPx()
            if (glass.dark) {
                drawLine(mark.copy(alpha = 0.35f), Offset(size.width - i - a * 1.6f, i + bar / 2), Offset(size.width - i, i + bar / 2), bar * 4)
            }
            drawLine(mark, Offset(size.width - i - a * 1.6f, i + bar / 2), Offset(size.width - i, i + bar / 2), bar)
        }
    }
}

/** A glowing rule: solid on the left, fading right (under the HUD, under titles). */
@Composable
fun GlowRule(modifier: Modifier = Modifier, width: Dp = 132.dp) {
    val glass = AppTheme.glass
    val c = if (glass.dark) Frost else glass.slab
    Canvas(modifier.width(width).height(6.dp)) {
        val y = size.height / 2
        val brush = Brush.horizontalGradient(0f to c, 0.55f to c.copy(alpha = 0.9f), 1f to c.copy(alpha = 0f))
        if (glass.dark) drawLine(brush, Offset(0f, y), Offset(size.width, y), strokeWidth = 6.dp.toPx(), alpha = 0.18f)
        drawLine(brush, Offset(0f, y), Offset(size.width, y), strokeWidth = 2.dp.toPx())
    }
}

/**
 * Title with a chromatic split (red left, cyan right) and a cyan glow in the sea; a clean
 * charcoal title in the white room. [flicker] adds the occasional signal drop.
 */
@Composable
fun ChromaText(
    text: String,
    style: TextStyle,
    modifier: Modifier = Modifier,
    color: Color? = null,
    flicker: Boolean = false,
) {
    val glass = AppTheme.glass
    val ink = color ?: glass.ink
    val time = LocalAmbientTime.current
    if (!glass.dark) {
        Text(text, modifier, style = style, color = ink)
        return
    }
    Box(modifier.graphicsLayer { alpha = if (flicker) flickerAlpha(time.value) else 1f }) {
        Text(text, Modifier.offset(x = (-1.5).dp), style = style, color = Color(0xFFFF4646).copy(alpha = 0.45f))
        Text(text, Modifier.offset(x = 1.5.dp), style = style, color = Color(0xFF46E6FF).copy(alpha = 0.50f))
        Text(
            text,
            style = style.copy(shadow = Shadow(MemoryLight.copy(alpha = 0.45f), Offset.Zero, blurRadius = 22f)),
            color = ink,
        )
    }
}

/** The mockup's `flicker`: steady, then two quick drops every 9 s. */
private fun flickerAlpha(t: Float): Float {
    val p = (t % 9f) / 9f
    return when {
        p in 0.94f..0.96f -> 0.55f
        p in 0.97f..0.985f -> 0.70f
        else -> 1f
    }
}

/**
 * Black Flag's circular percentage: a thin arc with a glow, value in the middle, label below.
 * [fraction] is 0..1; [value] is what's printed (so callers choose "95%" or "12").
 */
@Composable
fun ArcGauge(
    fraction: Float,
    value: String,
    label: String,
    modifier: Modifier = Modifier,
    diameter: Dp = 62.dp,
) {
    val glass = AppTheme.glass
    val arc = if (glass.dark) Frost else glass.slab
    val track = if (glass.dark) MemoryLight.copy(alpha = 0.16f) else Color(0xFF8C949A).copy(alpha = 0.3f)
    Column(
        modifier.semantics { contentDescription = "$label $value" },
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Box(Modifier.size(diameter), contentAlignment = Alignment.Center) {
            Canvas(Modifier.size(diameter)) {
                val sw = 2.dp.toPx()
                val inset = 4.dp.toPx()
                val arcSize = Size(size.width - 2 * inset, size.height - 2 * inset)
                val tl = Offset(inset, inset)
                drawArc(track, 0f, 360f, false, tl, arcSize, style = Stroke(sw))
                val sweep = 360f * fraction.coerceIn(0f, 1f)
                if (sweep > 0f) {
                    if (glass.dark) {
                        drawArc(MemoryLight.copy(alpha = 0.30f), -90f, sweep, false, tl, arcSize, style = Stroke(sw * 4, cap = StrokeCap.Butt))
                    }
                    drawArc(arc, -90f, sweep, false, tl, arcSize, style = Stroke(sw * 1.2f, cap = StrokeCap.Butt))
                }
            }
            Text(value, style = TextStyle(fontFamily = AnimusDisplay, fontSize = 16.sp, fontWeight = FontWeight.Normal), color = glass.ink)
        }
        Text(label, style = TextStyle(fontFamily = AnimusBody, fontSize = 12.sp), color = glass.inkMuted)
    }
}

/** "DAY ◆12": the entity's age, Syndicate's level diamond. */
@Composable
fun DayBadge(day: Int, modifier: Modifier = Modifier) {
    val glass = AppTheme.glass
    Row(
        modifier.semantics { contentDescription = "Day $day" },
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(
            "DAY",
            style = TextStyle(fontFamily = AnimusDisplay, fontWeight = FontWeight.Light, fontSize = 14.sp, letterSpacing = 3.sp),
            color = if (glass.dark) Color(0xFFBFE9EC) else FogInkMuted,
        )
        Box(Modifier.size(28.dp), contentAlignment = Alignment.Center) {
            Box(
                Modifier
                    .size(19.dp)
                    .rotate(45f)
                    .drawBehind {
                        if (glass.dark) drawCircle(MemoryLight.copy(alpha = 0.35f), radius = size.minDimension * 0.95f)
                    }
                    .background(if (glass.dark) Frost else Color.White)
                    .then(if (glass.dark) Modifier else Modifier.border(1.dp, Color(0xFF8C949A).copy(alpha = 0.6f))),
            )
            Text(
                day.toString(),
                style = TextStyle(fontFamily = AnimusDisplay, fontWeight = FontWeight.SemiBold, fontSize = 12.sp),
                color = if (glass.dark) SeaBase else FogInk,
            )
        }
    }
}

enum class HudIcon { Concepts, Links, Fading, Documents }

/** A glyph + number in the HUD strip — no boxes, Black Flag's resource counters. */
@Composable
fun HudStat(icon: HudIcon, value: String, description: String, modifier: Modifier = Modifier) {
    val glass = AppTheme.glass
    val c = if (glass.dark) Color(0xFFBFF7F7) else FogInk
    Row(
        modifier.semantics { contentDescription = "$value $description" },
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(5.dp),
    ) {
        Canvas(Modifier.size(13.dp)) {
            val sw = 1.dp.toPx()
            if (glass.dark) drawCircle(MemoryLight.copy(alpha = 0.22f), radius = size.minDimension * 0.7f)
            when (icon) {
                HudIcon.Concepts -> {
                    drawCircle(c, radius = size.minDimension * 0.19f)
                    drawCircle(c, radius = size.minDimension * 0.43f, style = Stroke(sw))
                }
                HudIcon.Links -> {
                    val a = Offset(size.width * 0.18f, size.height * 0.78f)
                    val b = Offset(size.width * 0.82f, size.height * 0.22f)
                    drawLine(c, a, b, sw)
                    drawCircle(c, radius = size.minDimension * 0.15f, center = a)
                    drawCircle(c, radius = size.minDimension * 0.15f, center = b)
                }
                HudIcon.Fading -> drawCircle(
                    c, radius = size.minDimension * 0.43f,
                    style = Stroke(sw, pathEffect = androidx.compose.ui.graphics.PathEffect.dashPathEffect(floatArrayOf(3f * density, 2f * density))),
                )
                HudIcon.Documents -> {
                    drawRect(c, Offset(size.width * 0.12f, 0f), Size(size.width * 0.76f, size.height), style = Stroke(sw))
                    drawLine(c, Offset(size.width * 0.3f, size.height * 0.35f), Offset(size.width * 0.7f, size.height * 0.35f), sw)
                    drawLine(c, Offset(size.width * 0.3f, size.height * 0.6f), Offset(size.width * 0.7f, size.height * 0.6f), sw)
                }
            }
        }
        Text(value, style = TelemetryStyle.copy(fontSize = 12.sp), color = if (glass.dark) Color(0xFFD8F6F7) else FogInk)
    }
}

/** A horizontal light sweeping down a panel every [period] seconds (the Animus scan). */
@Composable
fun Modifier.scanSweep(period: Float = 7f): Modifier {
    val glass = AppTheme.glass
    if (!glass.dark) return this
    val time = LocalAmbientTime.current
    return this.drawWithContent {
        drawContent()
        val p = (time.value % period) / period
        val a = when {
            p < 0.08f -> p / 0.08f
            p > 0.7f -> (1f - p) / 0.3f * 0.6f
            else -> 1f - (p - 0.08f) / 0.62f * 0.4f
        }
        val y = size.height * p
        val brush = Brush.horizontalGradient(
            listOf(MemoryLight.copy(alpha = 0f), Color(0xFFBEFAFA).copy(alpha = 0.9f * a), MemoryLight.copy(alpha = 0f)),
        )
        drawLine(brush, Offset(0f, y), Offset(size.width, y), strokeWidth = 8.dp.toPx(), alpha = 0.18f)
        drawLine(brush, Offset(0f, y), Offset(size.width, y), strokeWidth = 1.dp.toPx())
    }
}

/** A frosted Animus panel: frost + rim + (optionally) ticks and the scan sweep. */
@Composable
fun AnimusPanel(
    modifier: Modifier = Modifier,
    ticks: Boolean = true,
    scan: Boolean = false,
    content: @Composable () -> Unit,
) {
    val base = modifier.frost().rimLight()
    Box(
        base
            .then(if (scan) Modifier.scanSweep() else Modifier)
            .then(if (ticks) Modifier.cornerTicks() else Modifier),
    ) { content() }
}

enum class Glyph { Entity, Constellation, Options }

/**
 * The tab-bar glyphs, drawn to match the HUD: the entity (a ring holding a diamond), memory
 * (a small constellation), options (three sliders). [lit] adds the world-light glow.
 */
@Composable
fun AnimusGlyph(glyph: Glyph, tint: Color, lit: Boolean, modifier: Modifier = Modifier.size(22.dp)) {
    val glass = AppTheme.glass
    Canvas(modifier) {
        val sw = 1.4.dp.toPx()
        val w = size.width
        val h = size.height
        if (lit && glass.dark) drawCircle(MemoryLight.copy(alpha = 0.22f), radius = size.minDimension * 0.62f)
        when (glyph) {
            Glyph.Entity -> {
                drawCircle(tint, radius = w * 0.42f, style = Stroke(sw))
                val d = w * 0.17f
                val path = androidx.compose.ui.graphics.Path().apply {
                    moveTo(w / 2, h / 2 - d); lineTo(w / 2 + d, h / 2); lineTo(w / 2, h / 2 + d); lineTo(w / 2 - d, h / 2); close()
                }
                drawPath(path, tint)
            }
            Glyph.Constellation -> {
                val p = listOf(Offset(w * 0.16f, h * 0.72f), Offset(w * 0.44f, h * 0.22f), Offset(w * 0.84f, h * 0.44f), Offset(w * 0.6f, h * 0.84f))
                drawLine(tint, p[0], p[1], sw); drawLine(tint, p[1], p[2], sw); drawLine(tint, p[1], p[3], sw)
                drawLine(tint.copy(alpha = tint.alpha * 0.5f), p[2], p[3], sw)
                p.forEachIndexed { i, o -> drawCircle(tint, radius = if (i == 1) w * 0.11f else w * 0.075f, center = o) }
            }
            Glyph.Options -> {
                listOf(0.25f to 0.30f, 0.5f to 0.68f, 0.75f to 0.42f).forEach { (y, knob) ->
                    drawLine(tint, Offset(w * 0.08f, h * y), Offset(w * 0.92f, h * y), sw)
                    val k = w * 0.12f
                    drawRect(tint, topLeft = Offset(w * knob - k / 2, h * y - k / 2), size = Size(k, k))
                }
            }
        }
    }
}
