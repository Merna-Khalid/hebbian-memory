package com.mobilerag.ui.theme

import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.ui.graphics.Color

/**
 * "Animus" palette — two worlds, one accent.
 *
 * Dark is the **sea**: a deep teal-black lit by cyan "memory light" (the awake entity).
 * Light is the **white room**: grey-white fog, charcoal ink, the accent as a solid slab.
 * The accent is used sparingly — markers, one tick per frame, the fog's selection slab — the
 * world light (cyan in the sea) does the rest. See docs/ui-animus.md.
 */

// ── The sea (dark) ───────────────────────────────────────────────────────────
val SeaDeep = Color(0xFF020B10)
val SeaBase = Color(0xFF03121A)
val SeaMid = Color(0xFF04161E)
val SeaGlowTeal = Color(0xFF0E4A57)
val SeaGlowDeep = Color(0xFF0B3B47)
/** The world light: nodes, rims, gauges, the selection bar's glow. */
val MemoryLight = Color(0xFF6FF0F0)
/** Near-white with a cyan cast — selection bars and bright type. */
val Frost = Color(0xFFEFFFFF)
val SeaInk = Color(0xFFF2FBFC)
val SeaInkMuted = Color(0xFFA9CFD4)
val SeaInkDim = Color(0xFF6FA7AE)

// ── The white room (light) ───────────────────────────────────────────────────
val FogSky = Color(0xFFAEB8BE)
val FogMid = Color(0xFFCBD2D6)
val FogBase = Color(0xFFE6E9EB)
val FogFloor = Color(0xFFF6F7F7)
val FogInk = Color(0xFF22272B)
val FogInkMuted = Color(0xFF4A5157)

/**
 * A named accent. [mark] is the small accent on the dark sea (ticks, fading markers);
 * [slab] is the solid selection fill in the white room, dark enough for white text (≥ 4.5:1).
 * [start]/[end]/[glow] drive the legacy gradient API ([GlassTokens.accentBrush]) and are
 * resolved per mode by the theme.
 */
data class AccentSpec(
    val id: String,
    val label: String,
    val start: Color,
    val end: Color,
    val glow: Color,
    val darkScheme: androidx.compose.material3.ColorScheme,
    val lightScheme: androidx.compose.material3.ColorScheme,
    val mark: Color,
    val slab: Color,
)

private fun seaScheme(mark: Color) = darkColorScheme(
    primary = Color(0xFF7FF3F3),
    onPrimary = SeaBase,
    primaryContainer = SeaGlowTeal,
    onPrimaryContainer = Color(0xFFD8FBFB),
    secondary = mark,
    onSecondary = SeaBase,
    secondaryContainer = Color(0xFF0A2A34),
    onSecondaryContainer = SeaInk,
    tertiary = mark,
    onTertiary = SeaBase,
    tertiaryContainer = Color(0xFF0A2A34),
    onTertiaryContainer = SeaInk,
    background = SeaBase,
    onBackground = SeaInk,
    surface = Color(0xFF062029),
    onSurface = SeaInk,
    surfaceVariant = Color(0xFF0B2C36),
    onSurfaceVariant = SeaInkMuted,
    surfaceContainerLowest = Color(0xFF031A22),
    surfaceContainerLow = Color(0xFF052029),
    surfaceContainer = Color(0xFF072630),
    surfaceContainerHigh = Color(0xFF0A2D38),
    surfaceContainerHighest = Color(0xFF0D3440),
    outline = Color(0xFF3E6B73),
    outlineVariant = Color(0xFF1B3C44),
    error = Color(0xFFFF7A6A),
    onError = SeaBase,
    inverseSurface = Frost,
    inverseOnSurface = SeaBase,
    inversePrimary = SeaGlowTeal,
)

private fun fogScheme(slab: Color) = lightColorScheme(
    primary = slab,
    onPrimary = Color.White,
    primaryContainer = Color(0xFFF1DCD6),
    onPrimaryContainer = FogInk,
    secondary = Color(0xFF2A2F34),
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFDDE2E5),
    onSecondaryContainer = FogInk,
    tertiary = slab,
    onTertiary = Color.White,
    tertiaryContainer = Color(0xFFDDE2E5),
    onTertiaryContainer = FogInk,
    background = FogBase,
    onBackground = FogInk,
    surface = Color(0xFFF4F6F7),
    onSurface = FogInk,
    surfaceVariant = Color(0xFFDDE2E5),
    onSurfaceVariant = FogInkMuted,
    surfaceContainerLowest = Color(0xFFFFFFFF),
    surfaceContainerLow = Color(0xFFF6F7F8),
    surfaceContainer = Color(0xFFEEF1F2),
    surfaceContainerHigh = Color(0xFFE8EBED),
    surfaceContainerHighest = Color(0xFFE2E6E8),
    outline = Color(0xFF8C949A),
    outlineVariant = Color(0xFFC9CFD3),
    error = Color(0xFFB3261E),
    onError = Color.White,
)

private fun accent(id: String, label: String, mark: Color, slab: Color) = AccentSpec(
    id = id,
    label = label,
    start = Frost,
    end = MemoryLight,
    glow = MemoryLight,
    darkScheme = seaScheme(mark),
    lightScheme = fogScheme(slab),
    mark = mark,
    slab = slab,
)

// ── Registry ─────────────────────────────────────────────────────────────────

val VermilionAccent = accent("vermilion", "Vermilion", mark = Color(0xFFFF6A4A), slab = Color(0xFFB42A17))
val GoldAccent = accent("gold", "Gold", mark = Color(0xFFF2C46D), slab = Color(0xFF8C6210))
val CyanAccent = accent("cyan", "Cyan", mark = MemoryLight, slab = Color(0xFF1A7680))

val AllAccents = listOf(VermilionAccent, GoldAccent, CyanAccent)

/** Unknown ids — including the retired indigo/sakura/matcha/chrome/dynamic — fall back to vermilion. */
fun accentById(id: String): AccentSpec = AllAccents.firstOrNull { it.id == id } ?: VermilionAccent
