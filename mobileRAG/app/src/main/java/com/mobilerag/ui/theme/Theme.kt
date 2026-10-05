package com.mobilerag.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.mobilerag.R

/**
 * Design tokens the stock M3 [MaterialTheme] has no slot for: the glass fills, the rim light,
 * the world light and the accent's two roles. Read them through [AppTheme.glass].
 */
@Immutable
data class GlassTokens(
    val dark: Boolean,
    /** Mode-resolved accent: in the sea start/end/glow are the frost→cyan world light. */
    val accent: AccentSpec,
    /** Fill for a panel sitting on the atmosphere. */
    val panel: Color,
    /** Fill for a panel raised above another panel. */
    val panelHigh: Color,
    /** Fill for the least prominent chrome — chips, input wells. */
    val panelLow: Color,
    /** Top highlight line inside a panel. */
    val gloss: Color,
    /** Bright side of the 1px rim. */
    val rimBright: Color,
    /** Dim side of the 1px rim. */
    val rimDim: Color,
    /** Color of glows (the sea's cyan; a cool shadow in the white room). */
    val glow: Color,
    /** The accent as a small marker (ticks, fading dots). */
    val mark: Color,
    /** The accent as a solid selection slab (white text on it). */
    val slab: Color,
    /** Primary, secondary and dim ink on the atmosphere. */
    val ink: Color,
    val inkMuted: Color,
    val inkDim: Color,
) {
    /** The primary sweep: frost→cyan in the sea, the accent slab in the white room. */
    val accentBrush: Brush get() = Brush.linearGradient(listOf(accent.start, accent.end))

    /** Same sweep at low alpha — for fills that sit behind text. */
    fun accentWash(alpha: Float): Brush = Brush.linearGradient(
        listOf(accent.start.copy(alpha = alpha), accent.end.copy(alpha = alpha)),
    )

    /** The rim: bright at the top-left, fading toward the bottom-right. */
    val rimBrush: Brush get() = Brush.linearGradient(listOf(rimBright, rimDim, rimDim))

    /** The selection bar: solid on the left, fading out to the right (Black Flag's highlight). */
    fun selectionBrush(fadeFrom: Float = 0.58f): Brush {
        val c = if (dark) Color(0xFFECFFFF) else slab
        return Brush.horizontalGradient(
            0f to c.copy(alpha = 0.96f),
            fadeFrom to c.copy(alpha = 0.96f),
            1f to c.copy(alpha = 0f),
        )
    }

    /** Ink on top of [selectionBrush]. */
    val onSelection: Color get() = if (dark) SeaBase else Color.White
}

val LocalGlass = staticCompositionLocalOf<GlassTokens> {
    error("GlassTokens not provided — wrap the content in AppTheme { }")
}

/** Accessor object so call sites read `AppTheme.glass.panel` next to `MaterialTheme.colorScheme`. */
object AppTheme {
    val glass: GlassTokens
        @Composable get() = LocalGlass.current
}

private fun glassTokensFor(dark: Boolean, accent: AccentSpec): GlassTokens =
    if (dark) {
        GlassTokens(
            dark = true,
            accent = accent,
            panel = Color(0xFF092630).copy(alpha = 0.40f),
            panelHigh = Color(0xFF0E3A46).copy(alpha = 0.50f),
            panelLow = Color(0xFF051C24).copy(alpha = 0.45f),
            gloss = Color(0xFFD2FAFC).copy(alpha = 0.16f),
            rimBright = MemoryLight.copy(alpha = 0.34f),
            rimDim = MemoryLight.copy(alpha = 0.12f),
            glow = MemoryLight,
            mark = accent.mark,
            slab = accent.slab,
            ink = SeaInk,
            inkMuted = SeaInkMuted,
            inkDim = SeaInkDim,
        )
    } else {
        GlassTokens(
            dark = false,
            accent = accent.copy(start = accent.slab, end = accent.slab.copy(alpha = 0.82f), glow = accent.slab),
            panel = Color(0xFFFAFBFB).copy(alpha = 0.50f),
            panelHigh = Color(0xFFFFFFFF).copy(alpha = 0.72f),
            panelLow = Color(0xFFFFFFFF).copy(alpha = 0.38f),
            gloss = Color.White.copy(alpha = 0.95f),
            rimBright = Color.White.copy(alpha = 0.95f),
            rimDim = Color(0xFF8C949A).copy(alpha = 0.18f),
            glow = Color(0xFF505C64),
            mark = accent.slab,
            slab = accent.slab,
            ink = FogInk,
            inkMuted = FogInkMuted,
            inkDim = Color(0xFF6A7278),
        )
    }

// ── Type ─────────────────────────────────────────────────────────────────────

/** Display voice: condensed, light-to-semibold, tracked out (Barlow Semi Condensed, OFL). */
val AnimusDisplay = FontFamily(
    Font(R.font.barlow_semicondensed_light, FontWeight.Light),
    Font(R.font.barlow_semicondensed_regular, FontWeight.Normal),
    Font(R.font.barlow_semicondensed_medium, FontWeight.Medium),
    Font(R.font.barlow_semicondensed_semibold, FontWeight.SemiBold),
    Font(R.font.barlow_semicondensed_semibold, FontWeight.Bold),
)

/** Body voice (Barlow, OFL). Japanese falls back to the system CJK font. */
val AnimusBody = FontFamily(
    Font(R.font.barlow_regular, FontWeight.Normal),
    Font(R.font.barlow_medium, FontWeight.Medium),
    Font(R.font.barlow_medium, FontWeight.SemiBold),
    Font(R.font.barlow_medium, FontWeight.Bold),
)

/** Data voice (IBM Plex Mono, OFL). */
val AnimusMono = FontFamily(
    Font(R.font.ibm_plex_mono_regular, FontWeight.Normal),
    Font(R.font.ibm_plex_mono_medium, FontWeight.Medium),
    Font(R.font.ibm_plex_mono_medium, FontWeight.SemiBold),
)

/** Near-square: the menus this follows have hard edges; 2dp only softens the pixel stair. */
private val AppShapes = Shapes(
    extraSmall = RoundedCornerShape(2.dp),
    small = RoundedCornerShape(2.dp),
    medium = RoundedCornerShape(2.dp),
    large = RoundedCornerShape(3.dp),
    extraLarge = RoundedCornerShape(4.dp),
)

private fun appTypography(): Typography {
    val base = Typography()
    fun display(s: TextStyle, w: FontWeight, track: Float) =
        s.copy(fontFamily = AnimusDisplay, fontWeight = w, letterSpacing = track.sp)
    return base.copy(
        displayLarge = display(base.displayLarge, FontWeight.Light, 3f),
        displayMedium = display(base.displayMedium, FontWeight.Light, 2.5f),
        displaySmall = display(base.displaySmall, FontWeight.Light, 2f),
        headlineLarge = display(base.headlineLarge, FontWeight.Light, 1.6f),
        headlineMedium = display(base.headlineMedium, FontWeight.Light, 1.4f),
        headlineSmall = display(base.headlineSmall, FontWeight.Normal, 1.1f),
        titleLarge = display(base.titleLarge, FontWeight.Medium, 0.6f),
        titleMedium = display(base.titleMedium, FontWeight.Medium, 0.5f),
        titleSmall = display(base.titleSmall, FontWeight.Medium, 0.5f),
        bodyLarge = base.bodyLarge.copy(fontFamily = AnimusBody, lineHeight = 25.sp),
        bodyMedium = base.bodyMedium.copy(fontFamily = AnimusBody, lineHeight = 21.sp),
        bodySmall = base.bodySmall.copy(fontFamily = AnimusBody, lineHeight = 17.sp),
        labelLarge = display(base.labelLarge, FontWeight.SemiBold, 1.6f),
        labelMedium = base.labelMedium.copy(fontFamily = AnimusMono, fontWeight = FontWeight.Normal, letterSpacing = 1.6.sp),
        labelSmall = base.labelSmall.copy(fontFamily = AnimusMono, fontWeight = FontWeight.Normal, letterSpacing = 1.8.sp),
    )
}

/** Monospace style for the ARM-E / telemetry readouts that run through every screen. */
val TelemetryStyle: TextStyle = TextStyle(
    fontFamily = AnimusMono,
    fontSize = 11.sp,
    lineHeight = 15.sp,
    letterSpacing = 0.2.sp,
)

@Composable
fun AppTheme(
    themeMode: String,
    accent: String,
    content: @Composable () -> Unit,
) {
    val dark = when (themeMode) {
        "light" -> false
        "dark" -> true
        else -> isSystemInDarkTheme()
    }
    // Wallpaper-derived "dynamic" colour is retired: the atmosphere is the brand, and a
    // wallpaper hue would fight it. Old ids resolve to vermilion.
    val spec = accentById(accent)
    MaterialTheme(
        colorScheme = if (dark) spec.darkScheme else spec.lightScheme,
        typography = appTypography(),
        shapes = AppShapes,
    ) {
        CompositionLocalProvider(LocalGlass provides glassTokensFor(dark, spec)) {
            content()
        }
    }
}
