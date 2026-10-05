package com.mobilerag.ui.avatar

import android.graphics.BitmapFactory
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.ui.Alignment
import androidx.compose.ui.BiasAlignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.scale
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.unit.times
import com.mobilerag.ui.theme.AppTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * A chat participant's visual identity.
 *
 * [assetPath] is optional by design: drop a PNG at that path under `assets/` and it is used;
 * leave it absent and [glyph] renders inside a procedural neon orb instead. That keeps the app
 * building and looking finished before any artwork exists.
 *
 * [focusX]/[focusY] are the point in the source image (0..1, from the top-left) that should land
 * in the middle of the circle — portraits almost never have the face at the geometric center,
 * and a plain center-crop decapitates them.
 */
@Immutable
data class Persona(
    val id: String,
    val displayName: String,
    val subtitle: String,
    val glyph: String,
    val assetPath: String?,
    val focusX: Float = 0.5f,
    val focusY: Float = 0.5f,
    val zoom: Float = 1f,
    /** Draw [glyph] as a carved hanko seal instead of the neon orb (no artwork needed). */
    val seal: Boolean = false,
) {
    companion object {
        /**
         * Takemura — the Japanese tutor. His mark is a vermilion hanko seal carved with 竹
         * ("bamboo", the first character of 竹村), drawn in code: original, no third-party art.
         */
        val Tutor = Persona(
            id = "tutor",
            displayName = "Takemura",
            subtitle = "日本語 tutor",
            glyph = "竹",
            assetPath = null,
            seal = true,
        )

        /** Khepri — the personal/RAG assistant. Falls back to the ankh until art is supplied. */
        val Personal = Persona(
            id = "personal",
            displayName = "Khepri",
            subtitle = "second brain",
            glyph = "☥",
            assetPath = "avatars/personal.png",
            focusX = 0.5f,
            focusY = 0.38f,
            zoom = 1.12f,
        )
    }
}

/**
 * Decodes [assetPath] off the main thread, or emits null when the file isn't shipped.
 *
 * A missing avatar is the expected state, not an error — `assets.open` throwing FileNotFound is
 * how we detect "no artwork yet" and fall through to the procedural orb.
 */
@Composable
private fun rememberAssetBitmap(assetPath: String?): ImageBitmap? {
    val context = LocalContext.current
    val state by produceState<ImageBitmap?>(initialValue = null, assetPath) {
        value = if (assetPath == null) null else withContext(Dispatchers.IO) {
            runCatching {
                context.assets.open(assetPath).use { BitmapFactory.decodeStream(it) }?.asImageBitmap()
            }.getOrNull()
        }
    }
    return state
}

/**
 * Circular persona avatar with a gradient rim.
 *
 * When [active] the rim becomes a rotating conic sweep — used while that participant is
 * generating, so the "who is talking" signal lives on the avatar instead of a separate spinner.
 */
@Composable
fun PersonaAvatar(
    persona: Persona,
    modifier: Modifier = Modifier,
    size: Dp = 40.dp,
    active: Boolean = false,
    ringWidth: Dp = 2.dp,
) {
    val glass = AppTheme.glass
    val bitmap = rememberAssetBitmap(persona.assetPath)

    val transition = rememberInfiniteTransition(label = "avatar-ring")
    val angle by transition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(tween(2600, easing = LinearEasing), RepeatMode.Restart),
        label = "avatar-ring-angle",
    )

    Box(
        modifier
            .size(size)
            .shadow(
                elevation = if (active) 14.dp else 6.dp,
                shape = CircleShape,
                ambientColor = glass.glow.copy(alpha = 0.8f),
                spotColor = glass.glow.copy(alpha = 0.8f),
            )
            .clip(CircleShape)
            // The rim lives underneath: the inner circle is inset by ringWidth so this shows
            // through as a ring, which keeps a rotating sweep cheap (no stroke math).
            .background(
                if (active) {
                    Brush.sweepGradient(
                        listOf(glass.accent.start, glass.accent.end, glass.accent.start),
                    )
                } else {
                    Brush.linearGradient(listOf(glass.accent.start, glass.accent.end))
                },
            )
            .then(if (active) Modifier.rotate(angle) else Modifier),
        contentAlignment = Alignment.Center,
    ) {
        Box(
            Modifier
                .fillMaxSize()
                .padding(ringWidth)
                .clip(CircleShape)
                // Counter-rotate so the portrait stays upright while the rim spins.
                .then(if (active) Modifier.rotate(-angle) else Modifier)
                .background(MaterialTheme.colorScheme.background),
            contentAlignment = Alignment.Center,
        ) {
            if (bitmap != null) {
                Image(
                    bitmap = bitmap,
                    contentDescription = persona.displayName,
                    contentScale = ContentScale.Crop,
                    // Bias is -1..1 across the axis; focus is 0..1 from the top-left.
                    alignment = BiasAlignment(
                        horizontalBias = (persona.focusX * 2f - 1f).coerceIn(-1f, 1f),
                        verticalBias = (persona.focusY * 2f - 1f).coerceIn(-1f, 1f),
                    ),
                    modifier = Modifier.fillMaxSize().scale(persona.zoom),
                )
            } else if (persona.seal) {
                HankoSeal(persona.glyph, size)
            } else {
                ProceduralOrb(persona.glyph, size)
            }
        }
    }
}

/**
 * A hanko seal: deep ground, a vermilion square stamp with a thin inner border, the glyph
 * reversed out in paper white — like a name seal pressed onto a letter.
 */
@Composable
private fun HankoSeal(glyph: String, size: Dp) {
    val glass = AppTheme.glass
    val vermilion = Color(0xFFC8331E)
    Box(
        Modifier
            .fillMaxSize()
            .background(
                Brush.radialGradient(
                    listOf(
                        if (glass.dark) Color(0xFF0E3A46) else Color(0xFFF3EEE4),
                        if (glass.dark) Color(0xFF03121A) else Color(0xFFE2DCCF),
                    ),
                ),
            ),
        contentAlignment = Alignment.Center,
    ) {
        // Small (tab bar): a bigger stamp, no inner border, so the character stays legible.
        val small = size < 36.dp
        Box(
            Modifier
                .size(size * if (small) 0.74f else 0.60f)
                .rotate(-4f)
                .clip(RoundedCornerShape(size * 0.06f))
                .background(vermilion)
                .then(
                    if (small) Modifier
                    else Modifier.padding(size * 0.035f)
                        .border(maxOf(1f, size.value * 0.018f).dp, Color(0xFFFFF4EA).copy(alpha = 0.85f), RoundedCornerShape(size * 0.04f)),
                ),
            contentAlignment = Alignment.Center,
        ) {
            Text(
                glyph,
                fontSize = (size.value * if (small) 0.50f else 0.36f).sp,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Serif,
                color = Color(0xFFFFF4EA),
            )
        }
    }
}

/** The no-artwork fallback: a radial neon orb with the persona's glyph centered in it. */
@Composable
private fun ProceduralOrb(glyph: String, size: Dp) {
    val glass = AppTheme.glass
    Box(
        Modifier
            .fillMaxSize()
            .background(
                Brush.radialGradient(
                    listOf(
                        glass.accent.start.copy(alpha = 0.55f),
                        glass.accent.end.copy(alpha = 0.30f),
                        MaterialTheme.colorScheme.background,
                    ),
                ),
            ),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            glyph,
            fontSize = (size.value * 0.42f).sp,
            fontWeight = FontWeight.Medium,
            color = MaterialTheme.colorScheme.onSurface,
        )
    }
}

/**
 * The learner's own avatar: their initial in a glass disc. Deliberately quieter than the persona
 * avatars so the eye tracks the assistant's side of the conversation.
 */
@Composable
fun UserAvatar(
    name: String,
    modifier: Modifier = Modifier,
    size: Dp = 32.dp,
) {
    val glass = AppTheme.glass
    val initial = name.trim().firstOrNull()?.uppercase() ?: "•"
    Box(
        modifier
            .size(size)
            .clip(CircleShape)
            .background(glass.panelHigh)
            .border(1.dp, glass.rimBrush, CircleShape),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            initial,
            fontSize = (size.value * 0.40f).sp,
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}
