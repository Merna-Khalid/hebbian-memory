package com.mobilerag.ui.theme

import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.ClipOp
import androidx.compose.ui.graphics.Outline
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.addOutline
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.clipPath
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.draw.scale
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * The panel primitive (Animus): a faint outer bloom, a translucent fill — or, with [frost], the
 * atmosphere blurred behind it — a 1px rim and a bright hairline along the top edge.
 * Everything card-shaped in the app is one of these. Frost costs a blur pass per panel, so it
 * is opt-in for the few large panels; list cards use the plain tint.
 */
@Composable
fun Modifier.glassPanel(
    shape: Shape,
    fill: Color? = null,
    elevation: Dp = 12.dp,
    glowAlpha: Float = 0.35f,
    rim: Boolean = true,
    frost: Boolean = false,
): Modifier {
    val glass = AppTheme.glass
    val bloom = if (glass.dark) glowAlpha * 0.30f else glowAlpha * 0.22f
    return this
        .drawBehind {
            drawOuterGlow(
                shape = shape,
                color = glass.glow.copy(alpha = bloom),
                spread = elevation.toPx() * 1.8f,
            )
        }
        .then(if (frost) Modifier.frost(shape, fill) else Modifier.clip(shape).background(fill ?: glass.panel))
        .then(
            if (rim) {
                Modifier
                    .border(BorderStroke(1.dp, glass.rimBrush), shape)
                    .drawWithContent {
                        drawContent()
                        drawLine(glass.gloss, Offset(1f, 1f), Offset(size.width - 1f, 1f), strokeWidth = 1f)
                    }
            } else Modifier,
        )
}

/**
 * Paints a soft accent halo hugging the outside of [shape].
 *
 * This exists because `Modifier.shadow` is unusable under frosted glass: Android renders the
 * elevation shadow *beneath* the layer, and a fill that is 94% transparent lets it straight
 * through — it shows up as a hard grey box inside the card. Drawing the glow by hand and clipping
 * it to the region *outside* the outline keeps the panel interior perfectly clean.
 *
 * Built from concentric filled rounded-rects. Each ring is drawn at a *low* alpha and they
 * overlap, so opacity accumulates toward the panel edge and thins out with distance — that
 * overlap is what produces the falloff. Per-ring alpha is therefore scaled down by the ring
 * count; giving each ring the full alpha instead stacks to solid and reads as a hard neon
 * outline with visible banding.
 */
internal fun DrawScope.drawOuterGlow(
    shape: Shape,
    color: Color,
    spread: Float,
    rings: Int = 18,
) {
    if (spread <= 0f || color.alpha <= 0f) return
    val outline = shape.createOutline(size, layoutDirection, this)
    // Only rectangles and rounded rects can be expanded cheaply; arbitrary paths are skipped.
    val radius = when (outline) {
        is Outline.Rounded -> outline.roundRect.topLeftCornerRadius.x
        is Outline.Rectangle -> 0f
        else -> return
    }
    val hole = Path().apply { addOutline(outline) }
    // Chosen so the accumulated alpha right at the edge lands near `color.alpha`.
    val perRing = color.alpha * 1.6f / rings
    clipPath(hole, ClipOp.Difference) {
        for (i in rings downTo 1) {
            val t = i / rings.toFloat()
            val d = spread * t
            drawRoundRect(
                color = color.copy(alpha = perRing * (1f - t)),
                topLeft = Offset(-d, -d),
                size = Size(size.width + 2 * d, size.height + 2 * d),
                cornerRadius = CornerRadius(radius + d),
            )
        }
    }
}

/** Standard card. [onClick] adds a press-scale so taps feel physical. */
@Composable
fun GlassCard(
    modifier: Modifier = Modifier,
    shape: Shape = MaterialTheme.shapes.large,
    fill: Color? = null,
    elevation: Dp = 12.dp,
    glowAlpha: Float = 0.35f,
    onClick: (() -> Unit)? = null,
    frost: Boolean = false,
    ticks: Boolean = false,
    content: @Composable ColumnScope.() -> Unit,
) {
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    val scale by androidx.compose.animation.core.animateFloatAsState(
        targetValue = if (pressed) 0.985f else 1f,
        label = "press-scale",
    )
    Column(
        modifier
            .then(if (onClick != null) Modifier.scale(scale) else Modifier)
            .glassPanel(shape, fill, elevation, glowAlpha, frost = frost)
            .then(if (ticks) Modifier.cornerTicks(accent = false) else Modifier)
            .then(
                if (onClick != null) {
                    Modifier.clickable(interactionSource = interaction, indication = null, onClick = onClick)
                } else Modifier,
            ),
        content = content,
    )
}

/** Card + the standard 16dp inner padding and 8dp vertical rhythm. The common case. */
@Composable
fun GlassSection(
    title: String? = null,
    modifier: Modifier = Modifier,
    trailing: (@Composable RowScope.() -> Unit)? = null,
    content: @Composable ColumnScope.() -> Unit,
) {
    GlassCard(modifier.fillMaxWidth()) {
        Column(
            Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            if (title != null || trailing != null) {
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    if (title != null) SectionLabel(title, Modifier.weight(1f))
                    trailing?.invoke(this)
                }
            }
            content()
        }
    }
}

/** Small tracked-out uppercase label — the app's section voice. */
@Composable
fun SectionLabel(text: String, modifier: Modifier = Modifier) {
    Text(
        text.uppercase(),
        modifier = modifier,
        style = MaterialTheme.typography.labelSmall,
        color = AppTheme.glass.inkDim,
    )
}

/**
 * Primary action: filled with the neon sweep, glowing, and it scales on press. Use sparingly —
 * one per screen region, or the glow stops meaning anything.
 */
@Composable
fun NeonButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    icon: ImageVector? = null,
    loading: Boolean = false,
) {
    val glass = AppTheme.glass
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    val scale by androidx.compose.animation.core.animateFloatAsState(
        targetValue = if (pressed) 0.96f else 1f,
        label = "neon-press",
    )
    val shape = MaterialTheme.shapes.small
    Row(
        modifier
            .scale(scale)
            .alpha(if (enabled) 1f else 0.40f)
            .drawBehind {
                if (enabled && glass.dark) drawOuterGlow(shape, glass.glow.copy(alpha = 0.45f), 14.dp.toPx())
            }
            .clip(shape)
            .background(glass.accentBrush)
            .clickable(
                interactionSource = interaction,
                indication = null,
                enabled = enabled && !loading,
                onClick = onClick,
            )
            .defaultMinSize(minHeight = 48.dp)
            .padding(horizontal = 22.dp, vertical = 13.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Center,
    ) {
        // Frost bar in the sea (dark ink), accent slab in the white room (white ink).
        val ink = glass.onSelection
        if (loading) {
            CircularProgressIndicator(
                Modifier.size(16.dp),
                color = ink,
                strokeWidth = 2.dp,
            )
            Spacer(Modifier.width(10.dp))
        } else if (icon != null) {
            Icon(icon, contentDescription = null, tint = ink, modifier = Modifier.size(18.dp))
            Spacer(Modifier.width(8.dp))
        }
        Text(text, color = ink, style = MaterialTheme.typography.labelLarge, maxLines = 1)
    }
}

/** Secondary action: glass fill, accent text, no glow. */
@Composable
fun GhostButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    icon: ImageVector? = null,
) {
    val glass = AppTheme.glass
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    val scale by androidx.compose.animation.core.animateFloatAsState(
        targetValue = if (pressed) 0.96f else 1f,
        label = "ghost-press",
    )
    Row(
        modifier
            .scale(scale)
            .alpha(if (enabled) 1f else 0.40f)
            .clip(MaterialTheme.shapes.small)
            .background(glass.panelHigh)
            .border(BorderStroke(1.dp, glass.rimBrush), MaterialTheme.shapes.small)
            .clickable(interactionSource = interaction, indication = null, enabled = enabled, onClick = onClick)
            .defaultMinSize(minHeight = 48.dp)
            .padding(horizontal = 18.dp, vertical = 13.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Center,
    ) {
        if (icon != null) {
            Icon(icon, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(18.dp))
            Spacer(Modifier.width(8.dp))
        }
        Text(
            text,
            color = MaterialTheme.colorScheme.onSurface,
            style = MaterialTheme.typography.labelLarge,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
    }
}

/**
 * Compact stateful pill — the header toggles (retrieval mode, backend) and metadata tags.
 * [active] switches it from glass to the neon wash with an accent rim.
 */
@Composable
fun GlassPill(
    text: String,
    modifier: Modifier = Modifier,
    active: Boolean = false,
    leading: ImageVector? = null,
    tint: Color? = null,
    onClick: (() -> Unit)? = null,
) {
    val glass = AppTheme.glass
    val accent = tint ?: MaterialTheme.colorScheme.primary
    val chip = MaterialTheme.shapes.extraSmall
    Row(
        modifier
            .clip(chip)
            .then(
                if (active) {
                    Modifier
                        .background(glass.accentWash(0.22f))
                        .border(1.dp, accent.copy(alpha = 0.55f), chip)
                } else {
                    Modifier
                        .background(glass.panelLow)
                        .border(1.dp, glass.rimDim, chip)
                },
            )
            .then(if (onClick != null) Modifier.clickable(onClick = onClick) else Modifier)
            .padding(horizontal = 12.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (leading != null) {
            Icon(
                leading,
                contentDescription = null,
                tint = if (active) accent else MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(14.dp),
            )
            Spacer(Modifier.width(6.dp))
        }
        Text(
            text,
            style = MaterialTheme.typography.labelSmall,
            color = if (active) accent else MaterialTheme.colorScheme.onSurfaceVariant,
            maxLines = 1,
        )
    }
}

/** Circular glass icon button used across every header. */
@Composable
fun GlassIconButton(
    icon: ImageVector,
    contentDescription: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    active: Boolean = false,
    size: Dp = 42.dp,
) {
    val glass = AppTheme.glass
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    val scale by androidx.compose.animation.core.animateFloatAsState(
        targetValue = if (pressed) 0.90f else 1f,
        label = "icon-press",
    )
    Box(
        modifier
            .scale(scale)
            .alpha(if (enabled) 1f else 0.35f)
            .size(size)
            .clip(CircleShape)
            .background(if (active) glass.accentWash(0.26f) else Brush.linearGradient(listOf(glass.panelLow, glass.panelLow)))
            .border(BorderStroke(1.dp, if (active) glass.accentBrush else Brush.linearGradient(listOf(glass.ink.copy(alpha = 0.55f), glass.ink.copy(alpha = 0.35f)))), CircleShape)
            .clickable(interactionSource = interaction, indication = null, enabled = enabled, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Icon(
            icon,
            contentDescription = contentDescription,
            tint = if (active) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface,
            modifier = Modifier.size(size * 0.45f),
        )
    }
}

/** Icon-only primary action — the send buttons. Neon fill, glowing, press-scaled. */
@Composable
fun NeonIconButton(
    icon: ImageVector,
    contentDescription: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    size: Dp = 52.dp,
) {
    val glass = AppTheme.glass
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    val scale by androidx.compose.animation.core.animateFloatAsState(
        targetValue = if (pressed) 0.90f else 1f,
        label = "neon-icon-press",
    )
    Box(
        modifier
            .scale(scale)
            .alpha(if (enabled) 1f else 0.35f)
            .size(size)
            .drawBehind {
                if (enabled && glass.dark) drawOuterGlow(CircleShape, glass.glow.copy(alpha = 0.5f), 14.dp.toPx())
            }
            .clip(CircleShape)
            .background(glass.accentBrush)
            .clickable(interactionSource = interaction, indication = null, enabled = enabled, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Icon(
            icon,
            contentDescription = contentDescription,
            tint = glass.onSelection,
            modifier = Modifier.size(size * 0.42f),
        )
    }
}

/**
 * Text input in a recessed glass well. Replaces OutlinedTextField everywhere — the M3 outlined
 * field's floating label and hard border fight the frosted look.
 */
@Composable
fun GlassTextField(
    value: String,
    onValueChange: (String) -> Unit,
    placeholder: String,
    modifier: Modifier = Modifier,
    label: String? = null,
    singleLine: Boolean = true,
    minLines: Int = 1,
) {
    val glass = AppTheme.glass
    val shape = MaterialTheme.shapes.medium
    Column(modifier, verticalArrangement = Arrangement.spacedBy(6.dp)) {
        if (label != null) SectionLabel(label)
        Box(
            Modifier
                .fillMaxWidth()
                .clip(shape)
                .background(glass.panelLow)
                .border(1.dp, glass.rimDim, shape)
                .padding(horizontal = 16.dp, vertical = 14.dp),
        ) {
            if (value.isEmpty()) {
                Text(
                    placeholder,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.65f),
                )
            }
            androidx.compose.foundation.text.BasicTextField(
                value = value,
                onValueChange = onValueChange,
                modifier = Modifier.fillMaxWidth(),
                singleLine = singleLine,
                minLines = minLines,
                textStyle = LocalTextStyle.current.merge(
                    MaterialTheme.typography.bodyMedium.copy(color = MaterialTheme.colorScheme.onSurface),
                ),
                cursorBrush = androidx.compose.ui.graphics.SolidColor(MaterialTheme.colorScheme.primary),
            )
        }
    }
}

/** Label/value row — the readout format used by Home, Settings and the detail panes. */
@Composable
fun StatRow(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
    valueColor: Color? = null,
    mono: Boolean = false,
) {
    Row(
        modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            value,
            style = if (mono) TelemetryStyle.copy(fontSize = 13.sp) else MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Medium,
            color = valueColor ?: MaterialTheme.colorScheme.onSurface,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier.padding(start = 16.dp),
        )
    }
}

/** Telemetry text (ARM-E lines, timings) — monospace, dim, tracked. */
@Composable
fun Telemetry(
    text: String,
    modifier: Modifier = Modifier,
    color: Color? = null,
) {
    Text(
        text,
        modifier = modifier,
        style = TelemetryStyle,
        color = color ?: MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.85f),
    )
}

/** Determinate progress rendered as a neon bar in a glass track. */
@Composable
fun NeonProgress(progress: Float, modifier: Modifier = Modifier) {
    val glass = AppTheme.glass
    Box(
        modifier
            .fillMaxWidth()
            .height(3.dp)
            .background(glass.rimDim),
    ) {
        Box(
            Modifier
                .fillMaxWidth(progress.coerceIn(0f, 1f))
                .fillMaxSize()
                .drawBehind { if (glass.dark) drawOuterGlow(androidx.compose.ui.graphics.RectangleShape, glass.glow.copy(alpha = 0.5f), 5.dp.toPx(), rings = 6) }
                .background(glass.accentBrush),
        )
    }
}

/** Indeterminate progress: a neon comet sweeping a glass track. Used while models load. */
@Composable
fun NeonIndeterminate(modifier: Modifier = Modifier) {
    val glass = AppTheme.glass
    val transition = rememberInfiniteTransition(label = "sweep")
    val head by transition.animateFloat(
        initialValue = -0.35f,
        targetValue = 1.0f,
        animationSpec = infiniteRepeatable(tween(1400), RepeatMode.Restart),
        label = "sweep-head",
    )
    Box(
        modifier
            .fillMaxWidth()
            .height(3.dp)
            .clip(CircleShape)
            .background(glass.panelLow)
            .drawWithContent {
                drawContent()
                val w = size.width
                drawRect(
                    brush = Brush.horizontalGradient(
                        colors = listOf(Color.Transparent, glass.accent.start, glass.accent.end, Color.Transparent),
                        startX = head * w,
                        endX = (head + 0.35f) * w,
                    ),
                )
            },
    )
}

/** Error banner — red wash, red rim, never a full opaque slab. */
@Composable
fun ErrorBanner(message: String, modifier: Modifier = Modifier) {
    val shape = MaterialTheme.shapes.medium
    Row(
        modifier
            .fillMaxWidth()
            .clip(shape)
            .background(MaterialTheme.colorScheme.error.copy(alpha = 0.12f))
            .border(1.dp, MaterialTheme.colorScheme.error.copy(alpha = 0.45f), shape)
            .padding(14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            message,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.error,
        )
    }
}

/** Large screen title with the gradient applied to the text itself. */
@Composable
fun GradientTitle(
    text: String,
    modifier: Modifier = Modifier,
    style: androidx.compose.ui.text.TextStyle? = null,
) {
    ChromaText(text, style ?: MaterialTheme.typography.headlineMedium, modifier)
}

/**
 * Shared screen header: optional back affordance, gradient title, subtitle, trailing slot.
 * Applies the status-bar inset itself so every screen lines up under the notch identically.
 */
@Composable
fun ScreenHeader(
    title: String,
    modifier: Modifier = Modifier,
    subtitle: String? = null,
    onBack: (() -> Unit)? = null,
    trailing: (@Composable RowScope.() -> Unit)? = null,
) {
    Row(
        modifier
            .fillMaxWidth()
            .statusBarsPadding()
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (onBack != null) {
            GlassIconButton(
                icon = Icons.AutoMirrored.Filled.ArrowBack,
                contentDescription = "Back",
                onClick = onBack,
            )
            Spacer(Modifier.width(12.dp))
        }
        Column(Modifier.weight(1f)) {
            GradientTitle(title, style = MaterialTheme.typography.headlineMedium)
            GlowRule(Modifier.padding(top = 2.dp), width = 96.dp)
            if (subtitle != null) {
                Spacer(Modifier.height(3.dp))
                Text(
                    subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
        trailing?.invoke(this)
    }
}

/** Empty-state block: a dim glyph, a line of copy, centered in the remaining space. */
@Composable
fun EmptyState(
    glyph: String,
    message: String,
    modifier: Modifier = Modifier,
) {
    val glass = AppTheme.glass
    Box(modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(14.dp),
            modifier = Modifier.padding(horizontal = 40.dp),
        ) {
            Text(
                glyph,
                fontSize = 48.sp,
                style = LocalTextStyle.current.copy(brush = glass.accentWash(0.55f)),
            )
            CompositionLocalProvider(LocalContentColor provides MaterialTheme.colorScheme.onSurfaceVariant) {
                Text(
                    message,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                )
            }
        }
    }
}
