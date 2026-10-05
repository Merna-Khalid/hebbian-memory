package com.mobilerag.japanese

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.LayoutCoordinates
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntRect
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Popup
import androidx.compose.ui.window.PopupPositionProvider
import androidx.compose.ui.window.PopupProperties
import com.mobilerag.ui.theme.AppTheme
import com.mobilerag.ui.theme.GhostButton
import com.mobilerag.ui.theme.NeonButton
import com.mobilerag.ui.theme.glassPanel

/** A word the learner tapped, plus where it sits on screen so the overlay can point at it. */
@Immutable
data class WordTap(
    val messageId: Long,
    val tokenStart: Int,
    val word: String,
    /** Bounds of the tapped word in window coordinates. */
    val anchor: Rect,
)

/**
 * Renders [text] with every Japanese word individually tappable.
 *
 * Tappable tokens carry a faint accent wash so the affordance is visible without the paragraph
 * turning into a field of underlines; the currently open one is washed harder.
 */
@Composable
fun JapaneseTappableText(
    text: String,
    modifier: Modifier = Modifier,
    style: TextStyle = MaterialTheme.typography.bodyMedium,
    color: Color = MaterialTheme.colorScheme.onSurface,
    activeTokenStart: Int? = null,
    onWordTap: (word: String, tokenStart: Int, anchor: Rect) -> Unit,
) {
    val glass = AppTheme.glass
    val tokens = remember(text) { JapaneseSegmenter.segment(text) }

    // Words stay clean until tapped (the welcome screen says they're tappable); the open one
    // lights with the world light. A wash on every word read as a wall of grey blocks.
    val lit = if (glass.dark) com.mobilerag.ui.theme.MemoryLight.copy(alpha = 0.30f) else glass.slab.copy(alpha = 0.18f)
    val annotated: AnnotatedString = remember(text, activeTokenStart, lit) {
        buildAnnotatedString {
            tokens.forEach { token ->
                if (token.clickable) {
                    val active = token.start == activeTokenStart
                    withStyle(
                        SpanStyle(background = if (active) lit else Color.Transparent),
                    ) { append(token.text) }
                } else {
                    append(token.text)
                }
            }
        }
    }

    var layout by remember { mutableStateOf<TextLayoutResult?>(null) }
    var coords by remember { mutableStateOf<LayoutCoordinates?>(null) }

    Text(
        text = annotated,
        modifier = modifier
            .onGloballyPositioned { coords = it }
            .pointerInput(tokens) {
                detectTapGestures { pos ->
                    val result = layout ?: return@detectTapGestures
                    val node = coords ?: return@detectTapGestures
                    val offset = result.getOffsetForPosition(pos)
                    val token = tokens.firstOrNull { offset >= it.start && offset < it.end }
                        ?: return@detectTapGestures
                    if (!token.clickable) return@detectTapGestures

                    // Anchor on the word's first character. A wrapped token would give a
                    // meaningless union box, and pointing at the start always reads correctly.
                    val box = result.getBoundingBox(token.start)
                    val topLeft = node.localToWindow(box.topLeft)
                    onWordTap(
                        token.text,
                        token.start,
                        Rect(topLeft.x, topLeft.y, topLeft.x + box.width, topLeft.y + box.height),
                    )
                }
            },
        style = style.copy(color = color),
        onTextLayout = { layout = it },
    )
}

/**
 * Positions the overlay above the tapped word, flipping below when there isn't room, and clamps
 * it inside the window so a word near the edge doesn't push the card off-screen.
 */
private class AnchoredAbove(
    private val anchor: Rect,
    private val gapPx: Int,
    private val marginPx: Int,
) : PopupPositionProvider {
    override fun calculatePosition(
        anchorBounds: IntRect,
        windowSize: IntSize,
        layoutDirection: LayoutDirection,
        popupContentSize: IntSize,
    ): IntOffset {
        val x = (anchor.center.x - popupContentSize.width / 2f).toInt()
            .coerceIn(marginPx, (windowSize.width - popupContentSize.width - marginPx).coerceAtLeast(marginPx))
        val above = anchor.top.toInt() - popupContentSize.height - gapPx
        val y = if (above >= marginPx) above else (anchor.bottom.toInt() + gapPx)
        return IntOffset(
            x,
            y.coerceIn(marginPx, (windowSize.height - popupContentSize.height - marginPx).coerceAtLeast(marginPx)),
        )
    }
}

/**
 * The translation card. Looks the word up on open; shows the model download state on first use
 * and offers to hand the word to the tutor for a fuller explanation.
 */
@Composable
fun TranslationOverlay(
    tap: WordTap,
    onDismiss: () -> Unit,
    onAskTutor: ((String) -> Unit)? = null,
) {
    val context = LocalContext.current
    val density = LocalDensity.current
    val glass = AppTheme.glass
    val modelState by JaTranslator.modelState.collectAsState()

    var translation by remember(tap.word) { mutableStateOf<String?>(null) }
    var failed by remember(tap.word) { mutableStateOf(false) }

    LaunchedEffect(tap.word) {
        failed = false
        translation = null
        val result = JaTranslator.translate(context, tap.word)
        if (result == null) failed = true else translation = result
    }

    // Entrance motion is driven by a graphicsLayer rather than AnimatedVisibility: AnimatedVisibility
    // emits nothing on its first frame, so the Popup window gets created at 0×0 and never appears.
    var shown by remember(tap.word, tap.messageId) { mutableStateOf(false) }
    LaunchedEffect(tap.word, tap.messageId) { shown = true }
    val appear by animateFloatAsState(
        targetValue = if (shown) 1f else 0f,
        animationSpec = spring(dampingRatio = 0.75f, stiffness = 700f),
        label = "overlay-appear",
    )

    Popup(
        popupPositionProvider = AnchoredAbove(
            anchor = tap.anchor,
            gapPx = with(density) { 10.dp.roundToPx() },
            marginPx = with(density) { 12.dp.roundToPx() },
        ),
        onDismissRequest = onDismiss,
        properties = PopupProperties(focusable = true, dismissOnClickOutside = true),
    ) {
        Column(
                Modifier
                    .graphicsLayer {
                        alpha = appear
                        scaleX = 0.88f + 0.12f * appear
                        scaleY = 0.88f + 0.12f * appear
                    }
                    .widthIn(min = 180.dp, max = 300.dp)
                    // Opaque-ish fill: this floats over arbitrary content, so it needs its own
                    // ground rather than the translucent panel used on the aurora backdrop.
                    .glassPanel(
                        shape = MaterialTheme.shapes.medium,
                        fill = MaterialTheme.colorScheme.surface.copy(alpha = 0.97f),
                        elevation = 24.dp,
                        glowAlpha = 0.7f,
                    )
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text(
                    tap.word,
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                HorizontalDivider(color = glass.rimDim)

                when {
                    translation != null -> Text(
                        translation!!,
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.primary,
                    )

                    modelState is JaTranslator.ModelState.Downloading -> Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        CircularProgressIndicator(
                            Modifier.size(16.dp),
                            strokeWidth = 2.dp,
                            color = MaterialTheme.colorScheme.primary,
                        )
                        Text(
                            "Fetching the JA→EN pack — one time, then offline.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }

                    failed -> Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Text(
                            (modelState as? JaTranslator.ModelState.Failed)?.let {
                                "Translation pack unavailable — needs a network connection once."
                            } ?: "Couldn't translate that one.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.error,
                        )
                        NeonButton(
                            text = "Retry",
                            onClick = {
                                failed = false
                                translation = null
                            },
                        )
                    }

                    else -> Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        CircularProgressIndicator(
                            Modifier.size(16.dp),
                            strokeWidth = 2.dp,
                            color = MaterialTheme.colorScheme.primary,
                        )
                        Text(
                            "…",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }

                if (onAskTutor != null) {
                    Spacer(Modifier.height(2.dp))
                    GhostButton(
                        text = "Ask the tutor",
                        onClick = { onDismiss(); onAskTutor(tap.word) },
                        modifier = Modifier.fillMaxWidth(),
                    )
                }

                Text(
                    "on-device · ML Kit",
                    fontSize = 10.sp,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                )
        }
    }
}
