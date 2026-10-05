package com.mobilerag.hebbian.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate as drawRotate
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.mobilerag.hebbian.ArmeSummary
import com.mobilerag.japanese.JapaneseSegmenter
import com.mobilerag.japanese.JapaneseTappableText
import com.mobilerag.japanese.TranslationOverlay
import com.mobilerag.japanese.WordTap
import com.mobilerag.settings.rememberAppSettings
import com.mobilerag.ui.avatar.Persona
import com.mobilerag.ui.avatar.PersonaAvatar
import com.mobilerag.ui.theme.AnimusBody
import com.mobilerag.ui.theme.AnimusDisplay
import com.mobilerag.ui.theme.AnimusMono
import com.mobilerag.ui.theme.AppTheme
import com.mobilerag.ui.theme.ChromaText
import com.mobilerag.ui.theme.ErrorBanner
import com.mobilerag.ui.theme.GlowRule
import com.mobilerag.ui.theme.LocalAmbientTime
import com.mobilerag.ui.theme.MemoryLight
import com.mobilerag.ui.theme.NeonIndeterminate
import com.mobilerag.ui.theme.PauseAmbientWhile
import com.mobilerag.ui.theme.cornerTicks
import com.mobilerag.ui.theme.drawOuterGlow
import com.mobilerag.ui.theme.frost
import com.mobilerag.ui.theme.rimLight
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin

/**
 * Takemura — the tutor conversation in the Animus language (docs/ui-animus.md): a transmission
 * log, not a messenger. Takemura speaks from frosted panels; your lines are the bright
 * "selected" bar; under each reply, ARM-E reads out as a meter strip and the concepts that
 * turn formed are marked as remembered — you watch the memory being made.
 */
@Composable
fun TutorScreen(vm: TutorViewModel = viewModel()) {
    val ui by vm.ui.collectAsState()
    val settings = rememberAppSettings()
    var input by rememberSaveable { mutableStateOf("") }
    val listState = rememberLazyListState()
    PauseAmbientWhile(listState.isScrollInProgress)

    // The open translation card, or null — one at a time, drawn above the whole screen.
    var tap by remember { mutableStateOf<WordTap?>(null) }

    val busy = ui.building || ui.modelLoading || ui.generating || ui.memorizing

    Column(Modifier.fillMaxSize().imePadding().padding(horizontal = 14.dp)) {
        TutorHeader(ui = ui, speaking = busy)

        if (busy) NeonIndeterminate(Modifier.padding(horizontal = 4.dp))

        AnimatedVisibility(visible = ui.error != null, enter = fadeIn(), exit = fadeOut()) {
            ui.error?.let { ErrorBanner(it, Modifier.padding(vertical = 8.dp)) }
        }

        if (ui.messages.isEmpty()) {
            TutorWelcome(ready = ui.engineReady, modifier = Modifier.weight(1f))
        } else {
            LazyColumn(
                state = listState,
                modifier = Modifier.weight(1f).fillMaxWidth(),
                contentPadding = PaddingValues(vertical = 14.dp),
                verticalArrangement = Arrangement.spacedBy(18.dp),
            ) {
                items(ui.messages, key = { it.id }) { msg ->
                    if (msg.isUser) {
                        UserLine(msg, settings.learnerName.ifBlank { "you" })
                    } else {
                        TutorTransmission(
                            msg = msg,
                            activeTap = tap?.takeIf { it.messageId == msg.id },
                            onWordTap = { word, start, anchor -> tap = WordTap(msg.id, start, word, anchor) },
                        )
                    }
                }
            }
            LaunchedEffect(ui.messages.lastOrNull()?.text?.length) {
                listState.animateScrollToItem(ui.messages.size - 1)
            }
        }

        TutorComposer(
            value = input,
            onValueChange = { input = it },
            onSend = { vm.send(input); input = "" },
            enabled = ui.engineReady && !ui.generating && !ui.memorizing && !ui.modelLoading,
        )
    }

    tap?.let { active ->
        TranslationOverlay(
            tap = active,
            onDismiss = { tap = null },
            // The overlay gives the gloss; the tutor gives usage.
            onAskTutor = { word -> vm.send("「$word」の意味と使い方を教えてください。") },
        )
    }
}

// ── Header ───────────────────────────────────────────────────────────────────

@Composable
private fun TutorHeader(ui: TutorUiState, speaking: Boolean) {
    val glass = AppTheme.glass
    Column(Modifier.fillMaxWidth().statusBarsPadding().padding(top = 10.dp, bottom = 8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            PersonaAvatar(Persona.Tutor, size = 50.dp, active = speaking)
            Spacer(Modifier.width(14.dp))
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                ChromaText(
                    "TAKEMURA",
                    style = TextStyle(fontFamily = AnimusDisplay, fontWeight = FontWeight.Light, fontSize = 30.sp, letterSpacing = 4.sp),
                )
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                    LiveDot(active = speaking)
                    Text(
                        statusLine(ui).uppercase(),
                        style = TextStyle(fontFamily = AnimusMono, fontSize = 10.sp, letterSpacing = 1.6.sp),
                        color = glass.inkMuted,
                        maxLines = 1,
                    )
                }
            }
            Box(
                Modifier.border(1.dp, glass.rimBright).padding(horizontal = 10.dp, vertical = 6.dp),
            ) {
                Text("日本語", style = TextStyle(fontSize = 13.sp, letterSpacing = 2.sp), color = glass.ink)
            }
        }
        GlowRule(Modifier.padding(top = 10.dp), width = 150.dp)
    }
}

/** A diamond that breathes while Takemura is thinking or memorizing. */
@Composable
private fun LiveDot(active: Boolean) {
    val glass = AppTheme.glass
    val time = LocalAmbientTime.current
    Canvas(Modifier.size(9.dp)) {
        val pulse = if (active) 0.55f + 0.45f * sin(time.value * 2f * PI.toFloat() / 1.4f) else 1f
        val c = if (active) glass.mark else glass.glow
        drawRotate(45f) {
            if (glass.dark) drawRect(c.copy(alpha = 0.3f * pulse), topLeft = Offset(-size.width * 0.3f, -size.height * 0.3f), size = size * 1.6f)
            drawRect(c.copy(alpha = pulse))
        }
    }
}

private fun statusLine(ui: TutorUiState): String = when {
    ui.building -> "waking · building memory"
    ui.modelLoading -> "loading model"
    ui.generating -> "speaking"
    ui.memorizing -> "memorizing"
    ui.modelLoaded && ui.modelLoadMs > 0 -> "online · model in %.1f s".format(ui.modelLoadMs / 1000f)
    ui.engineReady -> "online"
    else -> "starting"
}

// ── Welcome ──────────────────────────────────────────────────────────────────

@Composable
private fun TutorWelcome(ready: Boolean, modifier: Modifier = Modifier) {
    val glass = AppTheme.glass
    Box(modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
            modifier = Modifier.padding(horizontal = 30.dp),
        ) {
            Box(
                Modifier.drawBehind {
                    if (glass.dark) {
                        drawCircle(
                            Brush.radialGradient(listOf(MemoryLight.copy(alpha = 0.35f), Color.Transparent), radius = size.minDimension * 0.95f),
                            radius = size.minDimension * 0.95f,
                        )
                    }
                    drawCircle(glass.glow.copy(alpha = 0.18f), radius = size.minDimension * 0.72f, style = Stroke(1f))
                },
            ) {
                PersonaAvatar(Persona.Tutor, size = 104.dp, active = !ready, ringWidth = 2.dp)
            }
            ChromaText(
                if (ready) "はじめましょう" else "準備中…",
                style = TextStyle(fontWeight = FontWeight.Normal, fontSize = 30.sp, letterSpacing = 3.sp),
                flicker = true,
            )
            Text(
                if (ready) "Speak in Japanese or English. What you learn here is remembered.\nTap any Japanese word to see what it means."
                else "Waking Takemura…",
                style = TextStyle(fontFamily = AnimusBody, fontSize = 14.sp, lineHeight = 21.sp),
                color = glass.inkMuted,
                textAlign = TextAlign.Center,
            )
        }
    }
}

// ── Lines ────────────────────────────────────────────────────────────────────

/** A log header: a small caps speaker name with a rule running off to the side. */
@Composable
private fun Speaker(name: String, alignEnd: Boolean) {
    val glass = AppTheme.glass
    Row(
        Modifier.fillMaxWidth().padding(bottom = 6.dp),
        horizontalArrangement = if (alignEnd) Arrangement.End else Arrangement.Start,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (alignEnd) Box(Modifier.width(40.dp).height(1.dp).background(Brush.horizontalGradient(listOf(Color.Transparent, glass.rimBright))))
        Text(
            name.uppercase(),
            modifier = Modifier.padding(horizontal = 8.dp),
            style = TextStyle(fontFamily = AnimusMono, fontSize = 10.sp, letterSpacing = 2.sp),
            color = glass.inkDim,
        )
        if (!alignEnd) Box(Modifier.width(40.dp).height(1.dp).background(Brush.horizontalGradient(listOf(glass.rimBright, Color.Transparent))))
    }
}

/** Your line: the bright bar, right-aligned, dark ink on frost (the accent slab in light mode). */
@Composable
private fun UserLine(msg: TutorMessage, name: String) {
    val glass = AppTheme.glass
    Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.End) {
        Speaker(name, alignEnd = true)
        Box(
            Modifier
                .widthIn(max = 300.dp)
                .drawBehind { if (glass.dark) drawOuterGlow(RectangleShape, glass.glow.copy(alpha = 0.22f), 12.dp.toPx()) }
                .background(if (glass.dark) Color(0xFFECFFFF).copy(alpha = 0.94f) else glass.slab)
                .padding(horizontal = 16.dp, vertical = 12.dp),
        ) {
            Text(
                msg.text,
                style = TextStyle(fontFamily = AnimusBody, fontSize = 16.sp, lineHeight = 23.sp),
                color = glass.onSelection,
            )
        }
    }
}

/** Takemura's line: a frosted panel with ticks, then the ARM-E strip and what was remembered. */
@Composable
private fun TutorTransmission(
    msg: TutorMessage,
    activeTap: WordTap?,
    onWordTap: (word: String, tokenStart: Int, anchor: androidx.compose.ui.geometry.Rect) -> Unit,
) {
    val glass = AppTheme.glass
    // The model writes markdown emphasis; this screen shows plain text, so drop the markers.
    val text = remember(msg.text) { msg.text.replace("**", "") }
    Column(Modifier.fillMaxWidth()) {
        Speaker(Persona.Tutor.displayName, alignEnd = false)
        Box(
            Modifier
                .widthIn(max = 330.dp)
                .then(
                    if (msg.isError) {
                        Modifier.background(MaterialTheme.colorScheme.error.copy(alpha = 0.12f))
                            .border(1.dp, MaterialTheme.colorScheme.error.copy(alpha = 0.45f))
                    } else {
                        Modifier.frost().rimLight().cornerTicks(inset = 4.dp, arm = 8.dp, accent = false)
                    },
                )
                .padding(horizontal = 16.dp, vertical = 14.dp),
        ) {
            when {
                text.isEmpty() && msg.streaming -> TransmissionDots()
                // Only the tutor's Japanese is tappable — English replies have nothing to segment.
                !msg.isError && JapaneseSegmenter.hasJapanese(text) ->
                    JapaneseTappableText(
                        text = text,
                        style = TextStyle(fontSize = 16.sp, lineHeight = 25.sp),
                        color = glass.ink,
                        activeTokenStart = activeTap?.tokenStart,
                        onWordTap = onWordTap,
                    )
                else -> Text(
                    text,
                    style = TextStyle(fontFamily = AnimusBody, fontSize = 16.sp, lineHeight = 24.sp),
                    color = if (msg.isError) MaterialTheme.colorScheme.error else glass.ink,
                )
            }
        }
        if (!msg.streaming) {
            msg.arme?.let { ArmeStrip(it, msg.lang, Modifier.padding(top = 10.dp)) }
            if (msg.concepts.isNotEmpty()) Remembered(msg.concepts.map { it.label }, Modifier.padding(top = 8.dp))
        }
    }
}

/**
 * ARM-E for this turn as a HUD strip: quadrant, then m_t / valence / arousal / dominance as small
 * glowing meters, and the mean weight change. Gated turns (attention too low to learn) are
 * marked with the accent.
 */
@Composable
private fun ArmeStrip(a: ArmeSummary, lang: String?, modifier: Modifier = Modifier) {
    val glass = AppTheme.glass
    Row(
        modifier.semantics {
            contentDescription = "ARM-E ${a.quadrant}, m_t %.2f, valence %.2f, arousal %.2f, dominance %.2f%s"
                .format(a.mT, a.valence, a.arousal, a.dominance, if (a.gated) ", gated" else "")
        },
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Box(
            Modifier.border(1.dp, if (a.gated) glass.mark else glass.rimBright).padding(horizontal = 6.dp, vertical = 2.dp),
        ) {
            Text(a.quadrant, style = TextStyle(fontFamily = AnimusMono, fontSize = 10.sp, fontWeight = FontWeight.Medium), color = if (a.gated) glass.mark else glass.ink)
        }
        Meter("m", ((a.mT - 0.2) / 2.8).toFloat())
        Meter("v", ((a.valence + 1) / 2).toFloat(), centered = true)
        Meter("a", a.arousal.toFloat())
        Meter("d", a.dominance.toFloat())
        Text(
            (if (a.deltaWMean >= 0) "+" else "−") + "%.3f".format(abs(a.deltaWMean)),
            style = TextStyle(fontFamily = AnimusMono, fontSize = 10.sp),
            color = glass.inkMuted,
        )
        if (a.gated) Text("GATED", style = TextStyle(fontFamily = AnimusMono, fontSize = 9.sp, letterSpacing = 1.5.sp), color = glass.mark)
        lang?.let { Text(it.uppercase(), style = TextStyle(fontFamily = AnimusMono, fontSize = 9.sp, letterSpacing = 1.5.sp), color = glass.inkDim) }
    }
}

/** A 30dp meter: a hairline track and a glowing fill; [centered] draws from the middle (valence). */
@Composable
private fun Meter(label: String, value: Float, centered: Boolean = false) {
    val glass = AppTheme.glass
    val v = value.coerceIn(0f, 1f)
    val fill = if (glass.dark) Color(0xFFEFFFFF) else glass.slab
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(label, style = TextStyle(fontFamily = AnimusMono, fontSize = 9.sp), color = glass.inkDim)
        Canvas(Modifier.width(26.dp).height(8.dp)) {
            val y = size.height / 2
            drawLine(glass.rimDim, Offset(0f, y), Offset(size.width, y), 1.dp.toPx())
            val from = if (centered) size.width / 2 else 0f
            val to = size.width * v
            if (glass.dark) drawLine(MemoryLight.copy(alpha = 0.3f), Offset(from, y), Offset(to, y), 5.dp.toPx())
            drawLine(fill, Offset(from, y), Offset(to, y), 2.dp.toPx(), cap = StrokeCap.Butt)
            if (centered) drawLine(glass.inkDim, Offset(size.width / 2, 0f), Offset(size.width / 2, size.height), 1f)
        }
    }
}

/** What this turn wrote into memory. */
@Composable
private fun Remembered(labels: List<String>, modifier: Modifier = Modifier) {
    val glass = AppTheme.glass
    FlowRow(
        modifier,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Text(
            "REMEMBERED",
            modifier = Modifier.align(Alignment.CenterVertically),
            style = TextStyle(fontFamily = AnimusMono, fontSize = 9.sp, letterSpacing = 1.8.sp),
            color = glass.inkDim,
        )
        labels.forEach { label ->
            Row(
                Modifier.border(1.dp, glass.rimDim).padding(horizontal = 7.dp, vertical = 3.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                Box(Modifier.size(5.dp).rotate(45f).background(glass.mark))
                Text(label, style = TextStyle(fontSize = 12.sp), color = glass.ink)
            }
        }
    }
}

/** Three diamonds pulsing in sequence — the line before the first token lands. */
@Composable
private fun TransmissionDots() {
    val glass = AppTheme.glass
    val time = LocalAmbientTime.current
    Row(horizontalArrangement = Arrangement.spacedBy(7.dp), verticalAlignment = Alignment.CenterVertically, modifier = Modifier.height(20.dp)) {
        repeat(3) { i ->
            Canvas(Modifier.size(7.dp)) {
                val a = 0.3f + 0.7f * ((sin((time.value * 2.2f - i * 0.35f) * 2f * PI.toFloat()) + 1f) / 2f)
                drawRotate(45f) { drawRect((if (glass.dark) Color(0xFFEFFFFF) else glass.slab).copy(alpha = a)) }
            }
        }
    }
}

// ── Composer ─────────────────────────────────────────────────────────────────

@Composable
private fun TutorComposer(
    value: String,
    onValueChange: (String) -> Unit,
    onSend: () -> Unit,
    enabled: Boolean,
) {
    val glass = AppTheme.glass
    Row(
        Modifier.fillMaxWidth().padding(top = 8.dp, bottom = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Box(
            Modifier
                .weight(1f)
                .frost()
                .rimLight()
                .cornerTicks(inset = 3.dp, arm = 7.dp)
                .padding(horizontal = 16.dp, vertical = 14.dp),
        ) {
            if (value.isEmpty()) {
                Text("日本語で話そう…", style = TextStyle(fontSize = 16.sp), color = glass.inkDim)
            }
            BasicTextField(
                value = value,
                onValueChange = onValueChange,
                modifier = Modifier.fillMaxWidth(),
                textStyle = TextStyle(fontFamily = AnimusBody, fontSize = 16.sp, color = glass.ink),
                cursorBrush = SolidColor(if (glass.dark) MemoryLight else glass.slab),
                maxLines = 5,
            )
        }
        DiamondSend(onClick = onSend, enabled = enabled && value.isNotBlank())
    }
}

/** Send, as the entity's diamond: frost with a glow in the sea, the accent slab in the fog. */
@Composable
private fun DiamondSend(onClick: () -> Unit, enabled: Boolean) {
    val glass = AppTheme.glass
    val fill = if (glass.dark) Color(0xFFEFFFFF) else glass.slab
    Box(
        Modifier
            .size(52.dp)
            .alpha(if (enabled) 1f else 0.35f)
            .clickable(enabled = enabled, role = Role.Button, onClickLabel = "Send", onClick = onClick)
            .semantics { contentDescription = "Send" },
        contentAlignment = Alignment.Center,
    ) {
        Canvas(Modifier.size(38.dp)) {
            val c = Offset(size.width / 2, size.height / 2)
            val r = size.minDimension / 2
            val diamond = Path().apply { moveTo(c.x, c.y - r); lineTo(c.x + r, c.y); lineTo(c.x, c.y + r); lineTo(c.x - r, c.y); close() }
            if (glass.dark && enabled) {
                drawCircle(Brush.radialGradient(listOf(MemoryLight.copy(alpha = 0.45f), Color.Transparent), center = c, radius = r * 1.5f), radius = r * 1.5f, center = c)
            }
            drawPath(diamond, fill)
            // Arrow
            val a = r * 0.36f
            val ink = glass.onSelection
            drawLine(ink, Offset(c.x - a, c.y), Offset(c.x + a, c.y), 2.dp.toPx())
            drawLine(ink, Offset(c.x + a, c.y), Offset(c.x + a * 0.35f, c.y - a * 0.65f), 2.dp.toPx())
            drawLine(ink, Offset(c.x + a, c.y), Offset(c.x + a * 0.35f, c.y + a * 0.65f), 2.dp.toPx())
        }
    }
}
