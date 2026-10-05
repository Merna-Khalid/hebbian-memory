package com.mobilerag.hebbian.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.mobilerag.hebbian.FadingConceptView
import com.mobilerag.hebbian.PracticeGradeResult
import com.mobilerag.hebbian.PracticeOffer
import com.mobilerag.ui.avatar.Persona
import com.mobilerag.ui.avatar.PersonaAvatar
import com.mobilerag.ui.theme.AppTheme
import com.mobilerag.ui.theme.EmptyState
import com.mobilerag.ui.theme.ErrorBanner
import com.mobilerag.ui.theme.GlassCard
import com.mobilerag.ui.theme.GlassPill
import com.mobilerag.ui.theme.NeonButton
import com.mobilerag.ui.theme.NeonIconButton
import com.mobilerag.ui.theme.NeonIndeterminate
import com.mobilerag.ui.theme.NeonProgress
import com.mobilerag.ui.theme.ScreenHeader
import com.mobilerag.ui.theme.SectionLabel
import com.mobilerag.ui.theme.Telemetry
import com.mobilerag.ui.theme.glassPanel

/** Recall probability colors: green when safe, amber when slipping, red when nearly gone. */
private val RecallGood = Color(0xFF4ADE80)
private val RecallWarn = Color(0xFFFBBF24)
private val RecallBad = Color(0xFFFB7185)

private fun recallColor(p: Double): Color = when {
    p >= 0.66 -> RecallGood
    p >= 0.33 -> RecallWarn
    else -> RecallBad
}

@Composable
fun PracticeScreen(vm: PracticeViewModel = viewModel()) {
    val ui by vm.ui.collectAsState()
    var answer by rememberSaveable { mutableStateOf("") }

    Column(Modifier.fillMaxSize().imePadding()) {
        ScreenHeader(
            title = "練習 Practice",
            subtitle = when {
                ui.building -> "building Hebbian engine…"
                ui.modelLoading -> "loading model…"
                else -> "the scheduler picks what you're about to forget"
            },
            trailing = { PersonaAvatar(Persona.Tutor, size = 40.dp, active = ui.busy) },
        )

        if (ui.building || ui.modelLoading || ui.busy) {
            NeonIndeterminate(Modifier.padding(horizontal = 20.dp))
        }

        AnimatedVisibility(visible = ui.error != null, enter = fadeIn(), exit = fadeOut()) {
            ui.error?.let { ErrorBanner(it, Modifier.padding(horizontal = 16.dp, vertical = 6.dp)) }
        }

        LazyColumn(
            verticalArrangement = Arrangement.spacedBy(12.dp),
            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 12.dp),
        ) {
            item(key = "controls") {
                NeonButton(
                    text = if (ui.busy && ui.grade == null && ui.offer == null) "Generating…" else "Next exercise",
                    onClick = { answer = ""; vm.nextExercise() },
                    enabled = ui.engineReady && !ui.busy && !ui.modelLoading,
                    loading = ui.busy && ui.grade == null && ui.offer == null,
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            if (ui.noExercise) {
                item(key = "empty") {
                    Text(
                        "Nothing is fading yet — chat with the tutor first and the queue will fill itself.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            ui.offer?.let { offer ->
                item(key = "offer") {
                    OfferCard(
                        offer = offer,
                        answer = answer,
                        onAnswer = { answer = it },
                        onSubmit = { vm.submit(answer); answer = "" },
                        busy = ui.busy,
                    )
                }
            }

            ui.grade?.let { grade ->
                item(key = "grade") { GradeCard(grade) }
            }

            item(key = "fading-header") {
                Spacer(Modifier.height(4.dp))
                SectionLabel("Fading concepts (${ui.fading.size})")
            }
            if (ui.fading.isEmpty() && ui.engineReady) {
                item(key = "fading-empty") {
                    Text(
                        "Queue is empty.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
            items(ui.fading, key = { it.nodeId }) { f -> FadingCard(f) }
        }
    }
}

/** A concept at risk. The recall bar is the point of the card; the numbers are supporting detail. */
@Composable
private fun FadingCard(f: FadingConceptView) {
    val color = recallColor(f.recallProb)
    GlassCard(Modifier.fillMaxWidth(), elevation = 8.dp, glowAlpha = 0.18f) {
        Row(Modifier.fillMaxWidth().padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(7.dp)) {
                Text(
                    f.label,
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        Modifier
                            .weight(1f)
                            .height(6.dp)
                            .clip(RoundedCornerShape(3.dp))
                            .background(AppTheme.glass.panelLow),
                    ) {
                        Box(
                            Modifier
                                .fillMaxWidth(f.recallProb.toFloat().coerceIn(0f, 1f))
                                .height(6.dp)
                                .clip(RoundedCornerShape(3.dp))
                                .background(color),
                        )
                    }
                    Spacer(Modifier.width(10.dp))
                    Telemetry("p=%.2f".format(f.recallProb), color = color)
                }
                Telemetry("strength=%.2f · staleness=%.2f".format(f.strength, f.staleness))
            }
            Spacer(Modifier.width(12.dp))
            GlassPill(f.conceptType)
        }
    }
}

@Composable
private fun OfferCard(
    offer: PracticeOffer,
    answer: String,
    onAnswer: (String) -> Unit,
    onSubmit: () -> Unit,
    busy: Boolean,
) {
    val glass = AppTheme.glass
    GlassCard(Modifier.fillMaxWidth(), elevation = 22.dp, glowAlpha = 0.6f) {
        Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            GlassPill(offer.exerciseType, active = true)
            Text(
                offer.prompt,
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.onSurface,
            )
            if (offer.hint.isNotEmpty()) {
                Text(
                    "Hint · ${offer.hint}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            if (offer.fading.isNotEmpty()) {
                LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    items(offer.fading, key = { it.first }) { (label, p) ->
                        GlassPill("$label · %.2f".format(p), tint = recallColor(p))
                    }
                }
            }
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.Bottom, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Box(
                    Modifier
                        .weight(1f)
                        .clip(RoundedCornerShape(3.dp))
                        .background(glass.panelLow)
                        .border(1.dp, glass.rimDim, RoundedCornerShape(3.dp))
                        .padding(horizontal = 16.dp, vertical = 14.dp),
                ) {
                    if (answer.isEmpty()) {
                        Text(
                            "Your answer…",
                            style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                        )
                    }
                    BasicTextField(
                        value = answer,
                        onValueChange = onAnswer,
                        modifier = Modifier.fillMaxWidth(),
                        textStyle = LocalTextStyle.current.merge(
                            MaterialTheme.typography.bodyLarge.copy(color = MaterialTheme.colorScheme.onSurface),
                        ),
                        cursorBrush = SolidColor(MaterialTheme.colorScheme.primary),
                        maxLines = 3,
                    )
                }
                NeonIconButton(
                    icon = Icons.Default.Check,
                    contentDescription = "Submit",
                    onClick = onSubmit,
                    enabled = answer.isNotBlank() && !busy,
                    size = 50.dp,
                )
            }
        }
    }
}

@Composable
private fun GradeCard(grade: PracticeGradeResult) {
    val accent = if (grade.correct) RecallGood else RecallBad
    val shape = MaterialTheme.shapes.large
    Column(
        Modifier
            .fillMaxWidth()
            .clip(shape)
            .background(accent.copy(alpha = 0.10f))
            .border(1.dp, accent.copy(alpha = 0.45f), shape)
            .padding(18.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text(
            if (grade.correct) "正解 · Correct" else "不正解 · Not quite",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold,
            color = accent,
        )
        if (grade.feedback.isNotEmpty()) {
            Text(
                grade.feedback,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface,
            )
        }
        grade.answer?.let {
            Text(
                "Answer · $it",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        if (grade.reinforced) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                NeonProgress(progress = 1f, modifier = Modifier.width(40.dp))
                Spacer(Modifier.width(10.dp))
                Telemetry("memory reinforced · Hebbian edges strengthened (m_t=1.5)")
            }
        }
    }
}
