package com.mobilerag.hebbian.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.mobilerag.ui.theme.AppTheme
import com.mobilerag.ui.theme.EmptyState
import com.mobilerag.ui.theme.ErrorBanner
import com.mobilerag.ui.theme.GhostButton
import com.mobilerag.ui.theme.GlassCard
import com.mobilerag.ui.theme.GlassIconButton
import com.mobilerag.ui.theme.GlassPill
import com.mobilerag.ui.theme.GlassSection
import com.mobilerag.ui.theme.NeonButton
import com.mobilerag.ui.theme.NeonIndeterminate
import com.mobilerag.ui.theme.NeonProgress
import com.mobilerag.ui.theme.ScreenHeader
import com.mobilerag.ui.theme.SectionLabel
import com.mobilerag.ui.theme.Telemetry

@Composable
fun SecondBrainScreen(vm: SecondBrainViewModel = viewModel()) {
    val ui by vm.ui.collectAsState()
    val listState = rememberLazyListState()
    LaunchedEffect(ui.rows.size) {
        if (ui.rows.isNotEmpty()) listState.animateScrollToItem(ui.rows.size - 1)
    }

    Column(Modifier.fillMaxSize()) {
        ScreenHeader(
            title = "Second Brain",
            subtitle = when {
                ui.building -> "building Hebbian engine…"
                ui.modelLoading -> "loading model…"
                else -> "simulated BCI feed · HR + attention gating"
            },
            trailing = { GlassPill("sim", active = true) },
        )

        if (ui.building || ui.modelLoading || ui.stepping) {
            NeonIndeterminate(Modifier.padding(horizontal = 20.dp))
        }

        AnimatedVisibility(visible = ui.error != null, enter = fadeIn(), exit = fadeOut()) {
            ui.error?.let { ErrorBanner(it, Modifier.padding(horizontal = 16.dp, vertical = 6.dp)) }
        }

        Column(
            Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            VitalsCard(ui)
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                NeonButton(
                    text = if (ui.playing) "Pause" else "Play",
                    icon = if (ui.playing) null else Icons.Default.PlayArrow,
                    onClick = vm::togglePlay,
                    enabled = ui.engineReady && (!ui.exhausted || ui.playing),
                )
                GhostButton("Step", vm::step, enabled = ui.engineReady && !ui.stepping && !ui.playing && !ui.exhausted)
                Spacer(Modifier.weight(1f))
                GlassIconButton(Icons.Default.Refresh, "Reset", vm::reset, enabled = ui.engineReady && !ui.stepping)
            }
        }

        if (ui.rows.isEmpty()) {
            EmptyState(
                glyph = "⌁",
                message = if (ui.engineReady) "No ingests yet — Step or Play through the scripted scenarios."
                else "Starting the second brain…",
                modifier = Modifier.weight(1f),
            )
        } else {
            LazyColumn(
                state = listState,
                modifier = Modifier.weight(1f),
                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                items(ui.rows, key = { it.index }) { row -> IngestRowCard(row) }
            }
        }
    }
}

/** Live physiology readout — the inputs the gate is deciding on. */
@Composable
private fun VitalsCard(ui: SecondBrainUiState) {
    GlassSection(
        title = "Signal",
        trailing = { if (ui.exhausted) GlassPill("done", active = true) },
    ) {
        Text(
            ui.scenarioLabel,
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onSurface,
        )
        Row(horizontalArrangement = Arrangement.spacedBy(20.dp), modifier = Modifier.fillMaxWidth()) {
            Vital("HR", ui.hrBpm?.let { "%.0f".format(it) } ?: "—", "bpm", Modifier.weight(1f))
            Vital("Attention", ui.attention?.let { "%.2f".format(it) } ?: "—", null, Modifier.weight(1f))
        }
        ui.attention?.let { NeonProgress(it.toFloat().coerceIn(0f, 1f)) }
    }
}

@Composable
private fun Vital(label: String, value: String, unit: String?, modifier: Modifier = Modifier) {
    Column(modifier, verticalArrangement = Arrangement.spacedBy(2.dp)) {
        SectionLabel(label)
        Row(verticalAlignment = Alignment.Bottom) {
            Text(
                value,
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onSurface,
            )
            if (unit != null) {
                Spacer(Modifier.width(4.dp))
                Telemetry(unit, Modifier.padding(bottom = 4.dp))
            }
        }
    }
}

/**
 * One ingest event. Gated events get a red spine and dimmed body — the whole point of the screen
 * is seeing at a glance which utterances made it past the gate.
 */
@Composable
private fun IngestRowCard(row: IngestLogRow) {
    val glass = AppTheme.glass
    val spine = if (row.gated) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary
    GlassCard(
        Modifier.fillMaxWidth(),
        elevation = 6.dp,
        glowAlpha = 0.15f,
        fill = glass.panelLow,
    ) {
        Row(Modifier.fillMaxWidth().padding(14.dp)) {
            Box(
                Modifier
                    .width(3.dp)
                    .height(if (row.concepts.isNotEmpty()) 76.dp else 58.dp)
                    .clip(CircleShape)
                    .background(spine),
            )
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        "#${row.index} ${row.scenario}",
                        style = MaterialTheme.typography.titleSmall,
                        color = MaterialTheme.colorScheme.onSurface,
                    )
                    Spacer(Modifier.weight(1f))
                    if (row.gated) GlassPill("gated", tint = MaterialTheme.colorScheme.error, active = true)
                }
                Telemetry(
                    "%s · m_t=%.2f · att=%.2f · hr=%s · Δw̄=%.4f".format(
                        row.quadrant, row.mT, row.attention,
                        row.hrBpm?.let { "%.0f".format(it) } ?: "—", row.deltaWMean,
                    ),
                )
                Text(
                    row.utterance,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                if (row.concepts.isNotEmpty()) {
                    Text(
                        "→ " + row.concepts.joinToString(", "),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.primary,
                    )
                }
            }
        }
    }
}
