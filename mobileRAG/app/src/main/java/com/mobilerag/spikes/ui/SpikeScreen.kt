package com.mobilerag.spikes.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.mobilerag.spikes.SpikeUiState
import com.mobilerag.spikes.SpikeViewModel
import com.mobilerag.ui.theme.AppTheme
import com.mobilerag.ui.theme.GlassCard
import com.mobilerag.ui.theme.GlassPill
import com.mobilerag.ui.theme.GhostButton
import com.mobilerag.ui.theme.NeonIndeterminate
import com.mobilerag.ui.theme.ScreenHeader
import com.mobilerag.ui.theme.Telemetry

private val Pass = Color(0xFF4ADE80)
private val Fail = Color(0xFFFB7185)

@Composable
fun SpikeScreen(vm: SpikeViewModel = viewModel()) {
    val states by vm.states.collectAsState()

    Column(Modifier.fillMaxSize()) {
        ScreenHeader(
            title = "Diagnostics",
            subtitle = "on-device verification harnesses · PASS/FAIL with latency and memory",
        )
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            items(states, key = { it.spike.id }) { state ->
                SpikeCard(state, onRun = { vm.runSpike(state.spike.id) })
            }
        }
    }
}

@Composable
fun SpikeCard(state: SpikeUiState, onRun: () -> Unit) {
    val glass = AppTheme.glass
    GlassCard(Modifier.fillMaxWidth(), elevation = 8.dp, glowAlpha = 0.18f) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                    Text(
                        state.spike.title,
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onSurface,
                    )
                    Text(
                        state.spike.description,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Spacer(Modifier.width(12.dp))
                state.result?.let { GlassPill(if (it.passed) "PASS" else "FAIL", active = true, tint = if (it.passed) Pass else Fail) }
            }

            if (state.running) {
                NeonIndeterminate()
            } else {
                GhostButton("Run", onRun)
            }

            state.result?.let { r ->
                Text(
                    r.summary,
                    style = MaterialTheme.typography.bodyMedium,
                    color = if (r.passed) Pass else Fail,
                )
            }

            if (state.logLines.isNotEmpty()) {
                // Log output gets its own recessed well so it reads as machine output, not copy.
                Column(
                    Modifier
                        .fillMaxWidth()
                        .clip(MaterialTheme.shapes.small)
                        .background(glass.panelLow)
                        .padding(12.dp),
                ) {
                    Telemetry(state.logLines.joinToString("\n"))
                }
            }
        }
    }
}
