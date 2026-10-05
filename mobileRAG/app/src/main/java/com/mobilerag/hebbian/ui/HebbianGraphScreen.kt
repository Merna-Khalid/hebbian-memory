package com.mobilerag.hebbian.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.mobilerag.hebbian.GraphEdgeView
import com.mobilerag.hebbian.GraphNodeView
import com.mobilerag.hebbian.store.ConceptNode
import com.mobilerag.ui.theme.AnimusBody
import com.mobilerag.ui.theme.AnimusDisplay
import com.mobilerag.ui.theme.AnimusMono
import com.mobilerag.ui.theme.AnimusPanel
import com.mobilerag.ui.theme.AppTheme
import com.mobilerag.ui.theme.ChromaText
import com.mobilerag.ui.theme.ErrorBanner
import com.mobilerag.ui.theme.GlowRule
import com.mobilerag.ui.theme.MemoryLight
import com.mobilerag.ui.theme.NeonIndeterminate
import com.mobilerag.ui.theme.PauseAmbientWhile
import com.mobilerag.ui.theme.Telemetry
import com.mobilerag.ui.theme.cornerTicks
import com.mobilerag.ui.theme.drawOuterGlow
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Hebbian weights saturate at W_CEIL (5.0) — the filter spans the whole range. */
private const val W_MAX = 5f

private fun formatTimestamp(millis: Long): String =
    SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault()).format(Date(millis))

/**
 * Memory — the concepts as a ranked list in frosted glass (docs/ui-animus.md). Each row's ring
 * is its connection strength relative to the strongest concept; tapping a row lights it with
 * the selection bar and opens its detail — description, strongest links, ARM-E history.
 */
@Composable
fun HebbianGraphScreen(vm: HebbianGraphViewModel = viewModel()) {
    val ui by vm.ui.collectAsState()
    val glass = AppTheme.glass
    val maxStrength = ui.strength.values.maxOrNull()?.takeIf { it > 0 } ?: 1.0
    val listState = rememberLazyListState()
    PauseAmbientWhile(listState.isScrollInProgress)
    // Bring an opened concept to the top so its detail has room below it.
    LaunchedEffect(ui.selectedId) {
        val i = ui.nodes.indexOfFirst { it.nodeId == ui.selectedId }
        if (i >= 0) listState.animateScrollToItem(i)
    }

    Column(Modifier.fillMaxSize().padding(horizontal = 14.dp)) {
        Column(Modifier.statusBarsPadding().padding(top = 14.dp, start = 4.dp, bottom = 12.dp)) {
            ChromaText(
                "記憶 MEMORY",
                style = TextStyle(fontFamily = AnimusDisplay, fontWeight = FontWeight.Light, fontSize = 44.sp, letterSpacing = 4.sp),
            )
            GlowRule(Modifier.padding(top = 2.dp), width = 110.dp)
            Text(
                when {
                    ui.building -> "waking the Hebbian engine…"
                    else -> "${ui.nodes.size} concepts · ${ui.edgeCount} links ≥ %.2f".format(ui.minWeight)
                },
                modifier = Modifier.padding(top = 6.dp),
                style = TextStyle(fontFamily = AnimusBody, fontSize = 14.sp),
                color = glass.inkMuted,
            )
        }

        if (ui.building || ui.loading) NeonIndeterminate(Modifier.padding(bottom = 8.dp))
        AnimatedVisibility(visible = ui.error != null, enter = fadeIn(), exit = fadeOut()) {
            ui.error?.let { ErrorBanner(it, Modifier.padding(bottom = 8.dp)) }
        }

        WeightFilter(
            value = ui.minWeight,
            enabled = ui.engineReady,
            onChange = vm::onMinWeightChange,
            onChangeFinished = vm::applyFilter,
        )
        Spacer(Modifier.height(14.dp))

        AnimusPanel(Modifier.fillMaxWidth().weight(1f).padding(bottom = 12.dp), ticks = false) {
            if (ui.nodes.isEmpty() && ui.engineReady && !ui.loading) {
                Box(Modifier.fillMaxSize().padding(32.dp), contentAlignment = Alignment.Center) {
                    Text(
                        if (ui.minWeight > 0f) "No concept has a link that strong yet."
                        else "No memories yet. Talk to Takemura, or run the second-brain sim, and they'll form here.",
                        style = TextStyle(fontFamily = AnimusBody, fontSize = 14.sp),
                        color = glass.inkMuted,
                    )
                }
            } else {
                LazyColumn(state = listState, contentPadding = PaddingValues(vertical = 8.dp)) {
                    itemsIndexed(ui.nodes, key = { _, n -> n.nodeId }) { i, node ->
                        val open = ui.selectedId == node.nodeId
                        Column {
                            ConceptRow(
                                index = i + 1,
                                node = node,
                                links = ui.linkCount[node.nodeId] ?: 0,
                                strength = ((ui.strength[node.nodeId] ?: 0.0) / maxStrength).toFloat(),
                                selected = open,
                                onClick = { vm.select(node.nodeId) },
                            )
                            AnimatedVisibility(open, enter = expandVertically() + fadeIn(), exit = shrinkVertically() + fadeOut()) {
                                ConceptDetail(ui.detailNode, ui.detailEdges, ui.labels, ui.detailLoading)
                            }
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun WeightFilter(
    value: Float,
    enabled: Boolean,
    onChange: (Float) -> Unit,
    onChangeFinished: () -> Unit,
) {
    val glass = AppTheme.glass
    val light = if (glass.dark) Color(0xFFEFFFFF) else glass.slab
    AnimusPanel(Modifier.fillMaxWidth(), ticks = false) {
        Column(Modifier.padding(start = 18.dp, end = 18.dp, top = 12.dp, bottom = 10.dp)) {
            Row(verticalAlignment = Alignment.Bottom) {
                Text("MIN WEIGHT", Modifier.weight(1f), style = TextStyle(fontFamily = AnimusMono, fontSize = 10.sp, letterSpacing = 2.sp), color = glass.inkMuted)
                Text("%.2f".format(value), style = TextStyle(fontFamily = AnimusDisplay, fontWeight = FontWeight.Light, fontSize = 22.sp), color = glass.ink)
            }
            Slider(
                value = value,
                onValueChange = onChange,
                onValueChangeFinished = onChangeFinished,
                valueRange = 0f..W_MAX,
                enabled = enabled,
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics { contentDescription = "Minimum link weight"; stateDescription = "%.2f".format(value) },
                thumb = {
                    Box(
                        Modifier
                            .size(14.dp)
                            .rotate(45f)
                            .drawBehind {
                                if (glass.dark) drawCircle(
                                    Brush.radialGradient(listOf(MemoryLight.copy(alpha = 0.45f), Color.Transparent), radius = size.minDimension * 1.2f),
                                    radius = size.minDimension * 1.2f,
                                )
                            }
                            .background(light),
                    )
                },
                track = { st ->
                    val frac = (st.value - st.valueRange.start) / (st.valueRange.endInclusive - st.valueRange.start)
                    Canvas(Modifier.fillMaxWidth().height(12.dp)) {
                        val y = size.height / 2
                        drawLine(glass.rimDim, Offset(0f, y), Offset(size.width, y), 1.dp.toPx())
                        val end = size.width * frac
                        if (glass.dark) drawLine(MemoryLight.copy(alpha = 0.3f), Offset(0f, y), Offset(end, y), 6.dp.toPx())
                        drawLine(light, Offset(0f, y), Offset(end, y), 2.dp.toPx())
                    }
                },
            )
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                listOf("0", "1", "2", "3", "4", "5").forEach {
                    Text(it, style = TextStyle(fontFamily = AnimusMono, fontSize = 10.sp), color = glass.inkDim)
                }
            }
        }
    }
}

/** One concept: index, label, meta line, and its strength ring. Selected = the bright bar. */
@Composable
private fun ConceptRow(
    index: Int,
    node: GraphNodeView,
    links: Int,
    strength: Float,
    selected: Boolean,
    onClick: () -> Unit,
) {
    val glass = AppTheme.glass
    val ink = if (selected) glass.onSelection else glass.ink
    val meta = if (selected) glass.onSelection.copy(alpha = 0.75f) else glass.inkMuted
    Row(
        Modifier
            .fillMaxWidth()
            .heightIn(min = 60.dp)
            .then(
                if (selected) {
                    Modifier
                        .drawBehind { if (glass.dark) drawOuterGlow(RectangleShape, glass.glow.copy(alpha = 0.25f), 14.dp.toPx()) }
                        .background(glass.selectionBrush(0.62f))
                } else Modifier,
            )
            .clickable(role = Role.Button, onClickLabel = if (selected) "Close details" else "Show details", onClick = onClick)
            .padding(horizontal = 18.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(
            "%02d".format(index),
            modifier = Modifier.width(22.dp),
            style = TextStyle(fontFamily = AnimusMono, fontSize = 11.sp),
            color = if (selected) ink else glass.inkDim,
        )
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(
                node.label,
                style = TextStyle(fontFamily = AnimusDisplay, fontWeight = FontWeight.Medium, fontSize = 19.sp, letterSpacing = 0.3.sp),
                color = ink,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            Text(
                "${node.conceptType} · seen ${node.activationCount}× · $links link${if (links == 1) "" else "s"}",
                style = TextStyle(fontFamily = AnimusMono, fontSize = 10.5.sp),
                color = meta,
            )
        }
        StrengthRing(strength, selected)
    }
}

@Composable
private fun StrengthRing(fraction: Float, selected: Boolean) {
    val glass = AppTheme.glass
    val arc = when {
        selected -> glass.onSelection
        glass.dark -> Color(0xFFEFFFFF)
        else -> glass.slab
    }
    val track = if (selected) glass.onSelection.copy(alpha = 0.18f) else glass.rimDim
    val pct = (fraction * 100).toInt()
    Box(Modifier.size(40.dp).semantics { contentDescription = "strength $pct percent" }, contentAlignment = Alignment.Center) {
        Canvas(Modifier.size(40.dp)) {
            val sw = 2.dp.toPx()
            val inset = 4.dp.toPx()
            val sz = Size(size.width - 2 * inset, size.height - 2 * inset)
            val tl = Offset(inset, inset)
            drawArc(track, 0f, 360f, false, tl, sz, style = Stroke(sw))
            if (glass.dark && !selected) drawArc(MemoryLight.copy(alpha = 0.3f), -90f, 360f * fraction, false, tl, sz, style = Stroke(sw * 3.5f))
            drawArc(arc, -90f, 360f * fraction, false, tl, sz, style = Stroke(sw * 1.1f))
        }
        Text("$pct", style = TextStyle(fontFamily = AnimusDisplay, fontSize = 13.sp), color = if (selected) glass.onSelection else glass.ink)
    }
}

/** The opened concept: description, Japanese summary, strongest links, ARM-E history. */
@Composable
private fun ConceptDetail(
    node: ConceptNode?,
    edges: List<GraphEdgeView>,
    labels: Map<String, String>,
    loading: Boolean,
) {
    val glass = AppTheme.glass
    Column(
        Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 8.dp)
            .background(if (glass.dark) Color(0xFF02101A).copy(alpha = 0.55f) else Color.White.copy(alpha = 0.55f))
            .border(1.dp, glass.rimDim)
            .cornerTicks(inset = 0.dp, arm = 10.dp)
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        if (loading || node == null) {
            NeonIndeterminate()
            return@Column
        }
        val description = node.textRaw.removePrefix("${node.label}:").trim()
        if (description.isNotEmpty()) {
            Text(description, style = TextStyle(fontFamily = AnimusBody, fontSize = 14.sp, lineHeight = 20.sp), color = glass.ink)
        }
        if (node.textSummary.isNotEmpty() && node.textSummary != description) {
            Text(node.textSummary, style = TextStyle(fontSize = 13.sp, lineHeight = 21.sp), color = glass.inkMuted)
        }

        // Both directions of each link, strongest first, named.
        val byOther = LinkedHashMap<String, GraphEdgeView>()
        edges.sortedByDescending { it.hebbWeight }.forEach { e ->
            val other = if (e.source == node.nodeId) e.target else e.source
            if (other != node.nodeId && other !in byOther) byOther[other] = e
        }
        if (byOther.isNotEmpty()) {
            Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
                SmallCaps("STRONGEST LINKS")
                byOther.entries.take(5).forEach { (other, e) -> LinkBar(labels[other] ?: other.take(8) + "…", e) }
            }
        }

        Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
            SmallCaps("ARM-E")
            Telemetry("m̄_t %.2f · d̄ %.2f · r̄_t %.2f · seen %d×".format(node.meanMT, node.meanDominance, node.meanRT, node.activationCount))
            Telemetry("formed ${formatTimestamp(node.createdAt)} · last seen ${formatTimestamp(node.updatedAt)}")
            node.recentEvents.takeLast(4).reversed().forEach { Telemetry(formatArmeEvent(it)) }
        }
    }
}

@Composable
private fun LinkBar(label: String, e: GraphEdgeView) {
    val glass = AppTheme.glass
    val frac = (e.hebbWeight / W_MAX).toFloat().coerceIn(0f, 1f)
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        Text(
            label,
            modifier = Modifier.width(118.dp),
            style = TextStyle(fontFamily = AnimusBody, fontSize = 13.sp),
            color = glass.ink,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
        Canvas(Modifier.weight(1f).height(8.dp)) {
            val y = size.height / 2
            drawLine(glass.rimDim, Offset(0f, y), Offset(size.width, y), 1.dp.toPx())
            val end = size.width * frac
            val c = if (glass.dark) Color(0xFFEFFFFF) else glass.slab
            if (glass.dark) drawLine(MemoryLight.copy(alpha = 0.35f), Offset(0f, y), Offset(end, y), 6.dp.toPx())
            drawLine(Brush.horizontalGradient(listOf(c, if (glass.dark) MemoryLight else c)), Offset(0f, y), Offset(end, y), 3.dp.toPx())
        }
        Text("%.2f".format(e.hebbWeight), style = TextStyle(fontFamily = AnimusMono, fontSize = 11.sp), color = glass.inkMuted)
    }
}

@Composable
private fun SmallCaps(text: String) {
    Text(text, style = TextStyle(fontFamily = AnimusMono, fontSize = 10.sp, letterSpacing = 2.sp), color = AppTheme.glass.inkDim)
}

/** recentEvents entries are compact JSON: {"m_t":..,"dominance":..,"r_t":..,"ts":..} (epoch s). */
private fun formatArmeEvent(raw: String): String = runCatching {
    val j = JSONObject(raw)
    "m_t %.2f · dom %.2f · r_t %.2f · %s".format(
        j.optDouble("m_t"), j.optDouble("dominance"), j.optDouble("r_t"),
        formatTimestamp((j.optDouble("ts") * 1000).toLong()),
    )
}.getOrDefault(raw)
