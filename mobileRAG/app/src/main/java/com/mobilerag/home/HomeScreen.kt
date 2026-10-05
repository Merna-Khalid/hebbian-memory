package com.mobilerag.home

import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.spring
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
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import com.mobilerag.hebbian.ArmeHistoryEntry
import com.mobilerag.hebbian.ui.Constellation
import com.mobilerag.ui.Routes
import com.mobilerag.ui.theme.AnimusBody
import com.mobilerag.ui.theme.AnimusDisplay
import com.mobilerag.ui.theme.AnimusMono
import com.mobilerag.ui.theme.AnimusPanel
import com.mobilerag.ui.theme.AppTheme
import com.mobilerag.ui.theme.ArcGauge
import com.mobilerag.ui.theme.ChromaText
import com.mobilerag.ui.theme.DayBadge
import com.mobilerag.ui.theme.GlassPill
import com.mobilerag.ui.theme.GlassSection
import com.mobilerag.ui.theme.GlowRule
import com.mobilerag.ui.theme.HudIcon
import com.mobilerag.ui.theme.HudStat
import com.mobilerag.ui.theme.PauseAmbientWhile
import com.mobilerag.ui.theme.StatRow
import com.mobilerag.ui.theme.Telemetry
import com.mobilerag.ui.theme.drawOuterGlow
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.TimeUnit

/**
 * Home — the entity's state (docs/ui-animus.md): the HUD, the memory as a constellation in a
 * frosted panel with its gauges, the menu, and the engine readouts below.
 */
@Composable
fun HomeScreen(
    onNavigate: (String) -> Unit,
    vm: HomeViewModel = viewModel(),
) {
    val state by vm.ui.collectAsState()

    // Refresh whenever the screen is resumed (returning from Chat/Tutor/Settings…).
    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) vm.refresh()
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    val listState = rememberLazyListState()
    PauseAmbientWhile(listState.isScrollInProgress)
    LazyColumn(
        state = listState,
        modifier = Modifier.fillMaxSize().padding(horizontal = 14.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        contentPadding = PaddingValues(bottom = 24.dp),
    ) {
        item { Hud(state) }
        item { EntityPanel(state) }
        item { MenuPanel(state, onNavigate) }
        item { ResumeLine(state, onNavigate) }
        item { EngineCard(state) }
        if (state.arme.isNotEmpty()) item { ArmeCard(state.arme) }
    }
}

// ── HUD ──────────────────────────────────────────────────────────────────────

@Composable
private fun Hud(state: HomeUiState) {
    Column(Modifier.fillMaxWidth().statusBarsPadding().padding(top = 10.dp)) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Row(Modifier.weight(1f), horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                HudStat(HudIcon.Concepts, count(state.concepts), "concepts")
                HudStat(HudIcon.Links, count(state.hebbianEdges), "links")
                HudStat(HudIcon.Fading, count(state.fading), "fading")
                HudStat(HudIcon.Documents, count(state.documents), "documents")
            }
            state.bornAt?.let { DayBadge(dayOf(it)) }
        }
        GlowRule(Modifier.padding(top = 6.dp))
    }
}

// ── The entity ───────────────────────────────────────────────────────────────

@Composable
private fun EntityPanel(state: HomeUiState) {
    val glass = AppTheme.glass
    AnimusPanel(Modifier.fillMaxWidth().height(386.dp), scan = true) {
        Box(Modifier.fillMaxSize()) {
            state.constellation?.let { layout ->
                Constellation(
                    layout,
                    Modifier.fillMaxWidth().height(274.dp),
                    top = 0.46f,
                    bottom = 0.93f,
                )
            }
            Column(Modifier.padding(start = 18.dp, top = 18.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(
                    buildString {
                        append(state.spaceName.ifEmpty { "—" }.uppercase())
                        state.bornAt?.let { append(" · BORN ").append(SimpleDateFormat("dd·MM·yyyy", Locale.US).format(Date(it))) }
                    },
                    style = TextStyle(fontFamily = AnimusMono, fontSize = 10.sp, letterSpacing = 2.sp),
                    color = glass.inkMuted,
                )
                ChromaText(
                    if (state.consolidating) "CONSOLIDATING" else "AWAKE",
                    style = TextStyle(
                        fontFamily = AnimusDisplay,
                        fontWeight = FontWeight.Light,
                        fontSize = if (state.consolidating) 40.sp else 54.sp,
                        lineHeight = 56.sp,
                        letterSpacing = 4.sp,
                    ),
                    flicker = true,
                )
            }
            if (state.constellation?.isEmpty != false && state.concepts == 0) {
                Text(
                    "No memories yet. Talk to Takemura and they'll start forming here.",
                    modifier = Modifier.align(Alignment.Center).padding(horizontal = 40.dp).offset(y = (-20).dp),
                    style = MaterialTheme.typography.bodyMedium,
                    color = glass.inkMuted,
                )
            }
            GaugeBand(state, Modifier.align(Alignment.BottomCenter))
        }
    }
}

@Composable
private fun GaugeBand(state: HomeUiState, modifier: Modifier = Modifier) {
    val glass = AppTheme.glass
    val total = state.concepts ?: 0
    fun frac(n: Int?) = if (total > 0 && n != null) n.toFloat() / total else 0f
    fun pct(n: Int?) = if (total > 0 && n != null) "${(100f * n / total).toInt()}%" else "—"
    Row(
        modifier
            .fillMaxWidth()
            .height(112.dp)
            .background(
                Brush.verticalGradient(
                    if (glass.dark) listOf(Color(0xFF020E14).copy(alpha = 0.2f), Color(0xFF020E14).copy(alpha = 0.62f))
                    else listOf(Color.White.copy(alpha = 0.15f), Color.White.copy(alpha = 0.55f)),
                ),
            )
            .drawBehind { drawLine(glass.rimDim, Offset(0f, 0f), Offset(size.width, 0f), 1f) },
        horizontalArrangement = Arrangement.SpaceEvenly,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        ArcGauge(frac(state.linked), pct(state.linked), "linked")
        ArcGauge(frac(state.fading), pct(state.fading), "fading")
        ArcGauge(frac(state.recalled), pct(state.recalled), "recalled")
    }
}

// ── Menu ─────────────────────────────────────────────────────────────────────

private data class MenuEntry(val key: String, val title: String, val route: String, val detail: String?, val marked: Boolean = false)

private val RowHeight = 48.dp

/**
 * The menu, Black Flag style: lowercase entries in a frosted panel, one bright bar that slides
 * to whatever you last opened (it remembers, like a pause menu's cursor).
 */
@Composable
private fun MenuPanel(state: HomeUiState, onNavigate: (String) -> Unit) {
    val glass = AppTheme.glass
    val entries = listOf(
        MenuEntry("tutor", "tutor", Routes.Tutor, "Takemura"),
        MenuEntry("notes", "notes", Routes.Chat, "Khepri"),
        MenuEntry("practice", "practice", Routes.Practice, state.fading?.takeIf { it > 0 }?.let { "$it fading" }, marked = (state.fading ?: 0) > 0),
        MenuEntry("memory", "memory", Routes.Memory, state.concepts?.let { "$it concepts" }),
        MenuEntry("brain", "second brain", Routes.Brain, null),
        MenuEntry("graph", "entity graph", Routes.Graph, state.entities?.let { "$it entities" }),
        MenuEntry("options", "options", Routes.Settings, null),
    )
    var selected by rememberSaveable { mutableStateOf("tutor") }
    val index = entries.indexOfFirst { it.key == selected }.coerceAtLeast(0)
    val barY by animateDpAsState(RowHeight * index, spring(dampingRatio = 0.8f, stiffness = 500f), label = "menu-bar")

    AnimusPanel(Modifier.fillMaxWidth(), ticks = false) {
        Box(Modifier.fillMaxWidth().padding(vertical = 10.dp)) {
            // The selection bar, behind the rows.
            Box(
                Modifier
                    .offset(y = barY)
                    .fillMaxWidth()
                    .height(RowHeight)
                    .drawBehind {
                        if (glass.dark) drawOuterGlow(androidx.compose.ui.graphics.RectangleShape, glass.glow.copy(alpha = 0.28f), 16.dp.toPx())
                    }
                    .background(glass.selectionBrush()),
            )
            Column {
                entries.forEach { e ->
                    MenuRow(e, selected = e.key == selected) {
                        selected = e.key
                        onNavigate(e.route)
                    }
                }
            }
        }
    }
}

@Composable
private fun MenuRow(e: MenuEntry, selected: Boolean, onClick: () -> Unit) {
    val glass = AppTheme.glass
    val ink = if (selected) glass.onSelection else glass.ink
    val sub = if (selected) glass.onSelection.copy(alpha = 0.78f) else glass.inkMuted
    Row(
        Modifier
            .fillMaxWidth()
            .height(RowHeight)
            .clickable(role = Role.Button, onClick = onClick)
            .padding(start = 18.dp, end = 14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            e.title,
            style = TextStyle(fontFamily = AnimusDisplay, fontWeight = if (selected) FontWeight.Medium else FontWeight.Normal, fontSize = 20.sp, letterSpacing = 0.8.sp),
            color = ink,
        )
        e.detail?.let {
            Spacer(Modifier.width(10.dp))
            if (e.marked && !selected) {
                Box(Modifier.size(7.dp).rotate(45f).background(glass.mark))
                Spacer(Modifier.width(7.dp))
                Text(it, style = TextStyle(fontFamily = AnimusMono, fontSize = 11.sp), color = glass.mark)
            } else {
                Text(it, style = TextStyle(fontFamily = AnimusBody, fontSize = 13.sp), color = sub)
            }
        }
        Spacer(Modifier.weight(1f))
        if (selected) {
            Box(
                Modifier.size(22.dp).border(1.dp, glass.onSelection.copy(alpha = 0.55f), CircleShape),
                contentAlignment = Alignment.Center,
            ) {
                Text("›", style = TextStyle(fontFamily = AnimusDisplay, fontSize = 16.sp, fontWeight = FontWeight.Medium), color = glass.onSelection)
            }
        }
    }
}

// ── Resume ───────────────────────────────────────────────────────────────────

@Composable
private fun ResumeLine(state: HomeUiState, onNavigate: (String) -> Unit) {
    val glass = AppTheme.glass
    Row(Modifier.fillMaxWidth().padding(horizontal = 4.dp), verticalAlignment = Alignment.CenterVertically) {
        Row(
            Modifier
                .clickable(role = Role.Button) { onNavigate(Routes.Tutor) }
                .height(44.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(9.dp),
        ) {
            Box(
                Modifier
                    .size(22.dp)
                    .drawBehind { if (glass.dark) drawCircle(glass.glow.copy(alpha = 0.25f), radius = size.minDimension * 0.75f) }
                    .border(1.5.dp, if (glass.dark) Color(0xFFEFFFFF) else glass.ink, CircleShape),
                contentAlignment = Alignment.Center,
            ) {
                Box(Modifier.size(8.dp).background(glass.mark, CircleShape))
            }
            Text("resume takemura", style = TextStyle(fontFamily = AnimusBody, fontSize = 13.sp), color = glass.inkMuted)
            Text("\\", style = TextStyle(fontFamily = AnimusBody, fontSize = 13.sp), color = glass.inkDim)
            Text(
                if (state.consolidating) "memory consolidating" else "memory awake",
                style = TextStyle(fontFamily = AnimusBody, fontSize = 13.sp),
                color = glass.ink,
            )
        }
        Spacer(Modifier.weight(1f))
        Text(
            state.lastConsolidatedAt?.let { "slept ${ago(it)}" } ?: "no sleep yet",
            style = TextStyle(fontFamily = AnimusMono, fontSize = 10.sp, letterSpacing = 1.sp),
            color = glass.inkDim,
        )
    }
}

// ── Engine + ARM-E (the readouts, kept below the fold) ───────────────────────

@Composable
private fun EngineCard(state: HomeUiState) {
    GlassSection(
        title = "Engine",
        trailing = { GlassPill(state.preset, active = true) },
    ) {
        StatRow("Model", state.selectedModel ?: "none installed")
        StatRow("Resident", state.residentModel)
        StatRow("Retrieval", state.retrievalMode)
        StatRow("Backend", state.backend)
        StatRow("Entity store", state.graphStoreId ?: "—", mono = true)
        StatRow("Hebbian store", state.hebbianStoreId ?: "—", mono = true)
        StatRow("Documents · chunks", "${count(state.documents)} · ${count(state.chunks)}")
        StatRow("Entities · edges", "${count(state.entities)} · ${count(state.edges)}")
    }
}

@Composable
private fun ArmeCard(entries: List<ArmeHistoryEntry>) {
    val glass = AppTheme.glass
    GlassSection(title = "ARM-E · recent") {
        entries.forEach { e ->
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                Box(
                    Modifier
                        .padding(end = 10.dp)
                        .size(6.dp)
                        .rotate(45f)
                        .background(if (e.gated) glass.mark else glass.glow),
                )
                Telemetry(
                    "${e.quadrant} · m_t ${e.mT}" + if (e.gated) " · gated" else "",
                    color = if (e.gated) glass.mark else null,
                )
            }
        }
    }
}

private fun count(value: Int?): String = value?.toString() ?: "—"

/** Day 1 is the day the first concept formed. */
private fun dayOf(bornAt: Long): Int =
    (TimeUnit.MILLISECONDS.toDays(System.currentTimeMillis() - bornAt) + 1).toInt().coerceAtLeast(1)

private fun ago(at: Long): String {
    val m = TimeUnit.MILLISECONDS.toMinutes(System.currentTimeMillis() - at).coerceAtLeast(0)
    return when {
        m < 1 -> "just now"
        m < 60 -> "${m}m ago"
        m < 60 * 24 -> "${m / 60}h ago"
        else -> "${m / (60 * 24)}d ago"
    }
}
