package com.mobilerag.graph.ui

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
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.sp
import com.mobilerag.ui.theme.AnimusBody
import com.mobilerag.ui.theme.AnimusDisplay
import com.mobilerag.ui.theme.AnimusMono
import com.mobilerag.ui.theme.AnimusPanel
import com.mobilerag.ui.theme.NeonIndeterminate
import androidx.lifecycle.viewmodel.compose.viewModel
import com.mobilerag.core.GraphStore
import com.mobilerag.ui.theme.AppTheme
import com.mobilerag.ui.theme.EmptyState
import com.mobilerag.ui.theme.ErrorBanner
import com.mobilerag.ui.theme.GhostButton
import com.mobilerag.ui.theme.GlassCard
import com.mobilerag.ui.theme.GlassPill
import com.mobilerag.ui.theme.NeonButton
import com.mobilerag.ui.theme.NeonProgress
import com.mobilerag.ui.theme.ScreenHeader
import com.mobilerag.ui.theme.SectionLabel
import com.mobilerag.ui.theme.Telemetry
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private fun formatTimestamp(millis: Long): String =
    SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault()).format(Date(millis))

@Composable
fun GraphScreen(vm: GraphViewModel = viewModel()) {
    val ui by vm.ui.collectAsState()
    val listState = rememberLazyListState()
    // 0 = list, 1 = constellation (3D)
    var tab by rememberSaveable { mutableStateOf(0) }
    LaunchedEffect(tab, ui.layout3d) { if (tab == 1) vm.ensureLayout3D() }
    // The detail card is inserted at the top of the list; scroll up so it's visible.
    LaunchedEffect(ui.selected?.id) {
        if (ui.selected != null) listState.animateScrollToItem(0)
    }

    Column(Modifier.fillMaxSize().imePadding()) {
        ScreenHeader(
            title = "Entity Graph",
            subtitle = "store · ${ui.storeId.ifEmpty { "…" }}",
        )

        Column(
            Modifier.padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            // The three counts are the screen's headline; tiles beat a run-on sentence.
            // The constellation gets the screen: tiles and actions only show with the list.
            if (tab == 0) Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                CountTile("Entities", ui.entityCount, Modifier.weight(1f))
                CountTile("Edges", ui.edgeCount, Modifier.weight(1f))
                CountTile("Communities", ui.communityCount, Modifier.weight(1f))
            }

            if (tab == 0) Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                NeonButton(
                    text = if (ui.rebuilding) "Rebuilding…" else "Rebuild graph",
                    onClick = vm::rebuild,
                    enabled = !ui.rebuilding,
                    loading = ui.rebuilding,
                )
                GhostButton(
                    text = if (ui.summarizing) "Summarizing…" else "Communities",
                    onClick = vm::rebuildCommunities,
                    enabled = !ui.summarizing,
                )
            }

            ui.rebuildProgress?.let { p ->
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    NeonProgress(if (p.chunksTotal > 0) p.chunksDone.toFloat() / p.chunksTotal else 0f)
                    Telemetry("${p.docName} · ${p.chunksDone}/${p.chunksTotal} chunks · doc ${p.docsDone + 1}/${p.docsTotal}")
                }
            }
            if (ui.summarizing) {
                Telemetry("loading the LLM and summarizing each community — a couple of minutes…")
            }
            ui.lastStats?.let { Telemetry("last build · $it") }
            if (ui.communityLastRun > 0) {
                Telemetry("summaries · ${formatTimestamp(ui.communityLastRun)}")
            }

            AnimatedVisibility(visible = ui.communityError != null, enter = fadeIn(), exit = fadeOut()) {
                ui.communityError?.let { ErrorBanner(it) }
            }

            ViewTabs(tab, onTab = { tab = it })
            if (tab == 0) SearchField(value = ui.query, onValueChange = vm::onQueryChange)
        }

        Spacer(Modifier.height(10.dp))

        if (tab == 1) {
            ConstellationPane(ui, Modifier.weight(1f))
        } else if (ui.entities.isEmpty()) {
            EmptyState(
                glyph = "◇",
                message = if (ui.query.isBlank()) "No entities yet — import documents or rebuild the graph."
                else "Nothing matches \"${ui.query}\".",
                modifier = Modifier.weight(1f),
            )
        } else {
            LazyColumn(
                state = listState,
                modifier = Modifier.weight(1f),
                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 4.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                ui.selected?.let { selected ->
                    item(key = "detail") {
                        EntityDetail(
                            entity = selected,
                            edges = ui.selectedEdges,
                            chunks = ui.selectedChunks,
                            loading = ui.detailLoading,
                            onClose = { vm.select(selected) },
                        )
                    }
                }
                items(ui.entities, key = { it.id }) { entity ->
                    EntityRow(entity, selected = ui.selected?.id == entity.id, onClick = { vm.select(entity) })
                }
            }
        }
    }
}

@Composable
private fun CountTile(label: String, value: Int, modifier: Modifier = Modifier) {
    val glass = AppTheme.glass
    Column(
        modifier
            .clip(MaterialTheme.shapes.medium)
            .background(glass.panelLow)
            .padding(vertical = 12.dp, horizontal = 10.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(3.dp),
    ) {
        Text(
            value.toString(),
            style = TextStyle(fontFamily = AnimusDisplay, fontWeight = FontWeight.Light, fontSize = 30.sp),
            color = glass.ink,
        )
        Text(
            label.uppercase(),
            style = TextStyle(fontFamily = AnimusMono, fontSize = 10.sp, letterSpacing = 1.sp),
            color = glass.inkDim,
            maxLines = 1,
            softWrap = false,
        )
    }
}

@Composable
private fun SearchField(value: String, onValueChange: (String) -> Unit) {
    val glass = AppTheme.glass
    val shape = RoundedCornerShape(3.dp)
    Box(
        Modifier
            .fillMaxWidth()
            .clip(shape)
            .background(glass.panelLow)
            .border(1.dp, glass.rimDim, shape)
            .padding(horizontal = 16.dp, vertical = 13.dp),
    ) {
        if (value.isEmpty()) {
            Text(
                "Search entities…",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
            )
        }
        BasicTextField(
            value = value,
            onValueChange = onValueChange,
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            textStyle = LocalTextStyle.current.merge(
                MaterialTheme.typography.bodyMedium.copy(color = MaterialTheme.colorScheme.onSurface),
            ),
            cursorBrush = SolidColor(MaterialTheme.colorScheme.primary),
        )
    }
}

@Composable
private fun EntityRow(entity: GraphStore.Entity, selected: Boolean, onClick: () -> Unit) {
    val glass = AppTheme.glass
    GlassCard(
        Modifier.fillMaxWidth(),
        onClick = onClick,
        elevation = if (selected) 18.dp else 6.dp,
        glowAlpha = if (selected) 0.55f else 0.15f,
        fill = if (selected) null else glass.panelLow,
    ) {
        Row(
            Modifier.fillMaxWidth().padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(
                    entity.name,
                    style = MaterialTheme.typography.titleMedium,
                    color = if (selected) MaterialTheme.colorScheme.primary
                    else MaterialTheme.colorScheme.onSurface,
                )
                Telemetry("${entity.mentionCount} mentions")
            }
            Spacer(Modifier.width(12.dp))
            GlassPill(entity.type, active = selected)
        }
    }
}

@Composable
private fun EntityDetail(
    entity: GraphStore.Entity,
    edges: List<EdgeGroup>,
    chunks: List<String>,
    loading: Boolean,
    onClose: () -> Unit,
) {
    val glass = AppTheme.glass
    GlassCard(Modifier.fillMaxWidth(), elevation = 22.dp, glowAlpha = 0.6f) {
        Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(
                        entity.name,
                        style = MaterialTheme.typography.headlineSmall,
                        color = MaterialTheme.colorScheme.onSurface,
                    )
                    Telemetry("${entity.type} · ${entity.mentionCount} mentions")
                }
                GhostButton("Close", onClose)
            }

            if (loading) {
                NeonProgress(0f)
            } else {
                if (edges.isNotEmpty()) {
                    SectionLabel("Related entities")
                    edges.forEach { group ->
                        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                group.otherName,
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurface,
                                modifier = Modifier.weight(1f),
                            )
                            GlassPill("${group.relation} ×${group.count}")
                        }
                    }
                }
                if (chunks.isNotEmpty()) {
                    Spacer(Modifier.height(2.dp))
                    SectionLabel("Source chunks")
                    chunks.forEachIndexed { i, chunk ->
                        if (i > 0) HorizontalDivider(color = glass.rimDim)
                        Telemetry(chunk, Modifier.padding(vertical = 4.dp))
                    }
                }
                if (edges.isEmpty() && chunks.isEmpty()) {
                    Text(
                        "No graph data for this entity.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

/** List / constellation switch: the selected side is the bright bar, the other a rim. */
@Composable
private fun ViewTabs(selected: Int, onTab: (Int) -> Unit) {
    val glass = AppTheme.glass
    Row(Modifier.fillMaxWidth().height(44.dp).border(1.dp, glass.rimDim)) {
        listOf("list", "constellation").forEachIndexed { i, label ->
            val on = i == selected
            Box(
                Modifier
                    .weight(1f)
                    .fillMaxHeight()
                    .then(if (on) Modifier.background(if (glass.dark) Color(0xFFECFFFF) else glass.slab) else Modifier)
                    .clickable(role = Role.Tab) { onTab(i) },
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    label,
                    style = TextStyle(fontFamily = AnimusDisplay, fontWeight = if (on) FontWeight.Medium else FontWeight.Normal, fontSize = 17.sp, letterSpacing = 0.8.sp),
                    color = if (on) glass.onSelection else glass.inkMuted,
                )
            }
        }
    }
}

/** The 3D constellation, a legend, and — once a star is tapped — its card. */
@Composable
private fun ConstellationPane(ui: GraphUiState, modifier: Modifier = Modifier) {
    val glass = AppTheme.glass
    var selected by remember(ui.layout3d) { mutableStateOf<Int?>(null) }
    Box(modifier.fillMaxWidth()) {
        val layout = ui.layout3d
        when {
            layout == null -> Box(Modifier.fillMaxSize().padding(24.dp), contentAlignment = Alignment.Center) {
                Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    NeonIndeterminate(Modifier.width(160.dp))
                    Telemetry("placing ${ui.entityCount} entities in space…")
                }
            }
            layout.isEmpty -> EmptyState(glyph = "◇", message = "No entities yet — import documents or rebuild the graph.")
            else -> {
                val centerY by animateFloatAsState(if (selected != null) 0.32f else 0.5f, label = "sphere-centre")
                EntityConstellation3D(layout, selected, onSelect = { selected = it }, modifier = Modifier.fillMaxSize(), centerY = centerY)
                Column(Modifier.padding(start = 18.dp, top = 4.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(
                        "${layout.nodes.size} ENTITIES · ${layout.links.size} LINKS",
                        style = TextStyle(fontFamily = AnimusMono, fontSize = 10.sp, letterSpacing = 1.6.sp),
                        color = glass.inkMuted,
                    )
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                        layout.communities.take(8).forEachIndexed { i, c ->
                            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(3.dp)) {
                                Box(Modifier.size(6.dp).rotate(45f).background(communityHue(i, glass.dark)))
                                Text("${c.size}", style = TextStyle(fontFamily = AnimusMono, fontSize = 9.sp), color = glass.inkDim)
                            }
                        }
                    }
                    Text("drag · pinch · tap a star", style = TextStyle(fontFamily = AnimusMono, fontSize = 9.sp, letterSpacing = 1.2.sp), color = glass.inkDim)
                }
                selected?.let { i -> StarCard(layout, i, Modifier.align(Alignment.BottomCenter).padding(horizontal = 14.dp, vertical = 12.dp)) }
            }
        }
    }
}

@Composable
private fun StarCard(layout: GraphLayout3D.Result, i: Int, modifier: Modifier = Modifier) {
    val glass = AppTheme.glass
    val node = layout.nodes[i]
    val community = layout.communities.getOrNull(node.community)
    val neighbours = remember(layout, i) {
        (layout.neighbours[i] ?: emptyList()).sortedByDescending { layout.nodes[it].weight }.take(6).map { layout.nodes[it].name }
    }
    AnimusPanel(modifier.fillMaxWidth()) {
        Column(Modifier.padding(horizontal = 18.dp, vertical = 16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(node.name, style = TextStyle(fontFamily = AnimusDisplay, fontWeight = FontWeight.Normal, fontSize = 24.sp, letterSpacing = 0.6.sp), color = glass.ink)
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Box(Modifier.size(7.dp).rotate(45f).background(communityHue(node.community, glass.dark)))
                Telemetry(
                    "${node.type} · ${node.mentions} mention${if (node.mentions == 1) "" else "s"}" +
                        (community?.let { " · community of ${it.size}" } ?: " · no community"),
                )
            }
            community?.summary?.takeIf { it.isNotBlank() }?.let {
                Text(it, style = TextStyle(fontFamily = AnimusBody, fontSize = 13.sp, lineHeight = 18.sp), color = glass.inkMuted, maxLines = 3, overflow = TextOverflow.Ellipsis)
            }
            if (neighbours.isNotEmpty()) {
                Text(
                    "linked · " + neighbours.joinToString("  ·  "),
                    style = TextStyle(fontFamily = AnimusBody, fontSize = 12.sp, lineHeight = 17.sp),
                    color = glass.ink,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }
    }
}
