package com.mobilerag.chat

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
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
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.horizontalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.List
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.mobilerag.rag.RagDatabase
import com.mobilerag.rag.RagPipeline
import com.mobilerag.settings.rememberAppSettings
import com.mobilerag.ui.avatar.Persona
import com.mobilerag.ui.avatar.PersonaAvatar
import com.mobilerag.ui.avatar.UserAvatar
import com.mobilerag.ui.theme.AppTheme
import com.mobilerag.ui.theme.ErrorBanner
import com.mobilerag.ui.theme.GlassIconButton
import com.mobilerag.ui.theme.GlassPill
import com.mobilerag.ui.theme.NeonIconButton
import com.mobilerag.ui.theme.NeonProgress
import com.mobilerag.ui.theme.SectionLabel
import com.mobilerag.ui.theme.Telemetry
import com.mobilerag.ui.theme.glassPanel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(vm: ChatViewModel = viewModel()) {
    val ui by vm.ui.collectAsState()
    val settings = rememberAppSettings()
    var input by rememberSaveable { mutableStateOf("") }
    var showDocs by remember { mutableStateOf(false) }
    var sheetCitation by remember { mutableStateOf<Citation?>(null) }
    val listState = rememberLazyListState()

    // MIME detection for .md is unreliable; import */* and filter by extension in the importer
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        if (uris.isNotEmpty()) vm.importDocuments(uris)
    }

    Column(Modifier.fillMaxSize().imePadding()) {
        ChatHeader(
            ui = ui,
            onImport = { picker.launch(arrayOf("*/*")) },
            onToggleDocs = { showDocs = !showDocs },
            docsOpen = showDocs,
            onSelectModel = vm::selectModel,
            onCycleRetrieval = {
                vm.setRetrievalMode(
                    when (ui.retrievalMode) {
                        RagPipeline.MODE_VECTOR -> RagPipeline.MODE_HYBRID
                        RagPipeline.MODE_HYBRID -> RagPipeline.MODE_GLOBAL
                        else -> RagPipeline.MODE_VECTOR
                    },
                )
            },
            onCycleBackend = {
                vm.setGenerationBackend(
                    when (ui.generationBackend) {
                        RagPipeline.BACKEND_LLAMACPP -> RagPipeline.BACKEND_MLKIT
                        RagPipeline.BACKEND_MLKIT -> RagPipeline.BACKEND_AUTO
                        else -> RagPipeline.BACKEND_LLAMACPP
                    },
                )
            },
        )

        AnimatedVisibility(visible = ui.error != null, enter = fadeIn(), exit = fadeOut()) {
            ui.error?.let { ErrorBanner(it, Modifier.padding(horizontal = 16.dp, vertical = 6.dp)) }
        }

        ui.indexing?.let { p ->
            IngestProgress(
                label = "Indexing ${p.docName}",
                detail = "${p.chunksDone}/${p.chunksTotal} chunks · doc ${p.docsDone + 1}/${p.docsTotal}",
                progress = if (p.chunksTotal > 0) p.chunksDone.toFloat() / p.chunksTotal else 0f,
            )
        }
        ui.graphIndexing?.let { p ->
            IngestProgress(
                label = "Building graph · ${p.docName}",
                detail = "${p.chunksDone}/${p.chunksTotal} chunks · doc ${p.docsDone + 1}/${p.docsTotal}",
                progress = if (p.chunksTotal > 0) p.chunksDone.toFloat() / p.chunksTotal else 0f,
            )
        }

        AnimatedVisibility(visible = showDocs, enter = expandVertically() + fadeIn(), exit = shrinkVertically() + fadeOut()) {
            DocumentPanel(ui.documents, ui.graphStats, onDelete = vm::deleteDocument)
        }

        if (ui.messages.isEmpty()) {
            ChatWelcome(hasDocs = ui.documents.isNotEmpty(), modifier = Modifier.weight(1f))
        } else {
            LazyColumn(
                state = listState,
                modifier = Modifier.weight(1f).fillMaxWidth(),
                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 12.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                items(ui.messages, key = { it.id }) { msg ->
                    MessageBubble(
                        msg = msg,
                        learnerName = settings.learnerName,
                        onCitation = { sheetCitation = it },
                    )
                }
            }
            LaunchedEffect(ui.messages.lastOrNull()?.text?.length) {
                listState.animateScrollToItem(ui.messages.size - 1)
            }
        }

        Composer(
            value = input,
            onValueChange = { input = it },
            onSend = { vm.ask(input); input = "" },
            enabled = !ui.generating,
            placeholder = "Ask about your notes…",
        )
    }

    sheetCitation?.let { c ->
        ModalBottomSheet(
            onDismissRequest = { sheetCitation = null },
            containerColor = MaterialTheme.colorScheme.surface,
        ) {
            Column(Modifier.padding(horizontal = 24.dp).padding(bottom = 40.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                SectionLabel("Source [${c.index}]")
                Text(c.docName, style = MaterialTheme.typography.titleLarge)
                Telemetry("relevance %.3f".format(c.score))
                Spacer(Modifier.height(4.dp))
                Text(c.text, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

// ── Header ───────────────────────────────────────────────────────────────────

@Composable
private fun ChatHeader(
    ui: ChatUiState,
    onImport: () -> Unit,
    onToggleDocs: () -> Unit,
    docsOpen: Boolean,
    onSelectModel: (String) -> Unit,
    onCycleRetrieval: () -> Unit,
    onCycleBackend: () -> Unit,
) {
    Column(Modifier.fillMaxWidth().statusBarsPadding()) {
        Row(
            Modifier.fillMaxWidth().padding(start = 20.dp, end = 14.dp, top = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            PersonaAvatar(Persona.Personal, size = 52.dp, active = ui.generating)
            Spacer(Modifier.width(14.dp))
            Column(Modifier.weight(1f)) {
                Text(
                    Persona.Personal.displayName,
                    style = MaterialTheme.typography.titleLarge,
                    color = MaterialTheme.colorScheme.onBackground,
                )
                Text(
                    when {
                        ui.generating -> "thinking…"
                        ui.modelLoading -> "loading model…"
                        else -> "${ui.documents.size} documents indexed"
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                )
            }
            GlassIconButton(Icons.Default.Add, "Import documents", onImport)
            Spacer(Modifier.width(8.dp))
            GlassIconButton(Icons.AutoMirrored.Filled.List, "Documents", onToggleDocs, active = docsOpen)
        }

        // Pipeline controls: one tap cycles each. Kept as pills so the active configuration is
        // always readable without opening settings.
        Row(
            Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState())
                .padding(horizontal = 20.dp, vertical = 10.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            ModelPill(ui.models, ui.selectedModel, onSelectModel)
            GlassPill(
                text = when (ui.retrievalMode) {
                    RagPipeline.MODE_HYBRID -> "Hybrid"
                    RagPipeline.MODE_GLOBAL -> "Global"
                    else -> "Vector"
                },
                active = true,
                onClick = onCycleRetrieval,
            )
            GlassPill(
                text = when (ui.generationBackend) {
                    RagPipeline.BACKEND_LLAMACPP -> "llama.cpp"
                    RagPipeline.BACKEND_MLKIT -> "ML Kit"
                    else -> "Auto"
                },
                onClick = onCycleBackend,
            )
        }
    }
}

@Composable
private fun ModelPill(models: List<String>, selected: String?, onSelect: (String) -> Unit) {
    var open by remember { mutableStateOf(false) }
    Box {
        GlassPill(
            text = selected ?: "no model",
            onClick = { if (models.isNotEmpty()) open = true },
        )
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            models.forEach { m ->
                DropdownMenuItem(text = { Text(m) }, onClick = { onSelect(m); open = false })
            }
        }
    }
}

// ── Progress & documents ─────────────────────────────────────────────────────

@Composable
private fun IngestProgress(label: String, detail: String, progress: Float) {
    Column(
        Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 6.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            SectionLabel(label)
            Telemetry("%d%%".format((progress * 100).toInt()))
        }
        NeonProgress(progress)
        Telemetry(detail)
    }
}

@Composable
private fun DocumentPanel(
    documents: List<RagDatabase.Document>,
    graphStats: String?,
    onDelete: (Long) -> Unit,
) {
    Column(
        Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 6.dp)
            .glassPanel(MaterialTheme.shapes.large, elevation = 10.dp)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        SectionLabel("Indexed documents")
        if (documents.isEmpty()) {
            Text(
                "Nothing indexed yet — import .txt or .md notes with +.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        documents.forEach { doc ->
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(doc.name, style = MaterialTheme.typography.bodyMedium, maxLines = 1)
                    Telemetry("${doc.chunkCount} chunks")
                }
                GlassIconButton(Icons.Default.Delete, "Delete ${doc.name}", { onDelete(doc.id) }, size = 36.dp)
            }
        }
        graphStats?.let { Telemetry("graph · $it") }
    }
}

// ── Welcome ──────────────────────────────────────────────────────────────────

@Composable
private fun ChatWelcome(hasDocs: Boolean, modifier: Modifier = Modifier) {
    Box(modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
            modifier = Modifier.padding(horizontal = 36.dp),
        ) {
            PersonaAvatar(Persona.Personal, size = 96.dp, ringWidth = 3.dp)
            Text(
                Persona.Personal.displayName,
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.onBackground,
            )
            Text(
                if (hasDocs) "Ask me anything about your notes."
                else "Import notes (.txt / .md) with + and I'll remember them for you.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = androidx.compose.ui.text.style.TextAlign.Center,
            )
        }
    }
}

// ── Bubbles ──────────────────────────────────────────────────────────────────

@Composable
private fun MessageBubble(msg: ChatMessage, learnerName: String, onCitation: (Citation) -> Unit) {
    val glass = AppTheme.glass
    val shape = if (msg.isUser) {
        RoundedCornerShape(3.dp, 3.dp, 0.dp, 3.dp)
    } else {
        RoundedCornerShape(3.dp, 3.dp, 3.dp, 0.dp)
    }

    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = if (msg.isUser) Arrangement.End else Arrangement.Start,
        verticalAlignment = Alignment.Bottom,
    ) {
        if (!msg.isUser) {
            PersonaAvatar(Persona.Personal, size = 34.dp, active = msg.streaming)
            Spacer(Modifier.width(10.dp))
        }

        Column(
            Modifier.widthIn(max = 300.dp),
            horizontalAlignment = if (msg.isUser) Alignment.End else Alignment.Start,
        ) {
            Box(
                Modifier
                    .then(
                        when {
                            msg.isError -> Modifier
                                .clip(shape)
                                .background(MaterialTheme.colorScheme.error.copy(alpha = 0.14f))
                                .border(1.dp, MaterialTheme.colorScheme.error.copy(alpha = 0.4f), shape)
                            msg.isUser -> Modifier
                                .clip(shape)
                                .background(glass.accentWash(0.26f))
                                .border(1.dp, glass.accent.start.copy(alpha = 0.35f), shape)
                            else -> Modifier.glassPanel(shape, elevation = 8.dp, glowAlpha = 0.25f)
                        },
                    )
                    .padding(horizontal = 16.dp, vertical = 12.dp),
            ) {
                if (msg.text.isEmpty() && msg.streaming) {
                    TypingDots()
                } else {
                    Text(
                        msg.text,
                        style = MaterialTheme.typography.bodyLarge,
                        color = if (msg.isError) MaterialTheme.colorScheme.error
                        else MaterialTheme.colorScheme.onSurface,
                    )
                }
            }

            if (msg.citations.isNotEmpty()) {
                LazyRow(
                    modifier = Modifier.padding(top = 8.dp),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    items(msg.citations, key = { it.index }) { c ->
                        GlassPill("[${c.index}] ${c.docName}", onClick = { onCitation(c) })
                    }
                }
            }
            msg.stats?.let { Telemetry(it, Modifier.padding(top = 6.dp, start = 4.dp)) }
        }

        if (msg.isUser) {
            Spacer(Modifier.width(10.dp))
            UserAvatar(learnerName.ifBlank { "You" }, size = 34.dp)
        }
    }
}

@Composable
private fun TypingDots() {
    val glass = AppTheme.glass
    val transition = rememberInfiniteTransition(label = "typing")
    Row(
        horizontalArrangement = Arrangement.spacedBy(5.dp),
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.height(20.dp),
    ) {
        repeat(3) { i ->
            val a by transition.animateFloat(
                initialValue = 0.25f,
                targetValue = 1f,
                animationSpec = infiniteRepeatable(
                    animation = tween(560, delayMillis = i * 150),
                    repeatMode = RepeatMode.Reverse,
                ),
                label = "dot$i",
            )
            Box(
                Modifier
                    .size(7.dp)
                    .alpha(a)
                    .clip(CircleShape)
                    .background(glass.accentBrush),
            )
        }
    }
}

// ── Composer ─────────────────────────────────────────────────────────────────

@Composable
private fun Composer(
    value: String,
    onValueChange: (String) -> Unit,
    onSend: () -> Unit,
    enabled: Boolean,
    placeholder: String,
) {
    val glass = AppTheme.glass
    Row(
        Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 10.dp),
        verticalAlignment = Alignment.Bottom,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Box(
            Modifier
                .weight(1f)
                .glassPanel(
                    shape = RoundedCornerShape(3.dp),
                    fill = glass.panelHigh,
                    elevation = 6.dp,
                    glowAlpha = 0.2f,
                )
                .padding(horizontal = 18.dp, vertical = 14.dp),
        ) {
            if (value.isEmpty()) {
                Text(
                    placeholder,
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                )
            }
            BasicTextField(
                value = value,
                onValueChange = onValueChange,
                modifier = Modifier.fillMaxWidth(),
                textStyle = LocalTextStyle.current.merge(
                    MaterialTheme.typography.bodyLarge.copy(color = MaterialTheme.colorScheme.onSurface),
                ),
                cursorBrush = SolidColor(MaterialTheme.colorScheme.primary),
                maxLines = 5,
            )
        }
        NeonIconButton(
            icon = Icons.AutoMirrored.Filled.Send,
            contentDescription = "Send",
            onClick = onSend,
            enabled = enabled && value.isNotBlank(),
        )
    }
}
