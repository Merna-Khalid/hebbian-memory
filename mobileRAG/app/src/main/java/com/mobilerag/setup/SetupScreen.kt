package com.mobilerag.setup

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.work.WorkInfo
import androidx.work.WorkManager
import com.mobilerag.ui.theme.AnimusBody
import com.mobilerag.ui.theme.AnimusDisplay
import com.mobilerag.ui.theme.AnimusMono
import com.mobilerag.ui.theme.AnimusPanel
import com.mobilerag.ui.theme.AppTheme
import com.mobilerag.ui.theme.ArcGauge
import com.mobilerag.ui.theme.ChromaText
import com.mobilerag.ui.theme.ErrorBanner
import com.mobilerag.ui.theme.GlowRule
import com.mobilerag.ui.theme.NeonButton
import kotlinx.coroutines.flow.map

/**
 * First run: the models aren't on the phone yet. Explains what they are, downloads them once
 * (ModelDownloadWorker), and hands over to the app when they're all in place.
 */
@Composable
fun SetupScreen(onReady: () -> Unit, onSkip: () -> Unit) {
    val context = LocalContext.current
    val glass = AppTheme.glass
    val info by remember {
        WorkManager.getInstance(context).getWorkInfosForUniqueWorkFlow(ModelDownloadWorker.NAME).map { it.firstOrNull() }
    }.collectAsState(initial = null)

    val state = info?.state
    val running = state == WorkInfo.State.RUNNING || state == WorkInfo.State.ENQUEUED
    // Enqueued after a failed attempt = waiting out a retry delay (dropped Wi-Fi, server hiccup).
    val waiting = state == WorkInfo.State.ENQUEUED && (info?.runAttemptCount ?: 0) > 0
    val done = info?.progress?.getLong(ModelDownloadWorker.KEY_DONE, -1L) ?: -1L
    val total = info?.progress?.getLong(ModelDownloadWorker.KEY_TOTAL, -1L) ?: -1L
    val current = info?.progress?.getString(ModelDownloadWorker.KEY_FILE)
    val failed = state == WorkInfo.State.FAILED
    val error = info?.outputData?.getString(ModelDownloadWorker.KEY_ERROR)

    LaunchedEffect(state) {
        if (state == WorkInfo.State.SUCCEEDED || ModelCatalog.ready(context)) onReady()
    }

    val remaining = remember(state) { ModelCatalog.remainingBytes(context) }
    val free = remember(state) { ModelCatalog.freeBytes(context) }
    val enoughSpace = free > remaining + 300_000_000L
    val fraction = when {
        total > 0 && done >= 0 -> done.toFloat() / total
        else -> 1f - remaining.toFloat() / ModelCatalog.totalBytes
    }.coerceIn(0f, 1f)

    val notifyPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {
        ModelDownloadWorker.start(context)
    }
    fun begin() {
        val needsAsk = Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        if (needsAsk) notifyPermission.launch(Manifest.permission.POST_NOTIFICATIONS) else ModelDownloadWorker.start(context)
    }

    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .statusBarsPadding()
            .navigationBarsPadding()
            .padding(horizontal = 18.dp, vertical = 18.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text("FIRST RUN", style = TextStyle(fontFamily = AnimusMono, fontSize = 10.sp, letterSpacing = 2.sp), color = glass.inkMuted)
        Column {
            ChromaText(
                "AWAKENING",
                style = TextStyle(fontFamily = AnimusDisplay, fontWeight = FontWeight.Light, fontSize = 46.sp, letterSpacing = 5.sp),
                flicker = true,
            )
            GlowRule(Modifier.padding(top = 2.dp))
        }
        Text(
            "Takemura and Khepri think on your phone, never in the cloud. Before they can wake, " +
                "they need their minds: three models, downloaded once. After that, everything runs offline.",
            style = TextStyle(fontFamily = AnimusBody, fontSize = 15.sp, lineHeight = 22.sp),
            color = glass.ink,
        )

        AnimusPanel(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                    ArcGauge(
                        fraction, "${(fraction * 100).toInt()}%",
                        when {
                            waiting -> "reconnecting"
                            running -> "downloading"
                            else -> "ready to fetch"
                        },
                    )
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text("%.1f GB".format(ModelCatalog.totalBytes / 1e9), style = TextStyle(fontFamily = AnimusDisplay, fontWeight = FontWeight.Light, fontSize = 28.sp), color = glass.ink)
                        Text(
                            when {
                                waiting -> "connection lost · retrying shortly"
                                running && current != null -> "now · $current"
                                else -> "%.1f GB left · %.1f GB free".format(remaining / 1e9, free / 1e9)
                            },
                            style = TextStyle(fontFamily = AnimusMono, fontSize = 11.sp),
                            color = glass.inkMuted,
                        )
                    }
                }
                ModelCatalog.Group.entries.forEach { g ->
                    val items = ModelCatalog.items.filter { it.group == g }
                    val have = remember(state, done) { items.all { ModelCatalog.isPresent(context, it) } }
                    Row(verticalAlignment = Alignment.Top, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        Box(Modifier.padding(top = 6.dp).size(7.dp).rotate(45f).background(if (have) glass.glow else glass.rimBright))
                        Column {
                            Text(
                                "${g.title}  ·  %.0f MB".format(items.sumOf { it.bytes } / 1e6),
                                style = TextStyle(fontFamily = AnimusDisplay, fontSize = 17.sp, letterSpacing = 0.4.sp),
                                color = glass.ink,
                            )
                            Text(g.role, style = TextStyle(fontFamily = AnimusBody, fontSize = 12.sp), color = glass.inkMuted)
                        }
                    }
                }
            }
        }

        if (failed) ErrorBanner("The download stopped: ${error ?: "unknown error"}. Tap to try again — it resumes where it left off.")
        if (!enoughSpace) ErrorBanner("Not enough free space: %.1f GB needed, %.1f GB free.".format((remaining + 300_000_000L) / 1e9, free / 1e9))

        NeonButton(
            text = when {
                waiting -> "Retry now"
                running -> "Pause"
                done > 0 || remaining < ModelCatalog.totalBytes -> "Resume download"
                else -> "Download (%.1f GB)".format(remaining / 1e9)
            },
            onClick = { if (running && !waiting) ModelDownloadWorker.pause(context) else begin() },
            enabled = running || enoughSpace,
            modifier = Modifier.fillMaxWidth(),
        )
        Text(
            "Use Wi-Fi. You can leave the app — the download keeps going and shows its progress in the notifications.",
            style = TextStyle(fontFamily = AnimusBody, fontSize = 13.sp, lineHeight = 19.sp),
            color = glass.inkMuted,
        )
        Spacer(Modifier.height(4.dp))
        Text(
            "look around first",
            modifier = Modifier.clickable(role = Role.Button, onClick = onSkip).padding(vertical = 10.dp),
            style = TextStyle(fontFamily = AnimusMono, fontSize = 11.sp, letterSpacing = 1.5.sp),
            color = glass.inkDim,
        )
    }
}
