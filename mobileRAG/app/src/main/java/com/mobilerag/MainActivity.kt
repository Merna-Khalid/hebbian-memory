package com.mobilerag

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.SystemBarStyle
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.asPaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.setValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.draw.drawBehind
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.mobilerag.chat.ChatScreen
import com.mobilerag.generation.LlmResidency
import com.mobilerag.graph.ui.GraphScreen
import com.mobilerag.hebbian.SleepWorker
import com.mobilerag.hebbian.ui.HebbianGraphScreen
import com.mobilerag.hebbian.ui.PracticeScreen
import com.mobilerag.hebbian.ui.SecondBrainScreen
import com.mobilerag.hebbian.ui.TutorScreen
import com.mobilerag.help.HelpScreen
import com.mobilerag.home.HomeScreen
import com.mobilerag.japanese.JaTranslator
import com.mobilerag.rag.CommunityIndexWorker
import com.mobilerag.settings.AppSettings
import com.mobilerag.settings.SettingsScreen
import com.mobilerag.settings.SettingsSubScreen
import com.mobilerag.settings.rememberAppSettings
import com.mobilerag.spikes.ui.SpikeScreen
import com.mobilerag.ui.Routes
import com.mobilerag.ui.avatar.Persona
import com.mobilerag.ui.avatar.PersonaAvatar
import com.mobilerag.ui.theme.AppTheme
import com.mobilerag.setup.ModelCatalog
import com.mobilerag.setup.SetupScreen
import com.mobilerag.ui.theme.AnimusBackground
import com.mobilerag.ui.theme.AnimusGlyph
import com.mobilerag.ui.theme.Glyph
import com.mobilerag.ui.theme.cornerTicks
import com.mobilerag.ui.theme.glassPanel

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        // The Animus atmosphere runs edge to edge; the system bars sit on top of it transparently.
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        // Daily community-summary rebuild (KEEP: restarts don't reset the schedule)
        CommunityIndexWorker.enqueuePeriodic(applicationContext)
        // Nightly Hebbian "sleep": cortical consolidation while idle + charging
        SleepWorker.enqueuePeriodic(applicationContext)
        setContent {
            val settings = rememberAppSettings()
            val dark = when (settings.themeMode) {
                AppSettings.THEME_LIGHT -> false
                AppSettings.THEME_DARK -> true
                else -> isSystemInDarkTheme()
            }
            // enableEdgeToEdge() above picks bar icon colors from the SYSTEM theme; re-apply
            // with the app's so a forced light theme on a dark phone (or vice versa) doesn't
            // leave invisible status-bar icons.
            DisposableEffect(dark) {
                val style = SystemBarStyle.auto(
                    android.graphics.Color.TRANSPARENT,
                    android.graphics.Color.TRANSPARENT,
                ) { dark }
                enableEdgeToEdge(statusBarStyle = style, navigationBarStyle = style)
                onDispose {}
            }
            AppTheme(themeMode = settings.themeMode, accent = settings.themeAccent) {
                AnimusBackground {
                    // First run: no models yet → the Awakening screen downloads them.
                    var modelsReady by remember { mutableStateOf(ModelCatalog.ready(applicationContext)) }
                    if (modelsReady) AppNavHost()
                    else SetupScreen(onReady = { modelsReady = true }, onSkip = { modelsReady = true })
                }
            }
        }
    }

    // ComponentActivity implements ComponentCallbacks2; trim events arrive here without registration.
    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        // Never reach the tab ViewModels from here: `by viewModels()` is lazy, so touching them
        // on the first trim event (e.g. backgrounding) CREATED both, and each init warmed up
        // the GGUF with its own system prompt — the chat/tutor double load. The engine is
        // process-wide; unloadIfIdle skips while any tab is generating.
        if (level >= TRIM_MEMORY_MODERATE) LlmResidency.unloadIfIdle(applicationContext)
        // The translation model is another resident chunk competing with the GGUF.
        if (level >= TRIM_MEMORY_RUNNING_LOW) JaTranslator.close()
    }
}

@Composable
fun AppNavHost(navController: NavHostController = rememberNavController()) {
    val backStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = backStackEntry?.destination?.route
    // Bottom bar only on the five top-level destinations; hidden on subroutes
    // (Practice/Brain/Graph/Spikes/Help and settings/*).
    val showBottomBar = currentRoute in Routes.bottomBar
    // ViewModels are scoped to the activity (not the NavBackStackEntry) so tab state —
    // chat history, tutor session, second-brain stream — survives navigation.
    val activity = LocalContext.current as ComponentActivity

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        // Transparent so the atmosphere shows through; the nav bar provides its own glass.
        containerColor = Color.Transparent,
        contentColor = MaterialTheme.colorScheme.onBackground,
        bottomBar = {
            if (showBottomBar) {
                GlassNavBar(
                    currentRoute = currentRoute,
                    onNavigate = { route ->
                        navController.navigate(route) {
                            popUpTo(Routes.Home) { saveState = true }
                            launchSingleTop = true
                            restoreState = true
                        }
                    },
                )
            }
        },
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Routes.Home,
            modifier = Modifier.padding(innerPadding),
        ) {
            composable(Routes.Home) {
                HomeScreen(
                    onNavigate = { route -> navController.navigate(route) },
                    vm = viewModel(viewModelStoreOwner = activity),
                )
            }
            composable(Routes.Chat) { ChatScreen(vm = viewModel(viewModelStoreOwner = activity)) }
            composable(Routes.Tutor) { TutorScreen(vm = viewModel(viewModelStoreOwner = activity)) }
            composable(Routes.Memory) { HebbianGraphScreen(vm = viewModel(viewModelStoreOwner = activity)) }
            composable(Routes.Settings) {
                SettingsScreen(onNavigate = { route -> navController.navigate(route) })
            }
            composable(Routes.Practice) { PracticeScreen(vm = viewModel(viewModelStoreOwner = activity)) }
            composable(Routes.Brain) { SecondBrainScreen(vm = viewModel(viewModelStoreOwner = activity)) }
            composable(Routes.Graph) { GraphScreen(vm = viewModel(viewModelStoreOwner = activity)) }
            composable(Routes.Spikes) { SpikeScreen(vm = viewModel(viewModelStoreOwner = activity)) }
            composable(Routes.Help) {
                HelpScreen(onBack = { navController.popBackStack() })
            }
            listOf(
                Routes.SettingsAppearance,
                Routes.SettingsPerformance,
                Routes.SettingsLearner,
                Routes.SettingsSpaces,
                Routes.SettingsModels,
                Routes.SettingsData,
                Routes.SettingsAbout,
            ).forEach { route ->
                composable(route) { SettingsSubScreen(route, navController) }
            }
        }
    }
}

/**
 * Floating frosted tab bar: hard-edged glass with corner ticks; the selected tab carries a
 * glowing bar above its label. The Chat and Tutor tabs render their persona's avatar — the two
 * conversations are with *someone*, and the bar should say who.
 */
@Composable
private fun GlassNavBar(currentRoute: String?, onNavigate: (String) -> Unit) {
    val glass = AppTheme.glass
    Box(
        Modifier
            .fillMaxWidth()
            .padding(
                start = 14.dp,
                end = 14.dp,
                top = 6.dp,
                // Lift clear of the gesture handle rather than letting it overlap the tabs.
                bottom = WindowInsets.navigationBars.asPaddingValues().calculateBottomPadding() + 8.dp,
            ),
    ) {
        Row(
            Modifier
                .fillMaxWidth()
                .glassPanel(
                    shape = MaterialTheme.shapes.small,
                    elevation = 16.dp,
                    glowAlpha = 0.35f,
                    frost = true,
                )
                .cornerTicks(inset = 4.dp, arm = 7.dp)
                .padding(horizontal = 6.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Routes.bottomBar.forEach { route ->
                NavTab(
                    route = route,
                    selected = currentRoute == route,
                    onClick = { onNavigate(route) },
                    modifier = Modifier.weight(1f),
                )
            }
        }
    }
}

@Composable
private fun NavTab(
    route: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val glass = AppTheme.glass
    val tint by animateColorAsState(
        targetValue = if (selected) MaterialTheme.colorScheme.primary
        else MaterialTheme.colorScheme.onSurfaceVariant,
        label = "tab-tint",
    )
    val lift by animateFloatAsState(
        targetValue = if (selected) 1.06f else 1f,
        animationSpec = spring(dampingRatio = 0.55f, stiffness = 500f),
        label = "tab-lift",
    )
    val interaction = remember { MutableInteractionSource() }

    Column(
        modifier
            .clickable(interactionSource = interaction, indication = null, onClick = onClick)
            .padding(vertical = 4.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(3.dp),
    ) {
        Box(Modifier.scale(lift), contentAlignment = Alignment.Center) {
            when (route) {
                Routes.Chat -> PersonaAvatar(Persona.Personal, size = 26.dp, ringWidth = 1.5.dp)
                Routes.Tutor -> PersonaAvatar(Persona.Tutor, size = 26.dp, ringWidth = 1.5.dp)
                else -> AnimusGlyph(
                    glyph = when (route) {
                        Routes.Home -> Glyph.Entity
                        Routes.Memory -> Glyph.Constellation
                        else -> Glyph.Options
                    },
                    tint = tint,
                    lit = selected,
                    modifier = Modifier.size(22.dp).semantics { contentDescription = bottomBarLabel(route) },
                )
            }
        }
        Text(
            bottomBarLabel(route).uppercase(),
            style = MaterialTheme.typography.labelSmall.copy(fontSize = 9.sp, letterSpacing = 1.4.sp),
            color = tint,
            maxLines = 1,
        )
        // The active indicator: a short glowing bar under the selected tab.
        Spacer(
            Modifier
                .size(width = if (selected) 22.dp else 0.dp, height = 2.dp)
                .drawBehind {
                    if (selected && glass.dark) drawRect(glass.glow.copy(alpha = 0.35f), topLeft = androidx.compose.ui.geometry.Offset(0f, -3.dp.toPx()), size = androidx.compose.ui.geometry.Size(size.width, size.height + 6.dp.toPx()))
                }
                .background(if (selected) glass.accentBrush else androidx.compose.ui.graphics.SolidColor(Color.Transparent)),
        )
    }
}

private fun bottomBarLabel(route: String): String = when (route) {
    Routes.Home -> "Home"
    Routes.Chat -> "Khepri"
    Routes.Tutor -> "Tutor"
    Routes.Memory -> "Memory"
    else -> "Settings"
}
