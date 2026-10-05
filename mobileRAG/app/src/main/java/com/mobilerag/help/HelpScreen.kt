package com.mobilerag.help

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import com.mobilerag.ui.theme.AppTheme
import com.mobilerag.ui.theme.GradientTitle
import com.mobilerag.ui.theme.ScreenHeader

/** In-app documentation. Renders assets/help.md with a minimal markdown subset:
 *  `#`–`###` headings, `-` bullets, `**bold**`, `` `code` ``, `---` dividers, paragraphs. */
@Composable
fun HelpScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val markdown = remember {
        runCatching { context.assets.open("help.md").bufferedReader().readText() }
            .getOrElse { "# Help unavailable\n\nhelp.md is missing from the APK assets." }
    }
    Column(Modifier.fillMaxSize()) {
        ScreenHeader(title = "Help", subtitle = "how everything works", onBack = onBack)
        Column(
            Modifier
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp),
        ) {
            markdown.lines().forEach { line -> MarkdownLine(line) }
            Spacer(Modifier.height(48.dp))
        }
    }
}

@Composable
private fun MarkdownLine(line: String) {
    when {
        line.startsWith("### ") -> {
            Spacer(Modifier.height(10.dp))
            Text(
                inlineMarkup(line.removePrefix("### ")),
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.onSurface,
            )
        }
        line.startsWith("## ") -> {
            Spacer(Modifier.height(14.dp))
            Text(
                inlineMarkup(line.removePrefix("## ")),
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.primary,
            )
        }
        line.startsWith("# ") -> {
            Spacer(Modifier.height(16.dp))
            GradientTitle(line.removePrefix("# "), style = MaterialTheme.typography.headlineSmall)
        }
        line.trim() == "---" -> HorizontalDivider(
            Modifier.padding(vertical = 10.dp),
            color = AppTheme.glass.rimDim,
        )
        line.startsWith("- ") -> Text(
            inlineMarkup("•  " + line.removePrefix("- ")),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(start = 8.dp, top = 2.dp),
        )
        line.isBlank() -> Spacer(Modifier.height(6.dp))
        else -> Text(
            inlineMarkup(line),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

/** Parses `**bold**` and `` `code` `` spans into an AnnotatedString. */
@Composable
private fun inlineMarkup(text: String): AnnotatedString {
    val codeColor = MaterialTheme.colorScheme.tertiary
    return buildAnnotatedString {
        var i = 0
        while (i < text.length) {
            when {
                text.startsWith("**", i) -> {
                    val end = text.indexOf("**", i + 2)
                    if (end < 0) { append(text.substring(i)); break }
                    withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { append(text.substring(i + 2, end)) }
                    i = end + 2
                }
                text.startsWith("`", i) -> {
                    val end = text.indexOf('`', i + 1)
                    if (end < 0) { append(text.substring(i)); break }
                    withStyle(SpanStyle(fontFamily = FontFamily.Monospace, color = codeColor)) {
                        append(text.substring(i + 1, end))
                    }
                    i = end + 1
                }
                else -> {
                    val next = listOf(text.indexOf("**", i), text.indexOf('`', i))
                        .filter { it >= 0 }.minOrNull() ?: text.length
                    append(text.substring(i, next))
                    i = next
                }
            }
        }
    }
}
