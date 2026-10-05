package com.mobilerag.hebbian

import com.mobilerag.hebbian.emotion.StubEmotionClassifier
import com.mobilerag.hebbian.store.HebbianStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * No-models engine — port of tutor_engine.py's `build_stub_engine` (HEBBIAN_STUB=1).
 * Deterministic SHA-256-seeded hash embeddings (768-dim to match EmbeddingGemma),
 * a template generator that fakes extraction/practice JSON in the exact shapes
 * ConceptExtractor/TutorEngine parse, and the keyword [StubEmotionClassifier].
 * The whole Hebbian system (graph, ARM-E, Hebbian updates, practice loop) runs
 * on-device with no embedding or GGUF models.
 *
 * Difference from Python: stub_chat_fn keyed off the system prompt; on-device the
 * system prompts travel in-band inside the prompt (see TutorEngine KDoc), and
 * ConceptExtractor's `generate` receives only EXTRACTION_PROMPT_TEMPLATE — so the
 * stub keys off substrings of the single prompt string instead.
 */

/** Deterministic hash embedding — SHA-256(text) seeds the RNG so vectors survive restarts. */
fun stubEmbedding(text: String, dim: Int = 768): FloatArray {
    val digest = MessageDigest.getInstance("SHA-256").digest(text.toByteArray(Charsets.UTF_8))
    val seed = ((digest[0].toInt() and 0xFF) shl 24) or
        ((digest[1].toInt() and 0xFF) shl 16) or
        ((digest[2].toInt() and 0xFF) shl 8) or
        (digest[3].toInt() and 0xFF)
    val rng = Random(seed)
    // Box–Muller gaussians, then L2-normalize (same spirit as Python's random.gauss vector).
    val v = FloatArray(dim)
    var i = 0
    while (i < dim) {
        val u1 = rng.nextDouble().let { if (it <= 0.0) Double.MIN_VALUE else it }
        val u2 = rng.nextDouble()
        val r = sqrt(-2.0 * ln(u1))
        v[i] = (r * cos(2.0 * PI * u2)).toFloat()
        if (i + 1 < dim) v[i + 1] = (r * kotlin.math.sin(2.0 * PI * u2)).toFloat()
        i += 2
    }
    var norm = 0.0
    for (x in v) norm += x.toDouble() * x
    norm = sqrt(norm)
    if (norm > 0.0) for (j in v.indices) v[j] = (v[j] / norm).toFloat()
    return v
}

/** Truncate to the first 2 sentences (Python stub_summary_fn — same as IngestPipeline's fallback). */
fun stubSummary(text: String): String {
    val sentences = text.replace("?", ".").replace("!", ".").split(".")
    return sentences.take(2).map { it.trim() }.filter { it.isNotEmpty() }.joinToString(". ") + "."
}

/**
 * Template generator port of Python's stub_chat_fn. Produces JSON in exactly the shapes
 * ConceptExtractor.extract and TutorEngine.nextPractice/submitPracticeAnswer parse.
 */
fun stubGenerate(prompt: String, @Suppress("UNUSED_PARAMETER") maxTokens: Int): String {
    // ConceptExtractor asks for JSON — give it one concept per turn. The extraction
    // request is the extraction system rules followed by EXTRACTION_PROMPT_TEMPLATE,
    // so key off the template's first line.
    val templateAt = prompt.indexOf("Extract concepts and relations")
    if (templateAt >= 0) {
        // Pull the label from the text between the --- delimiters, not the
        // "Extract concepts..." instruction line itself.
        val parts = prompt.substring(templateAt).split("---")
        val source = if (parts.size >= 2) parts[1] else prompt
        val words = source.split(Regex("\\s+"))
            .map { it.trim('.', ',', '!', '?', '—', '"', '\'') }
            // Skip transcript role prefixes: with concept identity, labelling every turn
            // "User:" would collapse the stub graph to one node.
            .filter { it.length > 3 && it != "User:" && it != "Assistant:" }
        val label = (words.firstOrNull() ?: "conversation").take(40)
        return JSONObject()
            .put("concepts", JSONArray().put(
                JSONObject()
                    .put("label", label)
                    .put("concept_type", "fact")
                    .put("description", "Concept mentioned by the user: $label.")
                    .put("source_lang", detectLang(label))
            ))
            .put("relations", JSONArray())
            .toString()
    }
    if ("practice-exercise generator" in prompt) {
        var target = "(unknown)"
        for (line in prompt.lines()) {
            if (line.startsWith("- ") && ":" in line) {
                target = line.substring(2).substringBefore(":").trim()
                break
            }
        }
        return JSONObject()
            .put("exercise_type", "translation")
            .put("prompt", "(stub) Translate to Japanese using “$target”: I want to eat ramen.")
            .put("hint", "")
            .put("answer", "ラーメンを食べたいです。")
            .toString()
    }
    if ("practice-answer grader" in prompt) {
        var expected = ""
        var given = ""
        for (line in prompt.lines()) {
            if (line.startsWith("Expected answer:")) expected = line.substringAfter(":").trim()
            if (line.startsWith("Learner's answer:")) given = line.substringAfter(":").trim()
        }
        val correct = expected.isNotEmpty() && expected == given
        return JSONObject()
            .put("correct", correct)
            .put(
                "feedback",
                if (correct) "(stub) 正解！Memory reinforced."
                else "(stub) Not quite — expected: ${expected.ifEmpty { "?" }}",
            )
            .toString()
    }
    val mem = if ("記憶" in prompt) "記憶" else ""
    return "(stub) You said: “$prompt”. " +
        "Real responses need the LFM model — run without HEBBIAN_STUB. $mem"
}

/**
 * Wire a fully functional model-less [TutorEngine]: IngestPipeline (hash embeddings,
 * truncating summary, stub emotion) + ConceptExtractor (template JSON) +
 * PracticeScheduler (persisted under `stateDir/practice_state`) + ConsolidationScheduler
 * stub. The TutorEngine gets [stubGenerate] as both the streaming and complete source —
 * `llm` stays null.
 */
fun buildStubEngine(
    store: HebbianStore,
    stateDir: File,
    preset: DomainPreset = DOMAIN_PRESETS.getValue("japanese_tutor"),
): TutorEngine {
    val consolidation = ConsolidationScheduler(store)
    val pipeline = IngestPipeline(
        store = store,
        embedFn = { text -> stubEmbedding(text) },
        summaryFn = { text -> stubSummary(text) },
        emotionClassifier = StubEmotionClassifier(),
        consolidation = consolidation,
    )
    val extractor = ConceptExtractor(
        pipeline = pipeline,
        generate = ::stubGenerate,
        store = store,
        extractionSystem = preset.extractionSystem,
        conceptTypes = preset.conceptTypes,
        relationTypes = preset.relationTypes,
    )
    val practiceScheduler = PracticeScheduler(File(stateDir, "practice_state"))

    // Fake streaming: emit the canned response in small chunks so Token events flow.
    val stubSend: (String, Int) -> Flow<String> = { p, m ->
        flow {
            val full = stubGenerate(p, m)
            var i = 0
            while (i < full.length) {
                val end = minOf(i + 24, full.length)
                emit(full.substring(i, end))
                i = end
            }
        }
    }

    return TutorEngine(
        store = store,
        pipeline = pipeline,
        extractor = extractor,
        practiceScheduler = practiceScheduler,
        embedQuery = { text -> stubEmbedding(text) },
        preset = preset,
        llm = null,
        sendTokens = stubSend,
        completeFn = ::stubGenerate,
    )
}
