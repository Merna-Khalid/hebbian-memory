package com.mobilerag.hebbian

import android.content.Context
import android.util.Log
import com.mobilerag.hebbian.cortical.CorticalConsolidator
import com.mobilerag.hebbian.emotion.EmotionClassifier
import com.mobilerag.hebbian.emotion.PhysioFusedEmotionClassifier
import com.mobilerag.hebbian.emotion.StubEmotionClassifier
import com.mobilerag.hebbian.llm.HebbianLlmSession
import com.mobilerag.hebbian.signals.SignalSource
import com.mobilerag.hebbian.store.HebbianStoreFactory
import com.mobilerag.embeddings.EmbeddingGemmaEngine
import com.mobilerag.profile.SpaceManager
import com.mobilerag.rag.RagPipeline
import com.mobilerag.settings.AppSettings
import java.io.File

/**
 * Production engine factory — port of tutor_engine.py's `build_engine`.
 *
 * Wires: HebbianStore (LadybugDB/SQLite via [HebbianStoreFactory]) + EmbeddingGemma
 * (768-d, replacing BGE-M3) + HebbianLlmSession (single GGUF persona, replacing
 * LFM2.5's stateless chat_fn) + IngestPipeline + ConceptExtractor + PracticeScheduler
 * + ConsolidationScheduler with the space's CorticalConsolidator (cortical GAT).
 *
 * Per-call system prompts from Python become in-band prompt text (see
 * HebbianLlmSession KDoc): the extraction system prompt is baked into the session at
 * load time, summary/practice prompts are plain user-turn text. Emotion classification
 * uses the wave-1 [StubEmotionClassifier] (Python's vad-bert has no on-device port);
 * pass [signalSource] to fuse live heart rate into arousal instead.
 */
object HebbianEngineFactory {

    private const val TAG = "HebbianEngineFactory"

    /** model_setup.py summary_fn — JA prompt; the model handles English input too. */
    private const val SUMMARY_MAX_TOKENS = 80

    /** Texts up to this length are stored as their own summary (no LLM call) — comfortably
     *  above a label plus a 1–2 sentence extracted description. */
    private const val SUMMARY_PASSTHROUGH_CHARS = 320

    /** Learner-profile block appended to the persona system prompt when set in Settings. */
    private fun learnerBlock(context: Context): String =
        AppSettings(context).learnerPromptBlock()?.let { "\n\n$it" } ?: ""

    suspend fun build(
        context: Context,
        domain: String = "japanese_tutor",
        signalSource: SignalSource? = null,
    ): TutorEngine {
        val preset = DOMAIN_PRESETS.getValue(domain)
        val spaceDir = SpaceManager.activeDir(context)
        val store = HebbianStoreFactory.create(context, spaceDir)
        val embedding = EmbeddingGemmaEngine.create(context)
        val llm = HebbianLlmSession(context)

        var emotion: EmotionClassifier = StubEmotionClassifier()
        if (signalSource != null) {
            emotion = PhysioFusedEmotionClassifier(
                textClassifier = emotion,
                physioSource = signalSource,
            )
        }

        // 1–2 sentence summary via the resident LLM (model_setup.py summary_fn:
        // JA prompt, max_new_tokens=80). Falls back to the 2-sentence truncation when
        // the model isn't loaded — same fallback IngestPipeline uses without summaryFn.
        // Extracted concepts arrive as "label: 1–2 sentence description" — already a
        // summary. Re-summarizing them cost one full LLM turn per concept (up to 6 per
        // extraction batch, the bulk of the "memorizing" time on the 8B), so only text
        // longer than a summary (e.g. the raw-transcript fallback node) goes to the model.
        val summaryFn: suspend (String) -> String = { text ->
            if (text.length <= SUMMARY_PASSTHROUGH_CHARS) text.trim() else try {
                llm.complete(
                    "次のテキストを1〜2文で要約してください。\n\n" +
                        "テキスト: ${text.take(1000)}\n\n要約:",
                    SUMMARY_MAX_TOKENS,
                )
            } catch (t: Throwable) {
                if (t is kotlinx.coroutines.CancellationException) throw t
                Log.w(TAG, "summary via LLM failed (${t.message}) — truncation fallback")
                stubSummary(text)
            }
        }

        // One cortical consolidator per space, shared with the other persona's engine.
        val consolidation = ConsolidationScheduler(store, CorticalConsolidator.forSpace(spaceDir))
        val pipeline = IngestPipeline(
            store = store,
            // lambda, not a reference: embedDocument's optional title param breaks the (String) shape
            embedFn = { text -> embedding.embedDocument(text) },
            summaryFn = summaryFn,
            emotionClassifier = emotion,
            consolidation = consolidation,
        )
        val extractor = ConceptExtractor(
            pipeline = pipeline,
            generate = llm::complete,
            store = store,
            extractionSystem = preset.extractionSystem,
            conceptTypes = preset.conceptTypes,
            relationTypes = preset.relationTypes,
        )
        val practiceScheduler = PracticeScheduler(File(spaceDir, "practice_state"))

        return TutorEngine(
            store = store,
            pipeline = pipeline,
            extractor = extractor,
            practiceScheduler = practiceScheduler,
            embedQuery = embedding::embedQuery,
            preset = preset,
            domain = domain,
            llm = llm,
        )
    }

    /**
     * Load the chat GGUF for [engine]'s persona — call from the UI ViewModel before the
     * first chat. Model selection is RagPipeline.pickModel, so tutor and chat always share
     * one resident GGUF (a tab switch is then a prompt swap, not a reload). The resident system prompt is the persona template with
     * `{memory_section}` emptied — chat inlines the memory section into each user turn.
     *
     * Returns model load time in ms (0 when already resident).
     */
    suspend fun ensureModelsLoaded(context: Context, engine: TutorEngine): Long {
        val llm = requireNotNull(engine.llm) { "stub engine has no LLM to load" }
        val models = File(context.filesDir, "models").listFiles()
            ?.filter { it.extension == "gguf" && it.canRead() }
            ?.sortedBy { it.name } ?: emptyList()
        val saved = context.getSharedPreferences(RAG_PREFS, Context.MODE_PRIVATE)
            .getString(KEY_MODEL, null)
        val name = RagPipeline.pickModel(models.map { it.name }, saved)
        val model = models.firstOrNull { it.name == name }
            ?: throw IllegalStateException(
                "No .gguf model under ${File(context.filesDir, "models").absolutePath} — push one via adb"
            )
        return llm.ensureLoaded(
            model = model,
            sessionKey = engine.domain,
            systemPrompt = engine.preset.systemTemplate.replace("{memory_section}", "") + learnerBlock(context),
        )
    }

    /** Matches RagPipeline's prefs keys so the UI's model picker applies here too. */
    private const val RAG_PREFS = "rag"
    private const val KEY_MODEL = "selected_model"
}
