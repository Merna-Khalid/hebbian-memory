package com.mobilerag.hebbian

import android.util.Log
import com.mobilerag.hebbian.store.HebbianEdge
import com.mobilerag.hebbian.store.HebbianStore
import kotlinx.coroutines.delay
import org.json.JSONArray
import org.json.JSONObject

/**
 * Concept extraction agent (port of `core/agents/concept_extractor.py`).
 *
 * Takes raw text (a conversation turn, a lesson, an article) and extracts structured
 * concept nodes + relations via the on-device LLM. Each extracted concept is ingested
 * through [IngestPipeline] (embedding, ARM-E modulation, Hebbian update, bootstrap
 * edges). Relations between co-extracted concepts become ASSOCIATED_WITH edges at a
 * higher initial weight (1.5) than bootstrap edges (1.0) — explicit semantic
 * relations are stronger than implicit similarity; all other co-extracted pairs get
 * co-occurrence edges at 1.2. If extraction fails entirely, the raw text is stored
 * as a single unvalidated "event" node (weak bootstrap edges, m_t capped at 1.0).
 *
 * The LLM is injected as [generate]: (prompt, maxTokens) -> full text. Wire it to
 * `HebbianLlmSession.complete` at integration; the session's system prompt is the chat
 * persona, so [extractionSystem] is sent in-band at the head of each extraction prompt.
 */

// ── Extraction schema ─────────────────────────────────────────────────

val CONCEPT_TYPES = listOf(
    "vocabulary",    // 食べる, 美しい, 電車 — individual words/expressions
    "grammar",       // て-form, passive voice, conditional
    "fact",          // general knowledge statements
    "entity",        // named things: Tokyo, Mount Fuji, anime titles
    "procedure",     // how to do something step by step
    "example",       // example sentences, usage examples
    "cultural",      // cultural notes, customs, context
    "question",      // open questions the user raised
)

val RELATION_TYPES = listOf(
    "is_form_of",    // て-form → verb conjugation
    "is_type_of",    // godan verb → verb
    "used_in",       // 食べる → eating contexts
    "opposite_of",   // 好き ↔ 嫌い
    "related_to",    // general semantic relation
    "example_of",    // 食べます → polite form example
    "precedes",      // step 1 → step 2 in a procedure
)

const val MAX_RETRIES = 3

/** Extraction emits compact JSON (≤6 concepts + relations); the 512 cap from model_setup.py
 *  priced every turn for an output that rarely exceeds ~300 tokens — truncated JSON fails
 *  parsing and falls back to an event node, so this only needs to fit the schema, not prose. */
private const val EXTRACTION_MAX_TOKENS = 384

/** Co-occurrence edge weight — weaker than explicit relations (1.5), stronger than bootstrap (1.0). */
private const val CO_OCCURRENCE_W = 1.2

// ── Output data classes ───────────────────────────────────────────────

data class ExtractedConcept(
    val label: String,
    val conceptType: String,
    val description: String,    // 1-2 sentence description
    val sourceLang: String,     // "ja", "en", "mixed"
)

data class ExtractedRelation(
    val srcLabel: String,
    val relation: String,
    val dstLabel: String,
)

data class ExtractionResult(
    val concepts: List<ExtractedConcept>,
    val relations: List<ExtractedRelation>,
    val rawResponse: String,
    val attempts: Int,
    val success: Boolean,
)

/** Result after concepts have been ingested into the system. */
data class IngestedExtractionResult(
    val extracted: ExtractionResult,
    val ingestResults: List<IngestResult>,  // one per concept
    val nodeIds: List<String>,              // nodeId per concept
    val edgesCreated: Int,
    val elapsedMs: Double,
    val validationPassed: Boolean = true,   // false when the fallback event-node path ran
)

// ── Prompts ───────────────────────────────────────────────────────────

val EXTRACTION_SYSTEM = """You are a knowledge extraction assistant for a Japanese language learning system.
Extract concepts and relations from the given text.
You must respond with ONLY valid JSON — no explanation, no markdown, no extra text.

Concept types: vocabulary, grammar, fact, entity, procedure, example, cultural, question
Relation types: is_form_of, is_type_of, used_in, opposite_of, related_to, example_of, precedes

Rules:
- Extract 1-6 concepts. Do not over-extract.
- Each concept must have a clear, specific label (e.g. "食べる" not "Japanese verb")
- description must be 1-2 sentences explaining the concept clearly
- Only create relations between concepts you actually extracted
- source_lang: "ja" for Japanese, "en" for English, "mixed" for both"""

// ── Second-brain preset (general ambient/BCI domain) ──────────────────
// Same shape as the Japanese-tutor preset above, swapped for a general
// personal-knowledge domain. Keeps the phrase "knowledge extraction" so
// tutor_engine.py's stub_chat_fn (which keys off that substring) still
// recognizes extraction calls regardless of which preset is active.

val SECOND_BRAIN_CONCEPT_TYPES = listOf(
    "idea",       // a novel thought, insight, or connection
    "task",       // something to do
    "plan",       // a multi-step intention
    "fact",       // general knowledge statements
    "entity",     // named things: people, places, projects, tools
    "feeling",    // an emotional state or reaction worth remembering
    "question",   // open questions raised
    "event",      // something that happened
)

val SECOND_BRAIN_RELATION_TYPES = listOf(
    "leads_to",     // idea A → idea B
    "blocks",       // task A blocks task B
    "part_of",      // step → plan
    "opposite_of",  // contradictory ideas/feelings
    "related_to",   // general semantic relation
    "example_of",   // concrete instance of a general idea
    "precedes",     // step 1 → step 2
)

val SECOND_BRAIN_EXTRACTION_SYSTEM = """You are a knowledge extraction assistant for a personal second-brain memory system.
Extract concepts and relations from the given text — a stream of the user's own ambient thoughts.
You must respond with ONLY valid JSON — no explanation, no markdown, no extra text.

Concept types: idea, task, plan, fact, entity, feeling, question, event
Relation types: leads_to, blocks, part_of, opposite_of, related_to, example_of, precedes

Rules:
- Extract 1-6 concepts. Do not over-extract.
- Each concept must have a clear, specific label (e.g. "refactor parser to use accumulator" not "coding task")
- description must be 1-2 sentences explaining the concept clearly
- Only create relations between concepts you actually extracted
- source_lang: "en" unless the text is in another language"""

val EXTRACTION_PROMPT_TEMPLATE = """Extract concepts and relations from this text:

---
{text}
---

Respond with this exact JSON structure:
{
  "concepts": [
    {
      "label": "concept name",
      "concept_type": "one of the types above",
      "description": "1-2 sentence description",
      "source_lang": "ja|en|mixed"
    }
  ],
  "relations": [
    {
      "src_label": "label of source concept",
      "relation": "one of the relation types",
      "dst_label": "label of target concept"
    }
  ]
}

JSON only:"""

// ── ConceptExtractor ──────────────────────────────────────────────────

class ConceptExtractor(
    private val pipeline: IngestPipeline,
    private val generate: suspend (String, Int) -> String,
    private val store: HebbianStore,
    private val relationWeight: Double = 1.5,
    private val verbose: Boolean = false,
    private val extractionSystem: String = EXTRACTION_SYSTEM,
    private val conceptTypes: List<String> = CONCEPT_TYPES,
    private val relationTypes: List<String> = RELATION_TYPES,
) {

    // ── Extraction ────────────────────────────────────────────────────

    /**
     * Call the LLM to extract concepts and relations from [text]; retries up to
     * [MAX_RETRIES] times on invalid JSON. Returns success=false if all fail.
     */
    suspend fun extract(text: String): ExtractionResult {
        // replace(), not .format(): the template contains literal { } in its JSON
        // example, and braces in the extracted text would crash str.format.
        // Python sends extractionSystem as this call's system prompt; on-device the session's
        // system prompt is the persona, so the rules — including the concept/relation type
        // lists the template calls "the types above" — go in-band ahead of the template.
        // Without them every concept fell back to "fact" and lost its TypeProfiles decay.
        val prompt = extractionSystem + "\n\n" +
            EXTRACTION_PROMPT_TEMPLATE.replace("{text}", text.take(1500))
        var raw = ""

        for (attempt in 1..MAX_RETRIES) {
            try {
                raw = generate(prompt, EXTRACTION_MAX_TOKENS)

                // Strip markdown fences if the model added them
                val clean = FENCE_REGEX.replace(raw, "").trim()

                // Find the JSON object — sometimes the model adds preamble
                val match = JSON_OBJECT_REGEX.find(clean)
                    ?: throw IllegalArgumentException("No JSON object found in response")

                val parsed = JSONObject(match.value)
                val concepts = parseConcepts(parsed.optJSONArray("concepts"))
                val relations = parseRelations(
                    parsed.optJSONArray("relations"),
                    concepts.mapTo(HashSet()) { it.label },
                )

                if (verbose && concepts.isNotEmpty()) {
                    Log.d(TAG, "extracted ${concepts.size} concepts, ${relations.size} relations (attempt $attempt)")
                }

                return ExtractionResult(
                    concepts = concepts,
                    relations = relations,
                    rawResponse = raw,
                    attempts = attempt,
                    success = true,
                )
            } catch (e: Exception) {
                if (verbose) Log.d(TAG, "attempt $attempt failed: $e")
                if (attempt == MAX_RETRIES) {
                    return ExtractionResult(
                        concepts = emptyList(),
                        relations = emptyList(),
                        rawResponse = raw,
                        attempts = attempt,
                        success = false,
                    )
                }
                delay(500) // brief pause before retry
            }
        }

        // Should not reach here
        return ExtractionResult(emptyList(), emptyList(), "", MAX_RETRIES, false)
    }

    private fun parseConcepts(raw: JSONArray?): List<ExtractedConcept> {
        if (raw == null) return emptyList()
        val concepts = ArrayList<ExtractedConcept>()
        for (i in 0 until raw.length()) {
            val item = raw.optJSONObject(i) ?: continue
            val label = item.optString("label", "").trim()
            var conceptType = item.optString("concept_type", "fact").trim()
            val description = item.optString("description", "").trim()
            var sourceLang = item.optString("source_lang", "en").trim()

            if (label.isEmpty() || description.isEmpty()) continue
            if (conceptType !in conceptTypes) conceptType = "fact"
            if (sourceLang !in SOURCE_LANGS) sourceLang = "en"

            concepts += ExtractedConcept(
                label = label,
                conceptType = conceptType,
                description = description,
                sourceLang = sourceLang,
            )
        }
        return concepts.take(6) // hard cap
    }

    private fun parseRelations(
        raw: JSONArray?,
        validLabels: Set<String>,
    ): List<ExtractedRelation> {
        if (raw == null) return emptyList()
        val relations = ArrayList<ExtractedRelation>()
        for (i in 0 until raw.length()) {
            val item = raw.optJSONObject(i) ?: continue
            val src = item.optString("src_label", "").trim()
            var rel = item.optString("relation", "").trim()
            val dst = item.optString("dst_label", "").trim()

            if (src.isEmpty() || dst.isEmpty() || rel.isEmpty()) continue
            if (src !in validLabels || dst !in validLabels) continue // only between extracted concepts
            if (rel !in relationTypes) rel = "related_to"
            if (src != dst) relations += ExtractedRelation(src, rel, dst)
        }
        return relations
    }

    // ── Ingest ────────────────────────────────────────────────────────

    /**
     * Extract concepts from [text] and ingest each into the pipeline, then create
     * explicit relation edges (weight [relationWeight]) and co-occurrence edges
     * (weight 1.2) between all extracted pairs. Falls back to a single unvalidated
     * event node if extraction fails.
     */
    suspend fun extractAndIngest(
        text: String,
        userText: String? = null,
        sessionId: String? = null,
        sourceType: String = "user",
        sourceUri: String? = null,
        attention: Double? = null,
    ): IngestedExtractionResult {
        val t0 = System.currentTimeMillis()

        // ── 1. Extract ────────────────────────────────────────────────
        val extraction = extract(text)

        if (!extraction.success || extraction.concepts.isEmpty()) {
            if (verbose) Log.d(TAG, "extraction failed — storing as event node")
            // Fallback: store raw text as a single event node. validated=false →
            // pipeline bootstraps weak edges and caps m_t at 1.0: an unverified
            // turn must earn strength, not get it for free.
            val result = pipeline.ingest(
                text = text,
                label = text.take(60),
                conceptType = "event",
                sourceType = sourceType,
                sourceUri = sourceUri,
                userText = userText,
                sessionId = sessionId,
                attention = attention,
                validated = false,
            )
            return IngestedExtractionResult(
                extracted = extraction,
                ingestResults = listOf(result),
                nodeIds = listOf(result.nodeId),
                edgesCreated = 0,
                elapsedMs = (System.currentTimeMillis() - t0).toDouble(),
                validationPassed = false,
            )
        }

        // ── 2. Ingest each concept ────────────────────────────────────
        val ingestResults = ArrayList<IngestResult>()
        val nodeIds = ArrayList<String>()
        val labelToNid = HashMap<String, String>()

        for (concept in extraction.concepts) {
            // Build full text: label + description for embedding
            val fullText = "${concept.label}: ${concept.description}"

            val result = pipeline.ingest(
                text = fullText,
                label = concept.label,
                conceptType = concept.conceptType,
                sourceType = sourceType,
                sourceUri = sourceUri,
                userText = userText,
                sessionId = sessionId,
                attention = attention,
            )
            ingestResults += result
            nodeIds += result.nodeId
            labelToNid[concept.label] = result.nodeId

            if (verbose) {
                Log.d(TAG, "ingested [${concept.conceptType}] ${concept.label}  m_t=${result.mT}  Δw=${result.deltaWMean}")
            }
        }

        // ── 3. Create explicit semantic relation edges ────────────────
        var edgesCreated = 0
        for (rel in extraction.relations) {
            val srcId = labelToNid[rel.srcLabel]
            val dstId = labelToNid[rel.dstLabel]
            if (srcId == null || dstId == null) continue
            if (srcId == dstId) continue // both labels resolved to one existing concept

            // Both directions; higher weight than bootstrap — explicit > implicit.
            for ((s, d) in listOf(srcId to dstId, dstId to srcId)) {
                store.upsertEdge(
                    HebbianEdge(
                        srcId = s,
                        dstId = d,
                        hebbWeight = relationWeight,
                        eligibility = 0.0,
                        causalScore = 0.0,
                        coActivationCount = 1,
                        lastUpdated = System.currentTimeMillis(),
                        layer = "hippocampal",
                    )
                )
            }
            edgesCreated += 1
            if (verbose) Log.d(TAG, "relation: ${rel.srcLabel} —[${rel.relation}]→ ${rel.dstLabel}  w=$relationWeight")
        }

        // ── 4. Co-occurrence edges between ALL extracted concepts ─────
        // Every concept pair extracted from the same text gets a co-occurrence
        // edge (they appeared together, so they're associated) — unless an
        // explicit relation already covers that pair.
        for (i in nodeIds.indices) {
            val nidA = nodeIds[i]
            for (nidB in nodeIds.subList(i + 1, nodeIds.size)) {
                if (nidA == nidB) continue
                val already = extraction.relations.any { r ->
                    (labelToNid[r.srcLabel] == nidA && labelToNid[r.dstLabel] == nidB) ||
                        (labelToNid[r.srcLabel] == nidB && labelToNid[r.dstLabel] == nidA)
                }
                if (!already) {
                    for ((s, d) in listOf(nidA to nidB, nidB to nidA)) {
                        store.upsertEdge(
                            HebbianEdge(
                                srcId = s,
                                dstId = d,
                                hebbWeight = CO_OCCURRENCE_W,
                                eligibility = 0.0,
                                causalScore = 0.0,
                                coActivationCount = 1,
                                lastUpdated = System.currentTimeMillis(),
                                layer = "hippocampal",
                            )
                        )
                    }
                    edgesCreated += 1
                }
            }
        }

        if (verbose) {
            Log.d(TAG, "done — ${extraction.concepts.size} concepts, $edgesCreated edges, ${System.currentTimeMillis() - t0}ms")
        }

        return IngestedExtractionResult(
            extracted = extraction,
            ingestResults = ingestResults,
            nodeIds = nodeIds,
            edgesCreated = edgesCreated,
            elapsedMs = (System.currentTimeMillis() - t0).toDouble(),
        )
    }

    // ── Convenience: extract from conversation turn ───────────────────

    /**
     * Extract concepts from a full conversation turn (user + LLM). The combined
     * text gives more context for extraction; [userMessage] drives ARM-E emotion.
     */
    suspend fun extractFromTurn(
        userMessage: String,
        llmResponse: String,
        sessionId: String? = null,
    ): IngestedExtractionResult {
        val combined = "User: $userMessage\nAssistant: $llmResponse"
        return extractAndIngest(
            text = combined,
            userText = userMessage,
            sessionId = sessionId,
            sourceType = "user",
        )
    }

    private companion object {
        const val TAG = "ConceptExtractor"
        val SOURCE_LANGS = setOf("ja", "en", "mixed")
        val FENCE_REGEX = Regex("```(?:json)?|```")
        val JSON_OBJECT_REGEX = Regex("\\{.*\\}", RegexOption.DOT_MATCHES_ALL)
    }
}
