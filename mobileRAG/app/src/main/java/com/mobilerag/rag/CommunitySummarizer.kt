package com.mobilerag.rag

import android.util.Log
import com.mobilerag.core.GenerationBackend
import com.mobilerag.core.GraphStore
import com.mobilerag.embeddings.EmbeddingGemmaEngine
import com.mobilerag.graph.CommunityDetector
import kotlinx.coroutines.flow.fold

/**
 * Phase 4 community summaries: detects communities over the entity graph
 * ([CommunityDetector]), has the LLM write a short thematic summary per community from
 * strictly-bounded evidence (member entities, strongest internal relations, a couple of
 * source snippets), embeds each summary in document mode, and replaces the rag.db
 * `communities` table in one transaction.
 *
 * Model lifecycle is the caller's responsibility — the backend must have a model loaded
 * before [rebuild] runs; this class only calls [GenerationBackend.generate].
 */
class CommunitySummarizer(
    private val graph: GraphStore,
    private val db: RagDatabase,
    private val engine: EmbeddingGemmaEngine,
    private val generator: GenerationBackend,
) {

    /** Re-detects communities, regenerates all summaries, persists them. Returns the count. */
    suspend fun rebuild(): Int {
        val entities = graph.allEntities(limit = Int.MAX_VALUE)
        val edges = graph.allEdges()
        val communities = CommunityDetector().detect(entities, edges)
        if (communities.isEmpty()) {
            db.replaceCommunities(emptyList())
            Log.i(TAG, "no communities (${entities.size} entities, ${edges.size} edges)")
            return 0
        }

        val byId = entities.associateBy { it.id }
        val records = ArrayList<RagDatabase.NewCommunity>(communities.size)
        for (community in communities) {
            val evidence = buildEvidence(community, byId, edges)
            val prompt = buildString {
                append(SUMMARY_INSTRUCTIONS)
                append("\n\nEvidence:\n").append(evidence)
                append("\nTask: In at most 4 sentences, describe the common theme connecting these entities and how they relate.\n/no_think")
            }
            val raw = generator.generate(prompt, MAX_OUTPUT_TOKENS).fold(StringBuilder()) { acc, t -> acc.append(t) }.toString()
            val summary = cleanSummary(raw)
            if (summary.isEmpty()) {
                Log.w(TAG, "empty summary for community of ${community.memberIds.size} — skipped")
                continue
            }
            val embedding = engine.embedDocument(summary)
            records += RagDatabase.NewCommunity(summary, community.memberIds, community.memberIds.size, embedding)
        }
        db.replaceCommunities(records)
        Log.i(TAG, "persisted ${records.size} community summaries")
        return records.size
    }

    private suspend fun buildEvidence(
        community: CommunityDetector.Community,
        byId: Map<Long, GraphStore.Entity>,
        edges: List<GraphStore.Edge>,
    ): String {
        val members = community.memberIds.mapNotNull { byId[it] }
            .sortedWith(compareByDescending<GraphStore.Entity> { it.mentionCount }.thenBy { it.name })
        val out = StringBuilder()
        out.append("Entities:\n")
        for (e in members) {
            out.append("- ${e.name} (${e.type}, ${e.mentionCount} mentions)\n")
        }

        // Strongest internal relations: parallel edges on a pair are summed, like detection.
        val memberSet = community.memberIds.toSet()
        data class PairEdge(val fromId: Long, val toId: Long, val relation: String, val weight: Double)
        val pairs = LinkedHashMap<Pair<Long, Long>, PairEdge>()
        for (e in edges) {
            if (e.fromId !in memberSet || e.toId !in memberSet || e.fromId == e.toId) continue
            val key = minOf(e.fromId, e.toId) to maxOf(e.fromId, e.toId)
            val prev = pairs[key]
            pairs[key] = if (prev == null) PairEdge(key.first, key.second, e.relation, e.weight)
                else prev.copy(weight = prev.weight + e.weight)
        }
        val topEdges = pairs.values.sortedWith(compareByDescending<PairEdge> { it.weight }.thenBy { it.fromId }.thenBy { it.toId })
            .take(MAX_INTERNAL_EDGES)
        if (topEdges.isNotEmpty()) {
            out.append("Relations:\n")
            for (e in topEdges) {
                val from = byId[e.fromId]?.name ?: graph.getEntity(e.fromId)?.name ?: "#${e.fromId}"
                val to = byId[e.toId]?.name ?: graph.getEntity(e.toId)?.name ?: "#${e.toId}"
                out.append("- $from ${e.relation} $to (weight ${e.weight})\n")
            }
        }

        // 1–2 short source snippets from the most-mentioned members' chunks.
        val chunkIds = LinkedHashSet<Long>()
        for (e in members) {
            for (chunkId in graph.chunksForEntity(e.id)) {
                chunkIds += chunkId
                if (chunkIds.size >= MAX_SNIPPETS) break
            }
            if (chunkIds.size >= MAX_SNIPPETS) break
        }
        if (chunkIds.isNotEmpty()) {
            val texts = db.chunkTextsById(chunkIds.toList())
            out.append("Source snippets:\n")
            for (id in chunkIds) {
                val text = texts[id] ?: continue
                out.append("- ").append(text.take(SNIPPET_CHARS)).append('\n')
            }
        }

        // Hard cap: ~600 tokens at 4 chars/token (RagPipeline budgets context in tokens;
        // this evidence block must stay well under that scale).
        return if (out.length > EVIDENCE_CHAR_BUDGET) out.substring(0, EVIDENCE_CHAR_BUDGET) else out.toString()
    }

    /** Strips think blocks (if any), trims, caps at [MAX_SUMMARY_WORDS]. */
    private fun cleanSummary(raw: String): String {
        var s = raw.replace(Regex("<think>.*?</think>", RegexOption.DOT_MATCHES_ALL), "")
        s = s.replace(Regex("<think>.*$", RegexOption.DOT_MATCHES_ALL), "").trim()
        val words = s.split(Regex("\\s+")).filter { it.isNotEmpty() }
        if (words.size > MAX_SUMMARY_WORDS) s = words.take(MAX_SUMMARY_WORDS).joinToString(" ")
        return s
    }

    companion object {
        private const val TAG = "CommunitySummarizer"
        private const val MAX_INTERNAL_EDGES = 10
        private const val MAX_SNIPPETS = 2
        private const val SNIPPET_CHARS = 240
        private const val EVIDENCE_CHAR_BUDGET = 2400 // ~600 tokens at 4 chars/token
        private const val MAX_OUTPUT_TOKENS = 256
        private const val MAX_SUMMARY_WORDS = 120
        private const val SUMMARY_INSTRUCTIONS =
            "You summarize a community of related entities from a personal knowledge graph. Use ONLY the evidence below. If the evidence is insufficient, say so. Never use knowledge outside the evidence."
    }
}
