package com.mobilerag.hebbian

import com.mobilerag.hebbian.store.ConceptNode
import java.text.Normalizer

/**
 * Concept identity: decides whether an extracted concept is one the graph already holds.
 * Mirrors `core/concept_identity.py` — keep the two in sync.
 *
 * Without it every mention minted a new node, so re-mentioning 食べる never strengthened
 * anything: activation counts stayed at 1, ARM-E novelty was always 1.0, and retrieval
 * filled with near-duplicates.
 *
 * Rules, in order:
 *  1. Same normalized label ([labelKey]) and same concept type → reuse.
 *  2. Otherwise the best vector hit of the same type with a Neo4j-scale score ≥
 *     [EMBEDDING_REUSE_SCORE] (raw cos ≥ 0.95) → reuse. Deliberately far above practice's
 *     "confusable" cutoff (raw cos 0.82): 食べる and 食べた must stay separate.
 *  3. Never for unvalidated content or `event` nodes (raw-transcript fallbacks).
 */
object ConceptIdentity {

    /** (1 + cos) / 2 ≥ 0.975  ⇔  raw cos ≥ 0.95. */
    const val EMBEDDING_REUSE_SCORE = 0.975

    private val WHITESPACE = Regex("\\s+")

    /** Wrapping quotes, brackets and sentence punctuation only — NOT symbols like # + 〜,
     *  which carry meaning in labels ("C#", "C++", "〜ている"). */
    private const val EDGE_CHARS = "\"'“”‘’「」『』《》〈〉（）()［］[]【】{}.,。、!！?？:：;；"

    /** NFKC → lowercase → trim → collapse whitespace → strip wrapping punctuation. */
    fun labelKey(label: String): String {
        val collapsed = WHITESPACE.replace(
            Normalizer.normalize(label, Normalizer.Form.NFKC).lowercase().trim(), " ",
        )
        return collapsed.trim { it in EDGE_CHARS || it.isWhitespace() }
    }

    /**
     * @param labelMatches existing concepts whose [labelKey] equals this label's (any type).
     * @param vectorHits vector-search hits for the new text, best first, Neo4j score scale.
     * @return the existing node to reuse, or null to create a new one.
     */
    fun resolve(
        label: String,
        conceptType: String,
        validated: Boolean,
        labelMatches: List<ConceptNode>,
        vectorHits: List<Pair<ConceptNode, Double>>,
    ): ConceptNode? {
        if (!validated || conceptType == EVENT_TYPE) return null
        if (labelKey(label).isNotEmpty()) {
            labelMatches
                .filter { it.conceptType == conceptType }
                .maxWithOrNull(compareBy<ConceptNode> { it.activationCount }.thenByDescending { it.createdAt })
                ?.let { return it }
        }
        return vectorHits
            .firstOrNull { (node, score) -> node.conceptType == conceptType && score >= EMBEDDING_REUSE_SCORE }
            ?.first
    }

    const val EVENT_TYPE = "event"
}
