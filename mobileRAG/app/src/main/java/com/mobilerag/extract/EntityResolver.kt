package com.mobilerag.extract

import com.mobilerag.core.GraphStore
import com.mobilerag.graph.JaroWinkler

/**
 * Resolves extracted entity mentions to canonical graph entities: exact match on the
 * normalized name (lowercase, trimmed, whitespace-collapsed) + label, then a fuzzy
 * Jaro-Winkler pass (>= 0.92, same label), else a new entity via [GraphStore.upsertEntity].
 */
class EntityResolver(private val store: GraphStore) {

    /** Returns each extracted entity paired with its graph entity id. */
    suspend fun resolve(
        entities: List<Gliner2Extractor.ExtractedEntity>,
    ): List<Pair<Gliner2Extractor.ExtractedEntity, Long>> =
        entities.map { entity ->
            val normalizedName = normalize(entity.text)
            val candidates = store.findEntities(normalizedName, limit = 50)
            val exact = candidates.firstOrNull {
                it.name.trim().lowercase() == normalizedName && it.type == entity.label
            }
            val id = exact?.id ?: run {
                val fuzzy = candidates
                    .filter { it.type == entity.label }
                    .map { it to JaroWinkler.similarity(it.name.trim().lowercase(), normalizedName) }
                    .filter { it.second >= FUZZY_THRESHOLD }
                    .maxByOrNull { it.second }
                fuzzy?.first?.id ?: store.upsertEntity(entity.text, entity.label, normalizedName)
            }
            entity to id
        }

    companion object {
        private const val FUZZY_THRESHOLD = 0.92

        fun normalize(name: String): String =
            name.trim().lowercase().replace(Regex("\\s+"), " ")
    }
}
