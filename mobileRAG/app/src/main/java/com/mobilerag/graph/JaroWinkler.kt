package com.mobilerag.graph

/**
 * Jaro-Winkler string similarity, shared by the graph stores (fuzzy name search) and
 * the entity resolver (fuzzy dedup at indexing time).
 */
internal object JaroWinkler {

    fun similarity(s1: String, s2: String): Double {
        if (s1 == s2) return 1.0
        val jaro = jaro(s1, s2)
        var prefix = 0
        val maxPrefix = minOf(4, s1.length, s2.length)
        while (prefix < maxPrefix && s1[prefix] == s2[prefix]) prefix++
        return jaro + prefix * 0.1 * (1.0 - jaro)
    }

    private fun jaro(s1: String, s2: String): Double {
        if (s1.isEmpty() || s2.isEmpty()) return 0.0
        val range = (maxOf(s1.length, s2.length) / 2 - 1).coerceAtLeast(0)
        val matched1 = BooleanArray(s1.length)
        val matched2 = BooleanArray(s2.length)
        var matches = 0
        for (i in s1.indices) {
            val start = maxOf(0, i - range)
            val end = minOf(i + range + 1, s2.length)
            for (j in start until end) {
                if (!matched2[j] && s1[i] == s2[j]) {
                    matched1[i] = true
                    matched2[j] = true
                    matches++
                    break
                }
            }
        }
        if (matches == 0) return 0.0
        var transpositions = 0
        var k = 0
        for (i in s1.indices) {
            if (matched1[i]) {
                while (!matched2[k]) k++
                if (s1[i] != s2[k]) transpositions++
                k++
            }
        }
        return (matches.toDouble() / s1.length +
            matches.toDouble() / s2.length +
            (matches - transpositions / 2.0) / matches) / 3.0
    }
}
