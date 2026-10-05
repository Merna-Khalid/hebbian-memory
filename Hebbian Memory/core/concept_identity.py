"""
concept_identity.py
Decides whether an extracted concept is one the graph already holds.
Mirrors mobileRAG's ConceptIdentity.kt — keep the two in sync.

Without it every mention minted a new node, so re-mentioning 食べる never
strengthened anything: activation counts stayed at 1, ARM-E novelty was
always 1.0, and retrieval filled with near-duplicates.

Rules, in order:
  1. Same normalized label (label_key) and same concept type → reuse.
  2. Otherwise the best vector hit of the same type with a Neo4j-scale score
     ≥ EMBEDDING_REUSE_SCORE (raw cos ≥ 0.95) → reuse. Deliberately far above
     practice's "confusable" cutoff (raw cos 0.82): 食べる and 食べた must stay
     separate.
  3. Never for unvalidated content or "event" nodes (raw-transcript fallbacks).
"""

import re
import unicodedata
from typing import Optional

# (1 + cos) / 2 ≥ 0.975  ⇔  raw cos ≥ 0.95
EMBEDDING_REUSE_SCORE = 0.975
EVENT_TYPE = "event"

_WHITESPACE = re.compile(r"\s+")
# Wrapping quotes, brackets and sentence punctuation only — NOT symbols like
# # + 〜, which carry meaning in labels ("C#", "C++", "〜ている").
_EDGE_CHARS = "\"'“”‘’「」『』《》〈〉（）()［］[]【】{}.,。、!！?？:：;；"


def label_key(label: str) -> str:
    """NFKC → lowercase → trim → collapse whitespace → strip wrapping punctuation."""
    collapsed = _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", label).lower().strip())
    return collapsed.strip(_EDGE_CHARS + " \t\n\r\f\v")


def resolve(label, concept_type, validated, label_matches, vector_hits) -> Optional[object]:
    """
    label_matches : existing ConceptNodes whose label_key equals this label's (any type)
    vector_hits   : [(ConceptNode, score)] best first, Neo4j score scale
    Returns the existing node to reuse, or None to create a new one.
    """
    if not validated or concept_type == EVENT_TYPE:
        return None
    if label_key(label):
        same_type = [n for n in label_matches if n.concept_type == concept_type]
        if same_type:
            # most activations; ties → earliest created
            return max(same_type, key=lambda n: (n.activation_count, -n.created_at))
    for node, score in vector_hits:
        if node.concept_type == concept_type and score >= EMBEDDING_REUSE_SCORE:
            return node
    return None
