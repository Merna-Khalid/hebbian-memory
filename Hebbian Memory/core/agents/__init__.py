"""
agents/
Autonomous agents for the Hebbian RAG system.

Each agent has a single responsibility and is independently importable.

Current agents:
    concept_extractor   — extracts structured concepts from conversation turns
                          and ingests them as typed nodes with semantic edges

Future agents:
    spaced_repetition   — surfaces decayed concepts for review
    memory_inspector    — answers "what do I remember about X?"
    lesson_parser       — extracts concepts from structured lesson documents
"""

from .concept_extractor import ConceptExtractor, ExtractionResult, IngestedExtractionResult

__all__ = [
    "ConceptExtractor",
    "ExtractionResult",
    "IngestedExtractionResult",
]