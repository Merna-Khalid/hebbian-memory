"""
concept_extractor.py
Concept extraction agent for the Japanese learning Hebbian RAG system.

Takes raw text (a conversation turn, a lesson, an article) and extracts
structured concept nodes + relations using LFM2.5-1.2B-JP.

Each extracted concept is ingested via IngestPipeline so it gets:
    - BGE-M3 embedding
    - ARM-E modulation
    - Hippocampal GNN Hebbian update
    - Edge bootstrapping to similar existing nodes

Relations between co-extracted concepts are stored as ASSOCIATED_WITH
edges with a higher initial weight (1.5) than bootstrap edges (1.0) —
because explicit semantic relations are stronger than implicit similarity.

Design:
    - No LangChain — pure Python + LFM2.5
    - Structured JSON prompt with retry (up to MAX_RETRIES attempts)
    - Graceful degradation: if extraction fails, fall back to storing
      the raw text as a single event node
    - Language-aware: handles EN, JA, and mixed input
"""

import json
import re
import uuid
import time
from dataclasses import dataclass, field
from typing import Optional

# Agents import from parent core/ package
# When running as part of the app: from core.agents import ConceptExtractor
# When running standalone tests: adjust sys.path as shown in __main__
try:
    from graph_store import GraphStore, HebbianEdge
    from ingest_pipeline import IngestPipeline, IngestResult
except ImportError:
    # Running from inside agents/ directory
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from graph_store import GraphStore, HebbianEdge
    from ingest_pipeline import IngestPipeline, IngestResult


# ── Extraction schema ─────────────────────────────────────────────────

CONCEPT_TYPES = [
    "vocabulary",    # 食べる, 美しい, 電車 — individual words/expressions
    "grammar",       # て-form, passive voice, conditional
    "fact",          # general knowledge statements
    "entity",        # named things: Tokyo, Mount Fuji, anime titles
    "procedure",     # how to do something step by step
    "example",       # example sentences, usage examples
    "cultural",      # cultural notes, customs, context
    "question",      # open questions the user raised
]

RELATION_TYPES = [
    "is_form_of",    # て-form → verb conjugation
    "is_type_of",    # godan verb → verb
    "used_in",       # 食べる → eating contexts
    "opposite_of",   # 好き ↔ 嫌い
    "related_to",    # general semantic relation
    "example_of",    # 食べます → polite form example
    "precedes",      # step 1 → step 2 in a procedure
]

MAX_RETRIES = 3


# ── Output dataclasses ────────────────────────────────────────────────

@dataclass
class ExtractedConcept:
    label       : str
    concept_type: str
    description : str          # 1-2 sentence description
    source_lang : str          # "ja", "en", "mixed"


@dataclass
class ExtractedRelation:
    src_label   : str
    relation    : str
    dst_label   : str


@dataclass
class ExtractionResult:
    concepts    : list[ExtractedConcept]
    relations   : list[ExtractedRelation]
    raw_response: str
    attempts    : int
    success     : bool


@dataclass
class IngestedExtractionResult:
    """Result after concepts have been ingested into the system."""
    extracted       : ExtractionResult
    ingest_results  : list[IngestResult]   # one per concept
    node_ids        : list[str]             # node_id per concept
    edges_created   : int
    elapsed_ms      : float
    validation_passed: bool = True   # False when the fallback event-node path ran
                                     # (validation-gated plasticity downstream)


# ── Prompts ───────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = """You are a knowledge extraction assistant for a Japanese language learning system.
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

# ── Second-brain preset (general ambient/BCI domain) ──────────────────
# Same shape as the Japanese-tutor preset above, swapped for a general
# personal-knowledge domain. Keeps the phrase "knowledge extraction" so
# tutor_engine.py's stub_chat_fn (which keys off that substring) still
# recognizes extraction calls regardless of which preset is active.

SECOND_BRAIN_CONCEPT_TYPES = [
    "idea",       # a novel thought, insight, or connection
    "task",       # something to do
    "plan",       # a multi-step intention
    "fact",       # general knowledge statements
    "entity",     # named things: people, places, projects, tools
    "feeling",    # an emotional state or reaction worth remembering
    "question",   # open questions raised
    "event",      # something that happened
]

SECOND_BRAIN_RELATION_TYPES = [
    "leads_to",     # idea A → idea B
    "blocks",       # task A blocks task B
    "part_of",      # step → plan
    "opposite_of",  # contradictory ideas/feelings
    "related_to",   # general semantic relation
    "example_of",   # concrete instance of a general idea
    "precedes",     # step 1 → step 2
]

SECOND_BRAIN_EXTRACTION_SYSTEM = """You are a knowledge extraction assistant for a personal second-brain memory system.
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

EXTRACTION_PROMPT_TEMPLATE = """Extract concepts and relations from this text:

---
{text}
---

Respond with this exact JSON structure:
{{
  "concepts": [
    {{
      "label": "concept name",
      "concept_type": "one of the types above",
      "description": "1-2 sentence description",
      "source_lang": "ja|en|mixed"
    }}
  ],
  "relations": [
    {{
      "src_label": "label of source concept",
      "relation": "one of the relation types",
      "dst_label": "label of target concept"
    }}
  ]
}}

JSON only:"""


# ── ConceptExtractor ──────────────────────────────────────────────────

class ConceptExtractor:
    """
    Extracts structured concepts from text using LFM2.5-1.2B-JP
    and ingests them into the Hebbian RAG memory system.

    Usage:
        extractor = ConceptExtractor(
            pipeline     = ingest_pipeline,
            chat_fn      = model_setup.chat_fn,
            store        = graph_store,
        )

        result = extractor.extract_and_ingest(
            text       = "食べる means 'to eat'. It is a godan verb (Group 1).",
            user_text  = "Can you teach me this verb?",
            session_id = session_id,
            source_type= "user",
        )
    """

    def __init__(
        self,
        pipeline        : IngestPipeline,
        chat_fn,                           # callable from model_setup
        store           : GraphStore,
        relation_weight : float = 1.5,     # initial weight for explicit relations
        verbose         : bool  = True,
        extraction_system: str  = EXTRACTION_SYSTEM,
        concept_types   : Optional[list] = None,
        relation_types  : Optional[list] = None,
    ):
        self.pipeline        = pipeline
        self.chat_fn         = chat_fn
        self.store           = store
        self.relation_weight = relation_weight
        self.verbose         = verbose
        self.extraction_system = extraction_system
        self.concept_types     = concept_types  or CONCEPT_TYPES
        self.relation_types    = relation_types or RELATION_TYPES

    # ── Extraction ────────────────────────────────────────────────────

    def extract(self, text: str) -> ExtractionResult:
        """
        Call LFM2.5 to extract concepts and relations from text.
        Retries up to MAX_RETRIES times on invalid JSON.

        Returns ExtractionResult with success=False if all attempts fail.
        """
        # replace(), not .format(): the template contains literal { } in
        # its JSON example, and braces in the extracted text would crash
        # str.format with a KeyError/ValueError.
        prompt = EXTRACTION_PROMPT_TEMPLATE.replace(
            "{text}", text[:1500]
        )
        raw = ""

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = self.chat_fn(
                    user_message  = prompt,
                    system_prompt = self.extraction_system,
                    history       = None,
                )

                # Strip markdown fences if model added them
                clean = re.sub(r"```(?:json)?|```", "", raw).strip()

                # Find the JSON object — sometimes model adds preamble
                match = re.search(r"\{.*\}", clean, re.DOTALL)
                if not match:
                    raise ValueError("No JSON object found in response")

                parsed = json.loads(match.group())

                concepts  = self._parse_concepts(parsed.get("concepts", []))
                relations = self._parse_relations(
                    parsed.get("relations", []),
                    {c.label for c in concepts}
                )

                if self.verbose and concepts:
                    print(f"  [extractor] extracted {len(concepts)} concepts, "
                          f"{len(relations)} relations (attempt {attempt})")

                return ExtractionResult(
                    concepts    = concepts,
                    relations   = relations,
                    raw_response= raw,
                    attempts    = attempt,
                    success     = True,
                )

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                if self.verbose:
                    print(f"  [extractor] attempt {attempt} failed: {e}")
                if attempt == MAX_RETRIES:
                    return ExtractionResult(
                        concepts    = [],
                        relations   = [],
                        raw_response= raw,
                        attempts    = attempt,
                        success     = False,
                    )
                time.sleep(0.5)   # brief pause before retry

        # Should not reach here
        return ExtractionResult([], [], "", MAX_RETRIES, False)

    def _parse_concepts(self, raw: list) -> list[ExtractedConcept]:
        concepts = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            label        = str(item.get("label", "")).strip()
            concept_type = str(item.get("concept_type", "fact")).strip()
            description  = str(item.get("description", "")).strip()
            source_lang  = str(item.get("source_lang", "en")).strip()

            if not label or not description:
                continue
            if concept_type not in self.concept_types:
                concept_type = "fact"
            if source_lang not in ("ja", "en", "mixed"):
                source_lang = "en"

            concepts.append(ExtractedConcept(
                label       = label,
                concept_type= concept_type,
                description = description,
                source_lang = source_lang,
            ))
        return concepts[:6]   # hard cap

    def _parse_relations(
        self, raw: list, valid_labels: set
    ) -> list[ExtractedRelation]:
        relations = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            src = str(item.get("src_label", "")).strip()
            rel = str(item.get("relation",  "")).strip()
            dst = str(item.get("dst_label", "")).strip()

            if not src or not dst or not rel:
                continue
            if src not in valid_labels or dst not in valid_labels:
                continue   # only relations between extracted concepts
            if rel not in self.relation_types:
                rel = "related_to"
            if src != dst:
                relations.append(ExtractedRelation(src, rel, dst))

        return relations

    # ── Ingest ────────────────────────────────────────────────────────

    def extract_and_ingest(
        self,
        text        : str,
        user_text   : str  = "",
        session_id  : Optional[str] = None,
        source_type : str  = "user",
        source_uri  : Optional[str] = None,
        attention   : Optional[float] = None,   # [0,1] engagement signal, e.g. from a BCI
    ) -> IngestedExtractionResult:
        """
        Extract concepts from text and ingest each into the pipeline.

        Steps:
            1. LFM extracts concepts + relations from text
            2. Each concept is ingested via IngestPipeline
               (embedded, ARM-E modulated, Hebbian updated, bootstrapped)
            3. Explicit semantic relations → ASSOCIATED_WITH edges
               with higher initial weight than bootstrap edges
            4. All concept pairs from same extraction → co-occurrence edges
               (they appeared together, so they're associated)

        Falls back to single event node if extraction fails.
        """
        t0 = time.time()

        # ── 1. Extract ────────────────────────────────────────────────
        extraction = self.extract(text)

        if not extraction.success or not extraction.concepts:
            if self.verbose:
                print("  [extractor] extraction failed — storing as event node")
            # Fallback: store raw text as single event node.
            # validated=False → pipeline bootstraps weak edges and caps
            # m_t at 1.0; engine skips co-retrieval reinforcement. An
            # unverified turn must earn strength, not get it for free.
            result = self.pipeline.ingest(
                text         = text,
                label        = text[:60],
                concept_type = "event",
                source_type  = source_type,
                source_uri   = source_uri,
                user_text    = user_text,
                session_id   = session_id,
                attention    = attention,
                validated    = False,
            )
            elapsed = (time.time() - t0) * 1000
            return IngestedExtractionResult(
                extracted     = extraction,
                ingest_results= [result],
                node_ids      = [result.node_id],
                edges_created = 0,
                elapsed_ms    = elapsed,
                validation_passed = False,
            )

        # ── 2. Ingest each concept ────────────────────────────────────
        ingest_results = []
        node_ids       = []
        label_to_nid   = {}

        for concept in extraction.concepts:
            # Build full text: label + description for embedding
            full_text = f"{concept.label}: {concept.description}"

            result = self.pipeline.ingest(
                text         = full_text,
                label        = concept.label,
                concept_type = concept.concept_type,
                source_type  = source_type,
                source_uri   = source_uri,
                user_text    = user_text,
                session_id   = session_id,
                attention    = attention,
            )
            ingest_results.append(result)
            node_ids.append(result.node_id)
            label_to_nid[concept.label] = result.node_id

            if self.verbose:
                print(f"  [extractor] ingested [{concept.concept_type}] "
                      f"{concept.label}  m_t={result.m_t:.2f}  "
                      f"Δw={result.delta_w_mean:+.4f}")

        # ── 3. Create explicit semantic relation edges ─────────────────
        edges_created = 0
        for rel in extraction.relations:
            src_id = label_to_nid.get(rel.src_label)
            dst_id = label_to_nid.get(rel.dst_label)
            if not src_id or not dst_id:
                continue
            if src_id == dst_id:
                continue   # both labels resolved to one existing concept

            # Directed edge matching relation direction
            # Higher weight than bootstrap (1.5 vs 1.0) — explicit > implicit
            for s, d in [(src_id, dst_id), (dst_id, src_id)]:
                self.store.upsert_edge(HebbianEdge(
                    src_id              = s,
                    dst_id              = d,
                    hebb_weight         = self.relation_weight,
                    eligibility         = 0.0,
                    causal_score        = 0.0,
                    co_activation_count = 1,
                    last_updated        = time.time(),
                    layer               = "hippocampal",
                ))
            edges_created += 1
            if self.verbose:
                print(f"  [extractor] relation: {rel.src_label} "
                      f"—[{rel.relation}]→ {rel.dst_label}  w={self.relation_weight}")

        # ── 4. Co-occurrence edges between ALL extracted concepts ──────
        # Every concept pair that was extracted from the same text
        # gets a co-occurrence edge. Weaker than explicit relations (1.2)
        # but stronger than pure similarity bootstrapping (1.0).
        co_weight = 1.2
        for i, nid_a in enumerate(node_ids):
            for nid_b in node_ids[i+1:]:
                if nid_a == nid_b:
                    continue
                # Only create if no explicit relation already exists
                already = any(
                    (label_to_nid.get(r.src_label) == nid_a and
                     label_to_nid.get(r.dst_label) == nid_b)
                    or
                    (label_to_nid.get(r.src_label) == nid_b and
                     label_to_nid.get(r.dst_label) == nid_a)
                    for r in extraction.relations
                )
                if not already:
                    for s, d in [(nid_a, nid_b), (nid_b, nid_a)]:
                        self.store.upsert_edge(HebbianEdge(
                            src_id              = s,
                            dst_id              = d,
                            hebb_weight         = co_weight,
                            eligibility         = 0.0,
                            causal_score        = 0.0,
                            co_activation_count = 1,
                            last_updated        = time.time(),
                            layer               = "hippocampal",
                        ))
                    edges_created += 1

        elapsed = (time.time() - t0) * 1000

        if self.verbose:
            print(f"  [extractor] done — {len(extraction.concepts)} concepts, "
                  f"{edges_created} edges, {elapsed:.0f}ms")

        return IngestedExtractionResult(
            extracted     = extraction,
            ingest_results= ingest_results,
            node_ids      = node_ids,
            edges_created = edges_created,
            elapsed_ms    = elapsed,
        )

    # ── Convenience: extract from conversation turn ───────────────────

    def extract_from_turn(
        self,
        user_message  : str,
        llm_response  : str,
        session_id    : Optional[str] = None,
    ) -> IngestedExtractionResult:
        """
        Extract concepts from a full conversation turn (user + LFM).
        The combined text gives more context for extraction.
        user_message drives ARM-E emotion signal.
        """
        combined = f"User: {user_message}\nAssistant: {llm_response}"
        return self.extract_and_ingest(
            text       = combined,
            user_text  = user_message,
            session_id = session_id,
            source_type= "user",
        )


# ── Demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Standalone test with a mock chat_fn
    # In production this is model_setup.chat_fn

    def mock_chat_fn(user_message, system_prompt=None, history=None):
        """Returns hardcoded JSON to test parsing without loading LFM."""
        return '''{
  "concepts": [
    {
      "label": "食べる",
      "concept_type": "vocabulary",
      "description": "Japanese verb meaning 'to eat'. Ichidan/Group 2 verb in dictionary form.",
      "source_lang": "ja"
    },
    {
      "label": "て-form",
      "concept_type": "grammar",
      "description": "Japanese verb conjugation form used for connecting actions, making requests, and forming progressive tense.",
      "source_lang": "ja"
    },
    {
      "label": "食べて",
      "concept_type": "example",
      "description": "The て-form of 食べる. Used in: 食べてください (please eat), 食べています (I am eating).",
      "source_lang": "ja"
    }
  ],
  "relations": [
    {
      "src_label": "食べて",
      "relation": "is_form_of",
      "dst_label": "食べる"
    },
    {
      "src_label": "食べて",
      "relation": "example_of",
      "dst_label": "て-form"
    }
  ]
}'''

    print("=== ConceptExtractor demo (mock LFM) ===\n")

    # Test extraction parsing
    extractor_test = ConceptExtractor.__new__(ConceptExtractor)
    extractor_test.chat_fn  = mock_chat_fn
    extractor_test.verbose  = True
    extractor_test.relation_weight = 1.5
    extractor_test.concept_types  = CONCEPT_TYPES
    extractor_test.relation_types = RELATION_TYPES

    result = extractor_test.extract(
        "食べる means to eat. Its て-form is 食べて, "
        "used in 食べてください (please eat)."
    )

    print(f"Success: {result.success}  Attempts: {result.attempts}")
    print(f"\nConcepts ({len(result.concepts)}):")
    for c in result.concepts:
        print(f"  [{c.concept_type}] {c.label} ({c.source_lang})")
        print(f"    {c.description}")

    print(f"\nRelations ({len(result.relations)}):")
    for r in result.relations:
        print(f"  {r.src_label} —[{r.relation}]→ {r.dst_label}")

    print("\n=== JSON parsing robustness test ===")
    # Test retry logic with bad JSON
    call_count = [0]
    def flaky_chat_fn(user_message, system_prompt=None, history=None):
        call_count[0] += 1
        if call_count[0] < 3:
            return "Sorry, here is the answer: not valid json {{{"
        return '{"concepts": [{"label": "test", "concept_type": "fact", "description": "A test concept.", "source_lang": "en"}], "relations": []}'

    extractor_test.chat_fn = flaky_chat_fn
    result2 = extractor_test.extract("test input")
    print(f"Flaky fn succeeded after {result2.attempts} attempts: {result2.success}")
    print(f"Extracted: {[c.label for c in result2.concepts]}")

    print("\n=== Integration note ===")
    print("In japanese_tutor.py, replace the current ingest call with:")
    print()
    print("  result = extractor.extract_from_turn(")
    print("      user_message = user_input,")
    print("      llm_response = response,")
    print("      session_id   = session_id,")
    print("  )")
    print()
    print("This creates structured concept nodes instead of raw event nodes.")
    print("食べる, て-form, and 食べて become separate searchable memory nodes.")