"""
tutor_engine.py
Shared engine behind both the CLI (japanese_tutor.py) and the HTTP API
(server.py). Owns:

    - session lifecycle (start/end, per-session history + seed nodes)
    - the chat turn: retrieve → respond → extract/ingest → co-retrieval
      reinforcement
    - read models for the dashboard (graph, node detail, stats, ARM-E log)

Import note: core/ modules import each other bare (e.g. `from graph_store
import ...`), so we put this file's directory on sys.path first. This makes
`tutor_engine` importable both as `core.tutor_engine` (from project root)
and as `tutor_engine` (from inside core/).
"""

import os
import re
import sys
import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_store import GraphStore            # noqa: E402
from ingest_pipeline import IngestPipeline    # noqa: E402
from practice_scheduler import PracticeScheduler, cosine_sim  # noqa: E402
from agents.concept_extractor import (  # noqa: E402
    ConceptExtractor,
    EXTRACTION_SYSTEM, CONCEPT_TYPES, RELATION_TYPES,
    SECOND_BRAIN_EXTRACTION_SYSTEM, SECOND_BRAIN_CONCEPT_TYPES,
    SECOND_BRAIN_RELATION_TYPES,
)
from signals import PhysioSample, UtteranceEvent  # noqa: E402


# ── Language detection ────────────────────────────────────────────────

_JA_RE = re.compile(r"[぀-ヿ一-鿿]")

def detect_lang(text: str) -> str:
    """'ja' if the text contains kana or kanji, else 'en'."""
    return "ja" if _JA_RE.search(text or "") else "en"


# ── System prompt (moved from japanese_tutor.py) ─────────────────────

SYSTEM_TEMPLATE = """\
あなたは日本語学習をサポートするAIです。
ユーザーの日本語学習の進捗を記憶しており、過去の会話や学習内容を踏まえて回答します。
英語と日本語の両方で回答できます。ユーザーの言語に合わせてください。

{memory_section}\
"""

MEMORY_SECTION_TEMPLATE = """\

以下は関連する記憶された知識です：

{concepts}
"""

# ── Second-brain preset (general ambient/BCI domain) ──────────────────

SECOND_BRAIN_SYSTEM_TEMPLATE = """\
You are a personal second-brain assistant. You have access to the
user's own remembered thoughts, ideas, tasks, and notes, strengthened
and decayed over time based on how often and how emotionally they
mattered. Use this memory to give grounded, context-aware answers.

{memory_section}\
"""

SECOND_BRAIN_MEMORY_SECTION_TEMPLATE = """\

Relevant remembered context:

{concepts}
"""

# ── Retrieval practice (forgetting-curve-driven exercises) ────────────
# Fading concepts from the graph become practice targets; a correct
# answer reinforces their Hebbian edges, closing the loop:
# decay flags what to practice → practice re-encodes → decay slows.

PRACTICE_SYSTEM = """You are a Japanese practice-exercise generator for a language learner.
Generate ONE short exercise that makes the learner actively recall the target concepts.
Vary the exercise type: EN→JA translation, fill-in-the-blank, produce a sentence using a
grammar point, or transform a form (e.g. dictionary → て-form).
Keep it at beginner-intermediate level unless the concepts suggest otherwise.
You must respond with ONLY valid JSON — no explanation, no markdown.

JSON shape:
{"exercise_type": "translation|fill_in_blank|sentence_production|form_transform",
 "prompt": "the exercise question",
 "hint": "short nudge, may be empty string",
 "answer": "the expected answer"}"""

PRACTICE_TARGETS_HEADER = "Target concepts (use one or more of these):\n"

PRACTICE_GRADE_SYSTEM = """You are a Japanese practice-answer grader for a language learner.
Compare the learner's answer to the expected answer. Accept correct answers with minor
orthography differences (kana variants, punctuation, full/half width). Judge meaning,
not style. If the answer is partially right, mark it incorrect but say what was right.
You must respond with ONLY valid JSON — no explanation, no markdown.

JSON shape:
{"correct": true or false, "feedback": "1-3 sentences: what was right/wrong, and the
correct form if wrong. Match the learner's language (English or Japanese)."}"""


def build_system_prompt(
    retrieved       : list[dict],
    system_template : str = SYSTEM_TEMPLATE,
    memory_template : str = MEMORY_SECTION_TEMPLATE,
) -> str:
    if not retrieved:
        return system_template.format(memory_section="")

    concept_lines = []
    for r in retrieved:
        src = f" ({r['source_uri']})" if r['source_uri'] else ""
        concept_lines.append(
            f"[{r['concept_type']}] {r['label']}{src}\n{r['text_summary']}"
        )

    memory_section = memory_template.format(
        concepts="\n\n".join(concept_lines)
    )
    return system_template.format(memory_section=memory_section)


# ── Domain presets ────────────────────────────────────────────────────
# Two personas coexisting on the shared engine/pipeline: the original
# Japanese tutor, and the general second-brain / ambient-BCI mode.

DOMAIN_PRESETS = {
    "japanese_tutor": dict(
        system_template   = SYSTEM_TEMPLATE,
        memory_template   = MEMORY_SECTION_TEMPLATE,
        extraction_system = EXTRACTION_SYSTEM,
        concept_types     = CONCEPT_TYPES,
        relation_types    = RELATION_TYPES,
    ),
    "second_brain": dict(
        system_template   = SECOND_BRAIN_SYSTEM_TEMPLATE,
        memory_template   = SECOND_BRAIN_MEMORY_SECTION_TEMPLATE,
        extraction_system = SECOND_BRAIN_EXTRACTION_SYSTEM,
        concept_types     = SECOND_BRAIN_CONCEPT_TYPES,
        relation_types    = SECOND_BRAIN_RELATION_TYPES,
    ),
}


# ── Session state ─────────────────────────────────────────────────────

@dataclass
class SessionState:
    session_id : str
    history    : list[dict] = field(default_factory=list)
    seed_ids   : list[str]  = field(default_factory=list)
    pending_exercise: Optional[dict] = None   # active practice exercise


# ── JSON helper for practice LLM calls ────────────────────────────────

def _parse_llm_json(raw: str) -> dict:
    """Extract the first JSON object from an LLM response (fences/preamble tolerated)."""
    clean = re.sub(r"```(?:json)?|```", "", raw or "").strip()
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in response")
    return json.loads(match.group())


# ── TutorEngine ───────────────────────────────────────────────────────

class TutorEngine:
    """
    One engine per process. The CLI and the HTTP API are thin shells
    over these methods.

    Not thread-safe by design — the tutor is single-user. FastAPI runs
    endpoints in a threadpool but Neo4j driver calls are serialized by
    GraphStore sessions; concurrent chats would interleave ARM-E state,
    so the API should effectively see one conversation at a time.
    """

    SEED_IDS_MAX = 50    # cap on per-session seed node list

    def __init__(
        self,
        store           : GraphStore,
        pipeline        : IngestPipeline,
        extractor       : ConceptExtractor,
        chat_fn,                           # model_setup.chat_fn
        embedding_fn,                      # model_setup.embedding_fn
        retrieval_top_k : int   = 4,
        max_history     : int   = 20,
        co_retrieval_eta: float = 0.02,    # Hebbian bump per co-retrieval
        practice_top_k  : int   = 2,      # fading concepts per exercise
        interference_threshold: float = 0.82,   # cosine cap between practice targets
        system_template : str   = SYSTEM_TEMPLATE,
        memory_template : str   = MEMORY_SECTION_TEMPLATE,
    ):
        self.store            = store
        self.pipeline         = pipeline
        self.extractor        = extractor
        self.chat_fn          = chat_fn
        self.embedding_fn     = embedding_fn
        self.retrieval_top_k  = retrieval_top_k
        self.max_history      = max_history
        self.co_retrieval_eta = co_retrieval_eta
        self.practice_top_k   = practice_top_k
        self.interference_threshold = interference_threshold
        self.system_template  = system_template
        self.memory_template  = memory_template

        # Adaptive practice scheduling: per-type half-lives learned from
        # graded answers, persisted under ./practice_state/.
        self.scheduler = PracticeScheduler()

        self._sessions: dict[str, SessionState] = {}

    # ── Sessions ──────────────────────────────────────────────────────

    def start_session(self, trigger: str = "api") -> str:
        sid = self.pipeline.start_session(trigger=trigger)
        self._sessions[sid] = SessionState(session_id=sid)
        return sid

    def end_session(self, session_id: str):
        """Idempotent — ending an unknown session is a no-op."""
        if session_id in self._sessions:
            self.pipeline.end_session(session_id)
            del self._sessions[session_id]

    def _state(self, session_id: str) -> SessionState:
        # Auto-create state for unknown sessions: if the server restarted
        # mid-conversation the UI keeps working instead of 404ing. The
        # underlying Neo4j session is gone, so ingest proceeds untracked.
        if session_id not in self._sessions:
            print(f"[engine] unknown session {session_id[:8]}… — ad-hoc state")
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    # ── Chat turn ─────────────────────────────────────────────────────

    def chat(self, message: str, session_id: str) -> dict:
        """
        One full conversation turn:
            1. Retrieve relevant concepts (vector + Hebbian + causal)
            2. Generate response with memory injected into system prompt
            3. Extract concepts from the exchange and ingest them
            4. Reinforce edges between co-RETRIEVED nodes — the temporal
               Hebbian signal ("retrieved together → wired together"),
               gated by this turn's m_t
            5. Update conversation state

        Returns the ChatResponse dict the UI expects.
        """
        state = self._state(session_id)

        # ── 1. Retrieve ───────────────────────────────────────────────
        query_emb = self.embedding_fn(message)
        retrieved = self.store.retrieve_for_llm(
            query_embedding = query_emb,
            seed_node_ids   = state.seed_ids[-10:],
            top_k           = self.retrieval_top_k,
        )

        # ── 2. Respond ────────────────────────────────────────────────
        response = self.chat_fn(
            user_message  = message,
            history       = state.history,
            system_prompt = build_system_prompt(
                retrieved, self.system_template, self.memory_template
            ),
        )

        # ── 3. Extract + ingest ───────────────────────────────────────
        ext = self.extractor.extract_from_turn(
            user_message = message,
            llm_response = response,
            session_id   = session_id,
        )
        state.seed_ids.extend(ext.node_ids)
        if len(state.seed_ids) > self.SEED_IDS_MAX:
            state.seed_ids = state.seed_ids[-self.SEED_IDS_MAX:]

        turn = ext.ingest_results[0] if ext.ingest_results else None
        m_t  = turn.m_t if turn else 1.0

        # ── 4. Co-retrieval reinforcement ─────────────────────────────
        # Validation-gated: if extraction fell back to a raw event node,
        # the turn's content is unverified — wiring it into the graph
        # would reinforce whatever the extractor hallucinated. Skip.
        retrieved_ids = [r["node_id"] for r in retrieved]
        if len(retrieved_ids) >= 2 and getattr(ext, "validation_passed", True):
            self.store.reinforce_co_retrieved(
                retrieved_ids, m_t=m_t, eta=self.co_retrieval_eta,
            )

        # ── 5. Conversation state ─────────────────────────────────────
        state.history.append({"role": "user",      "content": message})
        state.history.append({"role": "assistant", "content": response})
        if len(state.history) > self.max_history:
            state.history = state.history[-self.max_history:]

        arme = None
        if turn is not None:
            arme = {
                "quadrant"    : turn.quadrant,
                "m_t"         : round(turn.m_t, 4),
                "gated"       : turn.gated,
                "valence"     : round(turn.valence, 4),
                "arousal"     : round(turn.arousal, 4),
                "dominance"   : round(turn.dominance, 4),
                "delta_w_mean": round(turn.delta_w_mean, 6),
            }

        concepts = [
            {
                "label"      : c.label,
                "type"       : c.concept_type,
                "description": c.description,
                "lang"       : c.source_lang,
            }
            for c in ext.extracted.concepts
        ]

        return {
            "response" : response,
            "arme"     : arme,
            "concepts" : concepts,
            "lang"     : detect_lang(response),
            "retrieved": [
                {
                    "node_id"     : r["node_id"],
                    "label"       : r["label"],
                    "text_summary": r["text_summary"],
                    "concept_type": r["concept_type"],
                    "score"       : r["score"],
                }
                for r in retrieved
            ],
        }

    # ── Retrieval practice (adaptive forgetting-curve exercises) ──────

    def _select_practice_targets(self) -> list[dict]:
        """
        Rank fading candidates by predicted recall probability (lowest
        first), then greedily pick up to practice_top_k respecting:
          - spacing guard (not practiced within the last few minutes)
          - semantic-interference filter (skip candidates whose embedding
            cosine with an already-selected target exceeds the threshold —
            confusable items like 食べる/食べた get interleaved across
            exercises, not crammed into one)
        """
        pool = self.store.get_fading_concepts(
            limit=self.practice_top_k * 6,
        )

        now = time.time()
        ranked = sorted(
            pool,
            key=lambda f: self.scheduler.recall_prob(
                now - f["concept"].updated_at,
                f["concept"].concept_type,
            ),
        )

        selected: list[dict] = []
        for cand in ranked:
            nid = cand["concept"].node_id

            if not self.scheduler.ready_again(nid):
                continue    # spacing guard — practiced too recently

            # semantic-interference filter
            too_similar = any(
                cosine_sim(cand["concept"].embedding,
                           sel["concept"].embedding)
                > self.interference_threshold
                for sel in selected
                if cand["concept"].embedding and sel["concept"].embedding
            )
            if too_similar:
                continue

            elapsed = now - cand["concept"].updated_at
            cand["recall_prob"] = round(self.scheduler.recall_prob(
                elapsed, cand["concept"].concept_type), 4)
            selected.append(cand)
            if len(selected) >= self.practice_top_k:
                break

        return selected

    def fading_concepts(self, limit: int = 10) -> list[dict]:
        """Dashboard-facing view of the practice queue."""
        return [
            {
                "node_id"      : f["concept"].node_id,
                "label"        : f["concept"].label,
                "concept_type" : f["concept"].concept_type,
                "text_summary" : f["concept"].text_summary,
                "avg_w"        : round(f["avg_w"], 4),
                "degree"       : f["degree"],
                "strength"     : f["strength"],
                "staleness"    : f["staleness"],
                "fading_score" : f["fading_score"],
                "recall_prob"  : round(self.scheduler.recall_prob(
                    time.time() - f["concept"].updated_at,
                    f["concept"].concept_type), 4),
            }
            for f in self.store.get_fading_concepts(limit=limit)
        ]

    def next_practice(self, session_id: str) -> Optional[dict]:
        """
        Generate one exercise targeting the concepts least likely to be
        recalled, subject to spacing and interference constraints.

        Returns None if there's nothing worth practicing yet.
        """
        state = self._state(session_id)

        targets = self._select_practice_targets()
        if not targets:
            return None

        target_lines = "\n".join(
            f"- {t['concept'].label} ({t['concept'].concept_type}): "
            f"{t['concept'].text_summary}"
            for t in targets
        )
        prompt = (PRACTICE_TARGETS_HEADER + target_lines
                  + "\n\nGenerate the exercise JSON now.")

        raw = self.chat_fn(
            user_message  = prompt,
            system_prompt = PRACTICE_SYSTEM,
            history       = None,
        )
        try:
            parsed = _parse_llm_json(raw)
            exercise = {
                "exercise_type": str(parsed.get("exercise_type", "translation")),
                "prompt"       : str(parsed.get("prompt", "")).strip(),
                "hint"         : str(parsed.get("hint", "")).strip(),
                "answer"       : str(parsed.get("answer", "")).strip(),
                "targets"      : [t["concept"].label for t in targets],
                "node_ids"     : [t["concept"].node_id for t in targets],
                "target_types" : [t["concept"].concept_type for t in targets],
                "recall_probs" : {t["concept"].node_id: t["recall_prob"]
                                  for t in targets},
            }
        except (ValueError, json.JSONDecodeError) as e:
            print(f"[engine] practice generation failed: {e}")
            return None

        if not exercise["prompt"] or not exercise["answer"]:
            print("[engine] practice generation returned empty prompt/answer")
            return None

        state.pending_exercise = exercise
        return {
            "exercise_type": exercise["exercise_type"],
            "prompt"       : exercise["prompt"],
            "hint"         : exercise["hint"],
            "targets"      : exercise["targets"],
            # recall probabilities exposed so the UI can visualize why
            # this came up (low p = the model expects you to forget it)
            "fading"       : [
                {"label": t["concept"].label, "score": t["recall_prob"]}
                for t in targets
            ],
        }

    def submit_practice_answer(self, session_id: str, answer: str) -> Optional[dict]:
        """
        Grade the pending exercise. A correct answer re-encodes memory:
        target nodes get activation bumps and their pairwise edges get a
        Hebbian reinforcement bump (m_t=1.5 — retrieval success is exactly
        the co-activation signal the graph is built on).
        """
        state = self._state(session_id)
        ex = state.pending_exercise
        if ex is None:
            return None

        grade_prompt = (
            f"Exercise: {ex['prompt']}\n"
            f"Expected answer: {ex['answer']}\n"
            f"Learner's answer: {answer}\n\n"
            "Respond with the grading JSON now."
        )
        try:
            raw = self.chat_fn(
                user_message  = grade_prompt,
                system_prompt = PRACTICE_GRADE_SYSTEM,
                history       = None,
            )
            parsed = _parse_llm_json(raw)
            correct   = bool(parsed.get("correct", False))
            feedback  = str(parsed.get("feedback", "")).strip()
        except (ValueError, json.JSONDecodeError) as e:
            print(f"[engine] practice grading failed: {e}")
            return {"correct": False,
                    "feedback": "(grading failed — try again)",
                    "reinforced": False}

        reinforced = False
        if correct:
            nids = [n for n in ex["node_ids"]]
            for nid in nids:
                self.store.increment_activation(nid)
            if len(nids) >= 2:
                self.store.reinforce_co_retrieved(
                    nids, m_t=1.5, eta=self.co_retrieval_eta,
                )
            reinforced = True

        # Adaptive half-lives learn from the outcome regardless of
        # correctness — wrong answers shorten that type's horizon.
        self.scheduler.record(
            node_ids      = ex["node_ids"],
            concept_types = ex.get("target_types",
                                   [ex["targets"][0]] if ex["targets"] else []),
            correct       = correct,
        )

        state.pending_exercise = None
        return {
            "correct"   : correct,
            "feedback"  : feedback,
            "answer"    : ex["answer"],
            "targets"   : ex["targets"],
            "reinforced": reinforced,
        }

    # ── Ambient ingestion (second-brain / BCI path) ────────────────────

    def ingest_ambient(
        self,
        utterance : UtteranceEvent,
        physio    : PhysioSample,
        session_id: str,
    ) -> dict:
        """
        Ingest one decoded ambient utterance (e.g. from subvocal EMG) with
        its accompanying physiological sample. No chat turn — there is no
        response to generate, just extraction + Hebbian ingest, gated by
        the attention signal ARM-E doesn't get from text chat.

        Mirrors chat()'s extract+ingest step (3) without retrieve/respond
        (1-2) or co-retrieval reinforcement (4), since there's no query to
        retrieve against and nothing was co-retrieved this turn.
        """
        state = self._state(session_id)
        attention = physio.attention if physio.attention is not None else 1.0

        ext = self.extractor.extract_and_ingest(
            text        = utterance.text,
            user_text   = utterance.text,
            session_id  = session_id,
            source_type = "sensor",
            attention   = attention,
        )
        state.seed_ids.extend(ext.node_ids)
        if len(state.seed_ids) > self.SEED_IDS_MAX:
            state.seed_ids = state.seed_ids[-self.SEED_IDS_MAX:]

        turn = ext.ingest_results[0] if ext.ingest_results else None

        arme = None
        if turn is not None:
            arme = {
                "quadrant"    : turn.quadrant,
                "m_t"         : round(turn.m_t, 4),
                "gated"       : turn.gated,
                "valence"     : round(turn.valence, 4),
                "arousal"     : round(turn.arousal, 4),
                "dominance"   : round(turn.dominance, 4),
                "delta_w_mean": round(turn.delta_w_mean, 6),
            }

        concepts = [
            {
                "label"      : c.label,
                "type"       : c.concept_type,
                "description": c.description,
                "lang"       : c.source_lang,
            }
            for c in ext.extracted.concepts
        ]

        return {
            "arme"     : arme,
            "concepts" : concepts,
            "attention": round(attention, 4),
            "hr_bpm"   : physio.hr_bpm,
            "utterance": utterance.text,
        }

    # ── Dashboard read models ─────────────────────────────────────────

    @staticmethod
    def _concept_to_json(c) -> dict:
        return {
            "node_id"         : c.node_id,
            "label"           : c.label,
            "concept_type"    : c.concept_type,
            "source_type"     : c.source_type,
            "source_lang"     : detect_lang(c.label),
            "activation_count": c.activation_count,
            "mean_m_t"        : c.mean_m_t,
            "mean_dominance"  : c.mean_dominance,
            "text_summary"    : c.text_summary,
            "created_at"      : c.created_at,
        }

    def graph_data(
        self,
        min_weight      : float = 0.5,
        limit           : int   = 200,
        layer           : str   = "hippocampal",
        include_isolated: bool  = True,
    ) -> dict:
        sg = self.store.get_global_graph(
            layer=layer, min_weight=min_weight, limit=limit,
        )
        nodes = list(sg.nodes)

        # get_global_graph only returns nodes touching edges above
        # min_weight — merge in the rest so fresh/weakly-connected
        # concepts still show up in the dashboard
        if include_isolated:
            have = {n.node_id for n in nodes}
            for c in self.store.list_concepts(limit=limit):
                if c.node_id not in have:
                    nodes.append(c)

        return {
            "nodes": [self._concept_to_json(n) for n in nodes],
            "edges": [
                {
                    "source"      : e.src_id,
                    "target"      : e.dst_id,
                    "hebb_weight" : e.hebb_weight,
                    "eligibility" : e.eligibility,
                    "causal_score": e.causal_score,
                }
                for e in sg.edges
            ],
        }

    def node_detail(self, node_id: str) -> Optional[dict]:
        c = self.store.get_concept(node_id)
        return self._concept_to_json(c) if c else None

    def system_stats(self) -> dict:
        # Two separate counts: a joined MATCH returns zero rows (and
        # therefore zero nodes) whenever the graph has no edges yet.
        nodes = self.store._run("MATCH (c:Concept) RETURN count(c) AS n")
        edges = self.store._run(
            "MATCH ()-[r:ASSOCIATED_WITH]->() RETURN count(r) AS n"
        )
        sess = self.store._run(
            "MATCH (s:Session) RETURN count(s) AS n_sessions, "
            "coalesce(sum(s.total_activations), 0) AS n_ingests"
        )
        return {
            "n_nodes"   : nodes[0]["n"] if nodes else 0,
            "n_edges"   : edges[0]["n"] if edges else 0,
            "n_sessions": sess[0]["n_sessions"] if sess else 0,
            "n_ingests" : sess[0]["n_ingests"] if sess else 0,
        }

    def arme_history(self, limit: int = 50) -> list[dict]:
        """Flatten ARME._history (in-memory) for the dashboard sparkline."""
        history = getattr(getattr(self.pipeline, "arme", None), "_history", [])
        out = []
        for s in history[-limit:]:
            out.append({
                "quadrant" : s.emotion.quadrant,
                "m_t"      : round(s.m_t, 4),
                "gated"    : s.gated,
                "valence"  : round(s.emotion.valence, 4),
                "arousal"  : round(s.emotion.arousal, 4),
                "dominance": round(s.dominance, 4),
                "r_t"      : round(s.r_t, 4),
                "timestamp": s.timestamp,
            })
        return out

    # ── Shutdown ──────────────────────────────────────────────────────

    def close(self):
        for sid in list(self._sessions):
            self.end_session(sid)
        self.scheduler.save()
        self.store.close()


# ── Engine factories ──────────────────────────────────────────────────

def build_engine(
    neo4j_uri      : str = None,
    neo4j_user     : str = None,
    neo4j_password : str = None,
    device         : str = "mps",
    cortical_dir   : str = "./cortical_state",
    consolidate_n  : int = 20,
    retrieval_top_k: int = 4,
    max_history    : int = 20,
    verbose        : bool = True,
    domain         : str = "japanese_tutor",   # "japanese_tutor" | "second_brain"
    signal_source  = None,   # SignalSource — if given, fuses HR into arousal
) -> TutorEngine:
    """
    Production engine: BGE-M3 embeddings, LFM2.5 chat/summaries,
    vad-bert emotion, cortical consolidation scheduler.

    Credentials default to env vars NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD.
    Must be called from the project root (model_setup uses ./core/vad_model).

    domain selects which persona/extraction preset (DOMAIN_PRESETS) is
    wired into the ConceptExtractor and TutorEngine — the two coexist on
    the same graph and pipeline, just with different prompts.

    signal_source: pass a SignalSource (e.g. SimulatedSignalSource, or
    the real EMG rig's adapter once it exists) to have arousal come from
    live heart rate instead of text sentiment alone — see
    physio.PhysioFusedEmotionClassifier. Text-only emotion classification
    (the japanese_tutor default) is used when this is None.
    """
    preset = DOMAIN_PRESETS[domain]

    from core.model_setup import (
        embedding_fn, summary_fn, chat_fn, emotion_classifier,
    )
    from core.cortical_gnn import CorticalGNN
    from core.consolidation_scheduler import ConsolidationScheduler

    if signal_source is not None:
        from core.physio import PhysioFusedEmotionClassifier
        emotion_classifier = PhysioFusedEmotionClassifier(
            text_classifier=emotion_classifier, physio_source=signal_source,
        )

    store = GraphStore(
        neo4j_uri      or os.environ.get("NEO4J_URI",      "neo4j://localhost:7687"),
        neo4j_user     or os.environ.get("NEO4J_USER",     "neo4j"),
        neo4j_password or os.environ.get("NEO4J_PASSWORD", "password"),
    )
    store.setup_schema()

    pipeline = IngestPipeline(
        graph_store        = store,
        embedding_fn       = embedding_fn,
        summary_fn         = summary_fn,
        feature_dim        = 1024,               # BGE-M3 output dimension
        device             = device,
        emotion_classifier = emotion_classifier, # reuse — no double load
    )

    cort_gnn = CorticalGNN(
        in_dim=1024, hidden_dim=256, out_dim=1024, heads=4,
        lambda_causal=0.3, lambda_consist=0.5,
    )
    scheduler = ConsolidationScheduler(
        store              = store,
        cortical_gnn       = cort_gnn,
        model_dir          = cortical_dir,
        every_n            = consolidate_n,
        feature_dim        = 1024,
        consolidate_epochs = 30,
        consolidate_lr     = 5e-4,
        min_hebb_weight    = 0.5,
        background         = False,
        verbose            = verbose,
    )
    scheduler.load()
    pipeline.scheduler = scheduler

    extractor = ConceptExtractor(
        pipeline=pipeline, chat_fn=chat_fn, store=store,
        relation_weight=1.5, verbose=verbose,
        extraction_system=preset["extraction_system"],
        concept_types=preset["concept_types"],
        relation_types=preset["relation_types"],
    )

    return TutorEngine(
        store=store, pipeline=pipeline, extractor=extractor,
        chat_fn=chat_fn, embedding_fn=embedding_fn,
        retrieval_top_k=retrieval_top_k, max_history=max_history,
        system_template=preset["system_template"],
        memory_template=preset["memory_template"],
    )


def build_stub_engine(
    neo4j_uri      : str = None,
    neo4j_user     : str = None,
    neo4j_password : str = None,
    retrieval_top_k: int = 4,
    domain         : str = "japanese_tutor",   # "japanese_tutor" | "second_brain"
    signal_source  = None,   # SignalSource — if given, fuses HR into arousal
) -> TutorEngine:
    """
    Lightweight engine for trying the UI without downloading 3GB of
    models. Deterministic hash embeddings (1024-dim, matches the vector
    index), template chat responses, rule-based emotion. Still needs
    Neo4j running. Enable with HEBBIAN_STUB=1.

    domain selects the persona/extraction preset, same as build_engine.
    signal_source: same as build_engine — fuses live heart rate into
    arousal via PhysioFusedEmotionClassifier when provided.
    """
    preset = DOMAIN_PRESETS[domain]
    import math
    import random
    from types import SimpleNamespace

    def stub_embedding_fn(text: str, dim: int = 1024) -> list[float]:
        # hashlib, not hash(): str hashing is salted per process
        # (PYTHONHASHSEED), which would change embeddings across restarts.
        seed = int.from_bytes(
            hashlib.sha256(text.encode("utf-8")).digest()[:4], "big"
        )
        random.seed(seed)
        v = [random.gauss(0, 1) for _ in range(dim)]
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]

    def stub_summary_fn(text: str) -> str:
        sentences = text.replace("?", ".").replace("!", ".").split(".")
        return ". ".join(s.strip() for s in sentences[:2] if s.strip()) + "."

    def stub_chat_fn(user_message, history=None, system_prompt=None) -> str:
        sp = system_prompt or ""
        # ConceptExtractor asks for JSON — give it one concept per turn.
        if "knowledge extraction" in sp:
            # user_message is EXTRACTION_PROMPT_TEMPLATE — pull the label
            # from the actual text between the --- delimiters, not the
            # "Extract concepts..." instruction line itself.
            parts  = user_message.split("---")
            source = parts[1] if len(parts) >= 2 else user_message
            words  = [w.strip(".,!?—\"'") for w in source.split()]
            # Skip transcript role prefixes: with concept identity, labelling
            # every turn "User:" would collapse the stub graph to one node.
            words  = [w for w in words if len(w) > 3 and w not in ("User:", "Assistant:")]
            label = words[0] if words else "conversation"
            return json.dumps({
                "concepts": [{
                    "label"       : label[:40],
                    "concept_type": "fact",
                    "description" : f"Concept mentioned by the user: {label[:40]}.",
                    "source_lang" : detect_lang(label),
                }],
                "relations": [],
            })
        if "practice-exercise generator" in sp:
            target = "(unknown)"
            for line in user_message.splitlines():
                if line.startswith("- ") and ":" in line:
                    target = line[2:].split(":")[0].strip()
                    break
            return json.dumps({
                "exercise_type": "translation",
                "prompt"       : f"(stub) Translate to Japanese using “{target}”: "
                                 f"I want to eat ramen.",
                "hint"         : "",
                "answer"       : "ラーメンを食べたいです。",
            })
        if "practice-answer grader" in sp:
            expected = ""
            for line in user_message.splitlines():
                if line.startswith("Expected answer:"):
                    expected = line.split(":", 1)[1].strip()
                    break
            given = ""
            for line in user_message.splitlines():
                if line.startswith("Learner's answer:"):
                    given = line.split(":", 1)[1].strip()
                    break
            correct = bool(expected) and expected == given
            return json.dumps({
                "correct": correct,
                "feedback": ("(stub) 正解！Memory reinforced."
                             if correct else
                             f"(stub) Not quite — expected: {expected or '?'}"),
            })
        mem = "記憶" if "記憶" in sp else ""
        return (f"(stub) You said: “{user_message}”. "
                f"Real responses need the LFM model — run without HEBBIAN_STUB. {mem}")

    def stub_emotion_classifier(text: str):
        t = (text or "").lower()
        valence, arousal = 0.1, 0.4
        if any(w in t for w in ("!", "amazing", "fascinating", "great")):
            valence, arousal = 0.7, 0.8
        elif any(w in t for w in ("confused", "stuck", "wrong", "hate")):
            valence, arousal = -0.6, 0.75
        elif "?" in t or "？" in t:
            valence, arousal = 0.3, 0.65
        return SimpleNamespace(valence=valence, arousal=arousal, dominance=0.5)

    emotion_classifier = stub_emotion_classifier
    if signal_source is not None:
        from core.physio import PhysioFusedEmotionClassifier
        emotion_classifier = PhysioFusedEmotionClassifier(
            text_classifier=stub_emotion_classifier, physio_source=signal_source,
        )

    store = GraphStore(
        neo4j_uri      or os.environ.get("NEO4J_URI",      "neo4j://localhost:7687"),
        neo4j_user     or os.environ.get("NEO4J_USER",     "neo4j"),
        neo4j_password or os.environ.get("NEO4J_PASSWORD", "password"),
    )
    store.setup_schema()

    pipeline = IngestPipeline(
        graph_store        = store,
        embedding_fn       = stub_embedding_fn,
        summary_fn         = stub_summary_fn,
        feature_dim        = 1024,
        emotion_classifier = emotion_classifier,
    )
    extractor = ConceptExtractor(
        pipeline=pipeline, chat_fn=stub_chat_fn, store=store,
        relation_weight=1.5, verbose=False,
        extraction_system=preset["extraction_system"],
        concept_types=preset["concept_types"],
        relation_types=preset["relation_types"],
    )
    return TutorEngine(
        store=store, pipeline=pipeline, extractor=extractor,
        chat_fn=stub_chat_fn, embedding_fn=stub_embedding_fn,
        retrieval_top_k=retrieval_top_k,
        system_template=preset["system_template"],
        memory_template=preset["memory_template"],
    )
