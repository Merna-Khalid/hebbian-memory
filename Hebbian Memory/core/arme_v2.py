"""
ARM-E: Autonomous Reinforcement Modulation Engine — v2

Three improvements from literature review:

1. VALENCE×AROUSAL INTERACTION (from neuroscience: Kensinger 2004, PMC3530455)
   Old: e_t = tanh(α|v| + β·a)   — magnitude only, ignores quadrant
   New: quadrant-aware formula that treats the four emotional states differently
        High arousal + negative valence (Q2) SUPPRESSES learning
        High arousal + positive valence  (Q1) AMPLIFIES learning
        Low arousal states produce moderate, asymmetric effects

2. SPARSE ELIGIBILITY GATING (from MOHN, arxiv 1909.09902)
   m_t is still global (one scalar per session — preserves Point 5)
   But the gate g_ij controls WHICH edges can receive the signal
   Only edges with sufficient eligibility trace AND signal confidence update
   This is not a contradiction: dopamine is global, synaptic eligibility is local

3. ANTI-HEBBIAN NORMALIZATION (from SoftHebb, Moraitis et al. 2022)
   Edges that DON'T receive reinforcement get a mild anti-Hebbian decay
   This normalizes weights relative to each other, not just absolutely
   Prevents "rich get richer" without needing a hard weight ceiling
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import math
import time


# ── Emotional state dataclass ─────────────────────────────────────────

@dataclass
class EmotionalState:
    valence   : float   # [-1, +1]  negative ← → positive
    arousal   : float   # [0,   1]  calm ← → excited
    dominance : float   # [0,   1]  controlled ← → in-control (w_dom=0.0 until Phase 4)
    quadrant  : str     # Q1/Q2/Q3/Q4
    e_t       : float   # [-1, +1] final signal


@dataclass
class ARMEState:
    r_t        : float
    u_t        : float
    delta_t    : float
    emotion    : EmotionalState
    s_t        : float
    m_t_raw    : float
    m_t        : float
    dominance  : float   # raw dominance from VADScore — logged, w_dom=0.0 until Phase 4
    gated      : bool    # was this step gated (signal too weak)?
    timestamp  : float = field(default_factory=time.time)


# ── Improvement 1: Quadrant-aware e_t ────────────────────────────────

def compute_e_t_quadrant(
    valence  : float,   # [-1, +1]
    arousal  : float,   # [0, 1]
    alpha_e  : float = 0.6,
    beta_e   : float = 0.4,
) -> EmotionalState:
    """
    Quadrant-aware emotional intensity.

    The key insight from the literature (Kensinger 2004, PMC3530455):
    Valence and arousal do NOT combine additively for memory encoding.
    High arousal amplifies the effect of valence — it doesn't just
    add a fixed intensity boost on top of it.

    Old formula: e_t = tanh(α|v| + β·a) · sign(v)
        Problem: |v|=0.8, a=0.9 gives same magnitude whether
                 v=+0.8 (curiosity) or v=-0.8 (anxiety)
                 But anxiety should SUPPRESS learning, not amplify it.

    New formula: separate the arousal effect by valence sign.

    Positive valence (v > 0):
        e_t = tanh(α·v + β·a)
        Both valence and arousal push e_t positive → amplify learning
        "Excited and happy" → strong encoding

    Negative valence (v < 0):
        e_t = -tanh(α·|v| + β·(1 - a))
        High arousal with negative valence → LESS suppression
        Wait — let's think carefully here.

        The neuroscience finding: high arousal IMPAIRS working memory
        for negative stimuli. This means:
            Q2 (high arousal, negative): suppress short-term encoding
                                         but may strengthen long-term (amygdala)
            Q3 (low arousal, negative):  mild suppression, forgetting

        For our purposes (RAG memory system):
        We want to suppress Hebbian encoding when user is confused/anxious
        because incorrect associations formed under anxiety are not useful.

        Q2: e_t = -tanh(α·|v| · (1 + β·a))   strong negative, arousal amplifies suppression
        Q3: e_t = -tanh(α·|v| · (1 - β·a))   mild negative, arousal slightly reduces suppression

        Summary:
        Q1 (v>0, a>0.5): strong positive, novelty gate ON
        Q2 (v<0, a>0.5): strong negative, novelty gate OFF
        Q3 (v<0, a<0.5): mild negative, normal decay
        Q4 (v>0, a<0.5): mild positive, steady learning
    """
    v_abs = abs(valence)
    threshold = 0.5   # arousal threshold for high/low

    if valence >= 0 and arousal >= threshold:
        # Q1: curiosity, excitement, joy
        quadrant = "Q1"
        e_t = math.tanh(alpha_e * valence + beta_e * arousal)

    elif valence >= 0 and arousal < threshold:
        # Q4: calm satisfaction, mild interest
        quadrant = "Q4"
        e_t = math.tanh(alpha_e * valence + beta_e * arousal * 0.5)

    elif valence < 0 and arousal >= threshold:
        # Q2: anxiety, rage, confusion with high activation
        # Arousal AMPLIFIES the suppression signal
        quadrant = "Q2"
        e_t = -math.tanh(alpha_e * v_abs * (1.0 + beta_e * arousal))

    else:
        # Q3: boredom, mild frustration, sadness
        # Low arousal means weak signal — mild decay only
        quadrant = "Q3"
        e_t = -math.tanh(alpha_e * v_abs * (1.0 - beta_e * arousal))

    # clip to [-1, 1] as safety
    e_t = max(-1.0, min(1.0, e_t))
    return EmotionalState(valence=valence, arousal=arousal,
                          dominance=0.5,
                          quadrant=quadrant, e_t=e_t)


# ── Improvement 2: Eligibility trace per edge ────────────────────────

class EligibilityTracker:
    """
    Tracks per-edge eligibility traces for sparse ARM-E gating.

    This does NOT make m_t per-edge (m_t remains global).
    It determines which edges are ELIGIBLE to receive m_t.

    Biologically: dopamine floods uniformly (global m_t),
    but only recently potentiated synapses have eligibility tags
    that "catch" the dopamine signal.

    g_ij accumulates when edge (i,j) is activated (co-activation > 0).
    g_ij decays exponentially over time.
    Edge is eligible only when g_ij > theta_elig.
    """

    def __init__(
        self,
        tau_elig   : float = 5.0,    # eligibility decay timescale (in steps)
        theta_elig : float = 0.2,    # minimum eligibility to receive ARM-E signal
        kappa      : float = 1.0,    # scale: how much each co-activation adds
        rho        : float = 0.8,    # partial reset factor after capture
    ):
        self.tau_elig   = tau_elig
        self.theta_elig = theta_elig
        self.kappa      = kappa
        self.rho        = rho
        self._traces: Dict[Tuple, float] = {}   # (i,j) → g_ij

    def update(
        self,
        edge_keys  : List[Tuple[int,int]],   # all edges in subgraph
        coact      : Tensor,                  # [E] co-activation values
    ) -> Dict[Tuple, float]:
        """
        Update eligibility traces after a Hebbian step.
        Returns dict of {(i,j): g_ij} for all edges.
        """
        # decay all existing traces
        for key in list(self._traces.keys()):
            self._traces[key] *= (1.0 - 1.0 / self.tau_elig)
            if self._traces[key] < 1e-4:
                del self._traces[key]

        # accumulate from current co-activations
        for idx, (i, j) in enumerate(edge_keys):
            c = max(0.0, coact[idx].item())   # only positive co-activation
            if c > 0:
                key = (i, j)
                self._traces[key] = self._traces.get(key, 0.0) + self.kappa * c

        return dict(self._traces)

    def get_eligible_mask(
        self, edge_keys: List[Tuple[int,int]]
    ) -> Tensor:
        """
        Returns boolean mask [E]: True if edge is eligible for ARM-E update.
        """
        return torch.tensor([
            self._traces.get((i,j), 0.0) >= self.theta_elig
            for (i, j) in edge_keys
        ], dtype=torch.bool)

    def partial_reset(self, edge_keys: List[Tuple[int,int]]):
        """Call after a capture event — partially decay eligible traces."""
        for (i, j) in edge_keys:
            if (i, j) in self._traces:
                self._traces[(i, j)] *= self.rho


# ── Improvement 3: Anti-Hebbian normalization ────────────────────────

def anti_hebbian_decay(
    edge_weight  : Tensor,              # [E] current weights
    eligible_mask: Tensor,              # [E] bool — which edges were reinforced
    lam_anti     : float = 0.003,       # anti-Hebbian decay rate
    w_floor      : float = 0.1,
) -> Tensor:
    """
    Apply mild anti-Hebbian decay to NON-eligible edges.

    From SoftHebb: edges that don't get reinforced this step
    get a small competitive decay. This normalizes weights
    relative to each other — edges that co-activate frequently
    stay strong; edges that don't co-activate slowly weaken.

    This is separate from Oja's decay (which acts on ALL edges).
    Combined effect:
      Reinforced edge:     Oja decay + ARM-E boost  → grows if signal is strong
      Non-reinforced edge: Oja decay + anti-Hebbian  → decays faster
    """
    decay = torch.where(
        eligible_mask,
        torch.zeros_like(edge_weight),        # no extra decay for reinforced edges
        lam_anti * edge_weight,               # anti-Hebbian for non-reinforced
    )
    return (edge_weight - decay).clamp(min=w_floor)


# ── ARM-E v2 ──────────────────────────────────────────────────────────

class ARME:
    """
    ARM-E v2 — all three improvements integrated.

    Changes from v1:
      - e_t: quadrant-aware (valence × arousal interaction)
      - m_t: sparse eligibility gate before applying to edges
      - Hebbian update: anti-Hebbian normalization on non-eligible edges
      - novelty gate: δ_t only amplifies in Q1/Q4 (positive valence)
    """

    def __init__(
        self,
        w_r        : float = 0.6,
        w_u        : float = -0.2,
        w_d        : float = 0.5,
        w_e        : float = 0.4,
        w_dom      : float = 0.0,    # dominance weight — zero until Phase 4 tuning
        bias       : float = -0.3,
        m_min      : float = 0.2,
        m_max      : float = 3.0,
        ema_lambda : float = 0.4,
        tau        : float = 300.0,
        alpha_r    : float = 0.7,
        # Sparsity gate
        gate_threshold: float = 0.15,   # min signal_confidence to activate ARM-E
        # Anti-Hebbian
        lam_anti   : float = 0.003,
        # Eligibility
        tau_elig   : float = 5.0,
        theta_elig : float = 0.2,
    ):
        self.w   = torch.tensor([w_r, w_u, w_d, w_e, w_dom], dtype=torch.float)
        self.b   = torch.tensor(bias, dtype=torch.float)
        self.m_min         = m_min
        self.m_max         = m_max
        self.ema_lambda    = ema_lambda
        self.tau           = tau
        self.alpha_r       = alpha_r
        self.gate_threshold = gate_threshold
        self.lam_anti      = lam_anti

        self._m_prev    : float = 1.0
        self._last_seen : Dict[str, float] = {}
        self._history   : List[ARMEState] = []
        self.eligibility = EligibilityTracker(tau_elig, theta_elig)

    # ── Signal computations (unchanged from v1) ───────────────────────

    def compute_r_t(self, query_emb, retrieved_embs, scores=None) -> float:
        if retrieved_embs.shape[0] == 0:
            return 0.5
        q = F.normalize(query_emb.unsqueeze(0), dim=1)
        H = F.normalize(retrieved_embs, dim=1)
        k = H.shape[0]
        cos_q = (q @ H.T).squeeze(0).clamp(min=0)
        R = (F.softmax(scores.float(), dim=0) * cos_q).sum().item() \
            if scores is not None else cos_q.mean().item()
        D = 1.0 if k < 2 else \
            1.0 - (H @ H.T)[~torch.eye(k, dtype=torch.bool)].clamp(min=0).mean().item()
        return float(self.alpha_r * R + (1 - self.alpha_r) * D)

    def compute_u_t(self, node_activations, total_activations) -> float:
        return min(1.0, node_activations / total_activations) \
               if total_activations > 0 else 0.0

    def compute_delta_t(self, node_id: str) -> float:
        now = time.time()
        delta = 1.0 if node_id not in self._last_seen else \
                1.0 - math.exp(-(now - self._last_seen[node_id]) / self.tau)
        self._last_seen[node_id] = now
        return float(delta)

    # ── Improvement 1: quadrant-aware e_t ────────────────────────────

    def compute_e_t(
        self,
        text      : str,
        valence   : Optional[float] = None,
        arousal   : Optional[float] = None,
        dominance : Optional[float] = None,
    ) -> EmotionalState:
        if valence is None or arousal is None:
            valence, arousal = self._stub_classifier(text)
        state = compute_e_t_quadrant(valence, arousal)
        # dominance: use override if provided (from EmotionClassifier), else neutral 0.5
        state.dominance = dominance if dominance is not None else 0.5
        return state

    def _stub_classifier(self, text: str) -> Tuple[float, float]:
        t = text.lower()
        valence, arousal = 0.0, 0.3
        for w in ["excellent","amazing","love","brilliant","fascinating","eureka"]:
            if w in t: valence += 0.8; arousal = max(arousal, 0.85); break
        for w in ["good","interesting","helpful","nice","clear","thanks"]:
            if w in t: valence += 0.4; arousal = max(arousal, 0.5); break
        for w in ["terrible","hate","useless","broken","wrong","frustrated","stuck"]:
            if w in t: valence -= 0.8; arousal = max(arousal, 0.75); break
        for w in ["unclear","not sure","hmm","weird","doesn't work","confused"]:
            if w in t: valence -= 0.3; arousal = max(arousal, 0.45); break
        for w in ["why","how","what if","curious","wonder","explain","?"]:
            if w in t: valence += 0.2; arousal = max(arousal, 0.7); break
        return max(-1.0, min(1.0, valence)), max(0.0, min(1.0, arousal))

    # ── Improvement 2: sparse gating ─────────────────────────────────

    def _compute_signal_confidence(
        self, e_t: float, r_t: float, attention: float = 1.0
    ) -> float:
        """
        Confidence that this modulation signal is genuine.
        Requires strong emotion, strong retrieval coherence, AND
        attention: content that's emotionally loaded but unattended
        shouldn't be deeply encoded either. attention defaults to 1.0
        (no-op) for callers that don't have an attention signal — text
        chat, for instance.
        """
        return abs(e_t) * r_t * attention

    # ── m_t: Final modulation scalar ─────────────────────────────────

    def compute_m_t(
        self,
        r_t     : float,
        u_t     : float,
        delta_t : float,
        emotion : EmotionalState,
        attention: float = 1.0,
    ) -> Tuple[float, ARMEState]:
        """
        Improvement 1: novelty gate — δ_t only amplifies in Q1/Q4.
        If user is negative (Q2/Q3), novelty does NOT boost learning.
        You don't want to deeply encode something that's confusing you.

        attention: [0, 1] engagement/focus signal (e.g. from a BCI).
        Defaults to 1.0 — a no-op for callers without one.
        """
        delta_gated = delta_t if emotion.quadrant in ("Q1", "Q4") else 0.0

        x_t = torch.tensor([r_t, u_t, delta_gated, emotion.e_t, emotion.dominance])
        s_t = (self.w @ x_t + self.b).item()
        m_raw = math.exp(s_t)
        m_ema = (1 - self.ema_lambda) * self._m_prev + self.ema_lambda * m_raw
        m_clamp = max(self.m_min, min(self.m_max, m_ema))

        # Improvement 2: sparse gate
        confidence = self._compute_signal_confidence(emotion.e_t, r_t, attention)
        gated = confidence < self.gate_threshold

        if gated:
            m_t = 1.0   # neutral — don't amplify OR suppress on weak signals
        else:
            m_t = m_clamp
            self._m_prev = m_t   # only update EMA when signal is real

        state = ARMEState(
            r_t=r_t, u_t=u_t, delta_t=delta_t,
            emotion=emotion, s_t=s_t, m_t_raw=m_raw,
            m_t=m_t, dominance=emotion.dominance, gated=gated,
        )
        self._history.append(state)
        return m_t, state

    # ── Main entry point ──────────────────────────────────────────────

    def step(
        self,
        query_emb        : Tensor,
        retrieved_embs   : Tensor,
        user_text        : str,
        node_id          : str,
        node_activations : int  = 1,
        total_activations: int  = 10,
        retrieval_scores : Optional[Tensor] = None,
        valence_override : Optional[float]  = None,
        arousal_override : Optional[float]  = None,
        dominance_override: Optional[float]  = None,
        attention        : float = 1.0,
    ) -> Tuple[float, ARMEState]:
        r_t     = self.compute_r_t(query_emb, retrieved_embs, retrieval_scores)
        u_t     = self.compute_u_t(node_activations, total_activations)
        delta_t = self.compute_delta_t(node_id)
        emotion = self.compute_e_t(user_text, valence_override, arousal_override,
                                   dominance_override)
        return self.compute_m_t(r_t, u_t, delta_t, emotion, attention)

    # ── Hebbian update with eligibility + anti-Hebbian ────────────────

    def modulated_hebbian_update(
        self,
        edge_index  : Tensor,    # [2, E]
        edge_weight : Tensor,    # [E]
        H_pre       : Tensor,    # [N, D] features used for co-activation
        m_t         : float,
        eta         : float = 0.05,
        lam_oja     : float = 0.008,   # scalar, or per-edge Tensor [E]
        w_floor     : float = 0.1,
        w_ceil      : float = 5.0,
        eligibility : Optional[Tensor] = None,   # [E] persisted traces
    ) -> Tuple[Tensor, Dict]:
        """
        Full Hebbian update with all three improvements:
        1. m_t already incorporates quadrant-aware e_t and novelty gate
        2. Eligibility mask: only eligible edges receive m_t
        3. Anti-Hebbian decay on non-eligible edges

        eligibility: pass the per-edge traces persisted in Neo4j when the
        subgraph's local node indices are meaningless across calls (the
        ingest pipeline case). When None, the internal EligibilityTracker
        is used instead (standalone / demo case — indices must be stable).

        Returns (new_edge_weight, stats). The updated eligibility trace
        values are in stats["eligibility"] ([E] tensor) so the caller
        can persist them.
        """
        src = edge_index[0]
        dst = edge_index[1]
        h_src = H_pre[src]    # [E, D]
        h_dst = H_pre[dst]    # [E, D]

        coact = (h_src * h_dst).sum(dim=1)   # [E]

        # Update eligibility traces
        edge_keys = list(zip(src.tolist(), dst.tolist()))
        if eligibility is not None:
            # External (persisted) traces: decay + accumulate, same math
            # as EligibilityTracker but stateless — caller owns the state.
            elig_vals = eligibility * (1.0 - 1.0 / self.eligibility.tau_elig) \
                        + self.eligibility.kappa * coact.clamp(min=0)
        else:
            self.eligibility.update(edge_keys, coact)
            elig_vals = torch.tensor(
                [self.eligibility._traces.get(k, 0.0) for k in edge_keys],
                dtype=torch.float,
            )
        eligible = elig_vals >= self.eligibility.theta_elig   # [E] bool

        # Oja's rule — applied to ALL edges (baseline stability).
        # lam_oja may be a scalar or a per-edge tensor (type-conditioned
        # decay — see type_profiles.py); both broadcast against [E].
        hebb  = eta * coact
        decay = lam_oja * edge_weight * (h_src ** 2).sum(1)
        delta_base = hebb - decay

        # ARM-E modulation — only on eligible edges
        # Non-eligible edges: get base update without modulation boost
        m_factor = torch.where(eligible,
                               torch.full_like(edge_weight, m_t),
                               torch.ones_like(edge_weight))
        new_w = edge_weight + m_factor * delta_base

        # Anti-Hebbian normalization on non-eligible edges
        new_w = anti_hebbian_decay(new_w, eligible, self.lam_anti, w_floor)

        new_w = new_w.clamp(w_floor, w_ceil)

        stats = {
            "n_eligible"   : eligible.sum().item(),
            "n_total"      : len(edge_keys),
            "mean_coact"   : coact.mean().item(),
            "mean_w"       : new_w.mean().item(),
            "eligible_mean_w" : new_w[eligible].mean().item() if eligible.any() else 0.0,
            "ineligible_mean_w": new_w[~eligible].mean().item() if (~eligible).any() else 0.0,
            "eligibility"  : elig_vals,   # [E] updated traces — persist these
        }
        return new_w, stats

    def reset_ema(self):
        self._m_prev = 1.0


# ── Demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import os
    os.makedirs("outputs", exist_ok=True)

    torch.manual_seed(42)

    # ── Part 1: Show quadrant differences ────────────────────────────
    print("=== Quadrant-aware e_t vs old formula ===\n")
    print(f"{'Scenario':<28} {'v':>6} {'a':>6} {'Q':>4} "
          f"{'e_t new':>10} {'e_t old':>10} {'diff':>8}")
    print("-" * 74)

    test_cases = [
        ("eureka! (Q1)",          +0.8, 0.9),
        ("curious, excited (Q1)", +0.5, 0.7),
        ("calm, satisfied (Q4)",  +0.6, 0.2),
        ("mildly happy (Q4)",     +0.3, 0.3),
        ("frustrated, stuck (Q2)",-0.7, 0.8),
        ("anxious, panicked (Q2)",-0.9, 0.95),
        ("bored, disengaged (Q3)",-0.4, 0.2),
        ("mildly sad (Q3)",       -0.3, 0.15),
        ("neutral",                0.0, 0.3),
    ]

    for name, v, a in test_cases:
        state_new = compute_e_t_quadrant(v, a)
        # old formula
        magnitude = math.tanh(0.6 * abs(v) + 0.4 * a)
        e_old = math.copysign(magnitude, v) if v != 0 else 0.0
        diff = state_new.e_t - e_old
        marker = " ←" if abs(diff) > 0.15 else ""
        print(f"{name:<28} {v:>6.2f} {a:>6.2f} {state_new.quadrant:>4} "
              f"{state_new.e_t:>10.4f} {e_old:>10.4f} {diff:>8.4f}{marker}")

    # ── Part 2: Full session with eligibility + anti-Hebbian ─────────
    print("\n=== Session simulation with sparse gating ===\n")
    arme = ARME()

    D = 8
    sessions = [
        ("cold start",            +0.0, 0.3, "node_A", 1, 5,  True),
        ("curious discovery",     +0.6, 0.8, "node_B", 1, 6,  True),
        ("same node, less novel", +0.4, 0.5, "node_B", 3, 7,  True),
        ("frustrated, stuck",     -0.7, 0.8, "node_C", 1, 8,  False),
        ("novelty + anxious",     -0.5, 0.9, "node_D", 1, 9,  False),
        ("calm understanding",    +0.5, 0.2, "node_E", 2, 10, True),
        ("eureka moment",         +0.9, 0.95,"node_F", 1, 11, True),
        ("weak signal short msg", +0.1, 0.2, "node_G", 1, 12, True),
    ]

    print(f"{'#':<3} {'scenario':<26} {'Q':<4} {'e_t':<8} {'r_t':<7} "
          f"{'conf':<7} {'gated':<7} {'m_t':<7} {'note'}")
    print("-" * 88)

    for idx, (name, v, a, nid, nact, total, good_ret) in enumerate(sessions):
        q = torch.randn(D)
        r = (q.unsqueeze(0).expand(3,-1) + torch.randn(3,D)*0.15
             if good_ret else torch.randn(3,D)*2.0)
        m_t, state = arme.step(
            query_emb=q, retrieved_embs=r, user_text=name,
            node_id=nid, node_activations=nact, total_activations=total,
            valence_override=v, arousal_override=a,
        )
        conf = abs(state.emotion.e_t) * state.r_t
        note = "novelty gated (neg)" if state.emotion.quadrant in ("Q2","Q3") else ""
        note = "weak signal" if state.gated else note
        print(f"{idx+1:<3} {name:<26} {state.emotion.quadrant:<4} "
              f"{state.emotion.e_t:<8.3f} {state.r_t:<7.3f} "
              f"{conf:<7.3f} {'YES' if state.gated else 'no':<7} "
              f"{state.m_t:<7.3f} {note}")

    # ── Part 3: Hebbian update comparison ────────────────────────────
    print("\n=== Eligible vs non-eligible edge weight divergence ===\n")
    arme2 = ARME()
    N, D2, E = 5, 4, 6
    edge_index = torch.tensor([[0,1,1,2,2,3],[1,0,2,3,4,2]], dtype=torch.long)
    H = F.normalize(torch.tensor([
        [0.05,0.05,0.95,0.05],
        [0.05,0.10,0.90,0.05],
        [0.50,0.50,0.50,0.50],
        [0.95,0.05,0.05,0.05],
        [0.90,0.10,0.05,0.05],
    ]), dim=1)
    ew = torch.ones(E)

    print(f"{'Step':<6} {'m_t':<8} {'eligible':<11} {'elig w':<12} {'non-elig w':<12} {'diff'}")
    print("-" * 58)

    for step in range(8):
        v = +0.7 if step < 4 else -0.6
        a = 0.8  if step < 4 else 0.8
        nid = f"node_{step}"
        q = H[0].clone() + torch.randn(D2)*0.1
        r = H[1:4].clone() + torch.randn(3,D2)*0.1
        m_t, state = arme2.step(
            query_emb=q, retrieved_embs=r, user_text="",
            node_id=nid, node_activations=1, total_activations=max(1,step+1),
            valence_override=v, arousal_override=a,
        )
        ew, hstats = arme2.modulated_hebbian_update(
            edge_index, ew, H, m_t,
        )
        ew_elig  = hstats.get("eligible_mean_w", 0)
        ew_nonel = hstats.get("ineligible_mean_w", 0)
        print(f"{step+1:<6} {m_t:<8.3f} "
              f"{hstats['n_eligible']}/{hstats['n_total']:<7} "
              f"{ew_elig:<12.4f} {ew_nonel:<12.4f} "
              f"{ew_elig - ew_nonel:.4f}")

    # ── Visualization ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10), facecolor="#F8F8F6")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # Panel 1: e_t comparison across quadrants
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#F8F8F6")
    names_plot = [c[0] for c in test_cases]
    e_new = [compute_e_t_quadrant(v,a).e_t for _,v,a in test_cases]
    e_old_list = [math.copysign(math.tanh(0.6*abs(v)+0.4*a), v) if v!=0 else 0.0
                  for _,v,a in test_cases]
    x = range(len(names_plot))
    ax1.bar([i-0.2 for i in x], e_old_list, 0.35,
            color="#888780", alpha=0.7, label="old formula")
    ax1.bar([i+0.2 for i in x], e_new, 0.35,
            color="#378ADD", alpha=0.8, label="quadrant-aware v2")
    ax1.axhline(0, color="#333", linewidth=0.8)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([n.split("(")[0].strip() for n in names_plot],
                         rotation=30, ha="right", fontsize=7)
    ax1.set_ylabel("e_t"); ax1.set_title("e_t: old vs quadrant-aware", fontsize=11)
    ax1.legend(fontsize=8); ax1.spines[["top","right"]].set_visible(False)

    # Panel 2: m_t over session with gating markers
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#F8F8F6")
    m_vals    = [s.m_t for s in arme._history[:8]]
    gated_arr = [s.gated for s in arme._history[:8]]
    bar_colors= ["#888780" if g else ("#1D9E75" if m>1 else "#D85A30")
                 for g,m in zip(gated_arr, m_vals)]
    bars = ax2.bar(range(len(m_vals)), m_vals, color=bar_colors, alpha=0.85)
    ax2.axhline(1.0, color="#333", linewidth=1, linestyle="--", alpha=0.5)
    for bar, g in zip(bars, gated_arr):
        if g:
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                     "gated", ha="center", va="bottom", fontsize=7, color="#888780")
    ax2.set_xticks(range(len(m_vals)))
    ax2.set_xticklabels([f"#{i+1}" for i in range(len(m_vals))], fontsize=8)
    ax2.set_title("m_t per session step\ngray = gated (weak signal)", fontsize=11)
    ax2.set_ylabel("m_t"); ax2.spines[["top","right"]].set_visible(False)

    # Panel 3: valence-arousal space colored by e_t difference
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#F8F8F6")
    import numpy as np
    vv = np.linspace(-1, 1, 40)
    aa = np.linspace(0, 1, 40)
    VV, AA = np.meshgrid(vv, aa)
    E_new = np.vectorize(lambda v,a: compute_e_t_quadrant(float(v),float(a)).e_t)(VV, AA)
    E_old = np.vectorize(lambda v,a: math.copysign(math.tanh(0.6*abs(v)+0.4*a), v)
                         if v!=0 else 0.0)(VV, AA)
    diff_map = E_new - E_old
    im = ax3.contourf(VV, AA, diff_map, levels=20, cmap="RdBu_r")
    plt.colorbar(im, ax=ax3, fraction=0.04)
    ax3.axvline(0, color="white", linewidth=0.8, alpha=0.5)
    ax3.axhline(0.5, color="white", linewidth=0.8, alpha=0.5)
    ax3.set_xlabel("valence"); ax3.set_ylabel("arousal")
    ax3.set_title("e_t diff: new − old\n(red=new higher, blue=new lower)", fontsize=11)
    for (txt, vx, ax_) in [("Q1",0.6,0.75),("Q2",-0.6,0.75),
                             ("Q3",-0.6,0.2),("Q4",0.6,0.2)]:
        ax3.text(vx, ax_, txt, ha="center", color="white",
                 fontweight="bold", fontsize=10)

    # Panel 4: eligible vs non-eligible weight divergence
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#F8F8F6")
    ax4.set_title("Eligible vs non-eligible\nweight divergence", fontsize=11)
    ax4.text(0.5, 0.5, "see console output\n(weight divergence table)",
             ha="center", va="center", transform=ax4.transAxes,
             fontsize=10, color="var(--color-text-secondary)" if False else "#888780")
    ax4.spines[["top","right","left","bottom"]].set_visible(False)
    ax4.set_xticks([]); ax4.set_yticks([])

    fig.suptitle("ARM-E v2: Quadrant-aware emotion + sparse gating + anti-Hebbian",
                 fontsize=13, y=1.01)
    path = "arme_v2_analysis.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="#F8F8F6")
    plt.close()
    print(f"\n→ saved {path}")