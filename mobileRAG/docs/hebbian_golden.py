"""Golden-value generator for the Kotlin Hebbian port (throwaway).

Runs the REAL core/arme_v2.py and core/hippocampal_gnn_pyg.py on fixed
deterministic inputs and prints all outputs. No RNG — fully reproducible.
Run with an interpreter that has torch + torch_geometric, e.g.:
  /opt/anaconda3/envs/RAG/bin/python docs/hebbian_golden.py
"""
import sys
import math

sys.path.insert(0, "/Users/mernahafez/Documents/Personal Projects/Full Hebbian System/Hebbian Memory/core")

import torch
import torch.nn.functional as F

import arme_v2
from arme_v2 import ARME, compute_e_t_quadrant
from hippocampal_gnn_pyg import HippocampalGNN
import type_profiles

T0 = 1_700_000_000.0

def fmt(v):
    return f"{float(v):.8f}"

def fmt_row(t):
    return "[" + ", ".join(fmt(x) for x in t.flatten().tolist()) + "]"

# ── Fixed graph: 4 nodes, 3 edges ─────────────────────────────────────
H = F.normalize(torch.tensor([
    [0.05, 0.05, 0.95, 0.05],
    [0.05, 0.10, 0.90, 0.05],
    [0.50, 0.50, 0.50, 0.50],
    [0.95, 0.05, 0.05, 0.05],
]), dim=1)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
edge_weight = torch.tensor([1.0, 0.8, 1.2])

print("=== inputs ===")
print("H (normalized):")
for row in H:
    print("  " + fmt_row(row))
print("edge_src:", edge_index[0].tolist())
print("edge_dst:", edge_index[1].tolist())
print("edge_weight:", [fmt(w) for w in edge_weight.tolist()])

# ── e_t quadrants ─────────────────────────────────────────────────────
print("\n=== compute_e_t_quadrant ===")
for name, v, a in [("Q1", 0.6, 0.8), ("Q4", 0.6, 0.2), ("Q2", -0.7, 0.8), ("Q3", -0.4, 0.2)]:
    st = compute_e_t_quadrant(v, a)
    print(f"{name} v={v} a={a}: quadrant={st.quadrant} e_t={fmt(st.e_t)}")

# ── ARME steps with injected time ─────────────────────────────────────
arme = ARME()
arme_v2.time.time = lambda: T0   # patch the module's time source

q = torch.tensor([0.10, 0.05, 0.90, 0.05])
retrieved = H[:3].clone()
scores = torch.tensor([0.9, 0.7, 0.5])

print("\n=== arme step 1 (Q1, first sight of node_A) ===")
m_t1, s1 = arme.step(
    query_emb=q, retrieved_embs=retrieved, user_text="ignored (overrides)",
    node_id="node_A", node_activations=1, total_activations=5,
    retrieval_scores=scores,
    valence_override=0.6, arousal_override=0.8, dominance_override=0.7,
)
print(f"r_t={fmt(s1.r_t)} u_t={fmt(s1.u_t)} delta_t={fmt(s1.delta_t)}")
print(f"quadrant={s1.emotion.quadrant} e_t={fmt(s1.emotion.e_t)} dominance={fmt(s1.dominance)}")
print(f"s_t={fmt(s1.s_t)} m_t_raw={fmt(s1.m_t_raw)} m_t={fmt(s1.m_t)} gated={s1.gated}")

arme_v2.time.time = lambda: T0 + 60.0
print("\n=== arme step 2 (Q2, same node 60 s later) ===")
m_t2, s2 = arme.step(
    query_emb=q, retrieved_embs=retrieved, user_text="ignored (overrides)",
    node_id="node_A", node_activations=1, total_activations=5,
    retrieval_scores=scores,
    valence_override=-0.7, arousal_override=0.8, dominance_override=0.3,
)
print(f"r_t={fmt(s2.r_t)} u_t={fmt(s2.u_t)} delta_t={fmt(s2.delta_t)}")
print(f"quadrant={s2.emotion.quadrant} e_t={fmt(s2.emotion.e_t)} dominance={fmt(s2.dominance)}")
print(f"s_t={fmt(s2.s_t)} m_t_raw={fmt(s2.m_t_raw)} m_t={fmt(s2.m_t)} gated={s2.gated}")

# ── propagate ─────────────────────────────────────────────────────────
gnn = HippocampalGNN(feature_dim=4, eta=0.05, lam=0.008)
H_post = gnn.propagate(H.clone(), edge_index, edge_weight)
print("\n=== propagate output (H_post) ===")
for row in H_post:
    print("  " + fmt_row(row))

# ── modulated_hebbian_update (external eligibility, per-edge lam) ─────
lam_oja = torch.tensor([0.008, 0.012, 0.008])
elig_in = torch.tensor([0.3, 0.1, 0.5])
new_w, stats = arme.modulated_hebbian_update(
    edge_index, edge_weight.clone(), H_post, m_t1,
    lam_oja=lam_oja, eligibility=elig_in,
)
print("\n=== modulated_hebbian_update (m_t from step 1) ===")
print(f"m_t={fmt(m_t1)} lam_oja={lam_oja.tolist()} eligibility_in={elig_in.tolist()}")
print("new_edge_weight:", fmt_row(new_w))
print("eligibility_out:", fmt_row(stats["eligibility"]))
print(f"n_eligible={stats['n_eligible']} n_total={stats['n_total']}")
print(f"mean_coact={fmt(stats['mean_coact'])}")
print(f"mean_w={fmt(stats['mean_w'])}")
print(f"eligible_mean_w={fmt(stats['eligible_mean_w'])}")
print(f"ineligible_mean_w={fmt(stats['ineligible_mean_w'])}")

# Second case: zero incoming traces — exercises eligible + anti-Hebbian paths
new_w2, stats2 = arme.modulated_hebbian_update(
    edge_index, edge_weight.clone(), H_post, m_t2,
    lam_oja=lam_oja, eligibility=torch.zeros(3),
)
print("\n=== modulated_hebbian_update (m_t from step 2, zero traces) ===")
print(f"m_t={fmt(m_t2)}")
print("new_edge_weight:", fmt_row(new_w2))
print("eligibility_out:", fmt_row(stats2["eligibility"]))
print(f"n_eligible={stats2['n_eligible']} n_total={stats2['n_total']}")
print(f"mean_coact={fmt(stats2['mean_coact'])}")
print(f"mean_w={fmt(stats2['mean_w'])}")
print(f"eligible_mean_w={fmt(stats2['eligible_mean_w'])}")
print(f"ineligible_mean_w={fmt(stats2['ineligible_mean_w'])}")

# ── type_profiles ─────────────────────────────────────────────────────
print("\n=== type_profiles ===")
print(f"SECONDS_PER_DAY={type_profiles.SECONDS_PER_DAY}")
for ct in ["vocabulary", "grammar", "event", "unknown_type"]:
    p = type_profiles.profile_for(ct)
    print(f"{ct}: tau_stale_s={p['tau_stale_s']} lam_mult={p['lam_mult']}")
print(f"staleness(86400, vocabulary)={fmt(type_profiles.staleness(86400.0, 'vocabulary'))}")
print(f"staleness(0, grammar)={fmt(type_profiles.staleness(0.0, 'grammar'))}")
print(f"lam_for(grammar, 0.008)={fmt(type_profiles.lam_for('grammar', 0.008))}")
print(f"lam_for(task, 0.008)={fmt(type_profiles.lam_for('task', 0.008))}")
