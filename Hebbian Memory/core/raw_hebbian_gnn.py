"""
Minimal Hebbian GNN — v5
Fixes: stronger semantic seeding, disconnect bridge from both clusters,
       visualization as a standalone function you can call any time.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

torch.manual_seed(42)

# ── Hyperparameters ──────────────────────────────────────────────────
N        = 5
D        = 4
ETA      = 0.05      # Hebbian learning rate
LAM      = 0.008     # Oja decay rate
W_FLOOR  = 0.5       # minimum edge weight
W_CEIL   = 4.0       # maximum edge weight (hard clamp)

# ── Graph topology ───────────────────────────────────────────────────
# Key fix: bridge node 2 connects to only ONE cluster (physics side).
# This lets the history cluster (0,1) and physics cluster (3,4)
# develop independently — topology no longer flattens the signal.
#
#   History: 0 ↔ 1
#   Bridge:  1 ↔ 2
#   Physics: 2 ↔ 3 ↔ 4 (node 4 also connects back to 3)
#
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 2)]

NODE_LABELS = {
    0: "N0\nhistory",
    1: "N1\nhistory",
    2: "N2\nbridge",
    3: "N3\nphysics",
    4: "N4\nphysics",
}

# Colors: blue = history, gray = bridge, orange = physics
NODE_COLORS = ["#378ADD", "#85B7EB", "#B4B2A9", "#D85A30", "#F0997B"]
DIM_COLORS  = ["#D85A30", "#1D9E75", "#378ADD", "#888780"]
DIM_NAMES   = ["d0-physics", "d1-neutral", "d2-history", "d3-neutral"]


# ── Init ─────────────────────────────────────────────────────────────

def init_H() -> torch.Tensor:
    """
    Stronger semantic seeding than v4.
    Physics nodes (3,4) load heavily on dim 0.
    History nodes (0,1) load heavily on dim 2.
    Bridge node (2) is uniform — equidistant from both topics.
    """
    H = torch.tensor([
        [0.05, 0.05, 0.95, 0.05],   # N0 — history
        [0.05, 0.10, 0.90, 0.05],   # N1 — history (slightly mixed)
        [0.50, 0.50, 0.50, 0.50],   # N2 — bridge  (uniform)
        [0.95, 0.05, 0.05, 0.05],   # N3 — physics
        [0.90, 0.10, 0.05, 0.05],   # N4 — physics (slightly mixed)
    ], dtype=torch.float)
    return F.normalize(H, dim=1)


def init_A() -> torch.Tensor:
    A = torch.zeros(N, N)
    for i, j in EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


# ── GNN layer ────────────────────────────────────────────────────────

def gnn_step(H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    One message-passing step.
    h_i' = tanh( mean_{j in N(i)} w_ij * h_j )
    No post-normalize — keeps feature magnitudes meaningful for Hebbian.
    """
    H2 = []
    for i in range(N):
        deg = A[i].sum().clamp(min=1)
        agg = (A[i].unsqueeze(1) * H).sum(dim=0) / deg
        H2.append(torch.tanh(agg))
    return torch.stack(H2)


# ── Hebbian + Oja update ─────────────────────────────────────────────

def hebbian_update(A: torch.Tensor, H_pre: torch.Tensor) -> torch.Tensor:
    """
    Oja's rule on PRE-step features.
    Δw_ij = η·(h_i·h_j) - λ·w_ij·‖h_i‖²
    Weights clamped to [W_FLOOR, W_CEIL].
    """
    A2 = A.clone()
    for i, j in EDGES:
        coact = torch.dot(H_pre[i], H_pre[j])
        hebb  = ETA * coact
        decay = LAM * A2[i, j] * torch.dot(H_pre[i], H_pre[i])
        dw    = hebb - decay
        new_w = (A2[i, j] + dw).clamp(W_FLOOR, W_CEIL)
        A2[i, j] = new_w
        A2[j, i] = new_w
    return A2


# ── Visualization ────────────────────────────────────────────────────

def visualize(H: torch.Tensor, A: torch.Tensor, step: int, ax_graph=None, ax_emb=None):
    """
    Draw the Hebbian GNN state.

    Parameters
    ----------
    H       : [N, D] node embeddings
    A       : [N, N] Hebbian weight matrix
    step    : current training step (for title)
    ax_graph: matplotlib Axes for the graph panel (created if None)
    ax_emb  : matplotlib Axes for the embedding panel (created if None)

    Returns
    -------
    fig : the matplotlib Figure (only if axes were not provided)
    """
    standalone = ax_graph is None
    if standalone:
        fig, (ax_graph, ax_emb) = plt.subplots(
            1, 2, figsize=(13, 5),
            gridspec_kw={"width_ratios": [1.3, 1]}
        )
        fig.patch.set_facecolor("#F8F8F6")
    else:
        fig = ax_graph.get_figure()

    # ── Left panel: graph ────────────────────────────────────────────
    ax_graph.set_facecolor("#F8F8F6")
    ax_graph.set_title(f"Hebbian GNN — step {step}", fontsize=13, pad=10)

    G = nx.Graph()
    G.add_nodes_from(range(N))
    edge_weights = {}
    for i, j in EDGES:
        w = A[i, j].item()
        G.add_edge(i, j, weight=w)
        edge_weights[(i, j)] = w

    pos = nx.spring_layout(G, seed=7, weight=None)   # fixed layout

    all_w  = list(edge_weights.values())
    w_min, w_max = min(all_w), max(all_w)
    w_range = max(w_max - w_min, 1e-6)

    edge_list  = list(edge_weights.keys())
    edge_widths = [1.5 + 6.0 * (edge_weights[e] - w_min) / w_range for e in edge_list]
    edge_alphas = [0.30 + 0.65 * (edge_weights[e] - w_min) / w_range for e in edge_list]
    edge_colors = [
        plt.cm.Blues(0.4 + 0.55 * (edge_weights[e] - w_min) / w_range)
        for e in edge_list
    ]

    nx.draw_networkx_edges(
        G, pos, edgelist=edge_list,
        width=edge_widths, alpha=edge_alphas,
        edge_color=edge_colors, ax=ax_graph
    )

    nx.draw_networkx_nodes(
        G, pos,
        node_color=NODE_COLORS, node_size=900,
        linewidths=1.5, edgecolors="#555", ax=ax_graph
    )

    nx.draw_networkx_labels(
        G, pos,
        labels={i: NODE_LABELS[i] for i in range(N)},
        font_size=7.5, font_color="white", font_weight="bold", ax=ax_graph
    )

    # edge weight labels
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels={e: f"{w:.3f}" for e, w in edge_weights.items()},
        font_size=7, font_color="#444", ax=ax_graph
    )

    # legend
    patches = [
        mpatches.Patch(color="#378ADD", label="History (dim 2)"),
        mpatches.Patch(color="#B4B2A9", label="Bridge (uniform)"),
        mpatches.Patch(color="#D85A30", label="Physics (dim 0)"),
    ]
    ax_graph.legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.7)
    ax_graph.axis("off")

    # ── Right panel: embeddings as heatmap ──────────────────────────
    ax_emb.set_facecolor("#F8F8F6")
    ax_emb.set_title("Node embeddings (4 dims)", fontsize=13, pad=10)

    H_np = H.detach().numpy()   # [N, D]
    im = ax_emb.imshow(H_np, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

    ax_emb.set_xticks(range(D))
    ax_emb.set_xticklabels(DIM_NAMES, fontsize=8, rotation=15, ha="right")
    ax_emb.set_yticks(range(N))
    ax_emb.set_yticklabels(
        [NODE_LABELS[i].replace("\n", " ") for i in range(N)], fontsize=8
    )

    # cell values
    for r in range(N):
        for c in range(D):
            val = H_np[r, c]
            color = "white" if abs(val) > 0.5 else "#333"
            ax_emb.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

    plt.colorbar(im, ax=ax_emb, fraction=0.03, pad=0.04, label="activation")
    ax_emb.set_xlabel("Feature dimension", fontsize=9)

    plt.tight_layout()
    return fig if standalone else None


# ── Training loop ─────────────────────────────────────────────────────

def run(steps: int = 10, visualize_every: int = 5) -> tuple:
    """
    Run the Hebbian GNN for `steps` steps.
    Saves a PNG snapshot every `visualize_every` steps + final state.

    Returns H, A after training.
    """
    H = init_H()
    A = init_A()

    print(f"{'Step':<6} {'w(N3↔N4)':<12} {'w(N0↔N1)':<12} "
          f"{'w(N1↔N2)':<12} {'coact(3,4)':<12} {'coact(0,1)'}")
    print("-" * 68)

    for s in range(1, steps + 1):
        H_pre = H.clone()
        H     = gnn_step(H, A)
        A     = hebbian_update(A, H_pre)

        c34 = torch.dot(H_pre[3], H_pre[4]).item()
        c01 = torch.dot(H_pre[0], H_pre[1]).item()
        print(f"{s:<6} {A[3,4].item():<12.4f} {A[0,1].item():<12.4f} "
              f"{A[1,2].item():<12.4f} {c34:<12.4f} {c01:.4f}")

        if s % visualize_every == 0 or s == steps:
            fig = visualize(H, A, step=s)
            path = f"./outputs/hebbian_step_{s:03d}.png"
            fig.savefig(path, dpi=130, bbox_inches="tight",
                        facecolor="#F8F8F6")
            plt.close(fig)
            print(f"  → saved {path}")

    return H, A


if __name__ == "__main__":
    H_final, A_final = run(steps=20, visualize_every=5)

    print("\n=== Final edge weights ===")
    for i, j in EDGES:
        print(f"  w({i}↔{j}) = {A_final[i,j].item():.4f}")

    print("\n=== Final cosine similarities ===")
    for i, j in EDGES:
        sim = F.cosine_similarity(
            H_final[i].unsqueeze(0), H_final[j].unsqueeze(0)
        ).item()
        print(f"  sim(N{i}, N{j}) = {sim:+.3f}")