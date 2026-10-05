"""Golden values for the Kotlin CorticalGnn port (phase 7a).

Runs the REAL core/cortical_gnn.py (PyTorch + PyG GATConv) and records everything the
Kotlin port must reproduce. The reference is the production `consolidate()` itself: its
negative-edge sampler is wrapped only to record what it draws, so the Kotlin test can
replay the same negatives. Dropout is 0 (the one source of randomness that can't be
replayed); everything else is the production path.

Per case it records: inputs, initial parameters, the forward output z0 and every
parameter gradient at step 1 (from a separate forward/backward on a copy of the initial
model with the epoch-0 negatives), per-epoch loss components for all epochs, and the
final z and parameters.

Run with an interpreter that has torch + torch_geometric:
  /opt/anaconda3/envs/RAG/bin/python docs/cortical_golden.py
Writes docs/cortical-golden-values.json next to this file.
"""
import copy
import json
import os
import sys

CORE = "/Users/mernahafez/Documents/Personal Projects/Full Hebbian System/Hebbian Memory/core"
sys.path.insert(0, CORE)

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data

import cortical_gnn
from cortical_gnn import CorticalGNN, consolidate


def flat(t):
    return [float(v) for v in t.detach().reshape(-1).tolist()]


def params_of(model):
    return {name: {"shape": list(p.shape), "values": flat(p)} for name, p in model.state_dict().items()}


def run_case(name, x, edge_index, edge_weight, z_prev, in_dim, hidden, heads, out_dim,
             epochs, lr, seed):
    torch.manual_seed(seed)
    model = CorticalGNN(in_dim=in_dim, hidden_dim=hidden, out_dim=out_dim, heads=heads,
                        dropout=0.0, lambda_causal=0.3, lambda_consist=0.5)
    init_model = copy.deepcopy(model)
    init_params = params_of(model)

    # Record the negatives the production sampler draws, epoch by epoch.
    recorded = []
    original = cortical_gnn._sample_negative_edges

    def recording(ei, n, num_neg):
        out = original(ei, n, num_neg)
        recorded.append([out[0].tolist(), out[1].tolist()])
        return out

    cortical_gnn._sample_negative_edges = recording
    try:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight.unsqueeze(1), num_nodes=x.shape[0])
        z_final, history = consolidate(data, model, z_prev=z_prev, epochs=epochs, lr=lr, verbose=False)
    finally:
        cortical_gnn._sample_negative_edges = original
    assert len(recorded) == epochs

    # Step-1 forward + gradients on the initial model with the epoch-0 negatives.
    init_model.train()
    z0 = init_model.encode(x, edge_index)
    anchor = z_prev if z_prev is not None else x.clone().detach()
    neg0 = torch.tensor(recorded[0], dtype=torch.long)
    loss0, parts0 = init_model.loss(z0, anchor, edge_index, edge_weight, neg_edge_index=neg0)
    loss0.backward()
    grads0 = {n: flat(p.grad) for n, p in init_model.named_parameters()}
    # named_parameters uses the same names as state_dict for this model
    assert parts0["loss_total"] == history[0]["loss_total"], "step-1 replay diverged from consolidate()"

    return {
        "name": name,
        "config": {"in_dim": in_dim, "hidden": hidden, "heads": heads, "out_dim": out_dim,
                   "epochs": epochs, "lr": lr, "dropout": 0.0,
                   "lambda_causal": 0.3, "lambda_consist": 0.5},
        "x": [flat(r) for r in x],
        "edge_src": edge_index[0].tolist(),
        "edge_dst": edge_index[1].tolist(),
        "edge_weight": flat(edge_weight),
        "z_prev": [flat(r) for r in z_prev] if z_prev is not None else None,
        "init_params": init_params,
        "negatives": recorded,
        "z0": [flat(r) for r in z0],
        "loss0": parts0,
        "grads0": grads0,
        "losses": history,
        "z_final": [flat(r) for r in z_final],
        "final_params": params_of(model),
    }


def main():
    cases = []

    # Case A: the demo graph from cortical_gnn.py's __main__ (fixed weights, no hippocampal run).
    xa = F.normalize(torch.tensor([
        [0.05, 0.05, 0.95, 0.05],
        [0.05, 0.10, 0.90, 0.05],
        [0.50, 0.50, 0.50, 0.50],
        [0.95, 0.05, 0.05, 0.05],
        [0.90, 0.10, 0.05, 0.05],
    ]), dim=1)
    eia = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                        [1, 0, 2, 3, 4, 2, 4, 3]], dtype=torch.long)
    wa = torch.tensor([1.0, 1.2, 0.8, 1.5, 0.6, 2.0, 3.1, 0.9])
    cases.append(run_case("demo5", xa, eia, wa, None,
                          in_dim=4, hidden=8, heads=2, out_dim=4, epochs=30, lr=1e-3, seed=42))

    # Case B: seeded random graph, 4 heads, explicit z_prev (second-consolidation path).
    g = torch.Generator().manual_seed(7)
    n, d = 12, 16
    xb = F.normalize(torch.randn(n, d, generator=g), dim=1)
    pairs = set()
    while len(pairs) < 18:
        i, j = torch.randint(0, n, (2,), generator=g).tolist()
        if i != j:
            pairs.add((min(i, j), max(i, j)))
    src, dst = [], []
    for i, j in sorted(pairs):
        src += [i, j]
        dst += [j, i]
    eib = torch.tensor([src, dst], dtype=torch.long)
    wb = 0.5 + 4.5 * torch.rand(eib.shape[1], generator=g)
    zb_prev = xb + 0.05 * torch.randn(n, d, generator=g)
    cases.append(run_case("random12", xb, eib, wb, zb_prev,
                          in_dim=d, hidden=8, heads=4, out_dim=d, epochs=30, lr=5e-4, seed=123))

    out = {
        "generator": "docs/cortical_golden.py",
        "source": "Hebbian Memory/core/cortical_gnn.py (CorticalGNN, consolidate)",
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "note": "float32 reference; the Kotlin port computes in Double — compare with tolerances",
        "cases": cases,
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cortical-golden-values.json")
    with open(path, "w") as f:
        json.dump(out, f)
    for c in cases:
        print(f"{c['name']}: loss0={c['loss0']['loss_total']:.6f} "
              f"final={c['losses'][-1]['loss_total']:.6f} params={sum(len(p['values']) for p in c['init_params'].values())}")
    print("wrote", path)


if __name__ == "__main__":
    main()
