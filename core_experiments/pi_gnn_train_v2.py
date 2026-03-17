#!/usr/bin/env python3
"""
Phase 3: Physics-Informed Spatiotemporal GNN (PI-STGNN)

Architecture:
  Dual-branch GAT: spatial conv (flow->target fan-in) + temporal conv (t->t+1)
  Gated fusion -> MLP classifier

Loss:
  L_total = L_data + alpha * L_flow + beta * L_latency
    - L_data:    Cross-entropy node classification
    - L_flow:    Queue-conservation residual under a finite-capacity bottleneck
    - L_latency: M/M/1-style queueing delay violation

Input:  st_graph.pt (from Phase 2)
Output: pi_gnn_model.pt, phase3_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv
except ImportError:
    print("ERROR: torch_geometric required")
    sys.exit(1)

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None

from internal.submission_common import resolve_capacity_bytes_per_sec


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GRAPH_FILE = os.path.join(BASE_DIR, "st_graph.pt")
DEFAULT_MODEL_FILE = os.path.join(BASE_DIR, "pi_gnn_model.pt")
DEFAULT_RESULTS_FILE = os.path.join(BASE_DIR, "phase3_results.json")

DEFAULT_HIDDEN_DIM = 64
DEFAULT_NUM_HEADS = 4
DEFAULT_DROPOUT = 0.3
DEFAULT_LR = 0.005
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_EPOCHS = 200

DEFAULT_ALPHA_FLOW = 0.05
DEFAULT_BETA_LATENCY = 0.05
DEFAULT_LINK_CAPACITY = 0.0


def resolve_feature_indices(graph: Data) -> dict[str, int]:
    if hasattr(graph, "feature_index") and isinstance(graph.feature_index, dict):
        idx = {str(k): int(v) for k, v in graph.feature_index.items()}
        if "ln(N+1)" in idx and "lnN" not in idx:
            idx["lnN"] = idx["ln(N+1)"]
        if "ln(T+1)" in idx and "lnT" not in idx:
            idx["lnT"] = idx["ln(T+1)"]
        return idx
    return {
        "ln(N+1)": 0,
        "lnN": 0,
        "ln(T+1)": 1,
        "lnT": 1,
        "entropy": 2,
        "D_observed": 3,
        "pkt_rate": 4,
        "avg_pkt_size": 5,
        "port_diversity": 6,
    }


def expected_calibration_error(prob: torch.Tensor, true: torch.Tensor, n_bins: int = 10) -> float:
    if int(true.numel()) == 0:
        return 0.0
    prob = prob.detach().float().cpu()
    true = true.detach().float().cpu()
    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = torch.tensor(0.0)
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        if i == n_bins - 1:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        if int(mask.sum().item()) == 0:
            continue
        acc = true[mask].mean()
        conf = prob[mask].mean()
        ece = ece + (mask.float().mean() * torch.abs(acc - conf))
    return float(ece.item())


def safe_auroc(prob: torch.Tensor, true: torch.Tensor) -> float:
    if average_precision_score is None or roc_auc_score is None:
        return 0.0
    y_true = true.detach().cpu().numpy()
    y_prob = prob.detach().cpu().numpy()
    if len(set(map(int, y_true.tolist()))) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_prob))


def safe_auprc(prob: torch.Tensor, true: torch.Tensor) -> float:
    if average_precision_score is None:
        return 0.0
    y_true = true.detach().cpu().numpy()
    y_prob = prob.detach().cpu().numpy()
    if len(set(map(int, y_true.tolist()))) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_prob))


def compute_arrival_byte_rate(x_raw: torch.Tensor, feat_idx: dict[str, int]) -> torch.Tensor:
    idx_rate = int(feat_idx.get("pkt_rate", 3))
    idx_size = int(feat_idx.get("avg_pkt_size", 4))
    pkt_rate = x_raw[:, idx_rate].clamp(min=0.0)
    avg_pkt_size_bytes = x_raw[:, idx_size].clamp(min=0.0) * 1000.0
    return pkt_rate * avg_pkt_size_bytes


def build_physics_context_features(
    graph: Data,
    capacity_bytes_per_sec: float,
    feat_idx: dict[str, int],
) -> torch.Tensor:
    """
    Build per-node physics context features from per-window aggregates:
      1) queue_pressure = relu(raw_rho - 1)
      2) latency_gap   = relu(D_theory - D_observed_scaled)
    """
    n = graph.num_nodes
    ctx = torch.zeros((n, 2), dtype=torch.float, device=graph.x.device)
    if not hasattr(graph, "window_idx") or capacity_bytes_per_sec <= 0.0:
        return ctx

    valid = graph.window_idx >= 0
    if int(valid.sum().item()) == 0:
        return ctx

    idx_dobs = int(feat_idx.get("D_observed", 2))
    byte_rate = compute_arrival_byte_rate(graph.x, feat_idx)
    d_obs = graph.x[:, idx_dobs].clamp(min=0.0)
    ref_dobs = torch.quantile(d_obs[valid], q=0.75).clamp(min=1e-6)

    uniq_windows = torch.unique(graph.window_idx[valid])
    for w in uniq_windows:
        w_mask = (graph.window_idx == w) & valid
        if int(w_mask.sum().item()) == 0:
            continue

        agg_byte_rate = torch.sum(byte_rate[w_mask])
        raw_rho = agg_byte_rate / (float(capacity_bytes_per_sec) + 1e-6)
        rho = torch.clamp(raw_rho, min=0.0, max=0.995)
        queue_pressure = F.relu(raw_rho - 1.0)

        d_theory = 1.0 / (1.0 - rho + 1e-6)
        d_obs_w = torch.mean(d_obs[w_mask]) / ref_dobs
        latency_gap = F.relu(d_theory - d_obs_w)

        ctx[w_mask, 0] = queue_pressure
        ctx[w_mask, 1] = latency_gap

    return ctx


class SpatioTemporalGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout

        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.s_conv1 = GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False, dropout=dropout)
        self.s_conv2 = GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False, dropout=dropout)
        self.s_norm1 = nn.LayerNorm(hidden_channels)
        self.s_norm2 = nn.LayerNorm(hidden_channels)

        self.t_conv1 = GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False, dropout=dropout)
        self.t_conv2 = GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False, dropout=dropout)
        self.t_norm1 = nn.LayerNorm(hidden_channels)
        self.t_norm2 = nn.LayerNorm(hidden_channels)

        self.gate = nn.Sequential(nn.Linear(hidden_channels * 2, hidden_channels), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_edges = edge_index[:, edge_type == 0]
        temporal_edges = edge_index[:, edge_type == 1]

        n = x.size(0)
        self_loops = torch.arange(n, device=x.device).unsqueeze(0).expand(2, -1)
        spatial_edges = torch.cat([spatial_edges, self_loops], dim=1)
        temporal_edges = torch.cat([temporal_edges, self_loops], dim=1)

        h = F.elu(self.input_proj(x))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h_s = self.s_conv1(h, spatial_edges)
        h_s = self.s_norm1(h_s)
        h_s = F.elu(h_s)
        h_s = F.dropout(h_s, p=self.dropout, training=self.training)
        h_s = self.s_conv2(h_s, spatial_edges)
        h_s = self.s_norm2(h_s)
        h_s = F.elu(h_s)

        h_t = self.t_conv1(h, temporal_edges)
        h_t = self.t_norm1(h_t)
        h_t = F.elu(h_t)
        h_t = F.dropout(h_t, p=self.dropout, training=self.training)
        h_t = self.t_conv2(h_t, temporal_edges)
        h_t = self.t_norm2(h_t)
        h_t = F.elu(h_t)

        gate_val = self.gate(torch.cat([h_s, h_t], dim=-1))
        h_fused = gate_val * h_s + (1.0 - gate_val) * h_t
        logits = self.head(h_fused)
        return logits, h_fused


class PhysicsInformedLoss(nn.Module):
    """
    Differentiable physics loss using an inferred queue state:
      - L_flow penalizes mismatch between queue-conservation buildup and observed delay-derived queue pressure.
      - L_latency penalizes impossible low measured delay under high utilization.
    """

    def __init__(self, alpha: float, beta: float, capacity_bytes_per_sec: float, feat_idx: dict[str, int], delta_t: float):
        super().__init__()
        self.alpha_base = float(alpha)
        self.beta_base = float(beta)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.capacity_bytes_per_sec = float(capacity_bytes_per_sec)
        self.delta_t = max(float(delta_t), 1e-3)
        self.idx_lnN = int(feat_idx.get("ln(N+1)", feat_idx.get("lnN", 0)))
        self.idx_dobs = int(feat_idx.get("D_observed", 2))
        self.idx_rate = int(feat_idx.get("pkt_rate", 3))
        self.idx_size = int(feat_idx.get("avg_pkt_size", 4))

    def set_scale(self, ratio: float) -> None:
        r = max(0.0, min(1.0, float(ratio)))
        self.alpha = self.alpha_base * r
        self.beta = self.beta_base * r

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        x_raw: torch.Tensor,
        window_idx: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if int(mask.sum().item()) == 0:
            zero = torch.tensor(0.0, device=logits.device)
            return zero, zero, zero, zero

        l_data = F.cross_entropy(logits[mask], y[mask], weight=class_weights)
        if self.capacity_bytes_per_sec <= 0.0:
            zero = torch.tensor(0.0, device=logits.device)
            return l_data, l_data, zero, zero

        valid = (window_idx >= 0) & mask
        uniq_windows = torch.unique(window_idx[valid])
        if int(uniq_windows.numel()) == 0:
            zero = torch.tensor(0.0, device=logits.device)
            return l_data, l_data, zero, zero

        attack_prob = F.softmax(logits, dim=1)[:, 1]
        byte_rate = compute_arrival_byte_rate(x_raw, {"pkt_rate": self.idx_rate, "avg_pkt_size": self.idx_size})
        d_obs = x_raw[:, self.idx_dobs].clamp(min=0.0)
        if int(valid.sum().item()) > 0:
            ref_dobs = torch.quantile(d_obs[valid], q=0.75).clamp(min=1e-6)
        else:
            ref_dobs = torch.tensor(1.0, device=logits.device)

        flow_terms = []
        lat_terms = []
        queue_prev = torch.tensor(0.0, device=logits.device)
        service_bytes = torch.tensor(self.capacity_bytes_per_sec * self.delta_t, device=logits.device)

        for w in torch.sort(uniq_windows).values:
            w_mask = (window_idx == w) & valid
            if int(w_mask.sum().item()) < 1:
                continue

            p = attack_prob[w_mask]
            if float(p.sum().item()) < 1e-8:
                continue

            rate_w = byte_rate[w_mask]
            d_obs_w = d_obs[w_mask]

            agg_byte_rate = torch.sum(p * rate_w)
            arrivals = agg_byte_rate * self.delta_t
            queue_next = F.relu(queue_prev + arrivals - service_bytes)

            d_mean = torch.sum(p * d_obs_w) / (torch.sum(p) + 1e-6)
            d_scaled = d_mean / ref_dobs
            observed_queue = F.relu(d_scaled - 1.0) * service_bytes
            flow_terms.append(((queue_next - observed_queue) / (service_bytes + 1e-6)) ** 2)

            raw_rho = agg_byte_rate / (self.capacity_bytes_per_sec + 1e-6)
            rho = torch.clamp(raw_rho, min=0.0, max=0.995)
            d_theory = 1.0 / (1.0 - rho + 1e-6)
            lat_terms.append(F.relu(d_theory - d_scaled))

            queue_prev = queue_next

        l_flow = torch.stack(flow_terms).mean() if flow_terms else torch.tensor(0.0, device=logits.device)
        l_latency = torch.stack(lat_terms).mean() if lat_terms else torch.tensor(0.0, device=logits.device)
        l_total = l_data + self.alpha * l_flow + self.beta * l_latency
        return l_total, l_data, l_flow, l_latency


@torch.no_grad()
def evaluate(model: nn.Module, graph: Data, mask: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    model.eval()
    x_model = graph.x_model if hasattr(graph, "x_model") else graph.x_norm
    logits, _ = model(x_model, graph.edge_index, graph.edge_type)

    if int(mask.sum().item()) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "threshold": float(threshold),
            "auroc": 0.0,
            "auprc": 0.0,
            "ece": 0.0,
        }

    prob = F.softmax(logits[mask], dim=1)[:, 1]
    pred = (prob >= float(threshold)).long()
    true = graph.y[mask]

    tp = int(((pred == 1) & (true == 1)).sum().item())
    fp = int(((pred == 1) & (true == 0)).sum().item())
    fn = int(((pred == 0) & (true == 1)).sum().item())
    tn = int(((pred == 0) & (true == 0)).sum().item())

    total = max(tp + fp + fn + tn, 1)
    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": float(threshold),
        "auroc": safe_auroc(prob, true),
        "auprc": safe_auprc(prob, true),
        "ece": expected_calibration_error(prob, true),
    }


@torch.no_grad()
def find_best_threshold(model: nn.Module, graph: Data, val_mask: torch.Tensor) -> tuple[float, dict[str, float]]:
    if int(val_mask.sum().item()) == 0:
        m = evaluate(model, graph, val_mask, threshold=0.5)
        return 0.5, m

    best_t = 0.5
    best_m = evaluate(model, graph, val_mask, threshold=0.5)
    for t in torch.linspace(0.05, 0.95, 19):
        m = evaluate(model, graph, val_mask, threshold=float(t.item()))
        if m["f1"] > best_m["f1"]:
            best_m = m
            best_t = float(t.item())
    return best_t, best_m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 Physics-Informed ST-GNN Training")
    parser.add_argument("--graph-file", default=DEFAULT_GRAPH_FILE)
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE)
    parser.add_argument("--results-file", default=DEFAULT_RESULTS_FILE)

    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)

    parser.add_argument("--alpha-flow", type=float, default=DEFAULT_ALPHA_FLOW)
    parser.add_argument("--beta-latency", type=float, default=DEFAULT_BETA_LATENCY)
    parser.add_argument("--capacity", type=float, default=DEFAULT_LINK_CAPACITY)
    parser.add_argument("--capacity-mode", choices=["auto", "mbps", "bytes_per_sec"], default="auto")
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=40)

    parser.add_argument("--train-mask-name", default="train_mask")
    parser.add_argument("--val-mask-name", default="val_mask")
    parser.add_argument("--test-mask-name", default="test_mask")
    parser.add_argument("--temporal-test-mask-name", default="temporal_test_mask")
    parser.add_argument("--train-poison-frac", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--physics-context", action="store_true")
    return parser.parse_args()


def get_device(force_cpu: bool) -> torch.device:
    if (not force_cpu) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(args: argparse.Namespace) -> None:
    graph_file = os.path.abspath(os.path.expanduser(args.graph_file))
    model_file = os.path.abspath(os.path.expanduser(args.model_file))
    results_file = os.path.abspath(os.path.expanduser(args.results_file))
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    device = get_device(force_cpu=bool(args.force_cpu))
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    print("=" * 65)
    print("  Phase 3: Physics-Informed ST-GNN Training (PI-STGNN)")
    print("=" * 65)

    print(f"\n[1/5] Loading spatiotemporal graph: {graph_file}")
    graph: Data = torch.load(graph_file, weights_only=False).to(device)
    feat_idx = resolve_feature_indices(graph)
    flow_mask = graph.window_idx >= 0 if hasattr(graph, "window_idx") else (torch.arange(graph.num_nodes, device=device) > 0)

    if not hasattr(graph, args.train_mask_name):
        raise AttributeError(f"Graph missing mask: {args.train_mask_name}")
    if not hasattr(graph, args.val_mask_name):
        raise AttributeError(f"Graph missing mask: {args.val_mask_name}")
    if not hasattr(graph, args.test_mask_name):
        raise AttributeError(f"Graph missing mask: {args.test_mask_name}")

    train_mask = getattr(graph, args.train_mask_name).bool() & flow_mask
    val_mask = getattr(graph, args.val_mask_name).bool() & flow_mask
    test_mask = getattr(graph, args.test_mask_name).bool() & flow_mask
    temporal_test_mask = getattr(graph, args.temporal_test_mask_name).bool() & flow_mask if hasattr(graph, args.temporal_test_mask_name) else test_mask

    y_train_effective = graph.y.clone()
    poisoned_nodes = 0
    if args.train_poison_frac > 0:
        atk_train_idx = ((graph.y == 1) & train_mask).nonzero(as_tuple=False).view(-1)
        if int(atk_train_idx.numel()) > 0:
            n_poison = int(round(float(args.train_poison_frac) * int(atk_train_idx.numel())))
            n_poison = max(0, min(n_poison, int(atk_train_idx.numel())))
            if n_poison > 0:
                perm = torch.randperm(int(atk_train_idx.numel()), device=atk_train_idx.device)
                pick = atk_train_idx[perm[:n_poison]]
                y_train_effective[pick] = 0
                poisoned_nodes = int(n_poison)

    capacity_bytes_per_sec, capacity_source = resolve_capacity_bytes_per_sec(
        float(args.capacity),
        mode=args.capacity_mode,
        graph=graph,
    )
    delta_t = float(getattr(graph, "delta_t", 1.0))

    n_benign = int((graph.y == 0).sum().item())
    n_attack = int((graph.y == 1).sum().item())
    print(f"  Nodes: {graph.num_nodes} ({n_benign} benign + {n_attack} attack + 1 target)")
    print(f"  Edges: {graph.num_edges} (spatial={(graph.edge_type == 0).sum()}, temporal={(graph.edge_type == 1).sum()})")
    print(f"  Features: {graph.x.shape[1]}D")
    print(f"  Feature index: {feat_idx}")
    print(
        f"  Masks: train={int(train_mask.sum().item())} val={int(val_mask.sum().item())} "
        f"test={int(test_mask.sum().item())} temporal_test={int(temporal_test_mask.sum().item())}"
    )
    print(f"  Capacity bytes/s: {capacity_bytes_per_sec:.2f} ({capacity_source})")
    print(f"  delta_t={delta_t:.3f}s delay_metric={getattr(graph, 'delay_metric', 'unknown')}")
    if poisoned_nodes > 0:
        print(f"  Train label poisoning: flipped attack->benign nodes = {poisoned_nodes} ({args.train_poison_frac:.2%} requested)")

    graph.x_model = graph.x_norm
    if bool(args.physics_context):
        ctx = build_physics_context_features(graph, capacity_bytes_per_sec=capacity_bytes_per_sec, feat_idx=feat_idx)
        ctx_flow = ctx[flow_mask]
        c_mean = ctx_flow.mean(dim=0)
        c_std = ctx_flow.std(dim=0).clamp(min=1e-6)
        ctx_norm = (ctx - c_mean) / c_std
        ctx_norm[~flow_mask] = 0.0
        graph.physics_context = ctx_norm
        graph.x_model = torch.cat([graph.x_norm, ctx_norm], dim=1)
        print(f"  Physics context enabled: +{ctx_norm.shape[1]} dims (total {graph.x_model.shape[1]}D)")

    print("\n[2/5] Building ST-GNN model")
    model = SpatioTemporalGNN(
        in_channels=graph.x_model.shape[1],
        hidden_channels=args.hidden_dim,
        out_channels=2,
        num_heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print("  Dual-branch GAT + Gated Fusion + MLP")
    print(f"  Hidden: {args.hidden_dim}, Params: {n_params:,}, Device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = PhysicsInformedLoss(args.alpha_flow, args.beta_latency, capacity_bytes_per_sec, feat_idx, delta_t=delta_t)
    n_pos = int(((graph.y == 1) & train_mask).sum().item())
    n_neg = int(((graph.y == 0) & train_mask).sum().item())
    w_pos = float(n_neg / max(n_pos, 1))
    class_weights = torch.tensor([1.0, max(1.0, w_pos)], dtype=torch.float, device=device)

    print(f"\n[3/5] Training: {args.epochs} epochs")
    print(
        f"  L = L_data + {args.alpha_flow}*L_flow + {args.beta_latency}*L_latency  "
        f"(capacity_bytes_per_sec={capacity_bytes_per_sec:.2f})"
    )
    print(f"  Class weights: benign=1.0 attack={class_weights[1].item():.3f}")
    print()
    print(
        f"  {'Epoch':>5s} | {'L_total':>8s} {'L_data':>8s} {'L_flow':>8s} {'L_lat':>8s} | "
        f"{'Tr Acc':>7s} {'Val Acc':>7s} {'Val F1':>7s} | {'LR':>8s}"
    )
    print(f"  {'-' * 5}-+-{'-' * 8}-{'-' * 8}-{'-' * 8}-{'-' * 8}-+-{'-' * 7}-{'-' * 7}-{'-' * 7}-+-{'-' * 8}")

    history: dict[str, list[Any]] = {
        "L_total": [],
        "L_data": [],
        "L_flow": [],
        "L_latency": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1": [],
        "val_threshold": [],
    }

    best_val_f1 = -1.0
    best_epoch = 0
    best_threshold = 0.5
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        if args.warmup_epochs > 0:
            criterion.set_scale(min(1.0, float(epoch) / float(args.warmup_epochs)))
        else:
            criterion.set_scale(1.0)

        x_model = graph.x_model if hasattr(graph, "x_model") else graph.x_norm
        logits, _ = model(x_model, graph.edge_index, graph.edge_type)
        l_total, l_data, l_flow, l_latency = criterion(
            logits=logits,
            y=y_train_effective,
            mask=train_mask,
            x_raw=graph.x,
            window_idx=graph.window_idx,
            class_weights=class_weights,
        )

        l_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        history["L_total"].append(float(l_total.item()))
        history["L_data"].append(float(l_data.item()))
        history["L_flow"].append(float(l_flow.item()))
        history["L_latency"].append(float(l_latency.item()))

        if epoch % 10 == 0 or epoch <= 5:
            val_th, va = find_best_threshold(model, graph, val_mask)
            tr = evaluate(model, graph, train_mask, threshold=val_th)

            history["train_acc"].append(tr["accuracy"])
            history["val_acc"].append(va["accuracy"])
            history["val_f1"].append(va["f1"])
            history["val_threshold"].append(float(val_th))

            if va["f1"] > best_val_f1:
                best_val_f1 = va["f1"]
                best_epoch = epoch
                best_threshold = float(val_th)
                torch.save(model.state_dict(), model_file)
                bad_epochs = 0
            else:
                bad_epochs += 1

            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  {epoch:5d} | {l_total.item():8.4f} {l_data.item():8.4f} "
                f"{l_flow.item():8.4f} {l_latency.item():8.4f} | "
                f"{tr['accuracy']:7.3f} {va['accuracy']:7.3f} {va['f1']:7.3f} | "
                f"{lr:8.6f}"
            )
            print(f"          threshold={val_th:.2f} alpha={criterion.alpha:.4f} beta={criterion.beta:.4f}")

        if args.patience > 0 and bad_epochs >= args.patience:
            print(f"  Early stop at epoch {epoch} (no val-f1 improvement for {args.patience} eval checkpoints)")
            break

    print(f"\n  Best: epoch {best_epoch}, Val F1 = {best_val_f1:.4f}, threshold={best_threshold:.2f}")

    print("\n[4/5] Final evaluation (best model)")
    model.load_state_dict(torch.load(model_file, weights_only=True, map_location=device))

    final_eval: dict[str, dict[str, float]] = {}
    for name, mask in [
        ("Train", train_mask),
        ("Val", val_mask),
        ("Test (random)", test_mask),
        ("Test (temporal)", temporal_test_mask),
    ]:
        m = evaluate(model, graph, mask, threshold=best_threshold)
        key = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        final_eval[key] = m
        print(
            f"\n  {name:20s} | Acc={m['accuracy']:.4f} P={m['precision']:.4f} "
            f"R={m['recall']:.4f} F1={m['f1']:.4f} AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f}"
        )
        print(f"  {'':20s} | TP={m['tp']:4d}  FP={m['fp']:4d}  FN={m['fn']:4d}  TN={m['tn']:4d}  ECE={m['ece']:.4f}")

    print("\n[5/5] Per-IP physics-informed verdict")
    model.eval()
    with torch.no_grad():
        x_model = graph.x_model if hasattr(graph, "x_model") else graph.x_norm
        logits, _ = model(x_model, graph.edge_index, graph.edge_type)
        probs = F.softmax(logits, dim=1)[:, 1]

    print(f"\n  {'IP':>15s} | {'Truth':>6s} | {'P(atk)':>7s} | {'Verdict':>7s} | Result")
    print(f"  {'-' * 15}-+-{'-' * 6}-+-{'-' * 7}-+-{'-' * 7}-+-------")

    correct = 0
    total_ips = 0
    for ip_i, ip in enumerate(graph.source_ips):
        ip_mask = graph.ip_idx == ip_i
        if int(ip_mask.sum().item()) == 0:
            continue

        mean_p = float(probs[ip_mask].mean().item())
        truth = int(graph.y[ip_mask][0].item())
        truth_s = "ATTACK" if truth == 1 else "BENIGN"
        pred_s = "ATTACK" if mean_p >= best_threshold else "BENIGN"
        ok = truth_s == pred_s

        if ok:
            correct += 1
        total_ips += 1
        print(f"  {ip:>15s} | {truth_s:>6s} | {mean_p:>7.4f} | {pred_s:>7s} | {'OK' if ok else 'MISS'}")

    print(f"\n  Per-IP accuracy: {correct}/{total_ips} ({correct / max(total_ips, 1) * 100:.1f}%)")
    per_ip_acc = correct / max(total_ips, 1)

    results = {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_threshold": float(best_threshold),
        "final_eval": final_eval,
        "per_ip_accuracy": float(per_ip_acc),
        "history": history,
        "config": {
            "hidden": args.hidden_dim,
            "heads": args.heads,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "alpha": args.alpha_flow,
            "beta": args.beta_latency,
            "capacity": args.capacity,
            "capacity_mode": args.capacity_mode,
            "capacity_bytes_per_sec": capacity_bytes_per_sec,
            "capacity_source": capacity_source,
            "warmup_epochs": int(args.warmup_epochs),
            "patience": int(args.patience),
            "train_mask_name": args.train_mask_name,
            "val_mask_name": args.val_mask_name,
            "test_mask_name": args.test_mask_name,
            "temporal_test_mask_name": args.temporal_test_mask_name,
            "class_weight_attack": float(class_weights[1].item()),
            "train_poison_frac": float(args.train_poison_frac),
            "poisoned_nodes": int(poisoned_nodes),
            "seed": int(args.seed),
            "force_cpu": bool(args.force_cpu),
            "physics_context": bool(args.physics_context),
            "graph_file": graph_file,
            "delay_metric": getattr(graph, "delay_metric", "unknown"),
            "delta_t": delta_t,
        },
        "feature_index": feat_idx,
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 65}")
    print("  Physics-Informed Analysis Summary")
    print(f"{'=' * 65}")
    print(f"  Final L_data    = {history['L_data'][-1]:.4f}")
    print(f"  Final L_flow    = {history['L_flow'][-1]:.4f}")
    print(f"  Final L_latency = {history['L_latency'][-1]:.4f}")
    print("\n  Model:", model_file)
    print("  Results:", results_file)
    print(f"{'=' * 65}")


if __name__ == "__main__":
    train(parse_args())
