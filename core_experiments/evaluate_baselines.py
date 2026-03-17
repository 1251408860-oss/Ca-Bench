#!/usr/bin/env python3
"""
Evaluate node-classification baselines on a graph split:
  - RandomForest
  - GCN
  - GraphSAGE
  - GATv2
  - PI-GNN (pretrained)

For fairness, all probabilistic models use the validation split to tune the
decision threshold instead of forcing a fixed 0.5 cutoff.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier

try:
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"scikit-learn metrics required: {exc}")

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"torch_geometric is required: {exc}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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


def expected_calibration_error(prob: np.ndarray, true: np.ndarray, n_bins: int = 10) -> float:
    if true.size == 0:
        return 0.0
    prob = prob.astype(np.float64, copy=False)
    true = true.astype(np.float64, copy=False)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        if i == n_bins - 1:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(true[mask]))
        conf = float(np.mean(prob[mask]))
        ece += float(np.mean(mask.astype(np.float64)) * abs(acc - conf))
    return float(ece)


def thresholded_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= float(threshold)).astype(np.int64)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    fpr = fp / max(fp + tn, 1)

    if len(np.unique(y_true)) >= 2:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
        fpr_pts, tpr_pts, _ = roc_curve(y_true, y_prob)
        roc_pts = {"fpr": fpr_pts.tolist(), "tpr": tpr_pts.tolist()}
    else:
        auroc = 0.5
        auprc = float(np.mean(y_true)) if y_true.size else 0.0
        roc_pts = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]}

    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "roc_auc": float(auroc),
        "auprc": float(auprc),
        "ece": expected_calibration_error(y_prob, y_true),
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
    return {"metrics": metrics, "roc_points": roc_pts}


def pick_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.5
    best_t = 0.5
    best_f1 = thresholded_metrics(y_true, y_prob, threshold=best_t)["metrics"]["f1"]
    for t in np.linspace(0.05, 0.95, 19):
        f1 = thresholded_metrics(y_true, y_prob, threshold=float(t))["metrics"]["f1"]
        if f1 > best_f1 + 1e-12:
            best_f1 = f1
            best_t = float(t)
    return float(best_t)


def get_eval_mask(graph: Data) -> tuple[torch.Tensor, str]:
    if hasattr(graph, "temporal_test_mask") and int(graph.temporal_test_mask.sum().item()) > 0:
        return graph.temporal_test_mask.bool(), "temporal_test"
    return graph.test_mask.bool(), "test"


def get_val_mask(graph: Data) -> torch.Tensor:
    if hasattr(graph, "val_mask"):
        return graph.val_mask.bool()
    return torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.x.device)


def build_prediction_block(graph: Data, mask: torch.Tensor, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    idx = mask.nonzero(as_tuple=False).view(-1).detach().cpu().numpy().astype(np.int64)
    y_true = graph.y[mask].detach().cpu().numpy().astype(np.int64)
    y_pred = (y_prob >= float(threshold)).astype(np.int64)
    block: dict[str, Any] = {
        "node_idx": idx.tolist(),
        "y_true": y_true.tolist(),
        "y_prob": y_prob.astype(np.float64).tolist(),
        "y_pred": y_pred.tolist(),
    }

    if hasattr(graph, "window_idx"):
        block["window_idx"] = graph.window_idx[mask].detach().cpu().numpy().astype(np.int64).tolist()
    if hasattr(graph, "ip_idx"):
        block["ip_idx"] = graph.ip_idx[mask].detach().cpu().numpy().astype(np.int64).tolist()
    if hasattr(graph, "node_scenario_idx"):
        scenario_idx = graph.node_scenario_idx[mask].detach().cpu().numpy().astype(np.int64).tolist()
        block["scenario_idx"] = scenario_idx
        scenario_names = list(getattr(graph, "scenario_names", []))
        if scenario_names:
            block["scenario_name"] = [
                scenario_names[i] if 0 <= int(i) < len(scenario_names) else "unknown" for i in scenario_idx
            ]
    return block


def class_weights_from_mask(graph: Data, mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    n_pos = int(((graph.y == 1) & mask).sum().item())
    n_neg = int(((graph.y == 0) & mask).sum().item())
    w_pos = float(n_neg / max(n_pos, 1))
    return torch.tensor([1.0, max(1.0, w_pos)], dtype=torch.float32, device=device)


class GCNBaseline(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.dropout = float(dropout)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_channels // 2, 2),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        return self.head(h)


class GraphSAGEBaseline(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.dropout = float(dropout)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_channels // 2, 2),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        return self.head(h)


class GATv2Baseline(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout = float(dropout)
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=False, dropout=self.dropout)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=self.dropout)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_channels // 2, 2),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        h = F.elu(h)
        return self.head(h)


class PIGNNModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout = float(dropout)
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.s_conv1 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=self.dropout)
        self.s_conv2 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=self.dropout)
        self.s_norm1 = nn.LayerNorm(hidden_channels)
        self.s_norm2 = nn.LayerNorm(hidden_channels)
        self.t_conv1 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=self.dropout)
        self.t_conv2 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=self.dropout)
        self.t_norm1 = nn.LayerNorm(hidden_channels)
        self.t_norm2 = nn.LayerNorm(hidden_channels)
        self.gate = nn.Sequential(nn.Linear(hidden_channels * 2, hidden_channels), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_channels // 2, 2),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor | None = None) -> torch.Tensor:
        if edge_type is None:
            raise ValueError("PI-GNN requires edge_type")

        spatial_edges = edge_index[:, edge_type == 0]
        temporal_edges = edge_index[:, edge_type == 1]
        n = x.size(0)
        self_loops = torch.arange(n, device=x.device).unsqueeze(0).expand(2, -1)
        spatial_edges = torch.cat([spatial_edges, self_loops], dim=1)
        temporal_edges = torch.cat([temporal_edges, self_loops], dim=1)

        h = F.elu(self.input_proj(x))
        h = F.dropout(h, p=self.dropout, training=self.training)

        hs = self.s_conv1(h, spatial_edges)
        hs = self.s_norm1(hs)
        hs = F.elu(hs)
        hs = F.dropout(hs, p=self.dropout, training=self.training)
        hs = self.s_conv2(hs, spatial_edges)
        hs = self.s_norm2(hs)
        hs = F.elu(hs)

        ht = self.t_conv1(h, temporal_edges)
        ht = self.t_norm1(ht)
        ht = F.elu(ht)
        ht = F.dropout(ht, p=self.dropout, training=self.training)
        ht = self.t_conv2(ht, temporal_edges)
        ht = self.t_norm2(ht)
        ht = F.elu(ht)

        gate = self.gate(torch.cat([hs, ht], dim=-1))
        fused = gate * hs + (1.0 - gate) * ht
        return self.head(fused)


@torch.no_grad()
def model_probabilities(
    model: nn.Module,
    graph: Data,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor | None = None,
) -> torch.Tensor:
    model.eval()
    logits = model(graph.x_norm, edge_index, edge_type)
    return F.softmax(logits, dim=1)[:, 1]


def train_graph_model(
    model: nn.Module,
    graph: Data,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor | None = None,
) -> tuple[nn.Module, float, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_mask = graph.train_mask.bool()
    val_mask = get_val_mask(graph)
    class_weights = class_weights_from_mask(graph, train_mask, device=device)

    best_state: dict[str, torch.Tensor] | None = None
    best_threshold = 0.5
    best_val_f1 = -1.0
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(graph.x_norm, edge_index, edge_type)
        loss = F.cross_entropy(logits[train_mask], graph.y[train_mask], weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch <= 5 or epoch % 10 == 0:
            val_prob = model_probabilities(model, graph, edge_index=edge_index, edge_type=edge_type)[val_mask].detach().cpu().numpy()
            y_val = graph.y[val_mask].detach().cpu().numpy()
            val_threshold = pick_best_threshold(y_val, val_prob)
            val_f1 = thresholded_metrics(y_val, val_prob, val_threshold)["metrics"]["f1"]

            if val_f1 > best_val_f1 + 1e-12:
                best_val_f1 = float(val_f1)
                best_threshold = float(val_threshold)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

            if patience > 0 and bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_threshold), float(best_val_f1)


def load_threshold_from_results(path: str) -> float | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        value = data.get("best_threshold")
        return float(value) if value is not None else None
    except Exception:
        return None


def export_pi_predictions(
    graph_file: str,
    model_file: str,
    results_file: str,
    output_file: str,
    *,
    hidden_dim: int = 64,
    heads: int = 4,
    dropout: float = 0.3,
    force_cpu: bool = False,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    graph: Data = torch.load(graph_file, weights_only=False, map_location=device).to(device)
    eval_mask, eval_split = get_eval_mask(graph)
    threshold = float(load_threshold_from_results(results_file) or 0.5)

    model = PIGNNModel(
        in_channels=graph.x_norm.shape[1],
        hidden_channels=hidden_dim,
        heads=heads,
        dropout=dropout,
    ).to(device)
    state = torch.load(model_file, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        logits = model(graph.x_norm, graph.edge_index, graph.edge_type)
        prob = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    eval_np = eval_mask.detach().cpu().numpy().astype(bool)
    y_true = graph.y[eval_mask].detach().cpu().numpy().astype(np.int64)
    y_prob = prob[eval_np]
    block = thresholded_metrics(y_true, y_prob, threshold=threshold)

    payload: dict[str, Any] = {
        "graph_file": os.path.abspath(graph_file),
        "model_file": os.path.abspath(model_file),
        "results_file": os.path.abspath(results_file),
        "eval_split": eval_split,
        "metrics": block["metrics"],
        "roc_points": block["roc_points"],
        "predictions": build_prediction_block(graph, eval_mask, y_prob, threshold),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    graph: Data = torch.load(args.graph_file, weights_only=False).to(device)
    eval_mask, eval_split = get_eval_mask(graph)
    val_mask = get_val_mask(graph)
    edge_index = graph.edge_index_undirected if hasattr(graph, "edge_index_undirected") else graph.edge_index
    edge_type = graph.edge_type_undirected if hasattr(graph, "edge_type_undirected") else getattr(graph, "edge_type", None)

    y_eval = graph.y[eval_mask].detach().cpu().numpy()
    y_val = graph.y[val_mask].detach().cpu().numpy()

    results: dict[str, Any] = {
        "config": vars(args),
        "device": str(device),
        "eval_split": eval_split,
        "feature_index": resolve_feature_indices(graph),
        "thresholds": {},
        "metrics": {},
        "roc_points": {},
    }
    if args.save_predictions:
        results["predictions"] = {}

    x_train = graph.x_norm[graph.train_mask].detach().cpu().numpy()
    y_train = graph.y[graph.train_mask].detach().cpu().numpy()
    x_val = graph.x_norm[val_mask].detach().cpu().numpy()
    x_eval = graph.x_norm[eval_mask].detach().cpu().numpy()

    rf = RandomForestClassifier(
        n_estimators=args.rf_trees,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(x_train, y_train)
    rf_val_prob = rf.predict_proba(x_val)[:, 1] if x_val.shape[0] > 0 else np.zeros(0, dtype=np.float64)
    rf_threshold = pick_best_threshold(y_val, rf_val_prob) if x_val.shape[0] > 0 else 0.5
    rf_prob = rf.predict_proba(x_eval)[:, 1]
    rf_block = thresholded_metrics(y_eval, rf_prob, threshold=rf_threshold)
    results["thresholds"]["random_forest"] = float(rf_threshold)
    results["metrics"]["random_forest"] = rf_block["metrics"]
    results["roc_points"]["random_forest"] = rf_block["roc_points"]
    if args.save_predictions:
        results["predictions"]["random_forest"] = build_prediction_block(graph, eval_mask, rf_prob, rf_threshold)

    gcn = GCNBaseline(graph.x_norm.shape[1], hidden_channels=args.gcn_hidden, dropout=args.gcn_dropout).to(device)
    gcn, gcn_threshold, _ = train_graph_model(
        gcn,
        graph,
        device=device,
        epochs=args.gcn_epochs,
        lr=args.gcn_lr,
        weight_decay=args.gcn_weight_decay,
        patience=args.graph_patience,
        edge_index=edge_index,
    )
    gcn_prob = model_probabilities(gcn, graph, edge_index=edge_index).detach().cpu().numpy()[eval_mask.detach().cpu().numpy()]
    gcn_block = thresholded_metrics(y_eval, gcn_prob, threshold=gcn_threshold)
    results["thresholds"]["gcn"] = float(gcn_threshold)
    results["metrics"]["gcn"] = gcn_block["metrics"]
    results["roc_points"]["gcn"] = gcn_block["roc_points"]
    if args.save_predictions:
        results["predictions"]["gcn"] = build_prediction_block(graph, eval_mask, gcn_prob, gcn_threshold)

    graphsage = GraphSAGEBaseline(graph.x_norm.shape[1], hidden_channels=args.graphsage_hidden, dropout=args.graphsage_dropout).to(device)
    graphsage, sage_threshold, _ = train_graph_model(
        graphsage,
        graph,
        device=device,
        epochs=args.graphsage_epochs,
        lr=args.graphsage_lr,
        weight_decay=args.graphsage_weight_decay,
        patience=args.graph_patience,
        edge_index=edge_index,
    )
    sage_prob = model_probabilities(graphsage, graph, edge_index=edge_index).detach().cpu().numpy()[eval_mask.detach().cpu().numpy()]
    sage_block = thresholded_metrics(y_eval, sage_prob, threshold=sage_threshold)
    results["thresholds"]["graphsage"] = float(sage_threshold)
    results["metrics"]["graphsage"] = sage_block["metrics"]
    results["roc_points"]["graphsage"] = sage_block["roc_points"]
    if args.save_predictions:
        results["predictions"]["graphsage"] = build_prediction_block(graph, eval_mask, sage_prob, sage_threshold)

    gatv2 = GATv2Baseline(
        graph.x_norm.shape[1],
        hidden_channels=args.gatv2_hidden,
        heads=args.gatv2_heads,
        dropout=args.gatv2_dropout,
    ).to(device)
    gatv2, gat_threshold, _ = train_graph_model(
        gatv2,
        graph,
        device=device,
        epochs=args.gatv2_epochs,
        lr=args.gatv2_lr,
        weight_decay=args.gatv2_weight_decay,
        patience=args.graph_patience,
        edge_index=edge_index,
    )
    gat_prob = model_probabilities(gatv2, graph, edge_index=edge_index).detach().cpu().numpy()[eval_mask.detach().cpu().numpy()]
    gat_block = thresholded_metrics(y_eval, gat_prob, threshold=gat_threshold)
    results["thresholds"]["gatv2"] = float(gat_threshold)
    results["metrics"]["gatv2"] = gat_block["metrics"]
    results["roc_points"]["gatv2"] = gat_block["roc_points"]
    if args.save_predictions:
        results["predictions"]["gatv2"] = build_prediction_block(graph, eval_mask, gat_prob, gat_threshold)

    if not os.path.exists(args.pi_model_file):
        raise FileNotFoundError(f"PI-GNN model not found: {args.pi_model_file}")

    pi_model = PIGNNModel(
        in_channels=graph.x_norm.shape[1],
        hidden_channels=args.pi_hidden,
        heads=args.pi_heads,
        dropout=args.pi_dropout,
    ).to(device)
    state = torch.load(args.pi_model_file, map_location=device, weights_only=True)
    pi_model.load_state_dict(state, strict=False)
    pi_prob_full = model_probabilities(pi_model, graph, edge_index=graph.edge_index, edge_type=graph.edge_type).detach().cpu().numpy()

    pi_threshold = load_threshold_from_results(args.pi_results_file)
    if pi_threshold is None:
        pi_threshold = pick_best_threshold(y_val, pi_prob_full[val_mask.detach().cpu().numpy()]) if int(val_mask.sum().item()) > 0 else 0.5
    pi_prob = pi_prob_full[eval_mask.detach().cpu().numpy()]
    pi_block = thresholded_metrics(y_eval, pi_prob, threshold=pi_threshold)
    results["thresholds"]["pi_gnn"] = float(pi_threshold)
    results["metrics"]["pi_gnn"] = pi_block["metrics"]
    results["roc_points"]["pi_gnn"] = pi_block["roc_points"]
    if args.save_predictions:
        results["predictions"]["pi_gnn"] = build_prediction_block(graph, eval_mask, pi_prob, pi_threshold)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 72)
    print("Baseline Evaluation Complete")
    print("=" * 72)
    print(f"Eval split: {eval_split}")
    for name, m in results["metrics"].items():
        print(
            f"{name:14s} thr={m['threshold']:.2f} recall={m['recall']:.4f} "
            f"fpr={m['fpr']:.4f} auroc={m['roc_auc']:.4f} f1={m['f1']:.4f}"
        )
    print(f"Saved: {args.output_file}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate RF/GCN/GraphSAGE/GATv2/PI-GNN baselines")
    p.add_argument("--graph-file", default=os.path.join(BASE_DIR, "st_graph.pt"))
    p.add_argument("--pi-model-file", default=os.path.join(BASE_DIR, "pi_gnn_model.pt"))
    p.add_argument("--pi-results-file", default="")
    p.add_argument("--output-file", default=os.path.join(BASE_DIR, "baseline_eval_results.json"))

    p.add_argument("--rf-trees", type=int, default=400)

    p.add_argument("--gcn-epochs", type=int, default=120)
    p.add_argument("--gcn-lr", type=float, default=0.01)
    p.add_argument("--gcn-weight-decay", type=float, default=5e-4)
    p.add_argument("--gcn-hidden", type=int, default=64)
    p.add_argument("--gcn-dropout", type=float, default=0.3)

    p.add_argument("--graphsage-epochs", type=int, default=120)
    p.add_argument("--graphsage-lr", type=float, default=0.01)
    p.add_argument("--graphsage-weight-decay", type=float, default=5e-4)
    p.add_argument("--graphsage-hidden", type=int, default=64)
    p.add_argument("--graphsage-dropout", type=float, default=0.3)

    p.add_argument("--gatv2-epochs", type=int, default=120)
    p.add_argument("--gatv2-lr", type=float, default=0.005)
    p.add_argument("--gatv2-weight-decay", type=float, default=5e-4)
    p.add_argument("--gatv2-hidden", type=int, default=64)
    p.add_argument("--gatv2-heads", type=int, default=4)
    p.add_argument("--gatv2-dropout", type=float, default=0.3)

    p.add_argument("--graph-patience", type=int, default=12)

    p.add_argument("--pi-hidden", type=int, default=64)
    p.add_argument("--pi-heads", type=int, default=4)
    p.add_argument("--pi-dropout", type=float, default=0.3)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--force-cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run(args)
