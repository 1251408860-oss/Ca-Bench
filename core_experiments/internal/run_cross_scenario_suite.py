#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_baselines import export_pi_predictions
from prepare_hard_protocol_graph import apply_overlap_hardening

MODEL_NAMES = ["data_only", "random_forest", "gcn", "graphsage", "gatv2"]
DATA_ROOT = REPO_ROOT.parent / "mininet_testbed" / "real_collection"
RUN_ROOT = REPO_ROOT.parent / "paper_artifacts" / "runs"


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def scenario_name_from_path(path: str) -> str:
    p = Path(path)
    return p.stem or p.name


def capacity_for_graph(graph: Any) -> tuple[float, float]:
    cap = float(getattr(graph, "capacity_bytes_per_sec", 0.0) or 0.0)
    bw = float(getattr(graph, "manifest_core_bw_mbps", 0.0) or 0.0)
    return cap, bw


def merge_graphs(train_graphs: list[str], val_graphs: list[str], test_graphs: list[str]) -> Data:
    split_specs = [("train", train_graphs), ("val", val_graphs), ("test", test_graphs)]
    x_parts = []
    y_parts = []
    edge_parts = []
    edge_type_parts = []
    window_idx_parts = []
    local_window_idx_parts = []
    ip_idx_parts = []
    scenario_idx_parts = []
    node_capacity_parts = []
    node_bw_parts = []
    train_mask_parts = []
    val_mask_parts = []
    test_mask_parts = []

    source_ips: list[str] = []
    scenario_names: list[str] = []
    graph_files: list[str] = []
    node_ip: list[str] = []
    node_scenario_name: list[str] = []

    node_offset = 0
    window_offset = 0
    ip_offset = 0
    scenario_id = 0
    feature_names: list[str] | None = None
    feature_index: dict[str, int] | None = None
    target_ip = "10.0.0.100"
    delta_t = 1.0

    for split_name, paths in split_specs:
        for path in paths:
            graph = torch.load(path, weights_only=False, map_location="cpu")
            graph_files.append(os.path.abspath(path))
            scenario_name = scenario_name_from_path(path)
            scenario_names.append(scenario_name)

            if feature_names is None:
                feature_names = list(getattr(graph, "feature_names", []))
            if feature_index is None:
                feature_index = dict(getattr(graph, "feature_index", {}))
            target_ip = str(getattr(graph, "target_ip", target_ip))
            delta_t = float(getattr(graph, "delta_t", delta_t))

            n_nodes = int(graph.num_nodes)
            flow_mask = graph.window_idx >= 0 if hasattr(graph, "window_idx") else (torch.arange(n_nodes) > 0)
            x_parts.append(graph.x.detach().cpu())
            y_parts.append(graph.y.detach().cpu())
            edge_parts.append(graph.edge_index.detach().cpu() + node_offset)
            edge_type_parts.append(graph.edge_type.detach().cpu())

            local_window = graph.window_idx.detach().cpu().clone()
            global_window = local_window.clone()
            global_window[global_window >= 0] += int(window_offset)
            local_window_idx_parts.append(local_window)
            window_idx_parts.append(global_window)

            local_ip_idx = graph.ip_idx.detach().cpu().clone()
            global_ip_idx = local_ip_idx.clone()
            global_ip_idx[global_ip_idx >= 0] += int(ip_offset)
            ip_idx_parts.append(global_ip_idx)
            scenario_idx_parts.append(torch.full((n_nodes,), scenario_id, dtype=torch.long))

            cap, bw = capacity_for_graph(graph)
            node_capacity_parts.append(torch.full((n_nodes,), float(cap), dtype=torch.float32))
            node_bw_parts.append(torch.full((n_nodes,), float(bw), dtype=torch.float32))

            split_mask = torch.zeros(n_nodes, dtype=torch.bool)
            split_mask[flow_mask.detach().cpu()] = True
            train_mask_parts.append(split_mask if split_name == "train" else torch.zeros(n_nodes, dtype=torch.bool))
            val_mask_parts.append(split_mask if split_name == "val" else torch.zeros(n_nodes, dtype=torch.bool))
            test_mask_parts.append(split_mask if split_name == "test" else torch.zeros(n_nodes, dtype=torch.bool))

            local_source_ips = list(getattr(graph, "source_ips", []))
            prefixed_source_ips = [f"{scenario_name}:{ip}" for ip in local_source_ips]
            source_ips.extend(prefixed_source_ips)
            for node_i in range(n_nodes):
                node_scenario_name.append(scenario_name)
                ip_i = int(local_ip_idx[node_i].item())
                if 0 <= ip_i < len(prefixed_source_ips):
                    node_ip.append(prefixed_source_ips[ip_i])
                else:
                    node_ip.append(f"{scenario_name}:{target_ip}")

            node_offset += n_nodes
            window_valid = local_window[local_window >= 0]
            if int(window_valid.numel()) > 0:
                window_offset += int(window_valid.max().item()) + 1
            ip_offset += len(local_source_ips)
            scenario_id += 1

    x = torch.cat(x_parts, dim=0)
    y = torch.cat(y_parts, dim=0)
    edge_index = torch.cat(edge_parts, dim=1)
    edge_type = torch.cat(edge_type_parts, dim=0)
    window_idx = torch.cat(window_idx_parts, dim=0)
    local_window_idx = torch.cat(local_window_idx_parts, dim=0)
    ip_idx = torch.cat(ip_idx_parts, dim=0)
    node_scenario_idx = torch.cat(scenario_idx_parts, dim=0)
    node_capacity = torch.cat(node_capacity_parts, dim=0)
    node_bw = torch.cat(node_bw_parts, dim=0)
    train_mask = torch.cat(train_mask_parts, dim=0)
    val_mask = torch.cat(val_mask_parts, dim=0)
    test_mask = torch.cat(test_mask_parts, dim=0)

    flow_mask = window_idx >= 0
    if int(train_mask.sum().item()) == 0:
        raise RuntimeError("Merged graph has empty train mask")

    feat_mean = x[train_mask].mean(dim=0)
    feat_std = x[train_mask].std(dim=0).clamp(min=1e-6)
    x_norm = (x - feat_mean) / feat_std
    x_norm[~flow_mask] = 0.0

    edge_index_undirected = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_type_undirected = torch.cat([edge_type, edge_type], dim=0)

    graph = Data(
        x=x,
        x_norm=x_norm,
        y=y,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_index_undirected=edge_index_undirected,
        edge_type_undirected=edge_type_undirected,
        window_idx=window_idx,
        local_window_idx=local_window_idx,
        ip_idx=ip_idx,
        node_scenario_idx=node_scenario_idx,
        node_capacity_bytes_per_sec=node_capacity,
        node_core_bw_mbps=node_bw,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        temporal_test_mask=test_mask.clone(),
        feat_mean=feat_mean,
        feat_std=feat_std,
    )
    graph.split_protocol = "cross_scenario"
    graph.feature_names = feature_names or []
    graph.feature_index = feature_index or {}
    graph.source_ips = source_ips
    graph.scenario_names = scenario_names
    graph.node_scenario_name = node_scenario_name
    graph.node_ip = node_ip
    graph.target_ip = target_ip
    graph.delta_t = float(delta_t)
    graph.graph_files = graph_files
    graph.capacity_bytes_per_sec = 0.0
    graph.mixed_scenario_capacity = True
    graph.n_windows = int((window_idx[flow_mask].max().item() + 1) if int(flow_mask.sum().item()) > 0 else 0)
    return graph


def run_cmd(cmd: list[str], cwd: Path, log_file: Path, skip_if_exists: Path | None = None) -> None:
    if skip_if_exists is not None and skip_if_exists.exists():
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"command failed, see {log_file}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0, "mean": 0.0, "std": 0.0}
    if len(vals) == 1:
        return {"n": 1, "mean": float(vals[0]), "std": 0.0}
    return {"n": len(vals), "mean": float(statistics.mean(vals)), "std": float(statistics.stdev(vals))}


def pval_signflip(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or not x:
        return 1.0
    d = [a - b for a, b in zip(x, y)]
    n = len(d)
    obs = abs(sum(d) / n)
    ext = 0
    tot = 0
    for bits in itertools.product([-1.0, 1.0], repeat=n):
        tot += 1
        s = abs(sum(di * si for di, si in zip(d, bits)) / n)
        if s >= obs - 1e-12:
            ext += 1
    return float((ext + 1) / (tot + 1))


def pick_test_metrics(phase3: dict[str, Any]) -> dict[str, float]:
    fe = phase3.get("final_eval", {})
    out = fe.get("test_temporal") or fe.get("test_random") or {}
    tp = int(out.get("tp", 0))
    fp = int(out.get("fp", 0))
    tn = int(out.get("tn", 0))
    out = dict(out)
    out["fpr"] = float(fp / max(fp + tn, 1))
    return out


def metrics_from_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    fpr = fp / max(fp + tn, 1)
    return {
        "f1": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "fpr": float(fpr),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def attack_only_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    recall = tp / max(tp + fn, 1)
    return {
        "recall": float(recall),
        "miss_rate": float(1.0 - recall),
        "mean_prob": float(np.mean(y_prob)) if y_prob.size else 0.0,
        "median_prob": float(np.median(y_prob)) if y_prob.size else 0.0,
        "tp": tp,
        "fn": fn,
        "n_pos": int(y_true.size),
    }


def ensure_graphs(args: argparse.Namespace, project: Path, out_dir: Path, scenarios: list[str]) -> dict[str, Path]:
    graph_dir = out_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for scenario in scenarios:
        graph_file = graph_dir / f"{scenario}.pt"
        out[scenario] = graph_file
        run_cmd(
            [
                args.python_bin,
                "build_graph_v2.py",
                "--pcap-file",
                str(Path(args.real_collection_dir) / scenario / "full_arena_v2.pcap"),
                "--manifest-file",
                str(Path(args.real_collection_dir) / scenario / "arena_manifest_v2.json"),
                "--output-file",
                str(graph_file),
                "--target-ip",
                "10.0.0.100",
                "--delta-t",
                "1.0",
                "--seed",
                "42",
            ],
            cwd=project,
            log_file=out_dir / "logs" / f"build_{scenario}.log",
            skip_if_exists=graph_file if args.skip_existing else None,
        )
    return out


def scenario_metrics(predictions: dict[str, Any]) -> dict[str, dict[str, float]]:
    scen = np.asarray(predictions.get("scenario_name", []), dtype=object)
    y_true = np.asarray(predictions.get("y_true", []), dtype=np.int64)
    y_pred = np.asarray(predictions.get("y_pred", []), dtype=np.int64)
    out: dict[str, dict[str, float]] = {}
    if scen.size == 0:
        return out
    for name in sorted(set(scen.tolist())):
        mask = scen == name
        metric = metrics_from_binary(y_true[mask], y_pred[mask])
        metric["n_benign"] = int(np.sum(y_true[mask] == 0))
        metric["n_attack"] = int(np.sum(y_true[mask] == 1))
        metric["has_both_classes"] = bool(metric["n_benign"] > 0 and metric["n_attack"] > 0)
        out[str(name)] = metric
    return out


def split_block(predictions: dict[str, Any], scenario_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scen = np.asarray(predictions.get("scenario_name", []), dtype=object)
    if scen.size == 0 or not scenario_names:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)
    mask = np.isin(scen, np.asarray(scenario_names, dtype=object))
    y_true = np.asarray(predictions.get("y_true", []), dtype=np.int64)[mask]
    y_pred = np.asarray(predictions.get("y_pred", []), dtype=np.int64)[mask]
    y_prob = np.asarray(predictions.get("y_prob", []), dtype=np.float64)[mask]
    return y_true, y_pred, y_prob


def graph_flow_counts(graph: Any) -> dict[str, Any]:
    flow_mask = graph.window_idx >= 0 if hasattr(graph, "window_idx") else torch.ones(graph.num_nodes, dtype=torch.bool)
    benign = int(((graph.y == 0) & flow_mask).sum().item())
    attack = int(((graph.y == 1) & flow_mask).sum().item())
    return {
        "nodes": int(flow_mask.sum().item()),
        "benign": benign,
        "attack": attack,
        "has_both_classes": bool(benign > 0 and attack > 0),
        "attack_only": bool(benign == 0 and attack > 0),
        "benign_only": bool(benign > 0 and attack == 0),
    }


def audit_graph_files(graph_map: dict[str, Path], train_scenarios: list[str], val_scenarios: list[str], test_scenarios: list[str]) -> dict[str, Any]:
    split_by_scenario = {s: "train" for s in train_scenarios}
    split_by_scenario.update({s: "val" for s in val_scenarios})
    split_by_scenario.update({s: "test" for s in test_scenarios})
    audit: dict[str, Any] = {}
    for scenario_name, graph_file in graph_map.items():
        graph = torch.load(graph_file, weights_only=False, map_location="cpu")
        block = graph_flow_counts(graph)
        block["scenario"] = scenario_name
        block["split"] = split_by_scenario.get(scenario_name, "unknown")
        audit[scenario_name] = block
    return audit


def eval_split_audit(graph: Any) -> dict[str, Any]:
    scenario_names = list(getattr(graph, "scenario_names", []))
    scenario_idx = getattr(graph, "node_scenario_idx", None)
    if scenario_idx is None:
        return {}
    audit: dict[str, Any] = {}
    for sid, scenario_name in enumerate(scenario_names):
        scen_mask = scenario_idx == sid
        audit[scenario_name] = {}
        for split_name in ["train", "val", "test"]:
            split_mask = getattr(graph, f"{split_name}_mask").bool() & scen_mask
            benign = int(((graph.y == 0) & split_mask).sum().item())
            attack = int(((graph.y == 1) & split_mask).sum().item())
            audit[scenario_name][split_name] = {
                "nodes": int(split_mask.sum().item()),
                "benign": benign,
                "attack": attack,
                "has_both_classes": bool(benign > 0 and attack > 0),
                "attack_only": bool(benign == 0 and attack > 0),
                "benign_only": bool(benign > 0 and attack == 0),
            }
    return audit


def maybe_prepare_eval_graph(args: argparse.Namespace, merged_graph: Path, out_dir: Path) -> Path:
    if not args.hard_overlap:
        return merged_graph

    graph = torch.load(merged_graph, weights_only=False, map_location="cpu")
    train_mask, val_mask, test_mask = apply_overlap_hardening(
        graph,
        train_mask=graph.train_mask.bool(),
        val_mask=graph.val_mask.bool(),
        test_mask=graph.test_mask.bool(),
        train_keep_frac=float(args.train_keep_frac),
        val_keep_frac=float(args.val_keep_frac),
        test_keep_frac=float(args.test_keep_frac),
        min_keep_per_class=int(args.min_keep_per_class),
        feature_names=parse_csv(args.overlap_features),
    )
    graph.train_mask = train_mask.bool()
    graph.val_mask = val_mask.bool()
    graph.test_mask = test_mask.bool()
    graph.temporal_test_mask = graph.test_mask.clone()
    graph.hard_overlap = True
    graph.cross_scenario_hard_overlap = True
    graph.cross_scenario_overlap_features = parse_csv(args.overlap_features)
    graph.cross_scenario_keep_fracs = {
        "train": float(args.train_keep_frac),
        "val": float(args.val_keep_frac),
        "test": float(args.test_keep_frac),
    }

    eval_graph = out_dir / "graphs" / "cross_scenario_eval.pt"
    torch.save(graph, eval_graph)
    return eval_graph


def model_metric_lists(
    rows_data_only: list[dict[str, Any]],
    rows_baseline: list[dict[str, Any]],
    summary_key: str,
) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {m: {"f1": [], "recall": [], "fpr": []} for m in MODEL_NAMES}
    for row in rows_data_only:
        block = row.get(summary_key, {})
        if block:
            out["data_only"]["f1"].append(float(block.get("f1", 0.0)))
            out["data_only"]["recall"].append(float(block.get("recall", 0.0)))
            out["data_only"]["fpr"].append(float(block.get("fpr", 0.0)))
    for row in rows_baseline:
        blocks = row.get(summary_key, {})
        for model_name in MODEL_NAMES:
            if model_name == "data_only":
                continue
            block = blocks.get(model_name, {})
            if block:
                out[model_name]["f1"].append(float(block.get("f1", 0.0)))
                out[model_name]["recall"].append(float(block.get("recall", 0.0)))
                out[model_name]["fpr"].append(float(block.get("fpr", 0.0)))
    return out


def stats_from_metric_lists(metric_lists: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model_name, values in metric_lists.items():
        out[model_name] = {
            "f1": mean_std(values["f1"]),
            "recall": mean_std(values["recall"]),
            "fpr": mean_std(values["fpr"]),
        }
    return out


def compute_filtered_seed_metrics(
    rows_data_only: list[dict[str, Any]],
    rows_baseline: list[dict[str, Any]],
    paper_test_scenarios: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    filt_data: list[dict[str, Any]] = []
    filt_base: list[dict[str, Any]] = []
    for row in rows_data_only:
        pred = load_json(Path(row["prediction_file"]))["predictions"]
        y_true, y_pred, _ = split_block(pred, paper_test_scenarios)
        filt_data.append({"seed": row["seed"], "paper_metrics": metrics_from_binary(y_true, y_pred) if y_true.size else {}})
    for row in rows_baseline:
        baseline = load_json(Path(row["file"]))
        out: dict[str, Any] = {}
        for model_name, pred in baseline.get("predictions", {}).items():
            if model_name == "pi_gnn":
                continue
            y_true, y_pred, _ = split_block(pred, paper_test_scenarios)
            out[model_name] = metrics_from_binary(y_true, y_pred) if y_true.size else {}
        filt_base.append({"seed": row["seed"], "paper_metrics": out})
    return filt_data, filt_base


def compute_attack_only_probe(
    rows_data_only: list[dict[str, Any]],
    rows_baseline: list[dict[str, Any]],
    attack_only_scenarios: list[str],
) -> dict[str, Any]:
    if not attack_only_scenarios:
        return {}

    overall_rows: dict[str, list[dict[str, float]]] = {m: [] for m in MODEL_NAMES}
    per_scenario: dict[str, dict[str, list[dict[str, float]]]] = {
        scenario_name: {m: [] for m in MODEL_NAMES} for scenario_name in attack_only_scenarios
    }

    for row in rows_data_only:
        pred = load_json(Path(row["prediction_file"]))["predictions"]
        y_true, y_pred, y_prob = split_block(pred, attack_only_scenarios)
        if y_true.size:
            overall_rows["data_only"].append(attack_only_metrics(y_true, y_pred, y_prob))
        for scenario_name in attack_only_scenarios:
            sy, sp, sq = split_block(pred, [scenario_name])
            if sy.size:
                per_scenario[scenario_name]["data_only"].append(attack_only_metrics(sy, sp, sq))

    for row in rows_baseline:
        baseline = load_json(Path(row["file"]))
        for model_name, pred in baseline.get("predictions", {}).items():
            if model_name == "pi_gnn":
                continue
            y_true, y_pred, y_prob = split_block(pred, attack_only_scenarios)
            if y_true.size:
                overall_rows[model_name].append(attack_only_metrics(y_true, y_pred, y_prob))
            for scenario_name in attack_only_scenarios:
                sy, sp, sq = split_block(pred, [scenario_name])
                if sy.size:
                    per_scenario[scenario_name][model_name].append(attack_only_metrics(sy, sp, sq))

    summary: dict[str, Any] = {"overall": {}, "per_scenario": {}}
    for model_name in MODEL_NAMES:
        rows = overall_rows[model_name]
        summary["overall"][model_name] = {
            "recall": mean_std([float(r["recall"]) for r in rows]),
            "miss_rate": mean_std([float(r["miss_rate"]) for r in rows]),
            "mean_prob": mean_std([float(r["mean_prob"]) for r in rows]),
        }
    for scenario_name in attack_only_scenarios:
        summary["per_scenario"][scenario_name] = {}
        for model_name in MODEL_NAMES:
            rows = per_scenario[scenario_name][model_name]
            summary["per_scenario"][scenario_name][model_name] = {
                "recall": mean_std([float(r["recall"]) for r in rows]),
                "miss_rate": mean_std([float(r["miss_rate"]) for r in rows]),
                "mean_prob": mean_std([float(r["mean_prob"]) for r in rows]),
            }
    return summary


def raw_best_nonphysics(stats_block: dict[str, Any]) -> str:
    return max(["random_forest", "gcn", "graphsage", "gatv2"], key=lambda m: stats_block.get(m, {}).get("f1", {}).get("mean", 0.0))


def write_figures(out_dir: Path, paper_overall_stats: dict[str, Any], attack_only_probe: dict[str, Any]) -> None:
    if paper_overall_stats:
        models = ["data_only", "gcn", "graphsage", "gatv2"]
        means = [paper_overall_stats[m]["f1"]["mean"] for m in models]
        stds = [paper_overall_stats[m]["f1"]["std"] for m in models]
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.bar(np.arange(len(models)), means, yerr=stds, color=["#1f77b4", "#2ca02c", "#17becf", "#9467bd"])
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models)
        ax.set_ylim(0.0, min(1.0, max(0.65, max(means) + 0.1)))
        ax.set_ylabel("F1")
        ax.set_title("Cross-Scenario Stress Holdout F1")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_cross_scenario_overall.png", dpi=220)
        plt.close(fig)

    if attack_only_probe:
        models = ["data_only", "gcn", "graphsage", "gatv2"]
        means = [attack_only_probe["overall"][m]["recall"]["mean"] for m in models]
        stds = [attack_only_probe["overall"][m]["recall"]["std"] for m in models]
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.bar(np.arange(len(models)), means, yerr=stds, color=["#4c78a8", "#2ca02c", "#17becf", "#9467bd"])
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Recall")
        ax.set_title("Batch2 Attack-Only Probe Recall")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_attack_only_probe_recall.png", dpi=220)
        plt.close(fig)


def write_markdown(
    out_dir: Path,
    train_scenarios: list[str],
    val_scenarios: list[str],
    test_scenarios: list[str],
    graph_audit: dict[str, Any],
    split_audit: dict[str, Any],
    raw_overall_stats: dict[str, Any],
    paper_overall_stats: dict[str, Any],
    attack_only_probe: dict[str, Any],
    best_nonphysics: str,
    summary: dict[str, Any],
) -> None:
    warnings = summary.get("warnings", [])
    paper_test_scenarios = summary.get("paper_test_scenarios", [])
    attack_only_scenarios = summary.get("attack_only_test_scenarios", [])

    lines = [
        "# Cross-Scenario Summary",
        "",
        f"- Train scenarios: `{', '.join(train_scenarios)}`",
        f"- Val scenarios: `{', '.join(val_scenarios)}`",
        f"- Test scenarios: `{', '.join(test_scenarios)}`",
        f"- Hard overlap: `{bool(summary['config'].get('hard_overlap', False))}`",
        "",
        "## Scenario Audit (raw graph flows)",
    ]
    for scenario_name in train_scenarios + val_scenarios + test_scenarios:
        block = graph_audit[scenario_name]
        lines.append(
            f"- {scenario_name} [{block['split']}]: nodes={block['nodes']}, benign={block['benign']}, attack={block['attack']}, has_both_classes={block['has_both_classes']}"
        )

    lines.extend(["", "## Eval Split Audit"])
    for scenario_name in train_scenarios + val_scenarios + test_scenarios:
        block = split_audit.get(scenario_name, {})
        test_block = block.get("test", {})
        if test_block:
            lines.append(
                f"- {scenario_name} test: nodes={test_block['nodes']}, benign={test_block['benign']}, attack={test_block['attack']}, has_both_classes={test_block['has_both_classes']}"
            )

    if paper_overall_stats:
        lines.extend(["", "## Paper-Ready Overall (two-class test scenarios only)"])
        for model_name in MODEL_NAMES:
            stat = paper_overall_stats[model_name]["f1"]
            lines.append(f"- {model_name}: F1=`{stat['mean']:.4f}` +/- `{stat['std']:.4f}`")
        lines.append(f"- Best non-physics baseline: `{best_nonphysics}`")
        lines.append(f"- Paper-ready test scenarios: `{', '.join(paper_test_scenarios)}`")
    else:
        lines.extend(
            [
                "",
                "## Paper-Ready Overall",
                "- No paper-ready F1 summary is emitted because the test split does not contain any two-class scenario.",
            ]
        )

    lines.extend(["", "## Raw Overall (all test nodes)"])
    for model_name in MODEL_NAMES:
        stat = raw_overall_stats[model_name]["f1"]
        lines.append(f"- {model_name}: F1=`{stat['mean']:.4f}` +/- `{stat['std']:.4f}`")

    if attack_only_probe:
        lines.extend(["", "## Batch2 Attack-Only Probe"])
        lines.append(f"- Attack-only test scenarios: `{', '.join(attack_only_scenarios)}`")
        for model_name in MODEL_NAMES:
            stat = attack_only_probe["overall"][model_name]["recall"]
            lines.append(f"- {model_name}: Recall=`{stat['mean']:.4f}` +/- `{stat['std']:.4f}`")

    if warnings:
        lines.extend(["", "## Warnings"])
        for warning in warnings:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## Notes",
            "- Physics cross-scenario result is intentionally skipped in the main table because the current physics loss still assumes a single clean capacity scale, while merged scenario graphs mix different bottleneck capacities.",
        ]
    )
    (out_dir / "cross_scenario_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Run cross-scenario evaluation with scenario audit and optional stress hardening")
    p.add_argument("--project-dir", default=str(REPO_ROOT))
    p.add_argument("--python-bin", default="python")
    p.add_argument("--real-collection-dir", default=str(DATA_ROOT))
    p.add_argument("--output-dir", default=str(RUN_ROOT / "cross_scenario"))
    p.add_argument("--train-scenarios", default="scenario_d_three_tier_low2,scenario_e_three_tier_high2,scenario_f_two_tier_high2")
    p.add_argument("--val-scenarios", default="scenario_g_mimic_congest")
    p.add_argument("--test-scenarios", default="scenario_h_mimic_heavy_overlap")
    p.add_argument("--seeds", default="11,22,33,44,55")
    p.add_argument("--epochs", type=int, default=140)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--hard-overlap", action="store_true")
    p.add_argument("--train-keep-frac", type=float, default=0.85)
    p.add_argument("--val-keep-frac", type=float, default=0.90)
    p.add_argument("--test-keep-frac", type=float, default=0.75)
    p.add_argument("--min-keep-per-class", type=int, default=64)
    p.add_argument(
        "--overlap-features",
        default="ln(N+1),ln(T+1),entropy,pkt_rate,avg_pkt_size,port_diversity",
        help="comma-separated feature names used for overlap hardness",
    )
    args = p.parse_args()

    project = Path(args.project_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_scenarios = parse_csv(args.train_scenarios)
    val_scenarios = parse_csv(args.val_scenarios)
    test_scenarios = parse_csv(args.test_scenarios)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    all_scenarios = train_scenarios + val_scenarios + test_scenarios

    graph_map = ensure_graphs(args, project=project, out_dir=out_dir, scenarios=all_scenarios)
    graph_audit = audit_graph_files(graph_map, train_scenarios=train_scenarios, val_scenarios=val_scenarios, test_scenarios=test_scenarios)
    merged_graph = out_dir / "graphs" / "cross_scenario_merged.pt"
    if not (args.skip_existing and merged_graph.exists()):
        merged = merge_graphs(
            train_graphs=[str(graph_map[s]) for s in train_scenarios],
            val_graphs=[str(graph_map[s]) for s in val_scenarios],
            test_graphs=[str(graph_map[s]) for s in test_scenarios],
        )
        merged_graph.parent.mkdir(parents=True, exist_ok=True)
        torch.save(merged, merged_graph)
    eval_graph = maybe_prepare_eval_graph(args, merged_graph=merged_graph, out_dir=out_dir)
    eval_graph_obj = torch.load(eval_graph, weights_only=False, map_location="cpu")
    split_audit = eval_split_audit(eval_graph_obj)

    paper_test_scenarios = [s for s in test_scenarios if split_audit.get(s, {}).get("test", {}).get("has_both_classes", False)]
    attack_only_scenarios = [s for s in test_scenarios if split_audit.get(s, {}).get("test", {}).get("attack_only", False)]
    benign_only_scenarios = [s for s in test_scenarios if split_audit.get(s, {}).get("test", {}).get("benign_only", False)]
    warnings: list[str] = []
    if attack_only_scenarios:
        warnings.append(
            "These test scenarios are attack-only after graph construction and should not be used for paper F1 tables: "
            + ", ".join(attack_only_scenarios)
        )
    if benign_only_scenarios:
        warnings.append(
            "These test scenarios are benign-only after graph construction and should not be used for paper F1 tables: "
            + ", ".join(benign_only_scenarios)
        )
    if not paper_test_scenarios:
        warnings.append("No two-class test scenario is available in this split. Treat this suite as an auxiliary probe, not the main cross-scenario result.")

    rows_data_only = []
    rows_baseline = []
    per_scenario_rows: list[dict[str, Any]] = []
    for seed in seeds:
        stage_dir = out_dir / "stage3" / f"cross_scenario__data_only__seed{seed}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_results = stage_dir / "phase3_results.json"
        stage_model = stage_dir / "pi_gnn_model.pt"
        run_cmd(
            [
                args.python_bin,
                "pi_gnn_train_v2.py",
                "--graph-file",
                str(eval_graph),
                "--model-file",
                str(stage_model),
                "--results-file",
                str(stage_results),
                "--epochs",
                str(args.epochs),
                "--alpha-flow",
                "0.0",
                "--beta-latency",
                "0.0",
                "--capacity",
                "0",
                "--capacity-mode",
                "auto",
                "--warmup-epochs",
                "25",
                "--patience",
                "35",
                "--seed",
                str(seed),
                "--force-cpu",
            ],
            cwd=project,
            log_file=stage_dir / "run.log",
            skip_if_exists=stage_results if args.skip_existing else None,
        )
        phase3 = load_json(stage_results)

        pred_file = stage_dir / "predictions.json"
        if not (args.skip_existing and pred_file.exists()):
            export_pi_predictions(
                str(eval_graph),
                str(stage_model),
                str(stage_results),
                str(pred_file),
                force_cpu=True,
            )
        rows_data_only.append(
            {
                "seed": seed,
                "metrics": pick_test_metrics(phase3),
                "result_file": str(stage_results),
                "model_file": str(stage_model),
                "prediction_file": str(pred_file),
            }
        )

        baseline_dir = out_dir / "baseline" / f"seed{seed}"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_file = baseline_dir / "baseline_eval.json"
        run_cmd(
            [
                args.python_bin,
                "evaluate_baselines.py",
                "--graph-file",
                str(eval_graph),
                "--pi-model-file",
                str(stage_model),
                "--pi-results-file",
                str(stage_results),
                "--output-file",
                str(baseline_file),
                "--seed",
                str(seed),
                "--save-predictions",
                "--force-cpu",
            ],
            cwd=project,
            log_file=baseline_dir / "run.log",
            skip_if_exists=baseline_file if args.skip_existing else None,
        )
        baseline = load_json(baseline_file)
        rows_baseline.append({"seed": seed, "metrics": baseline.get("metrics", {}), "file": str(baseline_file)})

        data_pred = load_json(pred_file)["predictions"]
        for scenario_name, metric in scenario_metrics(data_pred).items():
            per_scenario_rows.append({"seed": seed, "model": "data_only", "scenario": scenario_name, "metrics": metric})
        for model_name, pred in baseline.get("predictions", {}).items():
            if model_name == "pi_gnn":
                continue
            for scenario_name, metric in scenario_metrics(pred).items():
                per_scenario_rows.append({"seed": seed, "model": model_name, "scenario": scenario_name, "metrics": metric})

    raw_metric_lists = model_metric_lists(rows_data_only, rows_baseline, summary_key="metrics")
    raw_overall_stats = stats_from_metric_lists(raw_metric_lists)

    summary: dict[str, Any] = {
        "config": vars(args),
        "merged_graph": str(merged_graph),
        "eval_graph": str(eval_graph),
        "train_scenarios": train_scenarios,
        "val_scenarios": val_scenarios,
        "test_scenarios": test_scenarios,
        "graph_audit": graph_audit,
        "eval_split_audit": split_audit,
        "paper_test_scenarios": paper_test_scenarios,
        "attack_only_test_scenarios": attack_only_scenarios,
        "benign_only_test_scenarios": benign_only_scenarios,
        "rows_data_only": rows_data_only,
        "rows_baseline": rows_baseline,
        "per_scenario_rows": per_scenario_rows,
        "overall_stats": raw_overall_stats,
        "paper_overall_stats": {},
        "per_scenario_stats": {},
        "attack_only_probe": {},
        "p_values": {},
        "physics_decision": {
            "cross_scenario_physics": "skipped",
            "reason": "mixed scenario capacities make current single-capacity physics loss hard to interpret cleanly",
        },
        "paper_recommendation": "main_table" if paper_test_scenarios else "attack_only_appendix",
        "warnings": warnings,
    }

    for scenario_name in test_scenarios:
        summary["per_scenario_stats"][scenario_name] = {}
        audit_block = split_audit.get(scenario_name, {}).get("test", {})
        for model_name in MODEL_NAMES:
            rows = [r for r in per_scenario_rows if r["model"] == model_name and r["scenario"] == scenario_name]
            vals = [float(r["metrics"].get("f1", 0.0)) for r in rows]
            rec = [float(r["metrics"].get("recall", 0.0)) for r in rows]
            fpr = [float(r["metrics"].get("fpr", 0.0)) for r in rows]
            summary["per_scenario_stats"][scenario_name][model_name] = {
                "class_balance": audit_block,
                "f1": mean_std(vals),
                "recall": mean_std(rec),
                "fpr": mean_std(fpr),
            }

    paper_overall_stats: dict[str, Any] = {}
    if paper_test_scenarios:
        filt_data, filt_base = compute_filtered_seed_metrics(rows_data_only, rows_baseline, paper_test_scenarios=paper_test_scenarios)
        paper_metric_lists = model_metric_lists(filt_data, filt_base, summary_key="paper_metrics")
        paper_overall_stats = stats_from_metric_lists(paper_metric_lists)
        summary["paper_overall_stats"] = paper_overall_stats

        data_f1 = paper_metric_lists["data_only"]["f1"]
        gcn_f1 = paper_metric_lists["gcn"]["f1"]
        sage_f1 = paper_metric_lists["graphsage"]["f1"]
        gat_f1 = paper_metric_lists["gatv2"]["f1"]
        summary["p_values"]["graphsage_vs_gcn_f1"] = pval_signflip(sage_f1, gcn_f1)
        summary["p_values"]["gatv2_vs_gcn_f1"] = pval_signflip(gat_f1, gcn_f1)
        best_nonphysics = raw_best_nonphysics(paper_overall_stats)
        best_vals = paper_metric_lists[best_nonphysics]["f1"]
        summary["best_nonphysics_baseline"] = best_nonphysics
        summary["p_values"]["best_nonphysics_vs_data_only_f1"] = pval_signflip(best_vals, data_f1)
    else:
        best_nonphysics = raw_best_nonphysics(raw_overall_stats)
        summary["best_nonphysics_baseline"] = best_nonphysics

    attack_only_probe = compute_attack_only_probe(rows_data_only, rows_baseline, attack_only_scenarios=attack_only_scenarios)
    summary["attack_only_probe"] = attack_only_probe

    out_json = out_dir / "cross_scenario_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_figures(out_dir=out_dir, paper_overall_stats=paper_overall_stats, attack_only_probe=attack_only_probe)
    write_markdown(
        out_dir=out_dir,
        train_scenarios=train_scenarios,
        val_scenarios=val_scenarios,
        test_scenarios=test_scenarios,
        graph_audit=graph_audit,
        split_audit=split_audit,
        raw_overall_stats=raw_overall_stats,
        paper_overall_stats=paper_overall_stats,
        attack_only_probe=attack_only_probe,
        best_nonphysics=best_nonphysics,
        summary=summary,
    )
    print(f"[DONE] cross-scenario suite: {out_dir}")


if __name__ == "__main__":
    main()
