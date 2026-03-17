#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch


FALLBACK_FEATURE_INDEX = {
    "ln(N+1)": 0,
    "ln(T+1)": 1,
    "entropy": 2,
    "D_observed": 3,
    "pkt_rate": 4,
    "avg_pkt_size": 5,
    "port_diversity": 6,
}


def load_manifest(manifest_file: str) -> dict[str, Any]:
    if not manifest_file or not os.path.exists(manifest_file):
        return {}
    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def role_for_ip(manifest: dict[str, Any], ip: str) -> str:
    roles = manifest.get("roles", {})
    if isinstance(roles, dict):
        return str(roles.get(ip, ""))
    return ""


def mask_counts(y: torch.Tensor, mask: torch.Tensor) -> dict[str, int]:
    return {
        "nodes": int(mask.sum().item()),
        "benign": int(((y == 0) & mask).sum().item()),
        "attack": int(((y == 1) & mask).sum().item()),
    }


def protocol_temporal_ood(graph, flow_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w = graph.window_idx[flow_mask]
    w_valid = w[w >= 0]
    if int(w_valid.numel()) == 0:
        zero = torch.zeros_like(flow_mask)
        return zero, zero, zero

    w_min = int(torch.min(w_valid).item())
    w_max = int(torch.max(w_valid).item())
    span = max(w_max - w_min + 1, 1)
    tr_end = w_min + int(0.6 * span)
    va_end = w_min + int(0.8 * span)

    train = flow_mask & (graph.window_idx >= w_min) & (graph.window_idx < tr_end)
    val = flow_mask & (graph.window_idx >= tr_end) & (graph.window_idx < va_end)
    test = flow_mask & (graph.window_idx >= va_end)
    return train, val, test


def protocol_topology_ood(graph, flow_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gid = graph.ip_idx.remainder(5)
    train = flow_mask & (gid <= 2)
    val = flow_mask & (gid == 3)
    test = flow_mask & (gid == 4)
    return train, val, test

def protocol_congestion_ood(graph, flow_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = resolve_feature_index(graph)
    rate_col = int(idx.get("pkt_rate", 4))

    win = graph.window_idx[flow_mask]
    x_rate = graph.x[flow_mask, rate_col]
    uniq = torch.unique(win[win >= 0])
    if int(uniq.numel()) == 0:
        zero = torch.zeros_like(flow_mask)
        return zero, zero, zero

    load = []
    for w in uniq.tolist():
        m = flow_mask & (graph.window_idx == int(w))
        agg = float(graph.x[m, rate_col].sum().item())
        load.append((int(w), agg))

    load.sort(key=lambda t: t[1])
    n = len(load)
    n_tr = max(1, int(0.6 * n))
    n_va = max(1, int(0.2 * n))
    tr_w = {w for w, _ in load[:n_tr]}
    va_w = {w for w, _ in load[n_tr:n_tr + n_va]}
    te_w = {w for w, _ in load[n_tr + n_va:]}

    train = flow_mask & torch.tensor([int(w.item()) in tr_w for w in graph.window_idx], dtype=torch.bool)
    val = flow_mask & torch.tensor([int(w.item()) in va_w for w in graph.window_idx], dtype=torch.bool)
    test = flow_mask & torch.tensor([int(w.item()) in te_w for w in graph.window_idx], dtype=torch.bool)
    return train, val, test


def protocol_attack_strategy_ood(
    graph,
    manifest: dict[str, Any],
    flow_mask: torch.Tensor,
    holdout_attack_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_ips = list(getattr(graph, "source_ips", []))
    if not source_ips:
        raise RuntimeError("graph.source_ips missing, cannot build attack_strategy_ood split")

    val_attack_type = "burst" if holdout_attack_type != "burst" else "slowburn"

    gid = graph.ip_idx.remainder(5)
    benign_train = (graph.y == 0) & flow_mask & (gid <= 2)
    benign_val = (graph.y == 0) & flow_mask & (gid == 3)
    benign_test = (graph.y == 0) & flow_mask & (gid == 4)

    attack_train = torch.zeros_like(flow_mask)
    attack_val = torch.zeros_like(flow_mask)
    attack_test = torch.zeros_like(flow_mask)

    for ip_i, ip in enumerate(source_ips):
        ip_mask = flow_mask & (graph.ip_idx == ip_i) & (graph.y == 1)
        if int(ip_mask.sum().item()) == 0:
            continue

        role = role_for_ip(manifest, ip)
        atype = ""
        if role.startswith("bot:"):
            atype = role.split(":", 1)[1].strip().lower()

        if atype == holdout_attack_type:
            attack_test |= ip_mask
        elif atype == val_attack_type:
            attack_val |= ip_mask
        else:
            attack_train |= ip_mask

    train = benign_train | attack_train
    val = benign_val | attack_val
    test = benign_test | attack_test
    return train, val, test


def resolve_feature_index(graph) -> dict[str, int]:
    idx = {}
    if hasattr(graph, "feature_index") and isinstance(graph.feature_index, dict):
        idx.update({str(k): int(v) for k, v in graph.feature_index.items()})
    for k, v in FALLBACK_FEATURE_INDEX.items():
        idx.setdefault(k, int(v))
    return idx


def overlap_scores(graph, train_mask: torch.Tensor, feature_names: list[str]) -> torch.Tensor:
    idx = resolve_feature_index(graph)
    cols = [idx[f] for f in feature_names if f in idx and idx[f] < graph.x_norm.shape[1]]
    if not cols:
        cols = [0, 1, 2]

    x = graph.x_norm[:, cols]
    flow_mask = graph.window_idx >= 0
    train = train_mask & flow_mask

    b_train = train & (graph.y == 0)
    a_train = train & (graph.y == 1)

    if int(b_train.sum().item()) == 0:
        b_train = flow_mask & (graph.y == 0)
    if int(a_train.sum().item()) == 0:
        a_train = flow_mask & (graph.y == 1)

    if int(b_train.sum().item()) == 0 or int(a_train.sum().item()) == 0:
        return torch.ones(graph.num_nodes, dtype=torch.float)

    c_b = x[b_train].mean(dim=0)
    c_a = x[a_train].mean(dim=0)
    d_b = torch.norm(x - c_b, dim=1)
    d_a = torch.norm(x - c_a, dim=1)
    return torch.abs(d_b - d_a)


def _select_indices(
    idx: torch.Tensor,
    score: torch.Tensor,
    mode: str,
    keep_frac: float,
    min_keep: int,
) -> torch.Tensor:
    n = int(idx.numel())
    if n == 0:
        return idx
    k = max(min_keep, int(round(keep_frac * n)))
    k = max(1, min(k, n))

    s = score[idx]
    order = torch.argsort(s, descending=False)

    if mode == "hard":
        pick = order[:k]
    elif mode == "easy":
        pick = order[-k:]
    else:
        lo = max(0, (n - k) // 2)
        hi = min(n, lo + k)
        pick = order[lo:hi]
    return idx[pick]


def apply_overlap_hardening(
    graph,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    train_keep_frac: float,
    val_keep_frac: float,
    test_keep_frac: float,
    min_keep_per_class: int,
    feature_names: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    score = overlap_scores(graph, train_mask=train_mask, feature_names=feature_names)

    def harden_split(mask: torch.Tensor, mode: str, frac: float) -> torch.Tensor:
        out = torch.zeros_like(mask)
        for cls in (0, 1):
            idx = ((mask) & (graph.y == cls)).nonzero(as_tuple=False).view(-1)
            pick = _select_indices(idx, score=score, mode=mode, keep_frac=frac, min_keep=min_keep_per_class)
            if int(pick.numel()) > 0:
                out[pick] = True
        return out

    train_new = harden_split(train_mask, mode="easy", frac=train_keep_frac)
    val_new = harden_split(val_mask, mode="mid", frac=val_keep_frac)
    test_new = harden_split(test_mask, mode="hard", frac=test_keep_frac)

    return train_new, val_new, test_new


def parse_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare harder anti-leakage graph splits")
    p.add_argument("--input-graph", required=True)
    p.add_argument("--output-graph", required=True)
    p.add_argument("--protocol", choices=["temporal_ood", "topology_ood", "congestion_ood", "attack_strategy_ood"], required=True)
    p.add_argument("--manifest-file", default="")
    p.add_argument("--holdout-attack-type", default="mimic")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--hard-overlap", action="store_true")
    p.add_argument("--train-keep-frac", type=float, default=0.80)
    p.add_argument("--val-keep-frac", type=float, default=0.85)
    p.add_argument("--test-keep-frac", type=float, default=0.95)
    p.add_argument("--min-keep-per-class", type=int, default=64)
    p.add_argument("--camouflage-test-attacks", action="store_true")
    p.add_argument("--camouflage-noise-scale", type=float, default=0.35)
    p.add_argument(
        "--overlap-features",
        default="ln(N+1),ln(T+1),entropy,pkt_rate,avg_pkt_size,port_diversity",
        help="comma-separated feature names used for overlap hardness",
    )
    args = p.parse_args()

    torch.manual_seed(int(args.seed))
    graph = torch.load(args.input_graph, weights_only=False, map_location="cpu")
    manifest = load_manifest(args.manifest_file)

    flow_mask = graph.window_idx >= 0 if hasattr(graph, "window_idx") else (torch.arange(graph.num_nodes) > 0)

    if args.protocol == "temporal_ood":
        train_mask, val_mask, test_mask = protocol_temporal_ood(graph, flow_mask)
    elif args.protocol == "topology_ood":
        train_mask, val_mask, test_mask = protocol_topology_ood(graph, flow_mask)
    elif args.protocol == "congestion_ood":
        train_mask, val_mask, test_mask = protocol_congestion_ood(graph, flow_mask)
    else:
        train_mask, val_mask, test_mask = protocol_attack_strategy_ood(
            graph, manifest=manifest, flow_mask=flow_mask, holdout_attack_type=args.holdout_attack_type.strip().lower()
        )

    if args.hard_overlap:
        feats = parse_list(args.overlap_features)
        train_mask, val_mask, test_mask = apply_overlap_hardening(
            graph,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            train_keep_frac=float(args.train_keep_frac),
            val_keep_frac=float(args.val_keep_frac),
            test_keep_frac=float(args.test_keep_frac),
            min_keep_per_class=int(args.min_keep_per_class),
            feature_names=feats,
        )

    if bool(args.camouflage_test_attacks) and hasattr(graph, "x_norm"):
        atk_test = test_mask & (graph.y == 1)
        benign_ref = train_mask & (graph.y == 0)
        if int(atk_test.sum().item()) > 0 and int(benign_ref.sum().item()) > 0:
            mu = graph.x_norm[benign_ref].mean(dim=0)
            sd = graph.x_norm[benign_ref].std(dim=0).clamp(min=1e-6)
            n = int(atk_test.sum().item())
            eps = torch.randn((n, graph.x_norm.shape[1]), dtype=graph.x_norm.dtype)
            graph.x_norm[atk_test] = mu.unsqueeze(0) + eps * sd.unsqueeze(0) * float(args.camouflage_noise_scale)
            graph.camouflage_test_attacks = True
            graph.camouflage_noise_scale = float(args.camouflage_noise_scale)

    graph.train_mask = train_mask.bool()
    graph.val_mask = val_mask.bool()
    graph.test_mask = test_mask.bool()
    graph.temporal_test_mask = test_mask.bool()
    graph.split_protocol = args.protocol
    graph.holdout_attack_type = args.holdout_attack_type.strip().lower()
    graph.hard_overlap = bool(args.hard_overlap)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_graph)), exist_ok=True)
    torch.save(graph, args.output_graph)

    print("=" * 70)
    print(f"Protocol: {args.protocol}")
    print(f"Hard overlap: {bool(args.hard_overlap)}")
    print(f"Camouflage test attacks: {bool(args.camouflage_test_attacks)}")
    print(f"Output  : {args.output_graph}")
    print("Train   :", mask_counts(graph.y, graph.train_mask))
    print("Val     :", mask_counts(graph.y, graph.val_mask))
    print("Test    :", mask_counts(graph.y, graph.test_mask))
    print("=" * 70)


if __name__ == "__main__":
    main()
