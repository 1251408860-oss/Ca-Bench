#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch


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


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare strict anti-leakage graph split")
    p.add_argument("--input-graph", required=True)
    p.add_argument("--output-graph", required=True)
    p.add_argument("--protocol", choices=["temporal_ood", "topology_ood", "attack_strategy_ood"], required=True)
    p.add_argument("--manifest-file", default="")
    p.add_argument("--holdout-attack-type", default="mimic")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(int(args.seed))
    graph = torch.load(args.input_graph, weights_only=False, map_location="cpu")
    manifest = load_manifest(args.manifest_file)

    flow_mask = graph.window_idx >= 0 if hasattr(graph, "window_idx") else (torch.arange(graph.num_nodes) > 0)

    if args.protocol == "temporal_ood":
        train_mask, val_mask, test_mask = protocol_temporal_ood(graph, flow_mask)
    elif args.protocol == "topology_ood":
        train_mask, val_mask, test_mask = protocol_topology_ood(graph, flow_mask)
    else:
        train_mask, val_mask, test_mask = protocol_attack_strategy_ood(
            graph, manifest=manifest, flow_mask=flow_mask, holdout_attack_type=args.holdout_attack_type.strip().lower()
        )

    graph.train_mask = train_mask.bool()
    graph.val_mask = val_mask.bool()
    graph.test_mask = test_mask.bool()
    graph.temporal_test_mask = test_mask.bool()
    graph.split_protocol = args.protocol
    graph.holdout_attack_type = args.holdout_attack_type.strip().lower()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_graph)), exist_ok=True)
    torch.save(graph, args.output_graph)

    print("=" * 70)
    print(f"Protocol: {args.protocol}")
    print(f"Output  : {args.output_graph}")
    print("Train   :", mask_counts(graph.y, graph.train_mask))
    print("Val     :", mask_counts(graph.y, graph.val_mask))
    print("Test    :", mask_counts(graph.y, graph.test_mask))
    print("=" * 70)


if __name__ == "__main__":
    main()
