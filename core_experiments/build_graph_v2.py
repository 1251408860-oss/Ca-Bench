#!/usr/bin/env python3
"""
Phase 2: Cross-Flow Spatiotemporal Heterogeneous Graph Construction

Input:  full_arena_v2.pcap
Output: st_graph.pt

Per-window flow node features (7D):
  [ln(N+1), ln(T+1), entropy, D_observed, pkt_rate, avg_pkt_size, port_diversity]

`D_observed` is the measured request-to-first-response latency proxy when available,
with connection-level fallback. It is no longer average packet inter-arrival time.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter, defaultdict, deque
from typing import Any

import torch

try:
    from torch_geometric.data import Data
except ImportError:
    print("ERROR: torch_geometric required")
    sys.exit(1)

try:
    from scapy.all import IP, TCP, Raw, rdpcap
except ImportError:
    print("ERROR: scapy required")
    sys.exit(1)

from internal.submission_common import (
    load_and_repair_manifest,
    manifest_capacity_bytes_per_sec,
    manifest_core_bw_mbps,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PCAP_FILE = os.path.join(BASE_DIR, "full_arena_v2.pcap")
OUTPUT_FILE = os.path.join(BASE_DIR, "st_graph.pt")
MANIFEST_FILE = os.path.join(BASE_DIR, "arena_manifest_v2.json")

DELTA_T = 1.0
TARGET_IP = "10.0.0.100"
REPAIR_MANIFEST_IN_PLACE = False

FEATURE_NAMES = [
    "ln(N+1)",
    "ln(T+1)",
    "entropy",
    "D_observed",
    "pkt_rate",
    "avg_pkt_size",
    "port_diversity",
]
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}

HTTP_METHOD_PREFIXES = (b"GET ", b"POST ", b"HEAD ", b"PUT ", b"DELETE ", b"OPTIONS ", b"PATCH ")


def shannon_entropy(data_bytes: bytes) -> float:
    if not data_bytes:
        return 0.0
    counts = Counter(data_bytes)
    n = len(data_bytes)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def classify_ip_fallback(ip: str) -> int:
    last = int(ip.split(".")[-1])
    if 30 <= last <= 99:
        return 1
    return 0


def label_ip(ip: str, manifest_labels: dict[str, int]) -> int:
    if ip in manifest_labels:
        return int(manifest_labels[ip])
    return classify_ip_fallback(ip)


def packet_fingerprint(pkt: Any) -> tuple[str, int, int, int, int] | None:
    if IP not in pkt or TCP not in pkt:
        return None
    return (
        str(pkt[IP].src),
        int(pkt[TCP].sport),
        int(pkt[TCP].dport),
        int(pkt[TCP].seq),
        int(round(float(pkt.time) * 1_000_000.0)),
    )


def connection_key_from_inbound(pkt: Any) -> tuple[str, int, int]:
    return str(pkt[IP].src), int(pkt[TCP].sport), int(pkt[TCP].dport)


def connection_key_from_outbound(pkt: Any) -> tuple[str, int, int]:
    return str(pkt[IP].dst), int(pkt[TCP].dport), int(pkt[TCP].sport)


def is_http_request_payload(payload: bytes) -> bool:
    return payload.startswith(HTTP_METHOD_PREFIXES)


def is_http_response_payload(payload: bytes) -> bool:
    return payload.startswith(b"HTTP/")


def compute_response_delay_maps(packets: list[Any]) -> tuple[dict[tuple[str, int, int, int, int], float], dict[tuple[str, int, int], float], dict[str, Any]]:
    pending: dict[tuple[str, int, int], deque[tuple[tuple[str, int, int, int, int], float]]] = defaultdict(deque)
    delay_by_packet: dict[tuple[str, int, int, int, int], float] = {}
    conn_delays: dict[tuple[str, int, int], list[float]] = defaultdict(list)

    request_count = 0
    matched_count = 0

    for pkt in packets:
        if IP not in pkt or TCP not in pkt or Raw not in pkt:
            continue

        payload = bytes(pkt[Raw].load)
        if not payload:
            continue

        if pkt[IP].dst == TARGET_IP and pkt[IP].src != TARGET_IP and is_http_request_payload(payload):
            key = connection_key_from_inbound(pkt)
            fp = packet_fingerprint(pkt)
            if fp is not None:
                pending[key].append((fp, float(pkt.time)))
                request_count += 1
            continue

        if pkt[IP].src == TARGET_IP and pkt[IP].dst != TARGET_IP and is_http_response_payload(payload):
            key = connection_key_from_outbound(pkt)
            if pending[key]:
                fp, t0 = pending[key].popleft()
                delay = max(0.0, float(pkt.time) - t0)
                delay_by_packet[fp] = delay
                conn_delays[key].append(delay)
                matched_count += 1

    conn_delay_mean = {k: (sum(v) / len(v)) for k, v in conn_delays.items() if v}
    samples = list(delay_by_packet.values())
    stats = {
        "delay_metric": "http_request_to_first_response_sec",
        "request_count": int(request_count),
        "matched_count": int(matched_count),
        "match_rate": float(matched_count / max(request_count, 1)),
        "mean_delay_sec": float(sum(samples) / len(samples)) if samples else 0.0,
        "median_delay_sec": float(sorted(samples)[len(samples) // 2]) if samples else 0.0,
    }
    return delay_by_packet, conn_delay_mean, stats


def extract_window_features(
    pkts: list[Any],
    delay_by_packet: dict[tuple[str, int, int, int, int], float],
    conn_delay_mean: dict[tuple[str, int, int], float],
) -> list[float]:
    if not pkts:
        return [0.0] * len(FEATURE_NAMES)

    n_packets = len(pkts)
    total_bytes = sum(len(p) for p in pkts)

    ts = sorted(float(p.time) for p in pkts)
    duration = max(ts[-1] - ts[0], 0.0) if len(ts) > 1 else 0.0

    payload = b""
    request_delays: list[float] = []
    for p in pkts:
        if Raw in p:
            raw = bytes(p[Raw].load)
            payload += raw
            if IP in p and TCP in p and p[IP].dst == TARGET_IP and is_http_request_payload(raw):
                fp = packet_fingerprint(p)
                key = connection_key_from_inbound(p)
                if fp is not None and fp in delay_by_packet:
                    request_delays.append(delay_by_packet[fp])
                elif key in conn_delay_mean:
                    request_delays.append(conn_delay_mean[key])

    entropy = shannon_entropy(payload)
    d_observed = (sum(request_delays) / len(request_delays)) if request_delays else 0.0

    pkt_rate = n_packets / max(duration, 0.001)
    avg_pkt_size = total_bytes / max(n_packets, 1)

    src_ports = set()
    for p in pkts:
        if TCP in p:
            src_ports.add(int(p[TCP].sport))
    port_diversity = len(src_ports) / max(n_packets, 1)

    return [
        math.log(total_bytes + 1.0),
        math.log(duration + 1.0),
        entropy,
        d_observed,
        pkt_rate,
        avg_pkt_size / 1000.0,
        port_diversity,
    ]


def build_spatiotemporal_graph() -> None:
    print("=" * 68)
    print("Phase 2: Build Graph from PCAP")
    print("=" * 68)

    if not os.path.exists(PCAP_FILE):
        raise FileNotFoundError(f"PCAP not found: {PCAP_FILE}")

    manifest, manifest_issues = load_and_repair_manifest(
        MANIFEST_FILE,
        target_ip=TARGET_IP,
        write_back=REPAIR_MANIFEST_IN_PLACE,
    )
    manifest_labels = manifest.get("ip_labels", {}) if isinstance(manifest.get("ip_labels", {}), dict) else {}

    print(f"[1/5] Load PCAP: {PCAP_FILE}")
    packets = rdpcap(PCAP_FILE)
    print(f"  packets={len(packets)}")

    delay_by_packet, conn_delay_mean, delay_stats = compute_response_delay_maps(list(packets))
    print(
        "  delay_metric={delay_metric} requests={request_count} matched={matched_count} match_rate={match_rate:.3f}".format(
            **delay_stats
        )
    )

    inbound = []
    for p in packets:
        if IP in p and TCP in p and p[IP].dst == TARGET_IP and p[IP].src != TARGET_IP:
            inbound.append(p)

    if not inbound:
        raise RuntimeError("No inbound TCP packets to target found")

    times = [float(p.time) for p in inbound]
    t_start = min(times)
    t_end = max(times)
    duration = max(t_end - t_start, 0.0)
    n_windows = max(1, int(math.ceil(duration / DELTA_T)))

    print(f"[2/5] Inbound packets={len(inbound)} duration={duration:.1f}s windows={n_windows}")

    window_packets: dict[tuple[str, int], list[Any]] = defaultdict(list)
    target_window_packets: dict[int, list[Any]] = defaultdict(list)

    for p in inbound:
        src = str(p[IP].src)
        w = min(int((float(p.time) - t_start) / DELTA_T), n_windows - 1)
        window_packets[(src, w)].append(p)
        target_window_packets[w].append(p)

    source_ips = sorted({ip for ip, _ in window_packets.keys()})
    print(f"  source_ips={len(source_ips)}")
    print(f"  labels={'manifest' if manifest_labels else 'fallback(ip range)'}")
    if manifest_issues:
        print(f"  manifest_repairs={len(manifest_issues)}")

    print("[3/5] Build nodes and edges")
    node_features: list[list[float]] = []
    node_labels: list[int] = []
    node_window_idx: list[int] = []
    node_ip_idx: list[int] = []

    spatial_src: list[int] = []
    spatial_dst: list[int] = []
    temporal_src: list[int] = []
    temporal_dst: list[int] = []

    all_target_pkts = [p for v in target_window_packets.values() for p in v]
    node_features.append(extract_window_features(all_target_pkts, delay_by_packet, conn_delay_mean))
    node_labels.append(0)
    node_window_idx.append(-1)
    node_ip_idx.append(-1)

    flow_node_map: dict[tuple[str, int], int] = {}
    for w in range(n_windows):
        for ip_i, src_ip in enumerate(source_ips):
            pkts = window_packets.get((src_ip, w))
            if not pkts:
                continue

            node_id = len(node_features)
            flow_node_map[(src_ip, w)] = node_id

            node_features.append(extract_window_features(pkts, delay_by_packet, conn_delay_mean))
            node_labels.append(label_ip(src_ip, manifest_labels))
            node_window_idx.append(w)
            node_ip_idx.append(ip_i)

            spatial_src.append(node_id)
            spatial_dst.append(0)

            prev = flow_node_map.get((src_ip, w - 1))
            if prev is not None:
                temporal_src.append(prev)
                temporal_dst.append(node_id)

    n_nodes = len(node_features)
    n_spatial = len(spatial_src)
    n_temporal = len(temporal_src)
    print(f"  nodes={n_nodes} flow_nodes={n_nodes - 1} edges={n_spatial + n_temporal}")

    print("[4/5] Assemble tensors")
    all_src = spatial_src + temporal_src
    all_dst = spatial_dst + temporal_dst
    edge_type = [0] * n_spatial + [1] * n_temporal

    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(node_labels, dtype=torch.long)
    edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    edge_type_t = torch.tensor(edge_type, dtype=torch.long)

    window_idx = torch.tensor(node_window_idx, dtype=torch.long)
    ip_idx = torch.tensor(node_ip_idx, dtype=torch.long)

    bi_src = all_src + all_dst
    bi_dst = all_dst + all_src
    bi_type = edge_type + edge_type

    flow_mask = torch.arange(n_nodes) > 0
    x_flow = x[flow_mask]
    feat_mean = x_flow.mean(dim=0)
    feat_std = x_flow.std(dim=0).clamp(min=1e-6)
    x_norm = (x - feat_mean) / feat_std
    x_norm[0] = 0.0

    n_flow = n_nodes - 1
    perm = torch.randperm(n_flow) + 1
    n_train = int(0.70 * n_flow)
    n_val = int(0.15 * n_flow)

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

    temporal_train = torch.zeros(n_nodes, dtype=torch.bool)
    temporal_test = torch.zeros(n_nodes, dtype=torch.bool)
    cutoff = int(0.7 * n_windows)
    for i, w in enumerate(node_window_idx):
        if w < 0:
            continue
        if w < cutoff:
            temporal_train[i] = True
        else:
            temporal_test[i] = True

    capacity_bps = manifest_capacity_bytes_per_sec(manifest) if manifest else None
    core_bw_mbps = manifest_core_bw_mbps(manifest) if manifest else None

    graph = Data(
        x=x,
        x_norm=x_norm,
        y=y,
        edge_index=edge_index,
        edge_type=edge_type_t,
        edge_index_undirected=torch.tensor([bi_src, bi_dst], dtype=torch.long),
        edge_type_undirected=torch.tensor(bi_type, dtype=torch.long),
        window_idx=window_idx,
        ip_idx=ip_idx,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        temporal_train_mask=temporal_train,
        temporal_test_mask=temporal_test,
        feat_mean=feat_mean,
        feat_std=feat_std,
    )

    graph.source_ips = source_ips
    graph.target_ip = TARGET_IP
    graph.delta_t = DELTA_T
    graph.n_windows = n_windows
    graph.feature_names = FEATURE_NAMES
    graph.feature_index = FEATURE_INDEX
    graph.label_source = "manifest" if manifest_labels else "fallback_ip_range"
    graph.delay_metric = delay_stats["delay_metric"]
    graph.delay_match_rate = float(delay_stats["match_rate"])
    graph.delay_request_count = int(delay_stats["request_count"])
    graph.delay_matched_count = int(delay_stats["matched_count"])
    graph.manifest_file = MANIFEST_FILE
    graph.manifest_issue_count = len(manifest_issues)
    graph.manifest_issues = manifest_issues
    graph.manifest_core_bw_mbps = float(core_bw_mbps) if core_bw_mbps is not None else 0.0
    graph.capacity_bytes_per_sec = float(capacity_bps) if capacity_bps is not None else 0.0

    print(f"[5/5] Save: {OUTPUT_FILE}")
    torch.save(graph, OUTPUT_FILE)

    benign = int(((y == 0) & flow_mask).sum().item())
    attack = int(((y == 1) & flow_mask).sum().item())

    print("=" * 68)
    print(f"Done: nodes={graph.num_nodes} edges={graph.num_edges} windows={n_windows}")
    print(f"Flow labels: benign={benign} attack={attack}")
    print(f"Features ({len(FEATURE_NAMES)}D): {FEATURE_NAMES}")
    print(f"feature_index={FEATURE_INDEX}")
    print(f"label_source={graph.label_source}")
    print(f"delay_match_rate={graph.delay_match_rate:.3f}")
    if graph.capacity_bytes_per_sec > 0:
        print(
            f"capacity_bytes_per_sec={graph.capacity_bytes_per_sec:.2f} "
            f"(manifest_core_bw_mbps={graph.manifest_core_bw_mbps:.2f})"
        )
    print("=" * 68)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build spatiotemporal graph from PCAP")
    parser.add_argument("--pcap-file", default=PCAP_FILE)
    parser.add_argument("--output-file", default=OUTPUT_FILE)
    parser.add_argument("--manifest-file", default=MANIFEST_FILE)
    parser.add_argument("--delta-t", type=float, default=DELTA_T)
    parser.add_argument("--target-ip", default=TARGET_IP)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repair-manifest-in-place", action="store_true")
    return parser.parse_args()


def main() -> None:
    global PCAP_FILE, OUTPUT_FILE, MANIFEST_FILE, DELTA_T, TARGET_IP, REPAIR_MANIFEST_IN_PLACE
    args = parse_args()
    PCAP_FILE = os.path.abspath(os.path.expanduser(args.pcap_file))
    OUTPUT_FILE = os.path.abspath(os.path.expanduser(args.output_file))
    MANIFEST_FILE = os.path.abspath(os.path.expanduser(args.manifest_file))
    DELTA_T = float(args.delta_t)
    TARGET_IP = str(args.target_ip)
    REPAIR_MANIFEST_IN_PLACE = bool(args.repair_manifest_in_place)

    torch.manual_seed(int(args.seed))
    build_spatiotemporal_graph()


if __name__ == "__main__":
    main()
