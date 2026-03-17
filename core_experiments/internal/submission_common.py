from __future__ import annotations

import copy
import json
import os
from typing import Any


DEFAULT_TARGET_IP = "10.0.0.100"
DEFAULT_CORE_BW_MBPS = 10.0


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def save_json(path: str, obj: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def infer_label_from_role(role: str) -> int | None:
    role_s = str(role).strip().lower()
    if role_s == "target":
        return 0
    if role_s == "benign_user":
        return 0
    if role_s.startswith("bot:"):
        return 1
    return None


def audit_and_repair_manifest(data: dict[str, Any], target_ip: str | None = None) -> tuple[dict[str, Any], list[str]]:
    repaired = copy.deepcopy(data)
    issues: list[str] = []

    topology = repaired.get("topology")
    if not isinstance(topology, dict):
        topology = {}
        repaired["topology"] = topology

    resolved_target = str(
        target_ip
        or topology.get("target_ip")
        or repaired.get("target_ip")
        or DEFAULT_TARGET_IP
    )
    topology["target_ip"] = resolved_target

    ip_labels = repaired.get("ip_labels")
    if not isinstance(ip_labels, dict):
        ip_labels = {}
        repaired["ip_labels"] = ip_labels

    roles = repaired.get("roles")
    if not isinstance(roles, dict):
        roles = {}
        repaired["roles"] = roles

    if str(roles.get(resolved_target, "")) != "target":
        issues.append(f"target role repaired for {resolved_target}")
        roles[resolved_target] = "target"

    if _safe_int(ip_labels.get(resolved_target)) != 0:
        issues.append(f"target label repaired for {resolved_target}")
        ip_labels[resolved_target] = 0

    for ip, role in list(roles.items()):
        if str(ip) == resolved_target:
            continue
        inferred = infer_label_from_role(str(role))
        if inferred is None:
            continue
        current = _safe_int(ip_labels.get(ip))
        if current != inferred:
            issues.append(f"label repaired for {ip}: {current} -> {inferred}")
            ip_labels[ip] = inferred

    return repaired, issues


def load_and_repair_manifest(
    path: str,
    target_ip: str | None = None,
    write_back: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    if not path or not os.path.exists(path):
        return {}, []

    data = load_json(path)
    repaired, issues = audit_and_repair_manifest(data, target_ip=target_ip)
    if write_back and issues:
        save_json(path, repaired)
    return repaired, issues


def manifest_core_bw_mbps(manifest: dict[str, Any]) -> float | None:
    topology = manifest.get("topology", {})
    if not isinstance(topology, dict):
        return None
    bottleneck = topology.get("core_bottleneck", {})
    if not isinstance(bottleneck, dict):
        return None
    try:
        bw = float(bottleneck.get("bw_mbps"))
    except Exception:
        return None
    if bw <= 0:
        return None
    return bw


def manifest_capacity_bytes_per_sec(manifest: dict[str, Any]) -> float | None:
    bw = manifest_core_bw_mbps(manifest)
    if bw is None:
        return None
    return bw * 1_000_000.0 / 8.0


def resolve_capacity_bytes_per_sec(
    capacity: float,
    mode: str = "auto",
    graph: Any | None = None,
    manifest: dict[str, Any] | None = None,
) -> tuple[float, str]:
    mode_s = str(mode).strip().lower()
    if mode_s not in {"auto", "mbps", "bytes_per_sec"}:
        mode_s = "auto"

    if mode_s == "mbps":
        return max(float(capacity), 0.0) * 1_000_000.0 / 8.0, f"cli_mbps:{float(capacity):g}"

    if mode_s == "bytes_per_sec":
        return max(float(capacity), 0.0), f"cli_bytes_per_sec:{float(capacity):g}"

    if float(capacity) > 0.0:
        if float(capacity) <= 1000.0:
            return float(capacity) * 1_000_000.0 / 8.0, f"cli_auto_mbps:{float(capacity):g}"
        return float(capacity), f"cli_auto_bytes_per_sec:{float(capacity):g}"

    if graph is not None:
        if hasattr(graph, "capacity_bytes_per_sec"):
            try:
                value = float(getattr(graph, "capacity_bytes_per_sec"))
                if value > 0:
                    return value, "graph.capacity_bytes_per_sec"
            except Exception:
                pass
        if hasattr(graph, "manifest_core_bw_mbps"):
            try:
                bw = float(getattr(graph, "manifest_core_bw_mbps"))
                if bw > 0:
                    return bw * 1_000_000.0 / 8.0, "graph.manifest_core_bw_mbps"
            except Exception:
                pass

    if manifest:
        value = manifest_capacity_bytes_per_sec(manifest)
        if value is not None and value > 0:
            return value, "manifest.core_bottleneck.bw_mbps"

    return DEFAULT_CORE_BW_MBPS * 1_000_000.0 / 8.0, f"fallback_default_mbps:{DEFAULT_CORE_BW_MBPS:g}"
