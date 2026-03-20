#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_baselines import export_pi_predictions
from make_paper_tables_figs import (
    PALETTE,
    add_panel_label,
    display_name,
    draw_line_with_band,
    model_color,
    paper_heatmap,
    save_figure,
    short_scenario,
    stage_name,
    stylize_axes,
    use_paper_style,
)


REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT.parent / "mininet_testbed" / "real_collection"
RUN_ROOT = REPO_ROOT.parent / "paper_artifacts" / "runs"
NETWORK_SCENARIOS_DEFAULT = (
    "scenario_d_three_tier_low2,"
    "scenario_e_three_tier_high2,"
    "scenario_f_two_tier_high2,"
    "scenario_g_mimic_congest,"
    "scenario_h_mimic_heavy_overlap"
)
NETWORK_MODELS = ["data_only", "random_forest", "gcn", "graphsage", "gatv2"]


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def run_cmd(cmd: list[str], cwd: Path, log_file: Path, skip_if_exists: Path | None = None) -> float:
    if skip_if_exists is not None and skip_if_exists.exists():
        return 0.0
    log_file.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log_file.open("w", encoding="utf-8") as f:
        result = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"command failed, see {log_file}")
    return float(time.time() - t0)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0, "mean": 0.0, "std": 0.0}
    if len(vals) == 1:
        return {"n": 1, "mean": float(vals[0]), "std": 0.0}
    return {"n": len(vals), "mean": float(statistics.mean(vals)), "std": float(statistics.stdev(vals))}


def parse_delay_ms(value: Any) -> float:
    text = str(value).strip().lower()
    if text.endswith("ms"):
        text = text[:-2]
    try:
        return float(text)
    except Exception:
        return 0.0


def manifest_metadata(manifest_file: Path) -> dict[str, Any]:
    data = read_json(manifest_file)
    top = data.get("topology", {}) if isinstance(data.get("topology"), dict) else {}
    core = top.get("core_bottleneck", {}) if isinstance(top.get("core_bottleneck"), dict) else {}
    run = data.get("run_config", {}) if isinstance(data.get("run_config"), dict) else {}
    return {
        "topology_type": str(top.get("type", "")),
        "users": int(top.get("users", 0) or 0),
        "bots": int(top.get("bots", 0) or 0),
        "core_bw_mbps": float(core.get("bw_mbps", 0.0) or 0.0),
        "delay_ms": parse_delay_ms(core.get("delay", 0.0)),
        "max_queue_size": int(core.get("max_queue_size", 0) or 0),
        "duration_sec": int(run.get("duration_sec", 0) or 0),
        "load_profile": str(run.get("load_profile", "")),
        "bot_type_mode": str(run.get("bot_type_mode", "")),
    }


def pick_phase3_test_metrics(results_path: Path) -> dict[str, float]:
    data = read_json(results_path)
    block = data.get("final_eval", {}).get("test_temporal") or data.get("final_eval", {}).get("test_random") or {}
    out = dict(block)
    fp = int(out.get("fp", 0))
    tn = int(out.get("tn", 0))
    out["fpr"] = float(fp / max(fp + tn, 1))
    return out


def pick_fed_test_metrics(results_file: Path) -> dict[str, float]:
    data = read_json(results_file)
    block = data.get("global_metrics", {}).get("test_temporal") or data.get("global_metrics", {}).get("test_random") or {}
    return {
        "f1": float(block.get("f1", 0.0)),
        "recall": float(block.get("recall", 0.0)),
        "fpr": float(block.get("fpr", 0.0)),
    }


def best_nonphysics(stat_block: dict[str, Any]) -> str:
    return max(["random_forest", "gcn", "graphsage", "gatv2"], key=lambda m: float(stat_block[m]["f1"]["mean"]))


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    xm = statistics.mean(x)
    ym = statistics.mean(y)
    num = sum((a - xm) * (b - ym) for a, b in zip(x, y))
    den_x = sum((a - xm) ** 2 for a in x) ** 0.5
    den_y = sum((b - ym) ** 2 for b in y) ** 0.5
    if den_x <= 1e-12 or den_y <= 1e-12:
        return 0.0
    return float(num / (den_x * den_y))


def partition_summary(graph_file: Path, num_clients: int) -> dict[str, Any]:
    graph = torch.load(graph_file, weights_only=False, map_location="cpu")
    train_mask = graph.train_mask.bool()
    flow_mask = graph.ip_idx >= 0
    rows = []
    sizes = []
    pos_ratios = []
    for cid in range(num_clients):
        mask = flow_mask & train_mask & ((graph.ip_idx % num_clients) == cid)
        n = int(mask.sum().item())
        benign = int(((graph.y == 0) & mask).sum().item())
        attack = int(((graph.y == 1) & mask).sum().item())
        rows.append({"client_id": cid, "train_nodes": n, "benign": benign, "attack": attack})
        if n > 0:
            sizes.append(float(n))
            pos_ratios.append(float(attack / max(n, 1)))
    size_mean = statistics.mean(sizes) if sizes else 0.0
    size_cv = float(statistics.stdev(sizes) / size_mean) if len(sizes) > 1 and size_mean > 0 else 0.0
    pos_std = float(statistics.stdev(pos_ratios)) if len(pos_ratios) > 1 else 0.0
    return {"clients": rows, "train_size_cv": size_cv, "attack_ratio_std": pos_std}


def graph_stats(graph_file: Path) -> dict[str, Any]:
    graph = torch.load(graph_file, weights_only=False, map_location="cpu")
    flow_mask = graph.window_idx >= 0 if hasattr(graph, "window_idx") else torch.ones(graph.num_nodes, dtype=torch.bool)
    return {
        "num_nodes": int(graph.num_nodes),
        "num_edges": int(graph.num_edges),
        "flow_nodes": int(flow_mask.sum().item()),
        "benign_flow_nodes": int(((graph.y == 0) & flow_mask).sum().item()),
        "attack_flow_nodes": int(((graph.y == 1) & flow_mask).sum().item()),
    }


def run_network_sensitivity(
    *,
    project: Path,
    python_bin: str,
    real_collection_dir: Path,
    output_dir: Path,
    scenarios: str,
    seeds: str,
    epochs: int,
    skip_existing: bool,
) -> None:
    use_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "graphs").mkdir(parents=True, exist_ok=True)
    (output_dir / "protocol_graphs").mkdir(parents=True, exist_ok=True)
    scenario_list = parse_csv(scenarios)
    seed_list = [int(x.strip()) for x in str(seeds).split(",") if x.strip()]

    summary: dict[str, Any] = {
        "config": {
            "project_dir": str(project),
            "python_bin": python_bin,
            "real_collection_dir": str(real_collection_dir),
            "output_dir": str(output_dir),
            "scenarios": scenarios,
            "seeds": seeds,
            "epochs": epochs,
            "skip_existing": skip_existing,
        },
        "rows": [],
        "scenario_stats": {},
        "group_stats": {"load_profile": {}, "topology_type": {}},
        "correlations": {},
    }

    for scenario_name in scenario_list:
        scenario_dir = real_collection_dir / scenario_name
        manifest_file = scenario_dir / "arena_manifest_v2.json"
        graph_file = output_dir / "graphs" / f"{scenario_name}.pt"
        protocol_graph = output_dir / "protocol_graphs" / f"{scenario_name}__congestion_hard.pt"

        build_sec = run_cmd(
            [
                python_bin,
                "build_graph_v2.py",
                "--pcap-file",
                str(scenario_dir / "full_arena_v2.pcap"),
                "--manifest-file",
                str(manifest_file),
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
            log_file=output_dir / "logs" / f"build_{scenario_name}.log",
            skip_if_exists=graph_file if skip_existing else None,
        )
        protocol_sec = run_cmd(
            [
                python_bin,
                "prepare_hard_protocol_graph.py",
                "--input-graph",
                str(graph_file),
                "--output-graph",
                str(protocol_graph),
                "--protocol",
                "congestion_ood",
                "--manifest-file",
                str(manifest_file),
                "--hard-overlap",
                "--train-keep-frac",
                "0.80",
                "--val-keep-frac",
                "0.90",
                "--test-keep-frac",
                "0.90",
                "--min-keep-per-class",
                "64",
                "--seed",
                "42",
            ],
            cwd=project,
            log_file=output_dir / "logs" / f"prep_{scenario_name}.log",
            skip_if_exists=protocol_graph if skip_existing else None,
        )

        metadata = manifest_metadata(manifest_file)
        rows_data: list[dict[str, Any]] = []
        rows_base: list[dict[str, Any]] = []
        for seed in seed_list:
            stage_dir = output_dir / "stage3" / f"{scenario_name}__data_only__seed{seed}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            stage_results = stage_dir / "phase3_results.json"
            stage_model = stage_dir / "pi_gnn_model.pt"
            train_sec = run_cmd(
                [
                    python_bin,
                    "pi_gnn_train_v2.py",
                    "--graph-file",
                    str(protocol_graph),
                    "--model-file",
                    str(stage_model),
                    "--results-file",
                    str(stage_results),
                    "--epochs",
                    str(epochs),
                    "--alpha-flow",
                    "0.0",
                    "--beta-latency",
                    "0.0",
                    "--capacity",
                    "0",
                    "--capacity-mode",
                    "auto",
                    "--warmup-epochs",
                    "20",
                    "--patience",
                    "30",
                    "--seed",
                    str(seed),
                    "--force-cpu",
                ],
                cwd=project,
                log_file=stage_dir / "run.log",
                skip_if_exists=stage_results if skip_existing else None,
            )
            base_dir = output_dir / "baseline" / f"{scenario_name}__seed{seed}"
            base_dir.mkdir(parents=True, exist_ok=True)
            baseline_file = base_dir / "baseline_eval.json"
            baseline_sec = run_cmd(
                [
                    python_bin,
                    "evaluate_baselines.py",
                    "--graph-file",
                    str(protocol_graph),
                    "--pi-model-file",
                    str(stage_model),
                    "--pi-results-file",
                    str(stage_results),
                    "--output-file",
                    str(baseline_file),
                    "--seed",
                    str(seed),
                    "--force-cpu",
                ],
                cwd=project,
                log_file=base_dir / "run.log",
                skip_if_exists=baseline_file if skip_existing else None,
            )
            data_metrics = pick_phase3_test_metrics(stage_results)
            base_metrics = read_json(baseline_file).get("metrics", {})
            rows_data.append({"seed": seed, "metrics": data_metrics, "train_duration_sec": train_sec})
            rows_base.append({"seed": seed, "metrics": base_metrics, "baseline_duration_sec": baseline_sec})
            summary["rows"].append(
                {
                    "scenario": scenario_name,
                    "seed": seed,
                    "metadata": metadata,
                    "data_only": data_metrics,
                    "baselines": base_metrics,
                }
            )

        stat_block: dict[str, Any] = {"metadata": metadata, "durations": {"build_sec": build_sec, "protocol_sec": protocol_sec}}
        stat_block["data_only"] = {
            "f1": mean_std([float(r["metrics"].get("f1", 0.0)) for r in rows_data]),
            "recall": mean_std([float(r["metrics"].get("recall", 0.0)) for r in rows_data]),
            "fpr": mean_std([float(r["metrics"].get("fpr", 0.0)) for r in rows_data]),
            "train_duration_sec": mean_std([float(r["train_duration_sec"]) for r in rows_data]),
        }
        for model_name in ["random_forest", "gcn", "graphsage", "gatv2"]:
            stat_block[model_name] = {
                "f1": mean_std([float(r["metrics"].get(model_name, {}).get("f1", 0.0)) for r in rows_base]),
                "recall": mean_std([float(r["metrics"].get(model_name, {}).get("recall", 0.0)) for r in rows_base]),
                "fpr": mean_std([float(r["metrics"].get(model_name, {}).get("fpr", 0.0)) for r in rows_base]),
                "eval_duration_sec": mean_std([float(r["baseline_duration_sec"]) for r in rows_base]),
            }
        stat_block["best_nonphysics_baseline"] = best_nonphysics(stat_block)
        summary["scenario_stats"][scenario_name] = stat_block

    for group_key in ["load_profile", "topology_type"]:
        groups: dict[str, dict[str, list[float]]] = {}
        for scenario_name, block in summary["scenario_stats"].items():
            group_value = str(block["metadata"].get(group_key, "unknown"))
            groups.setdefault(group_value, {m: [] for m in NETWORK_MODELS})
            groups[group_value]["data_only"].append(float(block["data_only"]["f1"]["mean"]))
            for model_name in ["random_forest", "gcn", "graphsage", "gatv2"]:
                groups[group_value][model_name].append(float(block[model_name]["f1"]["mean"]))
        for group_value, vals in groups.items():
            summary["group_stats"][group_key][group_value] = {m: mean_std(v) for m, v in vals.items()}

    bw = [float(block["metadata"]["core_bw_mbps"]) for block in summary["scenario_stats"].values()]
    delay = [float(block["metadata"]["delay_ms"]) for block in summary["scenario_stats"].values()]
    queue = [float(block["metadata"]["max_queue_size"]) for block in summary["scenario_stats"].values()]
    data_f1 = [float(block["data_only"]["f1"]["mean"]) for block in summary["scenario_stats"].values()]
    rf_f1 = [float(block["random_forest"]["f1"]["mean"]) for block in summary["scenario_stats"].values()]
    summary["correlations"] = {
        "core_bw_mbps_vs_data_only_f1": pearson(bw, data_f1),
        "delay_ms_vs_data_only_f1": pearson(delay, data_f1),
        "max_queue_size_vs_data_only_f1": pearson(queue, data_f1),
        "core_bw_mbps_vs_random_forest_f1": pearson(bw, rf_f1),
    }

    order = [scenario_name for scenario_name in scenario_list if scenario_name in summary["scenario_stats"]]
    scenario_labels = [short_scenario(s) for s in order]
    x = np.arange(len(order))

    best_baseline_vals = []
    best_baseline_stds = []
    for scenario_name in order:
        best_name = str(summary["scenario_stats"][scenario_name]["best_nonphysics_baseline"])
        best_baseline_vals.append(float(summary["scenario_stats"][scenario_name][best_name]["f1"]["mean"]))
        best_baseline_stds.append(float(summary["scenario_stats"][scenario_name][best_name]["f1"]["std"]))

    heatmap_models = ["data_only", "random_forest", "gcn", "graphsage", "gatv2"]
    heatmap_data = np.asarray(
        [[float(summary["scenario_stats"][scenario_name][model]["f1"]["mean"]) for scenario_name in order] for model in heatmap_models],
        dtype=float,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.6), gridspec_kw={"width_ratios": [1.2, 1.0]})
    stylize_axes(ax1, grid_axis="y")
    draw_line_with_band(
        ax1,
        x=x,
        y=[summary["scenario_stats"][scenario_name]["data_only"]["f1"]["mean"] for scenario_name in order],
        yerr=[summary["scenario_stats"][scenario_name]["data_only"]["f1"]["std"] for scenario_name in order],
        label="anchor GNN",
        color=model_color("data_only"),
    )
    draw_line_with_band(
        ax1,
        x=x,
        y=best_baseline_vals,
        yerr=best_baseline_stds,
        label="best baseline",
        color=PALETTE["rose"],
        marker="s",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_labels)
    ax1.set_ylim(0.94, 1.005)
    ax1.set_ylabel("F1")
    ax1.set_title("Sensitivity Across Network Conditions")
    ax1.legend(loc="lower left")
    add_panel_label(ax1, "a")

    paper_heatmap(ax2, heatmap_data, [display_name(model) for model in heatmap_models], scenario_labels)
    ax2.set_title("Per-Scenario F1 Heatmap")
    add_panel_label(ax2, "b")
    save_figure(fig, output_dir / "fig_network_condition_sensitivity.png")

    out_json = output_dir / "network_sensitivity_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# WASA Network Sensitivity Summary",
        "",
        "## Per Scenario",
    ]
    for scenario_name in order:
        block = summary["scenario_stats"][scenario_name]
        meta = block["metadata"]
        lines.append(
            f"- {scenario_name}: topology={meta['topology_type']}, load={meta['load_profile']}, "
            f"bw={meta['core_bw_mbps']}, delay_ms={meta['delay_ms']}, queue={meta['max_queue_size']}, "
            f"data_only_f1={block['data_only']['f1']['mean']:.4f}, "
            f"rf_f1={block['random_forest']['f1']['mean']:.4f}, "
            f"gcn_f1={block['gcn']['f1']['mean']:.4f}"
        )
    lines.extend(["", "## Correlations"])
    for key, value in summary["correlations"].items():
        lines.append(f"- {key}: {value:.4f}")
    (output_dir / "network_sensitivity_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_edge_budget(
    *,
    project: Path,
    python_bin: str,
    real_collection_dir: Path,
    output_dir: Path,
    scenario: str,
    seeds: str,
    aggregators: str,
    rounds_list: str,
    num_clients_list: str,
    local_epochs: int,
    skip_existing: bool,
) -> None:
    use_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "graphs").mkdir(parents=True, exist_ok=True)

    seed_list = [int(x.strip()) for x in str(seeds).split(",") if x.strip()]
    aggregator_list = parse_csv(aggregators)
    round_values = [int(x.strip()) for x in str(rounds_list).split(",") if x.strip()]
    client_values = [int(x.strip()) for x in str(num_clients_list).split(",") if x.strip()]

    scenario_dir = real_collection_dir / scenario
    graph_file = output_dir / "graphs" / f"{scenario}.pt"
    protocol_graph = output_dir / "graphs" / f"{scenario}__congestion_hard.pt"

    run_cmd(
        [
            python_bin,
            "build_graph_v2.py",
            "--pcap-file",
            str(scenario_dir / "full_arena_v2.pcap"),
            "--manifest-file",
            str(scenario_dir / "arena_manifest_v2.json"),
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
        log_file=output_dir / "logs" / "build_graph.log",
        skip_if_exists=graph_file if skip_existing else None,
    )
    run_cmd(
        [
            python_bin,
            "prepare_hard_protocol_graph.py",
            "--input-graph",
            str(graph_file),
            "--output-graph",
            str(protocol_graph),
            "--protocol",
            "congestion_ood",
            "--manifest-file",
            str(scenario_dir / "arena_manifest_v2.json"),
            "--hard-overlap",
            "--train-keep-frac",
            "0.80",
            "--val-keep-frac",
            "0.90",
            "--test-keep-frac",
            "0.90",
            "--min-keep-per-class",
            "64",
            "--seed",
            "42",
        ],
        cwd=project,
        log_file=output_dir / "logs" / "prepare_protocol.log",
        skip_if_exists=protocol_graph if skip_existing else None,
    )

    summary: dict[str, Any] = {
        "config": {
            "project_dir": str(project),
            "python_bin": python_bin,
            "real_collection_dir": str(real_collection_dir),
            "output_dir": str(output_dir),
            "scenario": scenario,
            "seeds": seeds,
            "aggregators": aggregators,
            "rounds_list": rounds_list,
            "num_clients_list": num_clients_list,
            "local_epochs": local_epochs,
            "skip_existing": skip_existing,
        },
        "rows": [],
        "setting_stats": {},
        "partition_audit": {},
    }

    for num_clients in client_values:
        summary["partition_audit"][str(num_clients)] = partition_summary(protocol_graph, num_clients=num_clients)
        for rounds in round_values:
            for aggregator in aggregator_list:
                setting_key = f"c{num_clients}_r{rounds}_{aggregator}"
                setting_rows = []
                for seed in seed_list:
                    run_dir = output_dir / "runs" / setting_key / f"seed{seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    result_file = run_dir / "results.json"
                    duration_sec = run_cmd(
                        [
                            python_bin,
                            "fed_pignn.py",
                            "--graph-file",
                            str(protocol_graph),
                            "--model-file",
                            str(run_dir / "model.pt"),
                            "--results-file",
                            str(result_file),
                            "--num-clients",
                            str(num_clients),
                            "--rounds",
                            str(rounds),
                            "--local-epochs",
                            str(local_epochs),
                            "--aggregation",
                            str(aggregator),
                            "--simulate-poison-frac",
                            "0.0",
                            "--poison-scale",
                            "0.0",
                            "--alpha-flow",
                            "0.0",
                            "--beta-latency",
                            "0.0",
                            "--capacity",
                            "0",
                            "--capacity-mode",
                            "auto",
                            "--warmup-rounds",
                            str(max(1, min(rounds, 2))),
                            "--seed",
                            str(seed),
                            "--client-cpus",
                            "2.0",
                            "--client-gpus",
                            "0.0",
                            "--force-cpu",
                        ],
                        cwd=project,
                        log_file=run_dir / "run.log",
                        skip_if_exists=result_file if skip_existing else None,
                    )
                    metrics = pick_fed_test_metrics(result_file)
                    row = {
                        "num_clients": num_clients,
                        "rounds": rounds,
                        "aggregation": aggregator,
                        "seed": seed,
                        "duration_sec": duration_sec,
                        **metrics,
                    }
                    summary["rows"].append(row)
                    setting_rows.append(row)
                summary["setting_stats"][setting_key] = {
                    "num_clients": num_clients,
                    "rounds": rounds,
                    "aggregation": aggregator,
                    "f1": mean_std([float(r["f1"]) for r in setting_rows]),
                    "recall": mean_std([float(r["recall"]) for r in setting_rows]),
                    "fpr": mean_std([float(r["fpr"]) for r in setting_rows]),
                    "duration_sec": mean_std([float(r["duration_sec"]) for r in setting_rows]),
                    "partition": summary["partition_audit"][str(num_clients)],
                }

    out_json = output_dir / "edge_suite_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, len(client_values), figsize=(9.8, 4.2), sharey=True)
    if len(client_values) == 1:
        axes = [axes]
    for idx, num_clients in enumerate(client_values):
        ax = axes[idx]
        stylize_axes(ax, grid_axis="y")
        for aggregator in aggregator_list:
            means = []
            stds = []
            for rounds in round_values:
                key = f"c{num_clients}_r{rounds}_{aggregator}"
                means.append(summary["setting_stats"][key]["f1"]["mean"])
                stds.append(summary["setting_stats"][key]["f1"]["std"])
            draw_line_with_band(
                ax,
                x=round_values,
                y=means,
                yerr=stds,
                label=display_name(aggregator),
                color=model_color(aggregator),
                marker="o" if aggregator == "fedavg" else "s",
            )
        ax.set_xticks(round_values)
        ax.set_xlabel("Communication rounds")
        ax.set_title(f"{num_clients} clients")
        ax.set_ylim(0.70, 0.99)
        ax.legend(loc="lower right")
        add_panel_label(ax, "a" if idx == 0 else "b")
        if idx == 0:
            ax.set_ylabel("F1")
    save_figure(fig, output_dir / "fig_edge_budget_tradeoff.png")

    lines = [
        "# WASA Edge Suite Summary",
        "",
        "## Settings",
    ]
    for key, block in summary["setting_stats"].items():
        lines.append(
            f"- {key}: F1={block['f1']['mean']:.4f} +/- {block['f1']['std']:.4f}, "
            f"recall={block['recall']['mean']:.4f}, fpr={block['fpr']['mean']:.4f}, "
            f"duration_sec={block['duration_sec']['mean']:.2f}"
        )
    (output_dir / "edge_suite_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_system_overhead(
    *,
    project: Path,
    python_bin: str,
    real_collection_dir: Path,
    output_dir: Path,
    scenario_build: str,
    scenario_stress: str,
    epochs: int,
    seed: int,
    skip_existing: bool,
) -> None:
    use_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "graphs").mkdir(parents=True, exist_ok=True)

    build_dir = real_collection_dir / scenario_build
    stress_dir = real_collection_dir / scenario_stress
    build_graph = output_dir / "graphs" / f"{scenario_build}.pt"
    stress_graph = output_dir / "graphs" / f"{scenario_stress}.pt"
    protocol_graph = output_dir / "graphs" / f"{scenario_stress}__congestion_hard.pt"

    timings: dict[str, Any] = {}
    timings["graph_build_primary_sec"] = run_cmd(
        [
            python_bin,
            "build_graph_v2.py",
            "--pcap-file",
            str(build_dir / "full_arena_v2.pcap"),
            "--manifest-file",
            str(build_dir / "arena_manifest_v2.json"),
            "--output-file",
            str(build_graph),
            "--target-ip",
            "10.0.0.100",
            "--delta-t",
            "1.0",
            "--seed",
            "42",
        ],
        cwd=project,
        log_file=output_dir / "logs" / "build_primary.log",
        skip_if_exists=build_graph if skip_existing else None,
    )
    timings["graph_build_stress_sec"] = run_cmd(
        [
            python_bin,
            "build_graph_v2.py",
            "--pcap-file",
            str(stress_dir / "full_arena_v2.pcap"),
            "--manifest-file",
            str(stress_dir / "arena_manifest_v2.json"),
            "--output-file",
            str(stress_graph),
            "--target-ip",
            "10.0.0.100",
            "--delta-t",
            "1.0",
            "--seed",
            "42",
        ],
        cwd=project,
        log_file=output_dir / "logs" / "build_stress.log",
        skip_if_exists=stress_graph if skip_existing else None,
    )
    timings["protocol_prepare_sec"] = run_cmd(
        [
            python_bin,
            "prepare_hard_protocol_graph.py",
            "--input-graph",
            str(stress_graph),
            "--output-graph",
            str(protocol_graph),
            "--protocol",
            "congestion_ood",
            "--manifest-file",
            str(stress_dir / "arena_manifest_v2.json"),
            "--hard-overlap",
            "--train-keep-frac",
            "0.80",
            "--val-keep-frac",
            "0.90",
            "--test-keep-frac",
            "0.90",
            "--min-keep-per-class",
            "64",
            "--seed",
            "42",
        ],
        cwd=project,
        log_file=output_dir / "logs" / "prepare_protocol.log",
        skip_if_exists=protocol_graph if skip_existing else None,
    )

    central_dir = output_dir / "central"
    central_dir.mkdir(parents=True, exist_ok=True)
    central_results = central_dir / "phase3_results.json"
    central_model = central_dir / "model.pt"
    timings["central_train_sec"] = run_cmd(
        [
            python_bin,
            "pi_gnn_train_v2.py",
            "--graph-file",
            str(protocol_graph),
            "--model-file",
            str(central_model),
            "--results-file",
            str(central_results),
            "--epochs",
            str(epochs),
            "--alpha-flow",
            "0.0",
            "--beta-latency",
            "0.0",
            "--capacity",
            "0",
            "--capacity-mode",
            "auto",
            "--warmup-epochs",
            "15",
            "--patience",
            "20",
            "--seed",
            str(seed),
            "--force-cpu",
        ],
        cwd=project,
        log_file=central_dir / "run.log",
        skip_if_exists=central_results if skip_existing else None,
    )
    pred_file = central_dir / "predictions.json"
    if skip_existing and pred_file.exists():
        timings["prediction_export_sec"] = 0.0
    else:
        t0 = time.time()
        export_pi_predictions(
            str(protocol_graph),
            str(central_model),
            str(central_results),
            str(pred_file),
            force_cpu=True,
        )
        timings["prediction_export_sec"] = float(time.time() - t0)
    baseline_file = central_dir / "baseline_eval.json"
    timings["baseline_eval_sec"] = run_cmd(
        [
            python_bin,
            "evaluate_baselines.py",
            "--graph-file",
            str(protocol_graph),
            "--pi-model-file",
            str(central_model),
            "--pi-results-file",
            str(central_results),
            "--output-file",
            str(baseline_file),
            "--seed",
            str(seed),
            "--force-cpu",
        ],
        cwd=project,
        log_file=central_dir / "baseline.log",
        skip_if_exists=baseline_file if skip_existing else None,
    )

    fed_dir = output_dir / "federated"
    fed_dir.mkdir(parents=True, exist_ok=True)
    fed_results = fed_dir / "results.json"
    timings["federated_sec"] = run_cmd(
        [
            python_bin,
            "fed_pignn.py",
            "--graph-file",
            str(protocol_graph),
            "--model-file",
            str(fed_dir / "model.pt"),
            "--results-file",
            str(fed_results),
            "--num-clients",
            "3",
            "--rounds",
            "2",
            "--local-epochs",
            "1",
            "--aggregation",
            "fedavg",
            "--simulate-poison-frac",
            "0.0",
            "--poison-scale",
            "0.0",
            "--alpha-flow",
            "0.0",
            "--beta-latency",
            "0.0",
            "--capacity",
            "0",
            "--capacity-mode",
            "auto",
            "--warmup-rounds",
            "1",
            "--seed",
            str(seed),
            "--client-cpus",
            "2.0",
            "--client-gpus",
            "0.0",
            "--force-cpu",
        ],
        cwd=project,
        log_file=fed_dir / "run.log",
        skip_if_exists=fed_results if skip_existing else None,
    )

    summary = {
        "config": {
            "project_dir": str(project),
            "python_bin": python_bin,
            "real_collection_dir": str(real_collection_dir),
            "output_dir": str(output_dir),
            "scenario_build": scenario_build,
            "scenario_stress": scenario_stress,
            "epochs": epochs,
            "seed": seed,
            "skip_existing": skip_existing,
        },
        "timings_sec": timings,
        "graph_stats": {
            "primary": graph_stats(build_graph),
            "stress": graph_stats(stress_graph),
            "stress_protocol": graph_stats(protocol_graph),
        },
        "central_metrics": read_json(central_results).get("final_eval", {}).get("test_temporal")
        or read_json(central_results).get("final_eval", {}).get("test_random")
        or {},
        "fed_metrics": read_json(fed_results).get("global_metrics", {}).get("test_temporal")
        or read_json(fed_results).get("global_metrics", {}).get("test_random")
        or {},
    }
    out_json = output_dir / "overhead_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    labels = [stage_name(key) for key in timings.keys()]
    vals = [float(timings[key]) for key in timings.keys()]
    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    stylize_axes(ax, grid_axis="x")
    ax.hlines(y_pos, 0.0, vals, color=PALETTE["grid"], linewidth=2.0)
    ax.scatter(vals, y_pos, color=PALETTE["blue"], s=58, zorder=3)
    for idx, value in enumerate(vals):
        ax.text(value + max(vals) * 0.025, idx, f"{value:.1f}s", va="center", fontsize=8, color=PALETTE["ink"])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, max(vals) * 1.20 if max(vals) > 0 else 1.0)
    ax.set_xlabel("Seconds")
    ax.set_title("System Overhead Summary")
    add_panel_label(ax, "a")
    save_figure(fig, output_dir / "fig_overhead_runtime.png")

    lines = [
        "# WASA Overhead Summary",
        "",
        "## Timings",
    ]
    for key, value in timings.items():
        lines.append(f"- {key}: {value:.2f}s")
    lines.extend(
        [
            "",
            "## Graph Stats",
            f"- primary_flow_nodes: {summary['graph_stats']['primary']['flow_nodes']}",
            f"- stress_flow_nodes: {summary['graph_stats']['stress']['flow_nodes']}",
            f"- stress_protocol_flow_nodes: {summary['graph_stats']['stress_protocol']['flow_nodes']}",
        ]
    )
    (output_dir / "overhead_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Run system-level experiments")
    p.add_argument("--project-dir", default=str(REPO_ROOT))
    p.add_argument("--python-bin", default="python")
    p.add_argument("--real-collection-dir", default=str(DATA_ROOT))
    p.add_argument("--output-root", default=str(RUN_ROOT))
    p.add_argument("--skip-existing", action="store_true")

    p.add_argument("--network-scenarios", default=NETWORK_SCENARIOS_DEFAULT)
    p.add_argument("--network-seeds", default="11,22,33")
    p.add_argument("--network-epochs", type=int, default=110)

    p.add_argument("--edge-scenario", default="scenario_h_mimic_heavy_overlap")
    p.add_argument("--edge-seeds", default="11,22,33")
    p.add_argument("--edge-aggregators", default="fedavg,median")
    p.add_argument("--edge-rounds-list", default="1,2,4")
    p.add_argument("--edge-num-clients-list", default="3,5")
    p.add_argument("--edge-local-epochs", type=int, default=2)

    p.add_argument("--overhead-scenario-build", default="scenario_e_three_tier_high2")
    p.add_argument("--overhead-scenario-stress", default="scenario_h_mimic_heavy_overlap")
    p.add_argument("--overhead-epochs", type=int, default=80)
    p.add_argument("--overhead-seed", type=int, default=11)
    args = p.parse_args()

    project = Path(args.project_dir).resolve()
    data_dir = Path(args.real_collection_dir).resolve()
    output_root = Path(args.output_root).resolve()
    network_out = output_root / "network_sensitivity"
    edge_out = output_root / "edge_budget"
    overhead_out = output_root / "system_overhead"
    network_out.mkdir(parents=True, exist_ok=True)
    edge_out.mkdir(parents=True, exist_ok=True)
    overhead_out.mkdir(parents=True, exist_ok=True)

    run_network_sensitivity(
        project=project,
        python_bin=args.python_bin,
        real_collection_dir=data_dir,
        output_dir=network_out,
        scenarios=args.network_scenarios,
        seeds=args.network_seeds,
        epochs=args.network_epochs,
        skip_existing=args.skip_existing,
    )
    run_edge_budget(
        project=project,
        python_bin=args.python_bin,
        real_collection_dir=data_dir,
        output_dir=edge_out,
        scenario=args.edge_scenario,
        seeds=args.edge_seeds,
        aggregators=args.edge_aggregators,
        rounds_list=args.edge_rounds_list,
        num_clients_list=args.edge_num_clients_list,
        local_epochs=args.edge_local_epochs,
        skip_existing=args.skip_existing,
    )
    run_system_overhead(
        project=project,
        python_bin=args.python_bin,
        real_collection_dir=data_dir,
        output_dir=overhead_out,
        scenario_build=args.overhead_scenario_build,
        scenario_stress=args.overhead_scenario_stress,
        epochs=args.overhead_epochs,
        seed=args.overhead_seed,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
