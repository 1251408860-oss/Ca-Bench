#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib as mpl

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


REPO_ROOT = Path(__file__).resolve().parent
RUN_ROOT = REPO_ROOT.parent / "paper_artifacts" / "runs"
OUTPUT_ROOT = REPO_ROOT.parent / "paper_artifacts"
MODELS_MAIN = ["data_only", "gcn", "graphsage", "gatv2"]

PALETTE = {
    "ink": "#2F3640",
    "slate": "#5B7386",
    "blue": "#4C78A8",
    "teal": "#3C8D8A",
    "sand": "#D8C3A5",
    "rose": "#B85757",
    "gold": "#C08A3E",
    "grid": "#D7DDE4",
    "fill": "#EEF2F5",
}

MODEL_COLORS = {
    "data_only": PALETTE["blue"],
    "anchor_gnn": PALETTE["blue"],
    "random_forest": PALETTE["gold"],
    "gcn": PALETTE["teal"],
    "graphsage": "#7A8AA0",
    "gatv2": PALETTE["rose"],
    "pi_gnn": "#7A8AA0",
    "fedavg": PALETTE["blue"],
    "median": PALETTE["rose"],
}

DISPLAY_NAMES = {
    "data_only": "anchor GNN",
    "anchor_gnn": "anchor GNN",
    "random_forest": "RF",
    "gcn": "GCN",
    "graphsage": "GraphSAGE",
    "gatv2": "GATv2",
    "pi_gnn": "PI-GNN",
    "fedavg": "FedAvg",
    "median": "Median",
}

STAGE_NAMES = {
    "graph_build_primary_sec": "Primary graph build",
    "graph_build_stress_sec": "Stress graph build",
    "protocol_prepare_sec": "Protocol prep",
    "central_train_sec": "Central train",
    "prediction_export_sec": "Prediction export",
    "baseline_eval_sec": "Baseline eval",
    "federated_sec": "Federated run",
}


def use_paper_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.linewidth": 0.8,
            "axes.titlepad": 8,
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "figure.dpi": 220,
            "savefig.dpi": 260,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def stylize_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.set_facecolor("white")
    ax.grid(True, axis=grid_axis, linestyle="-", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["ink"])
        ax.spines[side].set_linewidth(0.8)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=PALETTE["ink"],
    )


def save_figure(fig: plt.Figure, path: Path | str) -> None:
    fig.tight_layout()
    fig.savefig(Path(path))
    plt.close(fig)


def model_color(name: str) -> str:
    return MODEL_COLORS.get(name, PALETTE["slate"])


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, str(name).replace("_", " "))


def stage_name(name: str) -> str:
    return STAGE_NAMES.get(name, str(name).replace("_sec", "").replace("_", " "))


def short_scenario(name: str) -> str:
    if "scenario_" not in name:
        return name
    token = name.split("scenario_", 1)[1]
    head = token.split("_", 1)[0]
    return f"s_{head}"


def paper_heatmap(ax: plt.Axes, data: np.ndarray, row_labels: list[str], col_labels: list[str]) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "paper_blue_rose",
        ["#F7F8FA", "#C7D4E3", "#7FA0BE", "#4C78A8"],
    )
    ax.imshow(data, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(axis="x", rotation=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j,
                i,
                f"{data[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color=PALETTE["ink"],
            )
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_line_with_band(
    ax: plt.Axes,
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    yerr: list[float] | np.ndarray,
    label: str,
    color: str,
    marker: str = "o",
) -> None:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    err_arr = np.asarray(list(yerr), dtype=float)
    ax.plot(x_arr, y_arr, marker=marker, color=color, label=label)
    ax.fill_between(
        x_arr,
        np.clip(y_arr - err_arr, 0.0, 1.0),
        np.clip(y_arr + err_arr, 0.0, 1.0),
        color=color,
        alpha=0.12,
        linewidth=0,
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_read_json(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.exists() else None


def write_csv(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)


def ensure_dirs(output_root: Path) -> tuple[Path, Path]:
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir


def mean_metric(block: dict[str, Any], metric: str) -> float:
    return float(block.get(metric, {}).get("mean", 0.0))


def select_cross_scenario(cross: dict[str, Any]) -> str | None:
    preferred = "scenario_h_mimic_heavy_overlap"
    if preferred in cross.get("per_scenario_stats", {}):
        return preferred
    names = list(cross.get("per_scenario_stats", {}).keys())
    return names[0] if names else None


def build_table1(top: dict[str, Any], baseline: dict[str, Any], congestion: dict[str, Any], tables_dir: Path) -> None:
    stage3 = top["statistics"]["stage3"]
    rows = [["protocol", "model", "f1", "recall", "fpr"]]
    rows.append(
        [
            "temporal_ood",
            "anchor_gnn",
            mean_metric(stage3["temporal_ood"]["clean"]["data_only"], "f1"),
            mean_metric(stage3["temporal_ood"]["clean"]["data_only"], "recall"),
            0.0,
        ]
    )
    rows.append(
        [
            "temporal_ood",
            "gcn",
            mean_metric(baseline["stats"]["temporal_ood"]["gcn"], "f1"),
            mean_metric(baseline["stats"]["temporal_ood"]["gcn"], "recall"),
            mean_metric(baseline["stats"]["temporal_ood"]["gcn"], "fpr"),
        ]
    )
    rows.append(
        [
            "congestion_ood",
            "anchor_gnn",
            mean_metric(congestion["stage3_stats"]["data_only"], "f1"),
            mean_metric(congestion["stage3_stats"]["data_only"], "recall"),
            mean_metric(congestion["stage3_stats"]["data_only"], "fpr"),
        ]
    )
    for model in ["gatv2", "gcn", "graphsage"]:
        rows.append(
            [
                "congestion_ood",
                model,
                mean_metric(congestion["baseline_stats"][model], "f1"),
                mean_metric(congestion["baseline_stats"][model], "recall"),
                mean_metric(congestion["baseline_stats"][model], "fpr"),
            ]
        )
    write_csv(tables_dir / "table1_selected_detection.csv", rows)


def build_table2(cross: dict[str, Any], tables_dir: Path) -> None:
    scenario_name = select_cross_scenario(cross)
    if scenario_name is None:
        return
    block = cross["per_scenario_stats"][scenario_name]
    rows = [["scenario", "model", "f1", "recall", "fpr"]]
    for model in MODELS_MAIN:
        rows.append(
            [
                scenario_name,
                model,
                mean_metric(block[model], "f1"),
                mean_metric(block[model], "recall"),
                mean_metric(block[model], "fpr"),
            ]
        )
    write_csv(tables_dir / "table2_cross_scenario.csv", rows)


def build_table3(overhead: dict[str, Any], tables_dir: Path) -> None:
    primary = overhead.get("graph_stats", {}).get("primary", {})
    stress = overhead.get("graph_stats", {}).get("stress", {})
    protocol = overhead.get("graph_stats", {}).get("stress_protocol", {})
    rows = [["stage", "seconds", "flow_nodes", "edges"]]
    for key, value in overhead.get("timings_sec", {}).items():
        flow_nodes = ""
        edges = ""
        if key == "graph_build_primary_sec":
            flow_nodes = primary.get("flow_nodes", "")
            edges = primary.get("num_edges", "")
        elif key == "graph_build_stress_sec":
            flow_nodes = stress.get("flow_nodes", "")
            edges = stress.get("num_edges", "")
        elif key == "protocol_prepare_sec":
            flow_nodes = protocol.get("flow_nodes", "")
            edges = protocol.get("num_edges", "")
        rows.append([stage_name(key), float(value), flow_nodes, edges])
    write_csv(tables_dir / "table3_pipeline_cost.csv", rows)


def make_figure1(top: dict[str, Any], baseline: dict[str, Any], congestion: dict[str, Any], figures_dir: Path) -> None:
    stage3 = top["statistics"]["stage3"]
    easy_protocols = ["temporal_ood", "topology_ood", "attack_strategy_ood"]

    easy_means: list[float] = []
    congestion_means: list[float] = []
    for model in MODELS_MAIN:
        if model == "data_only":
            easy_means.append(float(np.mean([mean_metric(stage3[p]["clean"]["data_only"], "f1") for p in easy_protocols])))
            congestion_means.append(mean_metric(congestion["stage3_stats"]["data_only"], "f1"))
        else:
            easy_means.append(float(np.mean([mean_metric(baseline["stats"][p][model], "f1") for p in easy_protocols])))
            congestion_means.append(mean_metric(congestion["baseline_stats"][model], "f1"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.4), gridspec_kw={"width_ratios": [1.1, 1.0]})

    stylize_axes(ax1, grid_axis="y")
    for idx, model in enumerate(MODELS_MAIN):
        ax1.plot([0, 1], [easy_means[idx], congestion_means[idx]], color=model_color(model), marker="o", linewidth=2.0)
        ax1.text(1.04, congestion_means[idx], f"{display_name(model)} {congestion_means[idx]:.3f}", va="center", fontsize=8, color=model_color(model))
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["easy OOD mean", "congestion OOD"])
    ax1.set_xlim(-0.1, 1.35)
    ax1.set_ylim(0.0, 1.02)
    ax1.set_ylabel("F1")
    ax1.set_title("Generalization Gap Under Congestion")
    add_panel_label(ax1, "a")

    stylize_axes(ax2, grid_axis="x")
    y_pos = np.arange(len(MODELS_MAIN))
    ax2.barh(y_pos, congestion_means, color=[model_color(model) for model in MODELS_MAIN], alpha=0.92)
    for idx, value in enumerate(congestion_means):
        ax2.text(value + 0.015, y_pos[idx], f"{value:.3f}", va="center", fontsize=8, color=PALETTE["ink"])
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([display_name(model) for model in MODELS_MAIN])
    ax2.set_xlim(0.0, max(0.62, max(congestion_means) + 0.12))
    ax2.set_xlabel("F1")
    ax2.set_title("Congestion OOD Comparison")
    add_panel_label(ax2, "b")

    save_figure(fig, figures_dir / "figure1_congestion_gap.png")


def make_figure2(network: dict[str, Any], figures_dir: Path) -> None:
    order = [name for name in str(network["config"]["scenarios"]).split(",") if name in network["scenario_stats"]]
    if not order:
        order = list(network["scenario_stats"].keys())
    scenario_labels = [short_scenario(name) for name in order]
    x = np.arange(len(order))

    anchor_vals = [mean_metric(network["scenario_stats"][name]["data_only"], "f1") for name in order]
    anchor_stds = [float(network["scenario_stats"][name]["data_only"]["f1"]["std"]) for name in order]
    best_vals = []
    best_stds = []
    for name in order:
        best_name = str(network["scenario_stats"][name]["best_nonphysics_baseline"])
        best_vals.append(mean_metric(network["scenario_stats"][name][best_name], "f1"))
        best_stds.append(float(network["scenario_stats"][name][best_name]["f1"]["std"]))

    heatmap_models = ["data_only", "random_forest", "gcn", "graphsage", "gatv2"]
    heatmap_data = np.asarray(
        [[mean_metric(network["scenario_stats"][name][model], "f1") for name in order] for model in heatmap_models],
        dtype=float,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.6), gridspec_kw={"width_ratios": [1.2, 1.0]})
    stylize_axes(ax1, grid_axis="y")
    draw_line_with_band(ax1, x=x, y=anchor_vals, yerr=anchor_stds, label="anchor GNN", color=model_color("data_only"))
    draw_line_with_band(ax1, x=x, y=best_vals, yerr=best_stds, label="best baseline", color=PALETTE["rose"], marker="s")
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

    save_figure(fig, figures_dir / "figure2_network_sensitivity.png")


def make_figure3(edge: dict[str, Any], figures_dir: Path) -> None:
    num_clients_list = [int(x.strip()) for x in str(edge["config"]["num_clients_list"]).split(",") if x.strip()]
    rounds_list = [int(x.strip()) for x in str(edge["config"]["rounds_list"]).split(",") if x.strip()]
    aggregators = [x.strip() for x in str(edge["config"]["aggregators"]).split(",") if x.strip()]

    fig, axes = plt.subplots(1, len(num_clients_list), figsize=(9.8, 4.2), sharey=True)
    if len(num_clients_list) == 1:
        axes = [axes]

    for idx, num_clients in enumerate(num_clients_list):
        ax = axes[idx]
        stylize_axes(ax, grid_axis="y")
        for aggregator in aggregators:
            means = []
            stds = []
            for rounds in rounds_list:
                key = f"c{num_clients}_r{rounds}_{aggregator}"
                means.append(mean_metric(edge["setting_stats"][key], "f1"))
                stds.append(float(edge["setting_stats"][key]["f1"]["std"]))
            draw_line_with_band(
                ax,
                x=rounds_list,
                y=means,
                yerr=stds,
                label=display_name(aggregator),
                color=model_color(aggregator),
                marker="o" if aggregator == "fedavg" else "s",
            )
        ax.set_xticks(rounds_list)
        ax.set_xlabel("Communication rounds")
        ax.set_title(f"{num_clients} clients")
        ax.set_ylim(0.70, 0.99)
        ax.legend(loc="lower right")
        add_panel_label(ax, "a" if idx == 0 else "b")
        if idx == 0:
            ax.set_ylabel("F1")

    save_figure(fig, figures_dir / "figure3_edge_budget_tradeoff.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-dir", default=str(RUN_ROOT / "main_suite"))
    ap.add_argument("--cross-scenario-dir", default=str(RUN_ROOT / "cross_scenario"))
    ap.add_argument("--congestion-focus-dir", default=str(RUN_ROOT / "congestion_focus"))
    ap.add_argument("--network-sensitivity-dir", default=str(RUN_ROOT / "network_sensitivity"))
    ap.add_argument("--edge-suite-dir", default=str(RUN_ROOT / "edge_budget"))
    ap.add_argument("--overhead-dir", default=str(RUN_ROOT / "system_overhead"))
    ap.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    args = ap.parse_args()

    use_paper_style()
    output_root = Path(args.output_dir).resolve()
    tables_dir, figures_dir = ensure_dirs(output_root)

    top = maybe_read_json(Path(args.suite_dir).resolve() / "top_conf_summary.json")
    baseline = maybe_read_json(Path(args.suite_dir).resolve() / "baseline_significance" / "baseline_significance_summary.json")
    congestion = maybe_read_json(Path(args.congestion_focus_dir).resolve() / "congestion_focus_summary.json")
    cross = maybe_read_json(Path(args.cross_scenario_dir).resolve() / "cross_scenario_summary.json")
    network = maybe_read_json(Path(args.network_sensitivity_dir).resolve() / "network_sensitivity_summary.json")
    edge = maybe_read_json(Path(args.edge_suite_dir).resolve() / "edge_suite_summary.json")
    overhead = maybe_read_json(Path(args.overhead_dir).resolve() / "overhead_summary.json")

    if top is not None and baseline is not None and congestion is not None:
        build_table1(top, baseline, congestion, tables_dir)
        make_figure1(top, baseline, congestion, figures_dir)

    if cross is not None:
        build_table2(cross, tables_dir)

    if overhead is not None:
        build_table3(overhead, tables_dir)

    if network is not None:
        make_figure2(network, figures_dir)

    if edge is not None:
        make_figure3(edge, figures_dir)

    print(f"[DONE] paper outputs: {output_root}")


if __name__ == "__main__":
    main()
