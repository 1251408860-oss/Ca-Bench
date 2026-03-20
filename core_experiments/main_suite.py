#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_baselines import export_pi_predictions


REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT.parent / "mininet_testbed" / "real_collection"
RUN_ROOT = REPO_ROOT.parent / "paper_artifacts" / "runs"
BASELINE_MODELS = ["random_forest", "gcn", "graphsage", "gatv2", "pi_gnn"]


def run(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


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
    return (ext + 1) / (tot + 1)


def mean_std(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0, "mean": 0.0, "std": 0.0}
    if len(vals) == 1:
        return {"n": 1, "mean": float(vals[0]), "std": 0.0}
    return {"n": len(vals), "mean": float(statistics.mean(vals)), "std": float(statistics.stdev(vals))}


def run_baseline_significance(
    *,
    python_bin: str,
    project: Path,
    suite_dir: Path,
    seeds: list[int],
    include_congestion_ood: bool = True,
    save_predictions: bool = True,
) -> Path:
    protocols = ["temporal_ood", "topology_ood", "attack_strategy_ood"]
    if include_congestion_ood and (suite_dir / "protocol_graphs" / "congestion_ood.pt").exists():
        protocols.append("congestion_ood")
    out_dir = suite_dir / "baseline_significance"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for proto in protocols:
        graph = suite_dir / "protocol_graphs" / f"{proto}.pt"
        for seed in seeds:
            model_dir = suite_dir / "stage3" / f"{proto}__clean__physics_stable__seed{seed}"
            model = model_dir / "pi_gnn_model.pt"
            model_results = model_dir / "phase3_results.json"
            out = out_dir / f"{proto}_seed{seed}.json"
            log = out_dir / f"{proto}_seed{seed}.log"
            cmd = [
                python_bin,
                "evaluate_baselines.py",
                "--graph-file",
                str(graph),
                "--pi-model-file",
                str(model),
                "--pi-results-file",
                str(model_results),
                "--output-file",
                str(out),
                "--seed",
                str(seed),
                "--force-cpu",
            ]
            if save_predictions:
                cmd.append("--save-predictions")
            with log.open("w", encoding="utf-8") as f:
                result = subprocess.run(cmd, cwd=str(project), stdout=f, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                raise SystemExit(result.returncode)
            data = json.loads(out.read_text(encoding="utf-8"))
            runs.append({"protocol": proto, "seed": seed, "metrics": data.get("metrics", {}), "thresholds": data.get("thresholds", {})})

    summary: dict[str, Any] = {"runs": runs, "stats": {}}
    for proto in protocols:
        summary["stats"][proto] = {}
        proto_runs = [r for r in runs if r["protocol"] == proto]
        for model in BASELINE_MODELS:
            f1 = [float(r["metrics"].get(model, {}).get("f1", 0.0)) for r in proto_runs]
            fpr = [float(r["metrics"].get(model, {}).get("fpr", 0.0)) for r in proto_runs]
            rec = [float(r["metrics"].get(model, {}).get("recall", 0.0)) for r in proto_runs]
            auroc = [float(r["metrics"].get(model, {}).get("roc_auc", 0.0)) for r in proto_runs]
            threshold = [float(r["thresholds"].get(model, r["metrics"].get(model, {}).get("threshold", 0.5))) for r in proto_runs]
            summary["stats"][proto][model] = {
                "f1": mean_std(f1),
                "fpr": mean_std(fpr),
                "recall": mean_std(rec),
                "roc_auc": mean_std(auroc),
                "threshold": mean_std(threshold),
            }

        gcn_f1 = [float(r["metrics"].get("gcn", {}).get("f1", 0.0)) for r in proto_runs]
        gcn_fpr = [float(r["metrics"].get("gcn", {}).get("fpr", 0.0)) for r in proto_runs]
        summary["stats"][proto]["significance_vs_gcn"] = {}
        for model in ["graphsage", "gatv2", "pi_gnn"]:
            comp_f1 = [float(r["metrics"].get(model, {}).get("f1", 0.0)) for r in proto_runs]
            comp_fpr = [float(r["metrics"].get(model, {}).get("fpr", 0.0)) for r in proto_runs]
            summary["stats"][proto]["significance_vs_gcn"][model] = {
                "p_value_f1": pval_signflip(comp_f1, gcn_f1),
                "p_value_fpr": pval_signflip([-x for x in comp_fpr], [-y for y in gcn_fpr]),
            }

    out_file = out_dir / "baseline_significance_summary.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_file


def run_logged_cmd(cmd: list[str], cwd: Path, log_file: Path, skip_if_exists: Path | None = None) -> None:
    if skip_if_exists is not None and skip_if_exists.exists():
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as f:
        result = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pick_phase3_test_metrics(phase3: dict[str, Any]) -> dict[str, float]:
    final_eval = phase3.get("final_eval", {})
    return final_eval.get("test_temporal") or final_eval.get("test_random") or {}


def metric_with_fpr(metrics: dict[str, Any]) -> dict[str, float]:
    fp = int(metrics.get("fp", 0))
    tn = int(metrics.get("tn", 0))
    out = dict(metrics)
    out["fpr"] = float(fp / max(fp + tn, 1))
    return out


def ensure_congestion_graph(
    *,
    python_bin: str,
    project: Path,
    suite_dir: Path,
    output_dir: Path,
    scenario_dir: Path,
    explicit_graph: str,
    skip_existing: bool,
) -> Path:
    if explicit_graph and Path(explicit_graph).exists():
        return Path(explicit_graph)

    suite_graph = suite_dir / "protocol_graphs" / "congestion_ood.pt"
    if suite_graph.exists():
        return suite_graph

    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    base_graph = graph_dir / f"{scenario_dir.name}.pt"
    congestion_graph = graph_dir / "congestion_ood.pt"

    run_logged_cmd(
        [
            python_bin,
            "build_graph_v2.py",
            "--pcap-file",
            str(scenario_dir / "full_arena_v2.pcap"),
            "--manifest-file",
            str(scenario_dir / "arena_manifest_v2.json"),
            "--output-file",
            str(base_graph),
            "--target-ip",
            "10.0.0.100",
            "--delta-t",
            "1.0",
            "--seed",
            "42",
        ],
        cwd=project,
        log_file=output_dir / "logs" / f"build_{scenario_dir.name}.log",
        skip_if_exists=base_graph if skip_existing else None,
    )
    run_logged_cmd(
        [
            python_bin,
            "prepare_hard_protocol_graph.py",
            "--input-graph",
            str(base_graph),
            "--output-graph",
            str(congestion_graph),
            "--protocol",
            "congestion_ood",
            "--manifest-file",
            str(scenario_dir / "arena_manifest_v2.json"),
            "--holdout-attack-type",
            "mimic",
            "--seed",
            "42",
            "--hard-overlap",
            "--camouflage-test-attacks",
        ],
        cwd=project,
        log_file=output_dir / "logs" / "build_congestion_ood.log",
        skip_if_exists=congestion_graph if skip_existing else None,
    )
    return congestion_graph


def run_congestion_focus(
    *,
    python_bin: str,
    project: Path,
    suite_dir: Path,
    output_dir: Path,
    scenario_dir: Path,
    congestion_graph: str,
    seeds: str,
    epochs: int,
    skip_existing: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_list = [int(x.strip()) for x in seeds.split(",") if x.strip()]
    graph_file = ensure_congestion_graph(
        python_bin=python_bin,
        project=project,
        suite_dir=suite_dir,
        output_dir=output_dir,
        scenario_dir=scenario_dir,
        explicit_graph=congestion_graph,
        skip_existing=skip_existing,
    )

    rows_stage3 = []
    rows_baseline = []
    for seed in seed_list:
        for model_name, alpha, beta in [("data_only", 0.0, 0.0), ("physics_stable", 0.03, 0.02)]:
            exp_dir = output_dir / "stage3" / f"congestion_ood__{model_name}__seed{seed}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            result_file = exp_dir / "phase3_results.json"
            model_file = exp_dir / "pi_gnn_model.pt"
            run_logged_cmd(
                [
                    python_bin,
                    "pi_gnn_train_v2.py",
                    "--graph-file",
                    str(graph_file),
                    "--model-file",
                    str(model_file),
                    "--results-file",
                    str(result_file),
                    "--epochs",
                    str(epochs),
                    "--alpha-flow",
                    str(alpha),
                    "--beta-latency",
                    str(beta),
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
                log_file=exp_dir / "run.log",
                skip_if_exists=result_file if skip_existing else None,
            )
            phase3 = load_json(result_file)
            metrics = metric_with_fpr(pick_phase3_test_metrics(phase3))
            rows_stage3.append({"seed": seed, "model": model_name, "metrics": metrics, "result_file": str(result_file), "model_file": str(model_file)})

            pred_file = exp_dir / "predictions.json"
            if not (skip_existing and pred_file.exists()):
                export_pi_predictions(
                    str(graph_file),
                    str(model_file),
                    str(result_file),
                    str(pred_file),
                    force_cpu=True,
                )

        baseline_dir = output_dir / "baseline" / f"seed{seed}"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_file = baseline_dir / "baseline_eval.json"
        physics_dir = output_dir / "stage3" / f"congestion_ood__physics_stable__seed{seed}"
        run_logged_cmd(
            [
                python_bin,
                "evaluate_baselines.py",
                "--graph-file",
                str(graph_file),
                "--pi-model-file",
                str(physics_dir / "pi_gnn_model.pt"),
                "--pi-results-file",
                str(physics_dir / "phase3_results.json"),
                "--output-file",
                str(baseline_file),
                "--seed",
                str(seed),
                "--save-predictions",
                "--force-cpu",
            ],
            cwd=project,
            log_file=baseline_dir / "run.log",
            skip_if_exists=baseline_file if skip_existing else None,
        )
        baseline = load_json(baseline_file)
        rows_baseline.append({"seed": seed, "metrics": baseline.get("metrics", {}), "file": str(baseline_file)})

    summary: dict[str, Any] = {
        "config": {
            "python_bin": python_bin,
            "suite_dir": str(suite_dir),
            "output_dir": str(output_dir),
            "scenario_dir": str(scenario_dir),
            "congestion_graph": congestion_graph,
            "seeds": seeds,
            "epochs": epochs,
            "skip_existing": skip_existing,
        },
        "graph_file": str(graph_file),
        "stage3_rows": rows_stage3,
        "baseline_rows": rows_baseline,
        "stage3_stats": {},
        "baseline_stats": {},
        "p_values": {},
    }

    for model_name in ["data_only", "physics_stable"]:
        vals = [float(r["metrics"].get("f1", 0.0)) for r in rows_stage3 if r["model"] == model_name]
        rec = [float(r["metrics"].get("recall", 0.0)) for r in rows_stage3 if r["model"] == model_name]
        fpr = [float(r["metrics"].get("fpr", 0.0)) for r in rows_stage3 if r["model"] == model_name]
        summary["stage3_stats"][model_name] = {"f1": mean_std(vals), "recall": mean_std(rec), "fpr": mean_std(fpr)}

    for model_name in ["random_forest", "gcn", "graphsage", "gatv2", "pi_gnn"]:
        vals = [float(r["metrics"].get(model_name, {}).get("f1", 0.0)) for r in rows_baseline]
        rec = [float(r["metrics"].get(model_name, {}).get("recall", 0.0)) for r in rows_baseline]
        fpr = [float(r["metrics"].get(model_name, {}).get("fpr", 0.0)) for r in rows_baseline]
        summary["baseline_stats"][model_name] = {"f1": mean_std(vals), "recall": mean_std(rec), "fpr": mean_std(fpr)}

    data_f1 = [float(r["metrics"].get("f1", 0.0)) for r in rows_stage3 if r["model"] == "data_only"]
    phys_f1 = [float(r["metrics"].get("f1", 0.0)) for r in rows_stage3 if r["model"] == "physics_stable"]
    summary["p_values"]["physics_vs_data_only_f1"] = pval_signflip(phys_f1, data_f1)

    gcn_f1 = [float(r["metrics"].get("gcn", {}).get("f1", 0.0)) for r in rows_baseline]
    sage_f1 = [float(r["metrics"].get("graphsage", {}).get("f1", 0.0)) for r in rows_baseline]
    gat_f1 = [float(r["metrics"].get("gatv2", {}).get("f1", 0.0)) for r in rows_baseline]
    summary["p_values"]["graphsage_vs_gcn_f1"] = pval_signflip(sage_f1, gcn_f1)
    summary["p_values"]["gatv2_vs_gcn_f1"] = pval_signflip(gat_f1, gcn_f1)

    best_nonphysics = max(
        ["random_forest", "gcn", "graphsage", "gatv2"],
        key=lambda name: summary["baseline_stats"][name]["f1"]["mean"],
    )
    best_baseline_f1 = [float(r["metrics"].get(best_nonphysics, {}).get("f1", 0.0)) for r in rows_baseline]
    summary["best_nonphysics_baseline"] = best_nonphysics
    summary["p_values"]["best_nonphysics_vs_data_only_f1"] = pval_signflip(best_baseline_f1, data_f1)

    out_json = output_dir / "congestion_focus_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    data_seed_rows = sorted(
        [(int(r["seed"]), float(r["metrics"].get("f1", 0.0))) for r in rows_stage3 if r["model"] == "data_only"],
        key=lambda item: item[1],
    )
    rep_seed = data_seed_rows[len(data_seed_rows) // 2][0]
    models_plot = ["data_only", "physics_stable", "random_forest", "gcn", "graphsage", "gatv2"]
    means = [
        summary["stage3_stats"][m]["f1"]["mean"] if m in summary["stage3_stats"] else summary["baseline_stats"][m]["f1"]["mean"]
        for m in models_plot
    ]
    stds = [
        summary["stage3_stats"][m]["f1"]["std"] if m in summary["stage3_stats"] else summary["baseline_stats"][m]["f1"]["std"]
        for m in models_plot
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(np.arange(len(models_plot)), means, yerr=stds, color=["#1f77b4", "#d62728", "#8c564b", "#2ca02c", "#17becf", "#9467bd"])
    ax.set_xticks(np.arange(len(models_plot)))
    ax.set_xticklabels(models_plot, rotation=15)
    ax.set_ylim(0.0, 0.7)
    ax.set_ylabel("F1")
    ax.set_title("Congestion OOD: Mean F1 with Error Bars")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_congestion_main_errorbars.png", dpi=220)
    plt.close(fig)

    lines = [
        "# Congestion Focus Summary",
        "",
        f"- Graph: `{graph_file}`",
        f"- Seeds: `{','.join(str(seed) for seed in seed_list)}`",
        f"- Best non-physics baseline: `{best_nonphysics}`",
        "",
        "## Stage-3",
        f"- data_only: F1=`{summary['stage3_stats']['data_only']['f1']['mean']:.4f}` +/- `{summary['stage3_stats']['data_only']['f1']['std']:.4f}`",
        f"- physics_stable: F1=`{summary['stage3_stats']['physics_stable']['f1']['mean']:.4f}` +/- `{summary['stage3_stats']['physics_stable']['f1']['std']:.4f}`",
        f"- p(physics vs data_only, F1)=`{summary['p_values']['physics_vs_data_only_f1']:.6g}`",
        "",
        "## Baselines",
    ]
    for model_name in ["random_forest", "gcn", "graphsage", "gatv2"]:
        stat = summary["baseline_stats"][model_name]["f1"]
        lines.append(f"- {model_name}: F1=`{stat['mean']:.4f}` +/- `{stat['std']:.4f}`")
    lines.extend(
        [
            "",
            "## Notes",
            f"- Best non-physics baseline vs data_only p(F1)=`{summary['p_values']['best_nonphysics_vs_data_only_f1']:.6g}`",
            f"- Representative seed for case study: `{rep_seed}`",
            "- Physics remains a negative/secondary result on congestion.",
        ]
    )
    (output_dir / "congestion_focus_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_json


def main() -> None:
    p = argparse.ArgumentParser(description="Run the paper's main experiment chain")
    p.add_argument("--project-dir", default=str(REPO_ROOT))
    p.add_argument("--python-bin", default="python")
    p.add_argument("--real-collection-dir", default=str(DATA_ROOT))
    p.add_argument("--output-root", default=str(RUN_ROOT))
    p.add_argument("--seeds", default="11,22,33,44,55")
    p.add_argument("--congestion-scenario", default="scenario_e_three_tier_high2")
    p.add_argument("--congestion-graph", default="")
    p.add_argument("--congestion-seeds", default="11,22,33,44,55,66,77,88,99")
    p.add_argument("--congestion-epochs", type=int, default=140)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    project = Path(args.project_dir).resolve()
    output_root = Path(args.output_root).resolve()
    main_suite = output_root / "main_suite"
    congestion_out = output_root / "congestion_focus"
    cross_out = output_root / "cross_scenario"
    main_suite.mkdir(parents=True, exist_ok=True)
    congestion_out.mkdir(parents=True, exist_ok=True)
    cross_out.mkdir(parents=True, exist_ok=True)

    run(
        [
            args.python_bin,
            "internal/run_top_conference_suite.py",
            "--project-dir",
            str(project),
            "--python-bin",
            args.python_bin,
            "--output-dir",
            str(main_suite),
            "--real-collection-dir",
            str(Path(args.real_collection_dir).resolve()),
            "--seeds",
            args.seeds,
            "--include-congestion-ood",
            "--hard-overlap",
            "--camouflage-test-attacks",
            "--skip-federated",
        ],
        cwd=project,
    )

    run_baseline_significance(
        python_bin=args.python_bin,
        project=project,
        suite_dir=main_suite,
        seeds=[int(x.strip()) for x in args.seeds.split(",") if x.strip()],
        include_congestion_ood=True,
        save_predictions=True,
    )

    run_congestion_focus(
        python_bin=args.python_bin,
        project=project,
        suite_dir=main_suite,
        output_dir=congestion_out,
        scenario_dir=Path(args.real_collection_dir).resolve() / args.congestion_scenario,
        congestion_graph=args.congestion_graph,
        seeds=args.congestion_seeds,
        epochs=args.congestion_epochs,
        skip_existing=args.skip_existing,
    )

    cross_cmd = [
        args.python_bin,
        "internal/run_cross_scenario_suite.py",
        "--project-dir",
        str(project),
        "--python-bin",
        args.python_bin,
        "--real-collection-dir",
        str(Path(args.real_collection_dir).resolve()),
        "--output-dir",
        str(cross_out),
        "--train-scenarios",
        "scenario_d_three_tier_low2,scenario_e_three_tier_high2,scenario_f_two_tier_high2",
        "--val-scenarios",
        "scenario_g_mimic_congest",
        "--test-scenarios",
        "scenario_h_mimic_heavy_overlap",
        "--hard-overlap",
        "--train-keep-frac",
        "0.80",
        "--val-keep-frac",
        "0.85",
        "--test-keep-frac",
        "0.70",
        "--epochs",
        "120",
    ]
    if args.skip_existing:
        cross_cmd.append("--skip-existing")
    run(cross_cmd, cwd=project)


if __name__ == "__main__":
    main()
