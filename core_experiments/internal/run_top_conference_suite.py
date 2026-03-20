#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DATA_ROOT = REPO_ROOT.parent / "mininet_testbed" / "real_collection"
RUN_ROOT = REPO_ROOT.parent / "paper_artifacts" / "runs"


def run_cmd(cmd: list[str], cwd: Path, log_file: Path) -> tuple[int, float]:
    t0 = time.time()
    with log_file.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
    return int(p.returncode), float(time.time() - t0)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def summary_stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0, "mean": 0.0, "std": 0.0}
    if len(vals) == 1:
        return {"n": 1, "mean": float(vals[0]), "std": 0.0}
    return {"n": len(vals), "mean": float(statistics.mean(vals)), "std": float(statistics.stdev(vals))}


def paired_signflip_pvalue(x: list[float], y: list[float], n_perm: int = 4096) -> float:
    if len(x) != len(y) or len(x) == 0:
        return 1.0
    d = [float(a - b) for a, b in zip(x, y)]
    obs = abs(sum(d) / len(d))
    if obs <= 1e-12:
        return 1.0

    n = len(d)
    total = 0
    extreme = 0
    if n <= 12:
        for bits in itertools.product([-1.0, 1.0], repeat=n):
            total += 1
            stat = abs(sum(di * si for di, si in zip(d, bits)) / n)
            if stat >= obs - 1e-12:
                extreme += 1
    else:
        import random

        rng = random.Random(42)
        for _ in range(n_perm):
            total += 1
            stat = abs(sum(di * (1.0 if rng.random() > 0.5 else -1.0) for di in d) / n)
            if stat >= obs - 1e-12:
                extreme += 1
    return float((extreme + 1) / (total + 1))


def pick_test_metrics(phase3: dict[str, Any]) -> dict[str, float]:
    fe = phase3.get("final_eval", {})
    if isinstance(fe.get("test_temporal"), dict):
        return fe["test_temporal"]
    if isinstance(fe.get("test_random"), dict):
        return fe["test_random"]
    return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}


def pick_fed_metrics(fed: dict[str, Any]) -> dict[str, float]:
    gm = fed.get("global_metrics", {})
    if isinstance(gm.get("test_temporal"), dict):
        return gm["test_temporal"]
    if isinstance(gm.get("test_random"), dict):
        return gm["test_random"]
    return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "fpr": 0.0}


def main() -> None:
    p = argparse.ArgumentParser(description="Run high-standard suite")
    p.add_argument("--project-dir", default=str(REPO_ROOT))
    p.add_argument("--python-bin", default="python")
    p.add_argument("--output-dir", default=str(RUN_ROOT / "main_suite"))
    p.add_argument("--real-collection-dir", default=str(DATA_ROOT))
    p.add_argument("--seeds", default="11,22,33,44,55")
    p.add_argument("--stage3-epochs", type=int, default=140)
    p.add_argument("--fed-rounds", type=int, default=4)
    p.add_argument("--fed-local-epochs", type=int, default=2)
    p.add_argument("--num-clients", type=int, default=3)
    p.add_argument("--client-cpus", type=float, default=2.0)
    p.add_argument("--capacity", type=float, default=0.0)
    p.add_argument("--capacity-mode", choices=["auto", "mbps", "bytes_per_sec"], default="auto")
    p.add_argument("--fed-poison-mode", choices=["update_noise", "clean_label_local", "feature_camouflage_local"], default="update_noise")
    p.add_argument("--poison-node-frac", type=float, default=0.5)
    p.add_argument("--include-congestion-ood", action="store_true")
    p.add_argument("--hard-overlap", action="store_true")
    p.add_argument("--camouflage-test-attacks", action="store_true")
    p.add_argument("--skip-federated", action="store_true")
    p.add_argument("--scenario-low", default="scenario_d_three_tier_low2")
    p.add_argument("--scenario-high", default="scenario_e_three_tier_high2")
    p.add_argument("--scenario-two-tier", default="scenario_f_two_tier_high2")
    args = p.parse_args()

    project = Path(args.project_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    py = args.python_bin

    summary: dict[str, Any] = {
        "config": vars(args),
        "timestamps": {"start": time.strftime("%Y-%m-%d %H:%M:%S")},
        "real_capture_graphs": {},
        "protocol_graphs": {},
        "stage3_runs": [],
        "federated_runs": [],
        "statistics": {},
    }

    scenarios = {
        str(args.scenario_low): {"target_ip": "10.0.0.100"},
        str(args.scenario_high): {"target_ip": "10.0.0.100"},
        str(args.scenario_two_tier): {"target_ip": "10.0.0.100"},
    }
    for sc, cfg in scenarios.items():
        sc_dir = Path(args.real_collection_dir) / sc
        pcap = sc_dir / "full_arena_v2.pcap"
        manifest = sc_dir / "arena_manifest_v2.json"
        graph_out = out_dir / "graphs" / f"{sc}.pt"
        graph_out.parent.mkdir(parents=True, exist_ok=True)
        log = out_dir / "logs" / f"build_graph_{sc}.log"
        log.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            py,
            "build_graph_v2.py",
            "--pcap-file",
            str(pcap),
            "--manifest-file",
            str(manifest),
            "--output-file",
            str(graph_out),
            "--target-ip",
            cfg["target_ip"],
            "--delta-t",
            "1.0",
            "--seed",
            "42",
        ]
        rc, sec = run_cmd(cmd, cwd=project, log_file=log)
        if rc != 0:
            raise RuntimeError(f"Build graph failed for {sc}, see {log}")
        summary["real_capture_graphs"][sc] = {"graph_file": str(graph_out), "log": str(log), "duration_sec": sec}

    base_graph = Path(summary["real_capture_graphs"][str(args.scenario_high)]["graph_file"])
    base_manifest = Path(args.real_collection_dir) / str(args.scenario_high) / "arena_manifest_v2.json"

    protocols = {
        "temporal_ood": {"holdout": "mimic"},
        "topology_ood": {"holdout": "mimic"},
        "attack_strategy_ood": {"holdout": "mimic"},
    }
    if bool(args.include_congestion_ood):
        protocols["congestion_ood"] = {"holdout": "mimic"}
    for proto, pcfg in protocols.items():
        gout = out_dir / "protocol_graphs" / f"{proto}.pt"
        gout.parent.mkdir(parents=True, exist_ok=True)
        log = out_dir / "logs" / f"protocol_{proto}.log"
        prep_script = "prepare_leakage_protocol_graph.py"
        cmd = [py]
        if proto == "congestion_ood":
            prep_script = "prepare_hard_protocol_graph.py"
        cmd.extend(
            [
                prep_script,
                "--input-graph",
                str(base_graph),
                "--output-graph",
                str(gout),
                "--protocol",
                proto,
                "--manifest-file",
                str(base_manifest),
                "--holdout-attack-type",
                pcfg["holdout"],
                "--seed",
                "42",
            ]
        )
        if prep_script == "prepare_hard_protocol_graph.py":
            if bool(args.hard_overlap):
                cmd.append("--hard-overlap")
            if bool(args.camouflage_test_attacks):
                cmd.append("--camouflage-test-attacks")
        rc, sec = run_cmd(cmd, cwd=project, log_file=log)
        if rc != 0:
            raise RuntimeError(f"Protocol graph failed for {proto}, see {log}")
        summary["protocol_graphs"][proto] = {"graph_file": str(gout), "log": str(log), "duration_sec": sec}

    stage3_cfgs = {
        "data_only": {"alpha": 0.0, "beta": 0.0},
        "physics_stable": {"alpha": 0.03, "beta": 0.02},
    }
    for proto, pinfo in summary["protocol_graphs"].items():
        graph_file = pinfo["graph_file"]
        poison_cases = [("clean", 0.0)]
        if proto == "attack_strategy_ood":
            poison_cases.extend([("poison20", 0.20), ("poison35", 0.35)])

        for poison_name, poison_frac in poison_cases:
            for seed in seeds:
                for mname, mcfg in stage3_cfgs.items():
                    exp_id = f"{proto}__{poison_name}__{mname}__seed{seed}"
                    exp_dir = out_dir / "stage3" / exp_id
                    exp_dir.mkdir(parents=True, exist_ok=True)
                    model_file = exp_dir / "pi_gnn_model.pt"
                    result_file = exp_dir / "phase3_results.json"
                    log = exp_dir / "run.log"

                    cmd = [
                        py,
                        "pi_gnn_train_v2.py",
                        "--graph-file",
                        graph_file,
                        "--model-file",
                        str(model_file),
                        "--results-file",
                        str(result_file),
                        "--epochs",
                        str(args.stage3_epochs),
                        "--alpha-flow",
                        str(mcfg["alpha"]),
                        "--beta-latency",
                        str(mcfg["beta"]),
                        "--capacity",
                        str(args.capacity),
                        "--capacity-mode",
                        args.capacity_mode,
                        "--warmup-epochs",
                        "25",
                        "--patience",
                        "35",
                        "--train-poison-frac",
                        str(poison_frac),
                        "--seed",
                        str(seed),
                        "--force-cpu",
                    ]
                    rc, sec = run_cmd(cmd, cwd=project, log_file=log)
                    if rc != 0:
                        raise RuntimeError(f"Stage3 failed: {exp_id}, see {log}")
                    result = load_json(result_file)
                    metrics = pick_test_metrics(result)
                    summary["stage3_runs"].append(
                        {
                            "id": exp_id,
                            "protocol": proto,
                            "poison_case": poison_name,
                            "train_poison_frac": poison_frac,
                            "model": mname,
                            "seed": seed,
                            "duration_sec": sec,
                            "result_file": str(result_file),
                            "metrics": metrics,
                        }
                    )

    poison_cases: list[dict[str, float | str]] = []
    aggregators: list[str] = []
    if not bool(args.skip_federated):
        fed_graph = summary["protocol_graphs"]["temporal_ood"]["graph_file"]
        poison_cases = [
            {"name": "clean", "frac": 0.0, "scale": 0.0},
            {"name": "poison_mid", "frac": 0.2, "scale": 0.2},
            {"name": "poison_high", "frac": 0.4, "scale": 0.4},
        ]
        aggregators = ["fedavg", "median", "trimmed_mean", "shapley_proxy", "krum"]

        for seed in seeds:
            for pc in poison_cases:
                for agg in aggregators:
                    exp_id = f"{pc['name']}__{agg}__seed{seed}"
                    exp_dir = out_dir / "fed" / exp_id
                    exp_dir.mkdir(parents=True, exist_ok=True)
                    model_file = exp_dir / "fed_pignn_model.pt"
                    result_file = exp_dir / "phase4_federated_results.json"
                    log = exp_dir / "run.log"
                    cmd = [
                        py,
                        "fed_pignn.py",
                        "--graph-file",
                        fed_graph,
                        "--model-file",
                        str(model_file),
                        "--results-file",
                        str(result_file),
                        "--num-clients",
                        str(args.num_clients),
                        "--rounds",
                        str(args.fed_rounds),
                        "--local-epochs",
                        str(args.fed_local_epochs),
                        "--aggregation",
                        agg,
                        "--simulate-poison-frac",
                        str(pc["frac"]),
                        "--poison-scale",
                        str(pc["scale"]),
                        "--poison-mode",
                        args.fed_poison_mode,
                        "--poison-node-frac",
                        str(args.poison_node_frac),
                        "--alpha-flow",
                        "0.03",
                        "--beta-latency",
                        "0.02",
                        "--capacity",
                        str(args.capacity),
                        "--capacity-mode",
                        args.capacity_mode,
                        "--warmup-rounds",
                        "2",
                        "--seed",
                        str(seed),
                        "--client-cpus",
                        str(args.client_cpus),
                        "--client-gpus",
                        "0.0",
                        "--force-cpu",
                    ]
                    rc, sec = run_cmd(cmd, cwd=project, log_file=log)
                    if rc != 0:
                        raise RuntimeError(f"Federated run failed: {exp_id}, see {log}")
                    result = load_json(result_file)
                    metrics = pick_fed_metrics(result)
                    summary["federated_runs"].append(
                        {
                            "id": exp_id,
                            "seed": seed,
                            "poison_case": pc["name"],
                            "poison_frac": pc["frac"],
                            "poison_scale": pc["scale"],
                            "aggregation": agg,
                            "poison_mode": args.fed_poison_mode,
                            "duration_sec": sec,
                            "result_file": str(result_file),
                            "metrics": metrics,
                        }
                    )

    stage3_stats: dict[str, Any] = {}
    for proto in protocols.keys():
        stage3_stats[proto] = {}
        proto_poison_cases = sorted({str(r.get("poison_case", "clean")) for r in summary["stage3_runs"] if r["protocol"] == proto})
        for pcase in proto_poison_cases:
            stage3_stats[proto][pcase] = {}
            for mname in stage3_cfgs.keys():
                runs = [
                    r for r in summary["stage3_runs"]
                    if r["protocol"] == proto and str(r.get("poison_case", "clean")) == pcase and r["model"] == mname
                ]
                f1 = [float(r["metrics"].get("f1", 0.0)) for r in runs]
                rec = [float(r["metrics"].get("recall", 0.0)) for r in runs]
                stage3_stats[proto][pcase][mname] = {"f1": summary_stats(f1), "recall": summary_stats(rec)}

            phys = [
                float(r["metrics"].get("f1", 0.0))
                for r in summary["stage3_runs"]
                if r["protocol"] == proto and str(r.get("poison_case", "clean")) == pcase and r["model"] == "physics_stable"
            ]
            data = [
                float(r["metrics"].get("f1", 0.0))
                for r in summary["stage3_runs"]
                if r["protocol"] == proto and str(r.get("poison_case", "clean")) == pcase and r["model"] == "data_only"
            ]
            p_f1 = paired_signflip_pvalue(phys, data)
            phys_r = [
                float(r["metrics"].get("recall", 0.0))
                for r in summary["stage3_runs"]
                if r["protocol"] == proto and str(r.get("poison_case", "clean")) == pcase and r["model"] == "physics_stable"
            ]
            data_r = [
                float(r["metrics"].get("recall", 0.0))
                for r in summary["stage3_runs"]
                if r["protocol"] == proto and str(r.get("poison_case", "clean")) == pcase and r["model"] == "data_only"
            ]
            p_rec = paired_signflip_pvalue(phys_r, data_r)
            stage3_stats[proto][pcase]["significance_vs_data_only"] = {"p_value_f1": p_f1, "p_value_recall": p_rec}

    fed_stats: dict[str, Any] = {}
    for pc in poison_cases:
        fed_stats[pc["name"]] = {}
        for agg in aggregators:
            runs = [r for r in summary["federated_runs"] if r["poison_case"] == pc["name"] and r["aggregation"] == agg]
            f1 = [float(r["metrics"].get("f1", 0.0)) for r in runs]
            rec = [float(r["metrics"].get("recall", 0.0)) for r in runs]
            fpr = [float(r["metrics"].get("fpr", 0.0)) for r in runs]
            fed_stats[pc["name"]][agg] = {
                "f1": summary_stats(f1),
                "recall": summary_stats(rec),
                "fpr": summary_stats(fpr),
            }

        shapley = [float(r["metrics"].get("f1", 0.0)) for r in summary["federated_runs"] if r["poison_case"] == pc["name"] and r["aggregation"] == "shapley_proxy"]
        fedavg = [float(r["metrics"].get("f1", 0.0)) for r in summary["federated_runs"] if r["poison_case"] == pc["name"] and r["aggregation"] == "fedavg"]
        fed_stats[pc["name"]]["significance_shapley_vs_fedavg_f1"] = paired_signflip_pvalue(shapley, fedavg)

    summary["statistics"] = {"stage3": stage3_stats, "federated": fed_stats}
    summary["timestamps"]["end"] = time.strftime("%Y-%m-%d %H:%M:%S")

    out_file = out_dir / "top_conf_summary.json"
    save_json(out_file, summary)
    print(f"[DONE] High-standard suite complete: {out_file}")


if __name__ == "__main__":
    main()
