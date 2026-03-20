# Reproducibility Guide

This folder provides the locked environments and the optional evaluation container for `Ca-Bench`. The runnable entrypoints live under `core_experiments/` and `mininet_testbed/`.

## Locked Environment Files
- `requirements-lock-dl.txt`: training and evaluation dependencies
- `requirements-lock-plot.txt`: plotting dependencies
- `environment-lock-dl.yml`: exported conda environment for the original evaluation stack
- `Dockerfile.eval`: containerized evaluation and plotting environment without Mininet capture support

## One-Click Full Evaluation
From the repository root:

```bash
bash core_experiments/run_full_eval.sh
```

Optional overrides:

```bash
PY_BIN=python \
RUN_ROOT="$PWD/paper_artifacts/runs" \
SEEDS_STAGE3=11,22,33,44,55 \
USE_MANUSCRIPT_REFERENCE=1 \
bash core_experiments/run_full_eval.sh
```

Expected key outputs:
- `paper_artifacts/runs/main_suite/top_conf_summary.json`
- `paper_artifacts/runs/cross_scenario/cross_scenario_summary.json`
- `paper_artifacts/runs/congestion_focus/congestion_focus_summary.json`
- `paper_artifacts/runs/network_sensitivity/network_sensitivity_summary.json`
- `paper_artifacts/runs/edge_budget/edge_suite_summary.json`
- `paper_artifacts/runs/system_overhead/overhead_summary.json`
- `paper_artifacts/tables/*.csv`
- `paper_artifacts/figures/*.png`

## Optional Packet-Capture Regeneration
Run as root on a Mininet-capable Linux host:

```bash
sudo -E bash mininet_testbed/run_capture_batch2.sh
```

The default path reuses the bundled `mininet_testbed/llm_payloads.json` and does not require an API key. A compatible `LLM_API_KEY`, `DEEPSEEK_API_KEY`, or `OPENAI_API_KEY` is only needed when regenerating payloads from scratch, for example:

```bash
sudo -E REGENERATE_PAYLOADS=1 REQUIRE_REAL_LLM=1 LLM_API_KEY=... \
  bash mininet_testbed/run_capture_batch2.sh
```

Expected outputs:
- `mininet_testbed/real_collection/scenario_*/full_arena_v2.pcap`
- `mininet_testbed/real_collection/scenario_*/arena_manifest_v2.json`
- `mininet_testbed/real_collection/scenario_*/mininet.log`

## Docker (Evaluation and Plotting Only)
Build inside the repo root:

```bash
docker build -f repro/Dockerfile.eval -t cabench-eval .
```

Run:

```bash
docker run --rm -it -v "$PWD:/workspace/Ca-Bench" cabench-eval
```

Notes:
- The container is intended for model evaluation and figure generation only.
- Mininet capture requires host networking privileges and is intentionally excluded from the container workflow.
