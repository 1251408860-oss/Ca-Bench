# Reproducibility Guide

This folder provides one-click scripts and environment locks for the FedSTGCN "recharge" package.

## Locked Environment Files
- `requirements-lock-dl.txt`: training/evaluation dependencies.
- `requirements-lock-plot.txt`: paper plotting dependencies.
- `environment-lock-dl.yml`: conda export for DL environment.
- `Dockerfile.eval`: containerized eval/plot environment (no Mininet capture).

## One-Click Evaluation (Ubuntu1)
From `/home/user/FedSTGCN`:

```bash
bash repro/run_oneclick_recharge.sh
```

Optional overrides:

```bash
PY_BIN=/home/user/miniconda3/envs/DL/bin/python \
OUT_DIR=/home/user/FedSTGCN/top_conf_suite_recharge \
SEEDS_STAGE3=11,22,33,44,55 \
SEEDS_FED9=11,22,33,44,55,66,77,88,99 \
bash repro/run_oneclick_recharge.sh
```

Expected key outputs:
- `top_conf_suite_recharge/top_conf_summary.json`
- `top_conf_suite_recharge/baseline_significance/baseline_significance_summary.json`
- `top_conf_suite_recharge/fed_sig_ext9/fed_sig_ext9_summary.json`
- `top_conf_suite_recharge/fed_cross_protocol/fed_cross_protocol_summary.json`
- `top_conf_suite_recharge/paper_ready_plus/MASTER_SUMMARY.md`
- `top_conf_suite_recharge/paper_ready_plus/multiple_testing_corrections.md`

## Independent Batch2 Capture (Ubuntu root)
Run as root in Mininet-capable distro (typically `Ubuntu`):

```bash
sudo -E bash repro/run_capture_batch2.sh
```

Required env vars:
- `LLM_API_KEY` or `DEEPSEEK_API_KEY`

Recommended stable LLM transport defaults (already set in script):
- `KEEP_PROXY=0`
- `LLM_TRANSPORT=requests`
- `LLM_TIMEOUT_SEC=120`

Expected outputs:
- `real_collection/scenario_i_three_tier_low_b2/`
- `real_collection/scenario_j_three_tier_high_b2/`
- `real_collection/scenario_k_two_tier_high_b2/`
- `real_collection/scenario_l_mimic_heavy_b2/`

Each scenario should contain:
- `full_arena_v2.pcap`
- `arena_manifest_v2.json` (with `run_config.arena_seed`)
- `mininet.log`

## Windows Launcher
From Windows PowerShell:

```powershell
.\repro\run_oneclick.ps1
```

## Docker (Evaluation/Plot Only)
Build inside repo root:

```bash
docker build -f repro/Dockerfile.eval -t fedstgcn-eval .
```

Run:

```bash
docker run --rm -it -v "$PWD:/workspace/FedSTGCN" fedstgcn-eval
```

Note:
- Container image is for model/eval/plot reproducibility only.
- Mininet real capture requires host networking privileges and is intentionally excluded from container workflow.
