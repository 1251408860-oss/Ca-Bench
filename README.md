# Ca-Bench: Congestion-Aware Benchmarking of Web Bot Detection at the Edge

This repository contains the anonymous implementation and artifact package for the paper **"Congestion-Aware Benchmarking of Web Bot Detection at the Edge."**

To facilitate reproducibility and provide a clear review path, the package is organized into two core modules. Reviewers may navigate to the corresponding directory according to the component of the paper they intend to inspect or reproduce.

## Repository Navigation & Artifact Mapping

| Experiment Scope | Description | Directory Link |
| :--- | :--- | :--- |
| Offline benchmark pipeline | Graph construction from captured traffic, protocol split preparation, centralized and federated evaluation, and paper-ready tables and figures | [`core_experiments`](./core_experiments) |
| Mininet scenario generation | Traffic generation and packet capture for the five paper scenarios used throughout the benchmark | [`mininet_testbed`](./mininet_testbed) |

## Global Environment Overview

The artifact package is designed for Ubuntu/Linux workflows. The online scenario generation module requires a Mininet-enabled environment with root privileges, while the offline benchmark module requires a Python environment with the project dependencies installed.

For offline reproduction, the five paper scenario captures are bundled under `mininet_testbed/real_collection/scenario_*/full_arena_v2.pcap`. Reviewers may reproduce the paper tables and figures directly from these captures without rerunning Mininet.

Mininet regeneration remains available as an optional end-to-end validation path for reviewers who want to reproduce the online data-collection stage itself.

An example Conda environment file is provided as [`environment.yml`](./environment.yml). Reviewers may use it as a starting point for the Python runtime setup.

Locked evaluation environment files are also provided under [`repro`](./repro).

## Quick Installation

Create the Python environment:

```bash
conda env create -f environment.yml
conda activate cabench
```

Install the Mininet-side system dependencies on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y mininet openvswitch-switch tcpdump
```

The Mininet scenario-generation script requires a valid `LLM_API_KEY` or `DEEPSEEK_API_KEY` only when reviewers choose to regenerate packet captures from scratch.

### Offline benchmark dependencies

- Python 3 on Ubuntu/Linux
- `numpy`
- `matplotlib`
- `torch`
- `torch-geometric`
- `scikit-learn`
- `flwr`
- `scapy`

### Mininet scenario-generation dependencies

- Ubuntu/Linux with `sudo`
- Mininet
- `openvswitch-switch`
- `tcpdump`
- `locust`
- `scapy`
- `requests`
- A valid `LLM_API_KEY`, `DEEPSEEK_API_KEY`, or `OPENAI_API_KEY` only when reviewers explicitly regenerate LLM payloads from scratch

### Offline-first reproduction path

The repository already ships the five packet captures used by the paper under `mininet_testbed/real_collection/scenario_*/full_arena_v2.pcap`. Reviewers can therefore start directly from the offline benchmark pipeline.

For paper table and figure regeneration, the repository also includes the precomputed run summaries consumed by `core_experiments/make_paper_tables_figs.py` under `paper_artifacts/runs/`.

### Optional Mininet regeneration path

Reviewers who wish to reproduce the traffic-generation stage itself may rerun the Mininet module in a Mininet-capable Ubuntu/Linux environment. The default batch script reuses the bundled `mininet_testbed/llm_payloads.json` and therefore does not require any API key. Real LLM payload regeneration remains optional via `REGENERATE_PAYLOADS=1`, in which case a reviewer-provided compatible API key is required.

Detailed execution commands, expected inputs, and output locations are documented in the `README.md` file of each submodule.
