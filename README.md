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

Packet captures (`full_arena_v2.pcap`) are distributed as a GitHub Release asset for offline reproduction. Reviewers who want to regenerate the traces from scratch may still run the Mininet module locally.

An example Conda environment file is provided as [`environment.yml`](./environment.yml). Reviewers may use it as a starting point for the Python runtime setup.

## Dataset Download

Download the packet-capture bundle for offline reproduction from the GitHub Release page:

- Release: [`data-v1`](https://github.com/1251408860-oss/Ca-Bench/releases/tag/data-v1)

From the repository root:

```bash
wget https://github.com/1251408860-oss/Ca-Bench/releases/download/data-v1/real_collection.tar.gz
wget https://github.com/1251408860-oss/Ca-Bench/releases/download/data-v1/real_collection.tar.gz.sha256
sha256sum -c real_collection.tar.gz.sha256
tar -xzf real_collection.tar.gz
```

This will populate:

```bash
mininet_testbed/real_collection/scenario_*/full_arena_v2.pcap
```

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

The Mininet scenario-generation script also requires a valid `LLM_API_KEY` or `DEEPSEEK_API_KEY` in the shell environment.

### Offline benchmark dependencies

- Python 3 on Ubuntu/Linux
- `numpy`
- `matplotlib`
- `torch`
- `torch-geometric`
- `scikit-learn`
- `flwr[simulation]` (installs the `ray` backend required by the federated experiments)
- `scapy`

### Mininet scenario-generation dependencies

- Ubuntu/Linux with `sudo`
- Mininet
- `openvswitch-switch`
- `tcpdump`
- `locust`
- `scapy`
- `requests`
- A valid `LLM_API_KEY` or `DEEPSEEK_API_KEY` for the default scenario-generation script

### Special Mininet environment requirement

The offline benchmark can start directly from the downloadable packet-capture bundle above. Running the Mininet module is only required for reviewers who want to regenerate the five packet captures from scratch in a Mininet-capable Ubuntu/Linux environment.

Detailed execution commands, expected inputs, and output locations are documented in the `README.md` file of each submodule.

For manuscript-facing paper outputs, `core_experiments/make_paper_tables_figs.py` now takes the `congestion_ood` values for Table 1 and Figure 1 from `paper_artifacts/runs/main_suite/` by default so the generated summary stays aligned with the submitted paper. The separate `congestion_focus` run remains available as an auxiliary stress sweep. The submitted table values are additionally frozen in [`paper_artifacts/manuscript_reference.json`](./paper_artifacts/manuscript_reference.json), which is used only when regenerating the paper-ready tables.
