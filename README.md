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

Packet captures (`full_arena_v2.pcap`) are generated locally by the Mininet module and are therefore not tracked in the repository.

No packaged lockfile is currently provided in the repository. Reviewers should prepare the runtime environment manually according to the dependency summaries below.

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
- `tcpdump`
- `locust`
- `scapy`
- `requests`
- A valid `LLM_API_KEY` or `DEEPSEEK_API_KEY` for the default scenario-generation script

### Special Mininet environment requirement

The benchmark does not start from a downloadable dataset. Reviewers must first run the Mininet module in a Mininet-capable Ubuntu/Linux environment to generate the five packet captures under `mininet_testbed/real_collection/scenario_*/full_arena_v2.pcap`, and only then run the offline benchmark pipeline.

Detailed execution commands, expected inputs, and output locations are documented in the `README.md` file of each submodule.
