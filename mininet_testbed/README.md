# Mininet Testbed: Scenario Generation and Traffic Capture

## Module Overview and Experimental Scope

This module implements the Mininet-based scenario generation component of Ca-Bench. It constructs the five paper scenarios and records the packet captures consumed by the offline benchmark pipeline.

The generated traffic traces serve as the primary data source for graph construction and downstream model evaluation.

The repository already bundles the five paper captures under `real_collection/scenario_*/full_arena_v2.pcap`, so rerunning this module is optional for reviewers who only need offline reproduction.

## Environment and Dependency Preparation

Execution is intended for Ubuntu/Linux with Mininet installed and runnable via `sudo`. A Python environment is also required for the auxiliary traffic-generation scripts invoked by the topology runner.

Recommended installation on Ubuntu:

```bash
conda env create -f ../environment.yml
conda activate cabench
sudo apt-get update
sudo apt-get install -y mininet openvswitch-switch tcpdump
```

The Mininet module requires the following system and Python dependencies:

- Ubuntu/Linux
- `sudo` access
- Mininet
- `openvswitch-switch`
- `tcpdump`
- Python 3
- `locust`
- `scapy`
- `requests`

The default batch script also requires a valid `LLM_API_KEY` or `DEEPSEEK_API_KEY`, because `run_capture_batch2.sh` enables real LLM payload generation by default.

This module must be executed in a real Mininet-capable environment. It is not a pure Python preprocessing step.

## Execution Entry

Regenerate the five paper scenarios and corresponding packet captures:

```bash
sudo -E bash run_capture_batch2.sh
```

## Output Artifacts

- Scenario manifests: `real_collection/scenario_*/arena_manifest_v2.json`
- Packet captures: `real_collection/scenario_*/full_arena_v2.pcap`
- Mininet logs: `real_collection/scenario_*/mininet.log`

For most artifact evaluation workflows, reviewers can skip this regeneration step and directly use the bundled captures already present in `real_collection/`.
