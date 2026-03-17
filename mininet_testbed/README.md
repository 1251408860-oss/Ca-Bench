# Mininet Testbed: Scenario Generation and Traffic Capture

## Module Overview and Experimental Scope

This module implements the Mininet-based scenario generation component of Ca-Bench. It constructs the five paper scenarios and records the packet captures consumed by the offline benchmark pipeline.

The generated traffic traces serve as the primary data source for graph construction and downstream model evaluation.

## Environment and Dependency Preparation

Execution is intended for Ubuntu/Linux with Mininet installed and runnable via `sudo`. A Python environment is also required for the auxiliary traffic-generation scripts invoked by the topology runner.

The Mininet module requires the following system and Python dependencies:

- Ubuntu/Linux
- `sudo` access
- Mininet
- `tcpdump`
- Python 3
- `locust`
- `scapy`
- `requests`

The default batch script also requires a valid `LLM_API_KEY` or `DEEPSEEK_API_KEY`, because `run_capture_batch2.sh` enables real LLM payload generation by default.

This module must be executed in a real Mininet-capable environment. It is not a pure Python preprocessing step.

## Execution Entry

Generate the five paper scenarios and corresponding packet captures:

```bash
sudo -E bash run_capture_batch2.sh
```

## Output Artifacts

- Scenario manifests: `real_collection/scenario_*/arena_manifest_v2.json`
- Packet captures: `real_collection/scenario_*/full_arena_v2.pcap`
- Mininet logs: `real_collection/scenario_*/mininet.log`
