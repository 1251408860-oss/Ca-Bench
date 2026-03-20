# Ca-Bench Release Checklist

This checklist is intended for preparing the repository tree for a direct push to GitHub.

## Keep In The Repository

- `README.md`
- `environment.yml`
- `.gitignore`
- `core_experiments/`
- `mininet_testbed/`
- `mininet_testbed/real_collection/`
- `paper_artifacts/runs/`
- `paper_artifacts/tables/`
- `paper_artifacts/figures/`
- `paper_artifacts/manuscript_reference.json`
- `repro/`

## Publish As Release Assets, Not As Repo Files

- `real_collection.tar.gz`
- `real_collection.tar.gz.sha256`

Reason:
- the packet captures are already shipped inside `mininet_testbed/real_collection/`
- the tarball is only a mirror bundle for reviewers who prefer a separate download

## Keep Local Only

- `repro_runs/`
- `BLOCKSYS_HITRUST_EXPERIMENT_DEV_DOC.md`

Reason:
- these are local validation outputs or unrelated working notes
- they do not belong to the public artifact tree

## Pre-Push Verification

1. Confirm the five shipped captures exist under `mininet_testbed/real_collection/scenario_*/full_arena_v2.pcap`.
2. Confirm the shipped summaries exist under `paper_artifacts/runs/`.
3. Confirm tables exist under `paper_artifacts/tables/`.
4. Confirm figures exist under `paper_artifacts/figures/`.
5. Confirm `README.md`, `core_experiments/README.md`, `mininet_testbed/README.md`, and `repro/README.md` are consistent about:
   - bundled packet captures
   - optional release mirror
   - API key only being needed for payload regeneration
   - `python` being the default interpreter
6. Confirm no stale references remain to:
   - `FedSTGCN`
   - `run_oneclick_recharge`
   - `run_oneclick.ps1`
   - `/home/user/miniconda3/envs/DL/bin/python`

## Recommended Push Set

If you want a clean first commit or upload, the effective publish tree is:

- `README.md`
- `environment.yml`
- `.gitignore`
- `core_experiments/`
- `mininet_testbed/`
- `paper_artifacts/`
- `repro/`

## Recommended Release Flow

1. Push the repository tree without `repro_runs/`, the mirror tarball, or unrelated notes.
2. Create or update the GitHub release tag `data-v1`.
3. Upload `real_collection.tar.gz` and `real_collection.tar.gz.sha256` to that release.
4. Recheck the release links referenced in `README.md`.
