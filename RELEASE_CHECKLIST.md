# Ca-Bench: Release Preparation Notes

This document summarizes how the repository should be packaged when preparing a direct GitHub push or a public artifact release. The goal is to keep the tracked repository tree aligned with the online documentation while separating large mirrored assets and local validation outputs from the main source archive.

## Repository Tree Intended for Direct Publication

The tracked repository should retain the core documentation, environment description, experiment code, shipped packet captures, and paper-facing outputs. In practical terms, the public tree is expected to include `README.md`, `environment.yml`, `.gitignore`, `core_experiments/`, `mininet_testbed/`, `paper_artifacts/`, and `repro/`.

Within this tree, the five paper packet captures remain part of the repository checkout under `mininet_testbed/real_collection/`. The released run summaries under `paper_artifacts/runs/`, the exported tables under `paper_artifacts/tables/`, the exported figures under `paper_artifacts/figures/`, and the optional frozen manuscript reference at `paper_artifacts/manuscript_reference.json` are also intended to stay in version control because they support direct inspection of the published artifact.

## Mirrored Dataset Bundle

The files `real_collection.tar.gz` and `real_collection.tar.gz.sha256` should be treated as release assets rather than ordinary repository files. They are useful as a mirrored download path for reviewers who prefer a separate archive, but they duplicate the captures that are already shipped in `mininet_testbed/real_collection/`.

For that reason, the recommended release flow is to keep the repository tree focused on the checked-in artifact contents and upload the tarball pair to the GitHub release page, currently referenced as `data-v1`.

## Local-Only Working Outputs

The directory `repro_runs/` is a local validation workspace and should not be part of the public repository history. The same applies to unrelated working notes such as `BLOCKSYS_HITRUST_EXPERIMENT_DEV_DOC.md`. These files are useful during development, but they are not part of the published Ca-Bench artifact package and should remain excluded from normal pushes.

## Pre-Push Verification

Before pushing a release-oriented update, verify that the five shipped captures are still present under `mininet_testbed/real_collection/scenario_*/full_arena_v2.pcap`, that the public summaries remain available under `paper_artifacts/runs/`, and that the exported tables and figures are present under `paper_artifacts/tables/` and `paper_artifacts/figures/`.

It is also worth confirming that the main documentation files, including the repository root `README.md`, `core_experiments/README.md`, `mininet_testbed/README.md`, and `repro/README.md`, remain consistent about three points: the packet captures are bundled in the checkout, the GitHub release provides only an optional mirror, and API keys are only needed when regenerating the payload file rather than when replaying the default path.

Finally, check that stale project references from earlier drafts are not reintroduced into the published documentation. In particular, the public-facing files should not refer to `FedSTGCN`, `run_oneclick_recharge`, `run_oneclick.ps1`, or the old hard-coded interpreter path `/home/user/miniconda3/envs/DL/bin/python`.

## Recommended Release Flow

The simplest release path is to push the cleaned repository tree first, without `repro_runs/`, the mirrored tarball files, or unrelated local notes. After that, create or update the GitHub release entry and attach `real_collection.tar.gz` together with `real_collection.tar.gz.sha256`. Once the release assets are in place, perform one final check to ensure that the download links referenced in `README.md` still match the published release page.
