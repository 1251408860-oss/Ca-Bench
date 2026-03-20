#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$ROOT_DIR/core_experiments"
DATA_DIR="$ROOT_DIR/mininet_testbed/real_collection"
PY_BIN="${PY_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/paper_artifacts/runs}"
SEEDS_STAGE3="${SEEDS_STAGE3:-11,22,33,44,55}"
USE_MANUSCRIPT_REFERENCE="${USE_MANUSCRIPT_REFERENCE:-0}"

MANUSCRIPT_ARGS=()
if [[ "$USE_MANUSCRIPT_REFERENCE" == "1" ]]; then
  MANUSCRIPT_ARGS+=(--use-manuscript-reference)
fi

cd "$PROJECT_DIR"

"$PY_BIN" main_suite.py \
  --project-dir "$PROJECT_DIR" \
  --python-bin "$PY_BIN" \
  --real-collection-dir "$DATA_DIR" \
  --output-root "$RUN_ROOT" \
  --seeds "$SEEDS_STAGE3" \
  --skip-existing

"$PY_BIN" system_suite.py \
  --project-dir "$PROJECT_DIR" \
  --python-bin "$PY_BIN" \
  --real-collection-dir "$DATA_DIR" \
  --output-root "$RUN_ROOT" \
  --skip-existing

"$PY_BIN" make_paper_tables_figs.py \
  --suite-dir "$RUN_ROOT/main_suite" \
  --cross-scenario-dir "$RUN_ROOT/cross_scenario" \
  --congestion-focus-dir "$RUN_ROOT/congestion_focus" \
  --network-sensitivity-dir "$RUN_ROOT/network_sensitivity" \
  --edge-suite-dir "$RUN_ROOT/edge_budget" \
  --overhead-dir "$RUN_ROOT/system_overhead" \
  "${MANUSCRIPT_ARGS[@]}" \
  --output-dir "$ROOT_DIR/paper_artifacts"

echo "[DONE] runs: $RUN_ROOT"
echo "[DONE] paper outputs: $ROOT_DIR/paper_artifacts"
