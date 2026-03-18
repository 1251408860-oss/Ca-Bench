#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "[ERROR] run_capture_batch2.sh must be run as root (sudo)."
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTBED_DIR="$REPO_ROOT/mininet_testbed"
DATA_DIR="$TESTBED_DIR/real_collection"
PY_BIN="${PY_BIN:-/home/user/miniconda3/envs/DL/bin/python}"
PAYLOAD_FILE="$TESTBED_DIR/llm_payloads.json"

NUM_LLM_SESSIONS="${NUM_LLM_SESSIONS:-60}"
NUM_TOTAL_PAYLOADS="${NUM_TOTAL_PAYLOADS:-8000}"

export KEEP_PROXY="${KEEP_PROXY:-0}"
export LLM_TRANSPORT="${LLM_TRANSPORT:-requests}"
export LLM_TIMEOUT_SEC="${LLM_TIMEOUT_SEC:-120}"
export REQUIRE_REAL_LLM="${REQUIRE_REAL_LLM:-0}"
export REGENERATE_PAYLOADS="${REGENERATE_PAYLOADS:-0}"
export NUM_LLM_SESSIONS
export NUM_TOTAL_PAYLOADS

if [[ ("$REGENERATE_PAYLOADS" == "1" || ! -s "$PAYLOAD_FILE") && "$REQUIRE_REAL_LLM" == "1" && -z "${LLM_API_KEY:-${DEEPSEEK_API_KEY:-${OPENAI_API_KEY:-}}}" ]]; then
  echo "[ERROR] missing LLM_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY in environment"
  exit 1
fi

run() {
  echo "[RUN] $*"
  "$@"
}

capture() {
  local name="$1"
  local topo="$2"
  local load="$3"
  local bot_mode="$4"
  local seed="$5"
  local users="$6"
  local bots="$7"
  local duration_sec="$8"
  local core_bw="$9"
  local core_delay="${10}"
  local core_queue="${11}"

  local out_dir="$DATA_DIR/$name"
  mkdir -p "$out_dir"

  local pcap="$out_dir/full_arena_v2.pcap"
  local manifest="$out_dir/arena_manifest_v2.json"
  local tmp_pcap="$out_dir/full_arena_v2.pcap.tmp"
  local tmp_manifest="$out_dir/arena_manifest_v2.json.tmp"
  local log="$out_dir/mininet.log"

  echo "[SCENARIO] $name topo=$topo load=$load bot_mode=$bot_mode seed=$seed users=$users bots=$bots bw=$core_bw delay=$core_delay queue=$core_queue"

  rm -f "$tmp_pcap" "$tmp_manifest"

  set +e
  TOPOLOGY_MODE="$topo" \
  LOAD_PROFILE="$load" \
  BOT_TYPE_MODE="$bot_mode" \
  ARENA_SEED="$seed" \
  NUM_USERS="$users" \
  NUM_BOTS="$bots" \
  CORE_BW="$core_bw" \
  CORE_DELAY="$core_delay" \
  CORE_QUEUE="$core_queue" \
  BENIGN_ENGINE=locust \
  ATTACK_ENGINE=http \
  REQUIRE_REAL_LLM="$REQUIRE_REAL_LLM" \
  PYTHON_BIN="$PY_BIN" \
  PCAP_FILE="$tmp_pcap" \
  MANIFEST_FILE="$tmp_manifest" \
  "$PY_BIN" "$TESTBED_DIR/mininet_arena_v2.py" "$duration_sec" > "$log" 2>&1
  local rc=$?
  set -e

  if [[ ! -s "$tmp_pcap" || ! -s "$tmp_manifest" ]]; then
    rm -f "$tmp_pcap" "$tmp_manifest"
    echo "[ERROR] capture artifacts missing for $name (rc=$rc)"
    exit 1
  fi

  mv -f "$tmp_pcap" "$pcap"
  mv -f "$tmp_manifest" "$manifest"

  echo "[OK] $name rc=$rc pcap=$(du -h "$pcap" | awk '{print $1}')"
}

cd "$TESTBED_DIR"
if [[ "$REGENERATE_PAYLOADS" == "1" || ! -s "$PAYLOAD_FILE" ]]; then
  run "$PY_BIN" generate_llm_payloads.py
else
  echo "[INFO] using bundled payload file: $PAYLOAD_FILE"
fi

capture "scenario_d_three_tier_low2"     "three_tier" "low"    "mixed"       "20260310" "20" "60"  "180" "10" "5ms" "1000"
capture "scenario_e_three_tier_high2"    "three_tier" "high"   "mixed"       "20260311" "25" "80"  "180" "8"  "6ms" "900"
capture "scenario_f_two_tier_high2"      "two_tier"   "high"   "mixed"       "20260312" "25" "80"  "180" "8"  "6ms" "900"
capture "scenario_g_mimic_congest"       "three_tier" "medium" "all_mimic"   "20260313" "20" "100" "150" "4"  "8ms" "800"
capture "scenario_h_mimic_heavy_overlap" "three_tier" "high"   "mimic_heavy" "20260314" "35" "70"  "180" "6"  "8ms" "700"

echo "[DONE] paper scenarios completed under $DATA_DIR"
