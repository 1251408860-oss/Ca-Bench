#!/usr/bin/env python3
"""
Phase 1 - Full-Scale Mininet Physical Arena v2

Three-layer topology:
  Core(1) -> Aggregation(3) -> Edge(6) -> 91 hosts

Environment knobs:
  BENIGN_ENGINE=locust|script   (default: locust)
  ATTACK_ENGINE=http|scapy      (default: http)
  REQUIRE_REAL_LLM=0|1          (default: 0)
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Any

from mininet.link import Link, TCLink
from mininet.log import info, setLogLevel
from mininet.net import Mininet
from mininet.node import Controller, OVSKernelSwitch, UserSwitch

for v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(v, None)
os.environ["no_proxy"] = "*"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.expanduser("~")

DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 600
NUM_USERS = int(os.environ.get("NUM_USERS", "20"))
NUM_BOTS = int(os.environ.get("NUM_BOTS", "70"))
TARGET_IP = "10.0.0.100"

TOPOLOGY_MODE = os.environ.get("TOPOLOGY_MODE", "three_tier").strip().lower()
LOAD_PROFILE = os.environ.get("LOAD_PROFILE", "medium").strip().lower()

CORE_BW = int(os.environ.get("CORE_BW", "10"))
CORE_DELAY = os.environ.get("CORE_DELAY", "5ms")
CORE_QUEUE = int(os.environ.get("CORE_QUEUE", "1000"))

PCAP_FILE = os.path.abspath(os.environ.get("PCAP_FILE", os.path.join(BASE_DIR, "full_arena_v2.pcap")))
MANIFEST_FILE = os.path.abspath(os.environ.get("MANIFEST_FILE", os.path.join(BASE_DIR, "arena_manifest_v2.json")))

PYTHON3 = os.environ.get("PYTHON_BIN", os.path.join(HOME_DIR, "miniconda3", "envs", "DL", "bin", "python"))
LOCUST_BIN = os.environ.get("LOCUST_BIN", os.path.join(HOME_DIR, "miniconda3", "envs", "DL", "bin", "locust"))

ATTACK_SCRIPT = os.path.join(BASE_DIR, "bot_attack.py")
BENIGN_SCRIPT = os.path.join(BASE_DIR, "benign_traffic.py")
LOCUST_SCRIPT = os.path.join(BASE_DIR, "benign_user.py")
TARGET_SERVER_SCRIPT = os.path.join(BASE_DIR, "target_server.py")
PAYLOAD_FILE = os.path.join(BASE_DIR, "llm_payloads.json")

BENIGN_PCAP_FILE = os.environ.get("BENIGN_PCAP_FILE", "").strip()
BENIGN_ENGINE = os.environ.get("BENIGN_ENGINE", "locust").strip().lower()
ATTACK_ENGINE = os.environ.get("ATTACK_ENGINE", "http").strip().lower()
REQUIRE_REAL_LLM = os.environ.get("REQUIRE_REAL_LLM", "0").strip().lower() in ("1", "true", "yes")
FORCE_PLAIN_LINK = os.environ.get("FORCE_PLAIN_LINK", "0").strip().lower() in ("1", "true", "yes")
SKIP_CONNECTIVITY_TEST = os.environ.get("SKIP_CONNECTIVITY_TEST", "0").strip().lower() in ("1", "true", "yes")
BOT_TYPE_MODE = os.environ.get("BOT_TYPE_MODE", "mixed").strip().lower()
ARENA_SEED = int(os.environ.get("ARENA_SEED", "42"))
random.seed(ARENA_SEED)

if BENIGN_ENGINE not in ("locust", "script"):
    BENIGN_ENGINE = "locust"
if ATTACK_ENGINE not in ("http", "scapy"):
    ATTACK_ENGINE = "http"

def build_bot_types(num_bots: int, mode: str) -> dict[int, str]:
    if mode == "all_mimic":
        return {i: "mimic" for i in range(num_bots)}
    if mode == "all_slowburn":
        return {i: "slowburn" for i in range(num_bots)}
    if mode == "all_burst":
        return {i: "burst" for i in range(num_bots)}
    if mode == "mimic_heavy":
        # Harder semantic camouflage while keeping strategy diversity for OOD tests.
        n_mimic = int(round(0.70 * num_bots))
        n_slow = int(round(0.20 * num_bots))
        n_burst = max(0, num_bots - n_mimic - n_slow)
        types: dict[int, str] = {}
        for i in range(n_mimic):
            types[i] = "mimic"
        for i in range(n_mimic, n_mimic + n_slow):
            types[i] = "slowburn"
        for i in range(n_mimic + n_slow, n_mimic + n_slow + n_burst):
            types[i] = "burst"
        for i in range(num_bots):
            types.setdefault(i, "mimic")
        return types

    # mixed: slowburn 60%, burst 25%, mimic 15%
    n_slow = int(round(0.60 * num_bots))
    n_burst = int(round(0.25 * num_bots))
    n_mimic = max(0, num_bots - n_slow - n_burst)
    types: dict[int, str] = {}
    for i in range(n_slow):
        types[i] = "slowburn"
    for i in range(n_slow, n_slow + n_burst):
        types[i] = "burst"
    for i in range(n_slow + n_burst, n_slow + n_burst + n_mimic):
        types[i] = "mimic"
    for i in range(num_bots):
        types.setdefault(i, "slowburn")
    return types


BOT_TYPES = build_bot_types(NUM_BOTS, BOT_TYPE_MODE)

BOT_LINK_PROFILES = [
    {"bw": 5, "delay": "8ms", "loss": 1.0},
    {"bw": 8, "delay": "5ms", "loss": 0.5},
    {"bw": 10, "delay": "3ms", "loss": 0.2},
    {"bw": 3, "delay": "15ms", "loss": 2.0},
    {"bw": 12, "delay": "2ms", "loss": 0.0},
    {"bw": 4, "delay": "12ms", "loss": 1.5},
    {"bw": 7, "delay": "6ms", "loss": 0.8},
    {"bw": 6, "delay": "10ms", "loss": 1.2},
    {"bw": 9, "delay": "4ms", "loss": 0.3},
    {"bw": 2, "delay": "20ms", "loss": 2.5},
]

LOAD_SETTINGS = {
    "low": {"user_frac": 0.5, "locust_users_per_host": 1, "attack_gap_sec": 0.5},
    "medium": {"user_frac": 0.8, "locust_users_per_host": 1, "attack_gap_sec": 0.2},
    "high": {"user_frac": 1.0, "locust_users_per_host": 2, "attack_gap_sec": 0.05},
}



def select_switch_class():
    # Fallback to UserSwitch when ovs kernel module is unavailable.
    rc = os.system("ovs-vsctl show >/dev/null 2>&1")
    if rc == 0:
        return OVSKernelSwitch
    info("*** WARN: OVS unavailable, falling back to UserSwitch\n")
    return UserSwitch


def load_payload_metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {"exists": False, "llm_sessions": 0, "total_payloads": 0}
    if not os.path.exists(PAYLOAD_FILE):
        return meta
    try:
        with open(PAYLOAD_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        meta["exists"] = True
        if isinstance(payload, dict):
            m = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
            meta["llm_sessions"] = int(m.get("llm_sessions", 0))
            meta["total_payloads"] = int(m.get("total_payloads", len(payload.get("flat_payloads", []))))
            meta["metadata"] = m
            return meta
        if isinstance(payload, list):
            meta["total_payloads"] = len(payload)
        return meta
    except Exception as exc:
        meta["error"] = str(exc)
        return meta


def require_files() -> None:
    required = [ATTACK_SCRIPT, TARGET_SERVER_SCRIPT, PAYLOAD_FILE]
    if BENIGN_ENGINE == "locust":
        required += [LOCUST_SCRIPT]
    else:
        required += [BENIGN_SCRIPT]
    if BENIGN_PCAP_FILE:
        required += [BENIGN_PCAP_FILE]

    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        info("*** ERROR: missing files:\n")
        for p in missing:
            info(f"    {p}\n")
        raise FileNotFoundError("Required files missing")

    payload_meta = load_payload_metadata()
    if REQUIRE_REAL_LLM and int(payload_meta.get("llm_sessions", 0)) <= 0:
        raise RuntimeError("REQUIRE_REAL_LLM=1 but payload metadata has llm_sessions=0")



def write_manifest(
    users: list[Any],
    bots: list[Any],
    payload_meta: dict[str, Any],
    bot_profiles: dict[str, Any],
) -> None:
    ip_labels: dict[str, int] = {TARGET_IP: 0}
    roles: dict[str, str] = {TARGET_IP: "target"}

    for u in users:
        ip_labels[u.IP()] = 0
        roles[u.IP()] = "benign_user"
    for i, b in enumerate(bots):
        ip = b.IP()
        ip_labels[ip] = 1
        roles[ip] = f"bot:{BOT_TYPES.get(i, 'slowburn')}"

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "topology": {
            "type": TOPOLOGY_MODE,
            "target_ip": TARGET_IP,
            "users": NUM_USERS,
            "bots": NUM_BOTS,
            "core_bottleneck": {"bw_mbps": CORE_BW, "delay": CORE_DELAY, "max_queue_size": CORE_QUEUE},
        },
        "run_config": {
            "duration_sec": DURATION,
            "load_profile": LOAD_PROFILE,
            "bot_type_mode": BOT_TYPE_MODE,
            "arena_seed": ARENA_SEED,
            "benign_engine": BENIGN_ENGINE,
            "attack_engine": ATTACK_ENGINE,
            "require_real_llm": REQUIRE_REAL_LLM,
            "benign_pcap_file": BENIGN_PCAP_FILE,
        },
        "payload_metadata": payload_meta,
        "ip_labels": ip_labels,
        "roles": roles,
        "bot_profiles": bot_profiles,
        "pcap_file": PCAP_FILE,
    }

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)





def has_controller_binary() -> bool:
    return os.system("which controller >/dev/null 2>&1") == 0


def tc_constraints_supported() -> bool:
    # Probe whether tc qdisc kinds needed by TCLink are available.
    cmd_add = "tc qdisc add dev lo root handle 1: htb default 1 >/dev/null 2>&1"
    cmd_del = "tc qdisc del dev lo root >/dev/null 2>&1"
    rc = os.system(cmd_add)
    os.system(cmd_del)
    return rc == 0


def add_link_safe(net: Mininet, n1, n2, use_tc: bool, **opts):
    if use_tc:
        return net.addLink(n1, n2, **opts)
    return net.addLink(n1, n2)

def create_arena() -> None:
    require_files()
    payload_meta = load_payload_metadata()

    if os.path.exists(PCAP_FILE):
        os.remove(PCAP_FILE)

    info("=" * 70 + "\n")
    info("Phase 1: Mininet Arena v2\n")
    info(f"Duration={DURATION}s | benign={BENIGN_ENGINE} | attack={ATTACK_ENGINE}\n")
    info(f"Topology={TOPOLOGY_MODE} | LoadProfile={LOAD_PROFILE} | Users={NUM_USERS} | Bots={NUM_BOTS}\n")
    info(f"CoreBottleneck bw={CORE_BW}Mbps delay={CORE_DELAY} queue={CORE_QUEUE}\n")
    info(f"REQUIRE_REAL_LLM={int(REQUIRE_REAL_LLM)} | llm_sessions={payload_meta.get('llm_sessions', 0)}\n")
    info("=" * 70 + "\n")

    switch_cls = select_switch_class()
    use_tc = (not FORCE_PLAIN_LINK) and tc_constraints_supported()
    if not use_tc:
        info("*** WARN: tc qdisc capabilities missing, falling back to plain Link\n")

    has_ctrl = has_controller_binary()
    if has_ctrl:
        net = Mininet(controller=Controller, switch=switch_cls, link=TCLink if use_tc else Link)
        net.addController("c0")
    else:
        info("*** WARN: controller binary missing, running switches in standalone mode\n")
        net = Mininet(controller=None, switch=switch_cls, link=TCLink if use_tc else Link)

    # Core / aggregation / edge
    s_core = net.addSwitch("s1", failMode="standalone")
    edge_switches: list[Any] = []

    if TOPOLOGY_MODE == "three_tier":
        s_agg1 = net.addSwitch("s2", failMode="standalone")
        s_agg2 = net.addSwitch("s3", failMode="standalone")
        s_agg3 = net.addSwitch("s4", failMode="standalone")

        s_edge1 = net.addSwitch("s5", failMode="standalone")
        s_edge2 = net.addSwitch("s6", failMode="standalone")
        s_edge3 = net.addSwitch("s7", failMode="standalone")
        s_edge4 = net.addSwitch("s8", failMode="standalone")
        s_edge5 = net.addSwitch("s9", failMode="standalone")
        s_edge6 = net.addSwitch("s10", failMode="standalone")
        edge_switches = [s_edge1, s_edge2, s_edge3, s_edge4, s_edge5, s_edge6]
    elif TOPOLOGY_MODE == "two_tier":
        s_edge1 = net.addSwitch("s2", failMode="standalone")
        s_edge2 = net.addSwitch("s3", failMode="standalone")
        s_edge3 = net.addSwitch("s4", failMode="standalone")
        s_edge4 = net.addSwitch("s5", failMode="standalone")
        edge_switches = [s_edge1, s_edge2, s_edge3, s_edge4]
    elif TOPOLOGY_MODE == "flat_star":
        edge_switches = [s_core]
    else:
        raise ValueError(f"Unsupported TOPOLOGY_MODE={TOPOLOGY_MODE}")

    target = net.addHost("target", ip=TARGET_IP)

    users = [net.addHost(f"user{i+1}", ip=f"10.0.0.{10+i}") for i in range(NUM_USERS)]
    bots = [net.addHost(f"bot{i+1}", ip=f"10.0.0.{30+i}") for i in range(NUM_BOTS)]

    # Core bottleneck link for M/M/1 verification.
    add_link_safe(net, target, s_core, use_tc, bw=CORE_BW, delay=CORE_DELAY, loss=1, max_queue_size=CORE_QUEUE)

    if TOPOLOGY_MODE == "three_tier":
        add_link_safe(net, s_agg1, s_core, use_tc, bw=200, delay="1ms")
        add_link_safe(net, s_agg2, s_core, use_tc, bw=100, delay="2ms")
        add_link_safe(net, s_agg3, s_core, use_tc, bw=100, delay="3ms")

        add_link_safe(net, s_edge1, s_agg1, use_tc, bw=100, delay="1ms")
        add_link_safe(net, s_edge2, s_agg1, use_tc, bw=100, delay="1ms")
        add_link_safe(net, s_edge3, s_agg2, use_tc, bw=50, delay="2ms")
        add_link_safe(net, s_edge4, s_agg2, use_tc, bw=50, delay="3ms")
        add_link_safe(net, s_edge5, s_agg3, use_tc, bw=50, delay="2ms")
        add_link_safe(net, s_edge6, s_agg3, use_tc, bw=50, delay="3ms")
    elif TOPOLOGY_MODE == "two_tier":
        add_link_safe(net, edge_switches[0], s_core, use_tc, bw=120, delay="1ms")
        add_link_safe(net, edge_switches[1], s_core, use_tc, bw=100, delay="2ms")
        add_link_safe(net, edge_switches[2], s_core, use_tc, bw=80, delay="3ms")
        add_link_safe(net, edge_switches[3], s_core, use_tc, bw=80, delay="4ms")

    user_switches = edge_switches[: min(2, len(edge_switches))] if len(edge_switches) > 1 else edge_switches
    bot_switches = edge_switches[2:] if len(edge_switches) > 2 else edge_switches
    if not user_switches:
        user_switches = [s_core]
    if not bot_switches:
        bot_switches = [s_core]

    for i, user in enumerate(users):
        sw = user_switches[i % len(user_switches)]
        add_link_safe(net, user, sw, use_tc, bw=50, delay="1ms")

    bot_profiles: dict[str, Any] = {}
    for i, bot in enumerate(bots):
        sw = bot_switches[i % len(bot_switches)]

        profile = BOT_LINK_PROFILES[i % len(BOT_LINK_PROFILES)]
        add_link_safe(net, bot, sw, use_tc, **profile)
        bot_profiles[bot.IP()] = {
            "bot_id": i + 1,
            "bot_type": BOT_TYPES.get(i, "slowburn"),
            "link_profile": profile,
        }

    net.start()

    info("*** Start target service and tcpdump\n")
    target.cmd(f"{PYTHON3} {TARGET_SERVER_SCRIPT} 80 > /tmp/target_server.log 2>&1 &")
    time.sleep(2)
    target.cmd(f"tcpdump -U -i target-eth0 -w {PCAP_FILE} tcp > /tmp/tcpdump_v2.log 2>&1 &")
    time.sleep(2)

    if SKIP_CONNECTIVITY_TEST:
        info("*** Connectivity sampling skipped (SKIP_CONNECTIVITY_TEST=1)\n")
    else:
        info("*** Connectivity sampling\n")
        sample_hosts = users[:3] + bots[:3] + bots[42:44] + bots[60:62]
        for h in sample_hosts:
            res = h.cmd(f"ping -c 1 -W 2 {TARGET_IP}")
            ok = ("1 received" in res) or ("1 packets received" in res)
            info(f"    {h.name:8s} {h.IP():>12s} -> {'OK' if ok else 'FAIL'}\n")

    info(f"*** Start benign traffic ({BENIGN_ENGINE})\n")
    load_cfg = LOAD_SETTINGS.get(LOAD_PROFILE, LOAD_SETTINGS["medium"])
    active_users = max(1, int(round(len(users) * float(load_cfg["user_frac"]))))
    locust_users_per_host = int(load_cfg["locust_users_per_host"])
    attack_gap_sec = float(load_cfg["attack_gap_sec"])

    for i, user in enumerate(users[:active_users]):
        if BENIGN_ENGINE == "locust":
            run_time = max(DURATION + 30, 60)
            cmd = (
                f"{LOCUST_BIN} -f {LOCUST_SCRIPT} --headless -u {locust_users_per_host} -r 1 "
                f"--host http://{TARGET_IP} --run-time {run_time}s --only-summary "
                f"> /tmp/user{i+1}_locust.log 2>&1 &"
            )
        else:
            if BENIGN_PCAP_FILE:
                cmd = (
                    f"{PYTHON3} {BENIGN_SCRIPT} {TARGET_IP} 80 user{i+1} "
                    f"--pcap {BENIGN_PCAP_FILE} --iat-scale 1.0 > /tmp/user{i+1}.log 2>&1 &"
                )
            else:
                cmd = f"{PYTHON3} {BENIGN_SCRIPT} {TARGET_IP} 80 user{i+1} > /tmp/user{i+1}.log 2>&1 &"
        user.cmd(cmd)

    time.sleep(10)

    info("*** Start attack traffic\n")
    for i, bot in enumerate(bots):
        bot_type = BOT_TYPES.get(i, "slowburn")
        cmd = (
            f"{PYTHON3} {ATTACK_SCRIPT} {TARGET_IP} 80 {PAYLOAD_FILE} "
            f"bot{i+1} {bot_type} {ATTACK_ENGINE} > /tmp/bot{i+1}.log 2>&1 &"
        )
        bot.cmd(cmd)
        if attack_gap_sec > 0:
            time.sleep(attack_gap_sec)

    write_manifest(users=users, bots=bots, payload_meta=payload_meta, bot_profiles=bot_profiles)

    info(f"*** Arena running for {DURATION}s\n")
    start = time.time()
    try:
        while time.time() - start < DURATION:
            elapsed = int(time.time() - start)
            if elapsed > 0 and elapsed % 60 == 0:
                size_mb = os.path.getsize(PCAP_FILE) / 1048576.0 if os.path.exists(PCAP_FILE) else 0.0
                info(f"    [{elapsed:4d}s/{DURATION}s] pcap={size_mb:.1f}MB\n")
                time.sleep(1)
            time.sleep(1)
    except KeyboardInterrupt:
        info("*** interrupted by user\n")

    info("*** cleanup\n")
    for host in users + bots:
        host.cmd("pkill -f locust || true")
        host.cmd("pkill -f benign_traffic.py || true")
        host.cmd("pkill -f bot_attack.py || true")
        host.cmd("pkill -f python || true")

    target.cmd("pkill -INT tcpdump || true")
    time.sleep(2)
    target.cmd("pkill -f target_server.py || true")

    net.stop()

    if os.path.exists(PCAP_FILE):
        os.system(f"chmod 666 {PCAP_FILE} 2>/dev/null")
        size = os.path.getsize(PCAP_FILE)
        info("=" * 70 + "\n")
        info(f"PCAP: {PCAP_FILE}\n")
        info(f"Size: {size/1048576.0:.2f} MB\n")
        info(f"Manifest: {MANIFEST_FILE}\n")
        if size / 1048576.0 < 50:
            info("[WARN] PCAP < 50MB; consider longer duration for stronger queueing signal.\n")
        info("=" * 70 + "\n")
    else:
        info("[ERROR] PCAP missing, check /tmp/tcpdump_v2.log\n")


if __name__ == "__main__":
    setLogLevel("info")
    create_arena()
