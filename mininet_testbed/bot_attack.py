#!/usr/bin/env python3
"""
Phase 1 - Bot Attack Script (runs inside Mininet hosts)

Usage:
  python3 bot_attack.py TARGET PORT PAYLOAD_FILE BOT_ID BOT_TYPE [ATTACK_ENGINE]

ATTACK_ENGINE:
  http  (default) - urllib HTTP requests
  scapy           - crafted raw packets (fallback to http if unavailable)
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Any

# Clear proxy env vars inherited from host.
for v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(v, None)
os.environ["no_proxy"] = "*"

try:
    from urllib.request import Request, urlopen
except ImportError:  # pragma: no cover
    from urllib2 import Request, urlopen  # type: ignore

SCAPY_AVAILABLE = True
try:
    from scapy.all import IP, TCP, Raw, send  # type: ignore
except Exception:
    SCAPY_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_IP = sys.argv[1] if len(sys.argv) > 1 else "10.0.0.100"
TARGET_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 80
PAYLOAD_FILE = sys.argv[3] if len(sys.argv) > 3 else os.path.join(BASE_DIR, "llm_payloads.json")
BOT_ID = sys.argv[4] if len(sys.argv) > 4 else "bot0"
BOT_TYPE = sys.argv[5] if len(sys.argv) > 5 else "slowburn"
ATTACK_ENGINE = (sys.argv[6] if len(sys.argv) > 6 else "http").strip().lower()

SLEEP_MIN = float(os.getenv("ATTACK_SLEEP_MIN", "1.0"))
SLEEP_MAX = float(os.getenv("ATTACK_SLEEP_MAX", "4.0"))
if SLEEP_MIN < 0.1:
    SLEEP_MIN = 0.1
if SLEEP_MAX < SLEEP_MIN:
    SLEEP_MAX = SLEEP_MIN

if ATTACK_ENGINE not in ("http", "scapy"):
    ATTACK_ENGINE = "http"
if ATTACK_ENGINE == "scapy" and not SCAPY_AVAILABLE:
    ATTACK_ENGINE = "http"


def jitter_sleep() -> None:
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))


def load_payloads() -> list[dict[str, Any]]:
    with open(PAYLOAD_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if isinstance(data.get("flat_payloads"), list):
            return data["flat_payloads"]
        if isinstance(data.get("sessions"), list):
            out: list[dict[str, Any]] = []
            for session in data["sessions"]:
                for step in session.get("steps", []):
                    if isinstance(step, dict):
                        out.append(step)
            return out

    return []


def load_sessions() -> list[dict[str, Any]]:
    with open(PAYLOAD_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("sessions"), list):
        return data["sessions"]

    payloads = load_payloads()
    return [{"session_id": "fallback", "session_type": "misc", "steps": payloads}] if payloads else []


def _build_http_bytes(payload: dict[str, Any]) -> bytes:
    uri = str(payload.get("uri", "/"))
    if not uri.startswith("/"):
        uri = "/" + uri

    ua = str(payload.get("user_agent", "Mozilla/5.0"))
    headers = payload.get("headers", {}) if isinstance(payload.get("headers", {}), dict) else {}

    lines = [
        f"GET {uri} HTTP/1.1",
        f"Host: {TARGET_IP}:{TARGET_PORT}",
        f"User-Agent: {ua}",
        "Accept: */*",
        "Connection: keep-alive",
    ]
    for k, v in headers.items():
        kl = str(k).lower()
        if kl in ("host", "user-agent", "accept", "connection"):
            continue
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("")
    return "\r\n".join(lines).encode("utf-8", errors="ignore")


def _send_http(payload: dict[str, Any]) -> bool:
    uri = str(payload.get("uri", "/"))
    if not uri.startswith("/"):
        uri = "/" + uri

    ua = str(payload.get("user_agent", "Mozilla/5.0"))
    headers = payload.get("headers", {}) if isinstance(payload.get("headers", {}), dict) else {}
    url = f"http://{TARGET_IP}:{TARGET_PORT}{uri}"

    try:
        req = Request(url)
        req.add_header("User-Agent", ua)
        req.add_header("Host", TARGET_IP)
        req.add_header("Connection", "keep-alive")
        for k, v in headers.items():
            if str(k).lower() not in ("host", "user-agent", "connection"):
                req.add_header(str(k), str(v))
        with urlopen(req, timeout=8) as resp:
            resp.read()
        return True
    except Exception:
        return False


def _send_scapy(payload: dict[str, Any]) -> bool:
    if not SCAPY_AVAILABLE:
        return False
    try:
        raw = _build_http_bytes(payload)
        pkt = IP(dst=TARGET_IP) / TCP(sport=random.randint(1024, 65535), dport=TARGET_PORT, flags="PA") / Raw(load=raw)
        send(pkt, verbose=False)
        return True
    except Exception:
        return False


def send_request(payload: dict[str, Any]) -> bool:
    if ATTACK_ENGINE == "scapy":
        ok = _send_scapy(payload)
        if ok:
            return True
    return _send_http(payload)


def run_slowburn(payloads: list[dict[str, Any]]) -> None:
    idx = 0
    sent = 0
    err = 0
    random.shuffle(payloads)

    print(f"[{BOT_ID}] slowburn engine={ATTACK_ENGINE} sleep={SLEEP_MIN:.1f}-{SLEEP_MAX:.1f}s")
    while True:
        payload = payloads[idx % len(payloads)]
        idx += 1
        if send_request(payload):
            sent += 1
        else:
            err += 1

        if idx % 20 == 0:
            print(f"[{BOT_ID}] slowburn sent={sent} err={err}", flush=True)
        jitter_sleep()


def run_burst(payloads: list[dict[str, Any]]) -> None:
    idx = 0
    sent = 0
    err = 0
    burst_id = 0
    random.shuffle(payloads)

    print(f"[{BOT_ID}] burst engine={ATTACK_ENGINE}")
    while True:
        time.sleep(random.uniform(12.0, 28.0))
        burst_id += 1
        burst_size = random.randint(4, 12)
        for _ in range(burst_size):
            payload = payloads[idx % len(payloads)]
            idx += 1
            if send_request(payload):
                sent += 1
            else:
                err += 1
            time.sleep(random.uniform(0.2, 1.0))
        print(f"[{BOT_ID}] burst#{burst_id} size={burst_size} sent={sent} err={err}", flush=True)
        jitter_sleep()


def run_mimic(payloads: list[dict[str, Any]]) -> None:
    sessions = load_sessions()
    if not sessions:
        sessions = [{"steps": payloads}]
    random.shuffle(sessions)

    sent = 0
    err = 0
    s_idx = 0
    print(f"[{BOT_ID}] mimic engine={ATTACK_ENGINE} sessions={len(sessions)}")

    while True:
        session = sessions[s_idx % len(sessions)]
        s_idx += 1
        for step in session.get("steps", []):
            if send_request(step):
                sent += 1
            else:
                err += 1

            # Keep semantic delay but bound to anti-rate-detector jitter budget.
            think = step.get("think_time", random.uniform(SLEEP_MIN, SLEEP_MAX))
            try:
                think_v = float(think)
            except Exception:
                think_v = random.uniform(SLEEP_MIN, SLEEP_MAX)
            time.sleep(max(SLEEP_MIN, min(SLEEP_MAX, think_v)))

        print(f"[{BOT_ID}] mimic sessions={s_idx} sent={sent} err={err}", flush=True)
        jitter_sleep()


def main() -> None:
    payloads = load_payloads()
    if not payloads:
        print(f"[{BOT_ID}] ERROR no payloads loaded from {PAYLOAD_FILE}")
        sys.exit(1)

    print(
        f"[{BOT_ID}] start type={BOT_TYPE} engine={ATTACK_ENGINE} "
        f"target={TARGET_IP}:{TARGET_PORT} payloads={len(payloads)}"
    )

    try:
        if BOT_TYPE == "burst":
            run_burst(payloads)
        elif BOT_TYPE == "mimic":
            run_mimic(payloads)
        else:
            run_slowburn(payloads)
    except KeyboardInterrupt:
        print(f"[{BOT_ID}] stopped")


if __name__ == "__main__":
    main()
