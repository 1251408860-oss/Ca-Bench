#!/usr/bin/env python3
"""
Phase 1 - Benign Traffic Generator (runs INSIDE Mininet hosts)

Modes:
1) Synthetic browsing (default)
2) PCAP replay mode (when --pcap is provided)

Usage:
  python3 benign_traffic.py TARGET PORT USER_ID
  python3 benign_traffic.py TARGET PORT USER_ID --pcap ./benign_seed.pcap --iat-scale 1.0
"""

import argparse
import hashlib
import os
import random
import re
import sys
import time

# Clear proxy env vars
for v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(v, None)
os.environ["no_proxy"] = "*"

try:
    from urllib.request import Request, urlopen
except ImportError:
    from urllib2 import Request, urlopen

SCAPY_AVAILABLE = None

SEARCH_QUERIES = [
    "laptop", "shoes", "coffee+maker", "python+book", "keyboard",
    "headphones", "backpack", "desk+lamp", "water+bottle", "phone+case",
    "winter+coat", "gaming+mouse", "yoga+mat", "cookbook", "sunglasses",
    "bluetooth+speaker", "running+shoes", "desk+organizer", "plant+pot",
    "travel+adapter", "usb+cable", "notebook", "tea+kettle", "pillow",
    "watch", "wallet", "umbrella", "socks", "mug", "candle",
]

CATEGORIES = ["electronics", "clothing", "home", "books", "sports", "kitchen", "garden", "toys", "office", "health"]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 Mobile Safari/604.1",
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Benign traffic generator")
    parser.add_argument("target_ip", nargs="?", default="10.0.0.100")
    parser.add_argument("target_port", nargs="?", type=int, default=80)
    parser.add_argument("user_id", nargs="?", default="user0")
    parser.add_argument("--pcap", dest="pcap", default="", help="Seed PCAP path for benign replay")
    parser.add_argument("--iat-scale", dest="iat_scale", type=float, default=1.0, help="IAT scale for replay mode")
    parser.add_argument("--loop-pcap", dest="loop_pcap", action="store_true", default=True)
    parser.add_argument("--max-sleep", dest="max_sleep", type=float, default=10.0)
    return parser.parse_args()


def human_think_time():
    r = random.random()
    if r < 0.60:
        return random.uniform(0.5, 3.0)
    if r < 0.85:
        return random.uniform(3.0, 10.0)
    if r < 0.95:
        return random.uniform(10.0, 30.0)
    return random.uniform(30.0, 60.0)


def session_id(user_id):
    return hashlib.md5(f"{user_id}_{time.time()}_{random.random()}".encode()).hexdigest()[:16]


def send_request(url, ua, target_ip):
    try:
        req = Request(url)
        req.add_header("User-Agent", ua)
        req.add_header("Host", target_ip)
        req.add_header("Accept", "text/html,application/xhtml+xml,*/*;q=0.8")
        req.add_header("Connection", "keep-alive")
        resp = urlopen(req, timeout=6)
        resp.read()
        resp.close()
        return True
    except Exception:
        return False


def generate_url(user_id):
    sid = session_id(user_id)
    r = random.random()
    if r < 0.25:
        page = random.choice(["/", "/deals", "/bestsellers", "/new-arrivals"])
        return f"{page}?sid={sid}"
    if r < 0.55:
        q = random.choice(SEARCH_QUERIES)
        sort = random.choice(["relevance", "price_asc", "price_desc", "rating"])
        return f"/search?q={q}&sort={sort}&page={random.randint(1, 5)}&sid={sid}"
    if r < 0.75:
        pid = random.randint(10000, 99999)
        return f"/product/{pid}?ref=search&sid={sid}"
    if r < 0.88:
        cat = random.choice(CATEGORIES)
        return f"/category/{cat}?page={random.randint(1, 5)}&sid={sid}"
    if r < 0.95:
        pid = random.randint(10000, 99999)
        return f"/reviews/product/{pid}?page=1&sort=newest&sid={sid}"
    page = random.choice(["/about", "/contact", "/help/faq", "/privacy", "/sitemap"])
    return f"{page}?sid={sid}"


def extract_http_gets_from_pcap(pcap_path):
    """Extract GET paths and IATs from a seed pcap."""
    global SCAPY_AVAILABLE
    if not os.path.exists(pcap_path):
        return []

    try:
        from scapy.all import rdpcap, TCP, Raw, IP  # type: ignore
        SCAPY_AVAILABLE = True
    except Exception:
        SCAPY_AVAILABLE = False
        return []

    packets = rdpcap(pcap_path)
    events = []
    last_ts = None

    req_line_re = re.compile(r"^GET\s+([^\s]+)\s+HTTP/", re.IGNORECASE)
    ua_re = re.compile(r"^User-Agent:\s*(.+)$", re.IGNORECASE)

    for pkt in packets:
        if TCP not in pkt or Raw not in pkt or IP not in pkt:
            continue
        payload = bytes(pkt[Raw].load)
        if not payload.startswith(b"GET "):
            continue

        try:
            text = payload.decode("utf-8", errors="ignore")
        except Exception:
            continue

        lines = text.split("\r\n")
        if not lines:
            continue
        m = req_line_re.match(lines[0].strip())
        if not m:
            continue

        path = m.group(1).strip()
        if not path.startswith("/"):
            path = "/" + path

        ua = None
        for ln in lines[1:20]:
            m2 = ua_re.match(ln.strip())
            if m2:
                ua = m2.group(1).strip()
                break
        if not ua:
            ua = random.choice(USER_AGENTS)

        ts = float(pkt.time)
        if last_ts is None:
            iat = random.uniform(0.5, 2.0)
        else:
            iat = max(0.05, ts - last_ts)
        last_ts = ts

        events.append((path, ua, iat))

    return events


def run_pcap_replay(args):
    events = extract_http_gets_from_pcap(args.pcap)
    if not events:
        sys.stdout.write(f"[{args.user_id}] PCAP replay unavailable or empty, fallback to synthetic\n")
        sys.stdout.flush()
        run_synthetic(args)
        return

    sys.stdout.write(
        f"[{args.user_id}] PCAP replay mode events={len(events)} iat_scale={args.iat_scale} -> {args.target_ip}:{args.target_port}\n"
    )
    sys.stdout.flush()

    sent = 0
    err = 0
    idx = 0

    while True:
        path, ua, iat = events[idx % len(events)]
        idx += 1
        url = f"http://{args.target_ip}:{args.target_port}{path}"

        if send_request(url, ua, args.target_ip):
            sent += 1
        else:
            err += 1

        if idx % 100 == 0:
            sys.stdout.write(f"[{args.user_id}] replay sent={sent} err={err}\n")
            sys.stdout.flush()

        sleep_s = max(0.05, min(args.max_sleep, iat * args.iat_scale))
        time.sleep(sleep_s)

        if not args.loop_pcap and idx >= len(events):
            break


def run_synthetic(args):
    ua = random.choice(USER_AGENTS)
    sent = 0
    err = 0
    session_count = 0

    sys.stdout.write(f"[{args.user_id}] Synthetic benign traffic -> {args.target_ip}:{args.target_port}\n")
    sys.stdout.flush()

    while True:
        if random.random() < 0.70:
            # session-like navigation
            sid = session_id(args.user_id)
            session_len = random.randint(3, 8)
            for _ in range(session_len):
                if random.random() < 0.5:
                    q = random.choice(SEARCH_QUERIES)
                    path = f"/search?q={q}&page={random.randint(1,3)}&sid={sid}"
                else:
                    path = generate_url(args.user_id)
                url = f"http://{args.target_ip}:{args.target_port}{path}"
                if send_request(url, ua, args.target_ip):
                    sent += 1
                else:
                    err += 1
                time.sleep(human_think_time())
            session_count += 1
            if session_count % 3 == 0:
                sys.stdout.write(f"[{args.user_id}] sessions={session_count} sent={sent} err={err}\n")
                sys.stdout.flush()
            time.sleep(random.uniform(5.0, 30.0))
        else:
            path = generate_url(args.user_id)
            url = f"http://{args.target_ip}:{args.target_port}{path}"
            if send_request(url, ua, args.target_ip):
                sent += 1
            else:
                err += 1
            time.sleep(human_think_time())

        if random.random() < 0.02:
            ua = random.choice(USER_AGENTS)


def main():
    args = parse_args()
    if args.pcap:
        run_pcap_replay(args)
    else:
        run_synthetic(args)


if __name__ == "__main__":
    main()
