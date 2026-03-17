#!/usr/bin/env python3
from __future__ import annotations

import random

from locust import HttpUser, between, task

# Markov transitions for a realistic browsing chain:
# Homepage -> Search -> Product Detail with occasional loops.
TRANSITIONS: dict[str, list[tuple[str, float]]] = {
    "homepage": [("search", 0.70), ("homepage", 0.15), ("product", 0.15)],
    "search": [("product", 0.65), ("search", 0.20), ("homepage", 0.15)],
    "product": [("search", 0.50), ("homepage", 0.30), ("product", 0.20)],
}

SEARCH_TERMS = [
    "laptop",
    "shoes",
    "coffee+maker",
    "python+book",
    "mechanical+keyboard",
    "wireless+mouse",
    "desk+lamp",
]


def pick_next_state(state: str) -> str:
    options = TRANSITIONS.get(state, TRANSITIONS["homepage"])
    states, probs = zip(*options)
    return random.choices(states, probs, k=1)[0]


class BenignWebUser(HttpUser):
    # Human dwell time in [1, 5] seconds.
    wait_time = between(1.0, 5.0)

    def on_start(self) -> None:
        self.state = "homepage"

    @task
    def markov_session(self) -> None:
        if self.state == "homepage":
            self.client.get("/", name="Homepage")
        elif self.state == "search":
            term = random.choice(SEARCH_TERMS)
            page = random.randint(1, 4)
            self.client.get(f"/search?q={term}&page={page}", name="Search")
        else:
            pid = random.randint(10000, 99999)
            self.client.get(f"/product/{pid}?ref=search", name="Product_Detail")

        self.state = pick_next_state(self.state)
