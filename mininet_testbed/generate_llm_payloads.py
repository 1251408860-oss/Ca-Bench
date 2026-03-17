#!/usr/bin/env python3
"""
Phase 1 - Step 1: LLM Session-Chain Payload Generation.

Generates complete browsing session chains (not isolated payloads).
Two engines:
  1. LLM (DeepSeek) — 100 sessions with causal browsing logic
  2. Algorithmic — state-machine driven sessions as fallback

Output: ~/llm_payloads.json
  {
    "sessions": [...],        # grouped by session_id
    "flat_payloads": [...],   # backward-compatible flat list
    "metadata": {...}
  }

Run BEFORE starting the Mininet arena (outside Mininet, with full internet).
"""
import json
import random
import hashlib
import time
import os
import sys

# Proxy behavior:
# - default: clear proxies to avoid broken host-inherited settings
# - KEEP_PROXY=1: keep current proxy settings (needed in some WSL networks)
KEEP_PROXY = os.getenv('KEEP_PROXY', '0').strip().lower() in ('1', 'true', 'yes')
if not KEEP_PROXY:
    for v in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY'):
        os.environ.pop(v, None)
    os.environ['no_proxy'] = '*'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_TOTAL_PAYLOADS = int(os.getenv('NUM_TOTAL_PAYLOADS', '3000'))
NUM_LLM_SESSIONS = int(os.getenv('NUM_LLM_SESSIONS', '100'))
OUTPUT_FILE = os.path.expanduser(os.getenv('OUTPUT_FILE', os.path.join(BASE_DIR, 'llm_payloads.json')))
LLM_MODEL = os.getenv('LLM_MODEL', 'deepseek-chat')
LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'https://api.deepseek.com/v1')
LLM_API_KEY = os.getenv('LLM_API_KEY') or os.getenv('DEEPSEEK_API_KEY') or os.getenv('OPENAI_API_KEY')
LLM_TARGET_STEPS = int(os.getenv('LLM_TARGET_STEPS', '50'))
LLM_TIMEOUT_SEC = float(os.getenv('LLM_TIMEOUT_SEC', '120'))
LLM_TRANSPORT = os.getenv('LLM_TRANSPORT', 'requests').strip().lower()
if LLM_TRANSPORT not in ('openai', 'requests'):
    LLM_TRANSPORT = 'openai'
REQUIRE_REAL_LLM = os.getenv('REQUIRE_REAL_LLM', '0').strip().lower() in ('1', 'true', 'yes')

# ============================================================
# Session types and their characteristics
# ============================================================
SESSION_TYPES = ["shopping", "research", "impulse_buy", "comparison", "return_customer"]

SESSION_TYPE_WEIGHTS = {
    "shopping":        0.30,
    "research":        0.25,
    "impulse_buy":     0.15,
    "comparison":      0.15,
    "return_customer": 0.15,
}

# Think-time ranges by step type (seconds)
THINK_TIME_BY_STEP = {
    "search":    (1.0, 4.0),
    "category":  (2.0, 6.0),
    "product":   (3.0, 12.0),
    "cart":      (0.5, 2.0),
    "checkout":  (1.0, 3.0),
    "homepage":  (1.5, 5.0),
    "review":    (4.0, 15.0),
    "compare":   (3.0, 10.0),
    "wishlist":  (0.5, 2.0),
    "filter":    (0.8, 3.0),
    "api":       (0.2, 1.0),
    "misc":      (1.0, 4.0),
}

# ============================================================
# Real User-Agent database (30+ diverse entries)
# ============================================================
USER_AGENTS = [
    # Chrome Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.6723.116 Safari/537.36",
    "Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    # Chrome Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    # Firefox
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.2; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:119.0) Gecko/20100101 Firefox/119.0",
    # Safari
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
    # Edge
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.2210.91",
    # Mobile
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.144 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 14; SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.144 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; SM-A546B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.6045.193 Mobile Safari/537.36",
    # Opera
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 OPR/105.0.0.0",
    # Smart TV / IoT
    "Mozilla/5.0 (SMART-TV; Linux; Tizen 7.0) AppleWebKit/537.36 (KHTML, like Gecko) Version/7.0 TV Safari/537.36",
    # Older Chrome
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.5938.132 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.187 Safari/537.36",
    "Mozilla/5.0 (X11; CrOS x86_64 14541.0.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Misc
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Vivaldi/6.1.3035.204",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/115.0",
]

# ============================================================
# Realistic e-commerce vocabulary
# ============================================================
SEARCH_TERMS = [
    "wireless+bluetooth+headphones+noise+cancelling", "organic+green+tea+matcha+ceremonial+grade",
    "4k+ultrawide+monitor+34+inch+curved+IPS", "vintage+leather+messenger+bag+handmade",
    "smart+home+security+camera+system+outdoor+wifi", "mechanical+keyboard+cherry+mx+brown+tkl",
    "running+shoes+mens+cushioned+marathon", "cast+iron+skillet+12+inch+pre+seasoned",
    "usb+c+hub+multiport+adapter+thunderbolt+4", "yoga+mat+extra+thick+non+slip+eco+friendly",
    "portable+power+station+1000w+solar+generator", "stainless+steel+insulated+water+bottle+32oz",
    "noise+cancelling+earbuds+2024+premium+wireless", "ergonomic+office+chair+lumbar+support+mesh",
    "air+fryer+large+capacity+digital+touchscreen", "hiking+backpack+40l+waterproof+ultralight",
    "robot+vacuum+self+emptying+lidar+navigation", "standing+desk+electric+adjustable+memory+preset",
    "winter+jacket+mens+waterproof+down+insulated", "gaming+mouse+wireless+lightweight+25k+dpi",
    "espresso+machine+automatic+bean+to+cup+grinder", "led+strip+lights+smart+rgb+music+sync",
    "protein+powder+whey+isolate+unflavored+5lb", "tablet+stand+adjustable+aluminum+foldable",
    "smart+door+lock+fingerprint+keyless+entry", "bicycle+helmet+adult+mips+ventilated",
    "electric+toothbrush+sonic+pressure+sensor", "laptop+sleeve+15.6+inch+shockproof+waterproof",
    "garden+hose+expandable+100ft+brass+fittings", "wireless+charger+fast+charging+3+in+1+magsafe",
    "dutch+oven+enameled+cast+iron+6+quart", "ring+light+10+inch+tripod+phone+holder",
    "massage+gun+deep+tissue+percussion+quiet", "solar+panel+portable+100w+foldable+camping",
    "weighted+blanket+20+pounds+cooling+bamboo", "dash+cam+front+rear+4k+night+vision+gps",
    "resistance+bands+set+fabric+non+slip", "bookshelf+5+tier+industrial+metal+wood",
    "slow+cooker+programmable+6qt+ceramic+insert", "travel+pillow+memory+foam+neck+support",
    "mini+projector+1080p+portable+bluetooth", "chess+set+wooden+handcrafted+magnetic",
    "humidifier+ultrasonic+cool+mist+essential+oil", "label+maker+bluetooth+thermal+portable",
    "binoculars+10x42+waterproof+fogproof+bak4", "hand+mixer+electric+5+speed+stainless+steel",
    "ceiling+fan+52+inch+remote+control+dimmable", "first+aid+kit+comprehensive+300+pieces",
    "phone+mount+car+magnetic+dashboard+vent", "air+purifier+hepa+h13+large+room+quiet",
]

SORT_OPTIONS = ["price_asc", "price_desc", "relevance", "newest", "rating", "bestselling", "discount", "popularity"]
COLORS = ["red", "blue", "black", "white", "green", "silver", "gold", "navy", "gray", "pink", "brown", "beige", "burgundy", "teal", "coral"]
SIZES = ["XS", "S", "M", "L", "XL", "XXL", "6", "7", "8", "9", "10", "11", "12", "one-size", "28W", "30W", "32W", "34W"]
BRANDS = ["Samsung", "Apple", "Sony", "Nike", "Adidas", "Bose", "Logitech", "Anker", "Dell", "HP",
          "Lenovo", "Dyson", "Philips", "Bosch", "Canon", "LG", "JBL", "Corsair", "ASUS", "Razer"]
MATERIALS = ["cotton", "polyester", "leather", "stainless-steel", "aluminum", "wood", "ceramic",
             "silicone", "bamboo", "nylon", "titanium", "carbon-fiber", "merino-wool"]
CATEGORIES = ["electronics", "clothing", "shoes", "home-garden", "toys", "books", "sports",
              "automotive", "health-beauty", "jewelry", "office", "pet-supplies", "grocery",
              "baby", "tools", "appliances", "furniture", "outdoors", "musical-instruments"]
REFERRERS = [
    "https://www.google.com/search?q={}&sourceid=chrome&ie=UTF-8",
    "https://www.bing.com/search?q={}&form=QBLH&sp=-1",
    "https://www.facebook.com/marketplace/item/{}",
    "https://twitter.com/i/status/{}",
    "https://www.reddit.com/r/BuyItForLife/comments/{}/",
    "https://www.youtube.com/watch?v={}",
    "https://www.instagram.com/p/{}/",
    "https://news.ycombinator.com/item?id={}",
    "https://www.pinterest.com/pin/{}/",
    "https://www.tiktok.com/@deals/video/{}/",
]


# ============================================================
# Utility helpers
# ============================================================
def random_hex(n):
    return ''.join(random.choices('0123456789abcdef', k=n))

def random_session_id():
    return hashlib.md5(f"{random.random()}_{time.time()}".encode()).hexdigest()

def random_tracking_id():
    parts = [random_hex(random.randint(4, 8)) for _ in range(random.randint(2, 5))]
    return '-'.join(parts)

def random_uuid4():
    h = random_hex(32)
    return f"{h[:8]}-{h[8:12]}-4{h[13:16]}-{random.choice('89ab')}{h[17:20]}-{h[20:32]}"

def pick_think_time(step_type):
    """Pick think-time based on the step type."""
    lo, hi = THINK_TIME_BY_STEP.get(step_type, (1.0, 5.0))
    return round(random.uniform(lo, hi), 3)


# ============================================================
# URI generators (reused by both engines)
# ============================================================
def generate_search_uri(term=None):
    if term is None:
        term = random.choice(SEARCH_TERMS)
    params = [f"q={term}", f"page={random.randint(1, 25)}", f"sort={random.choice(SORT_OPTIONS)}"]
    filters_pool = [
        f"brand={random.choice(BRANDS)}", f"min_price={random.randint(5, 200)}",
        f"max_price={random.randint(200, 5000)}", f"color={random.choice(COLORS)}",
        f"size={random.choice(SIZES)}", f"rating_min={random.choice(['3.0', '3.5', '4.0', '4.5'])}",
        "shipping=free", "in_stock=true", f"material={random.choice(MATERIALS)}",
        f"condition={random.choice(['new', 'renewed', 'used-good', 'open-box'])}",
        f"seller={random.choice(['official', 'marketplace', 'premium', 'verified'])}",
    ]
    params.extend(random.sample(filters_pool, random.randint(3, 6)))
    params.append(f"sid={random_session_id()}")
    params.append(f"ref={random_tracking_id()}")
    params.append(f"ts={int(time.time()) + random.randint(-86400, 0)}")
    params.append(f"_reqid={random_hex(12)}")
    if random.random() > 0.5:
        params.append(f"ab_test={random.choice(['A', 'B', 'C'])}_{random_hex(4)}")
    if random.random() > 0.6:
        params.append(f"utm_source={random.choice(['google', 'facebook', 'email', 'instagram', 'tiktok'])}")
        params.append(f"utm_medium={random.choice(['cpc', 'organic', 'social', 'referral'])}")
        params.append(f"utm_campaign={random_hex(8)}")
    random.shuffle(params[3:])
    return f"/search?{'&'.join(params)}"


def generate_product_uri(product_id=None):
    if product_id is None:
        product_id = random.randint(100000, 9999999)
    cat = random.choice(CATEGORIES)
    slug_words = random.choice(SEARCH_TERMS).replace('+', '-').split('-')
    slug = '-'.join(random.sample(slug_words, min(len(slug_words), random.randint(3, 7))))
    params = [
        f"ref={random.choice(['search', 'category', 'recommend', 'history', 'ad', 'wishlist', 'compare'])}",
        f"sid={random_session_id()}", f"click_id={random_tracking_id()}", f"pos={random.randint(1, 48)}",
    ]
    if random.random() > 0.4:
        params.append(f"variant={random.choice(COLORS)}")
    if random.random() > 0.4:
        params.append(f"size={random.choice(SIZES)}")
    if random.random() > 0.6:
        params.append(f"warehouse={random.choice(['east', 'west', 'central', 'fulfillment-a'])}")
    if random.random() > 0.7:
        params.append(f"coupon={random_hex(8).upper()}")
    return f"/{cat}/product/{product_id}/{slug}?{'&'.join(params)}"


def generate_category_uri(cat=None):
    if cat is None:
        cat = random.choice(CATEGORIES)
    subcats = random.choice(SEARCH_TERMS).split('+')
    subcat = '-'.join(subcats[:random.randint(1, 3)])
    params = [
        f"page={random.randint(1, 15)}", f"sort={random.choice(SORT_OPTIONS)}",
        f"view={random.choice(['grid', 'list'])}", f"per_page={random.choice([24, 36, 48, 60, 96])}",
        f"sid={random_session_id()}",
    ]
    if random.random() > 0.5:
        params.append(f"brand={random.choice(BRANDS)}")
    return f"/category/{cat}/{subcat}?{'&'.join(params)}"


def generate_cart_uri(action=None):
    if action is None:
        action = random.choice(["view", "add", "update", "remove", "apply-coupon", "estimate-shipping"])
    params = [f"action={action}", f"sid={random_session_id()}", f"csrf={random_hex(16)}"]
    if action in ["add", "update", "remove"]:
        params.append(f"item_id={random.randint(100000, 9999999)}")
        params.append(f"qty={random.randint(1, 5)}")
        params.append(f"variant_id={random_hex(8)}")
    elif action == "apply-coupon":
        params.append(f"code={random_hex(8).upper()}")
    return f"/cart?{'&'.join(params)}"


def generate_checkout_uri():
    step = random.choice(["shipping", "payment", "review", "confirm"])
    params = [f"step={step}", f"sid={random_session_id()}", f"csrf={random_hex(16)}",
              f"order_ref={random_hex(12).upper()}"]
    return f"/checkout?{'&'.join(params)}"


def generate_misc_uri():
    templates = [
        f"/user/profile?sid={random_session_id()}&tab={random.choice(['orders', 'wishlist', 'settings', 'addresses'])}",
        f"/deals/flash-sale?category={random.choice(CATEGORIES)}&sid={random_session_id()}",
        f"/recommendations?user_hash={random_hex(16)}&algo={random.choice(['collaborative', 'content-based', 'hybrid'])}",
        f"/compare?items={','.join([str(random.randint(100000, 999999)) for _ in range(random.randint(2, 4))])}",
        f"/reviews/product/{random.randint(100000, 999999)}?page={random.randint(1, 10)}&sort={random.choice(['newest', 'helpful', 'rating'])}",
        f"/api/autocomplete?q={random.choice(SEARCH_TERMS).split('+')[0]}&limit=10&lang=en",
        f"/track/order/{random_uuid4()}",
        "/", "/bestsellers", "/new-arrivals",
    ]
    return random.choice(templates)


def generate_referrer():
    ref = random.choice(REFERRERS)
    if '{}' in ref:
        filler = random.choice([
            random.choice(SEARCH_TERMS).replace('+', '%20'),
            str(random.randint(10000000, 99999999)),
            random_hex(11),
        ])
        ref = ref.format(filler)
    return ref


def make_headers(ua, session_id=None):
    """Build realistic HTTP headers for a payload."""
    headers = {"User-Agent": ua}
    headers["Accept"] = random.choice([
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "application/json, text/javascript, */*; q=0.01",
        "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    ])
    headers["Accept-Language"] = random.choice([
        "en-US,en;q=0.9", "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        "zh-CN,zh;q=0.9,en;q=0.8", "ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7",
        "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7", "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
        "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7", "es-ES,es;q=0.9,en;q=0.8",
        "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7", "ru-RU,ru;q=0.9,en;q=0.8",
    ])
    headers["Accept-Encoding"] = "gzip, deflate, br"
    if random.random() > 0.25:
        headers["Referer"] = generate_referrer()
    if random.random() > 0.3:
        sid = session_id or random_session_id()
        ga_id = f"GA1.1.{random.randint(100000000, 999999999)}.{int(time.time()) - random.randint(0, 2592000)}"
        fbp = f"fb.1.{int(time.time()) - random.randint(0, 604800)}.{random.randint(100000000, 999999999)}"
        parts = [f"session={sid}", f"_ga={ga_id}", f"_fbp={fbp}"]
        if random.random() > 0.5:
            parts.append(f"_gid=GA1.1.{random.randint(100000000, 999999999)}.{int(time.time())}")
        if random.random() > 0.5:
            parts.append(f"cart_token={random_hex(16)}")
        if random.random() > 0.6:
            parts.append(f"ab_bucket={random.choice(['control', 'variant_a', 'variant_b'])}")
        headers["Cookie"] = "; ".join(parts)
    if random.random() > 0.5:
        headers["Cache-Control"] = random.choice(["no-cache", "max-age=0", "no-store"])
    if random.random() > 0.7:
        headers["X-Requested-With"] = "XMLHttpRequest"
    return headers


# ============================================================
# Algorithmic session-chain generator (state machine)
# ============================================================

# Transition matrices per session type: state -> [(next_state, probability), ...]
TRANSITIONS = {
    "shopping": {
        "start":    [("homepage", 0.3), ("search", 0.7)],
        "homepage": [("search", 0.6), ("category", 0.3), ("deals", 0.1)],
        "search":   [("product", 0.55), ("filter", 0.2), ("search", 0.15), ("category", 0.1)],
        "filter":   [("product", 0.6), ("search", 0.3), ("category", 0.1)],
        "category": [("product", 0.6), ("search", 0.2), ("category", 0.2)],
        "product":  [("cart", 0.35), ("product", 0.25), ("search", 0.15), ("review", 0.15), ("wishlist", 0.1)],
        "review":   [("product", 0.5), ("cart", 0.3), ("search", 0.2)],
        "cart":     [("checkout", 0.5), ("product", 0.3), ("search", 0.2)],
        "checkout": [("end", 1.0)],
        "wishlist": [("product", 0.4), ("search", 0.4), ("end", 0.2)],
        "deals":    [("product", 0.6), ("search", 0.4)],
    },
    "research": {
        "start":    [("search", 0.8), ("category", 0.2)],
        "search":   [("product", 0.4), ("search", 0.25), ("filter", 0.2), ("compare", 0.15)],
        "filter":   [("product", 0.5), ("search", 0.3), ("compare", 0.2)],
        "category": [("product", 0.5), ("search", 0.3), ("category", 0.2)],
        "product":  [("review", 0.3), ("compare", 0.25), ("product", 0.2), ("search", 0.15), ("end", 0.1)],
        "review":   [("product", 0.4), ("compare", 0.3), ("search", 0.2), ("end", 0.1)],
        "compare":  [("product", 0.4), ("search", 0.3), ("end", 0.3)],
    },
    "impulse_buy": {
        "start":    [("homepage", 0.4), ("search", 0.3), ("deals", 0.3)],
        "homepage": [("deals", 0.4), ("search", 0.3), ("product", 0.3)],
        "deals":    [("product", 0.7), ("search", 0.3)],
        "search":   [("product", 0.7), ("filter", 0.2), ("category", 0.1)],
        "filter":   [("product", 0.8), ("search", 0.2)],
        "category": [("product", 0.7), ("search", 0.3)],
        "product":  [("cart", 0.6), ("product", 0.2), ("review", 0.2)],
        "review":   [("cart", 0.6), ("product", 0.4)],
        "cart":     [("checkout", 0.7), ("product", 0.3)],
        "checkout": [("end", 1.0)],
    },
    "comparison": {
        "start":    [("search", 0.9), ("category", 0.1)],
        "search":   [("product", 0.4), ("filter", 0.3), ("compare", 0.2), ("category", 0.1)],
        "filter":   [("product", 0.5), ("compare", 0.3), ("search", 0.2)],
        "category": [("product", 0.5), ("compare", 0.3), ("search", 0.2)],
        "product":  [("compare", 0.35), ("product", 0.25), ("review", 0.2), ("search", 0.1), ("cart", 0.1)],
        "review":   [("compare", 0.4), ("product", 0.4), ("search", 0.2)],
        "compare":  [("product", 0.3), ("cart", 0.25), ("search", 0.25), ("end", 0.2)],
        "cart":     [("checkout", 0.5), ("compare", 0.3), ("end", 0.2)],
        "checkout": [("end", 1.0)],
    },
    "return_customer": {
        "start":    [("homepage", 0.3), ("product", 0.4), ("orders", 0.3)],
        "homepage": [("product", 0.4), ("search", 0.3), ("orders", 0.3)],
        "orders":   [("product", 0.4), ("search", 0.3), ("end", 0.3)],
        "search":   [("product", 0.5), ("filter", 0.25), ("category", 0.25)],
        "filter":   [("product", 0.6), ("search", 0.4)],
        "category": [("product", 0.6), ("search", 0.4)],
        "product":  [("cart", 0.4), ("review", 0.2), ("product", 0.2), ("wishlist", 0.1), ("end", 0.1)],
        "review":   [("product", 0.5), ("cart", 0.3), ("end", 0.2)],
        "cart":     [("checkout", 0.6), ("product", 0.2), ("end", 0.2)],
        "checkout": [("end", 1.0)],
        "wishlist": [("product", 0.5), ("cart", 0.3), ("end", 0.2)],
    },
}


def _pick_next_state(session_type, current_state):
    """Weighted random next state from transition matrix."""
    trans = TRANSITIONS.get(session_type, TRANSITIONS["shopping"])
    options = trans.get(current_state, [("end", 1.0)])
    states, probs = zip(*options)
    return random.choices(states, weights=probs, k=1)[0]


def _state_to_uri(state, ctx):
    """Convert a state name to a URI, updating context dict for continuity."""
    if state == "homepage":
        return "/", "homepage"
    elif state == "search":
        term = ctx.get("search_term", random.choice(SEARCH_TERMS))
        ctx["search_term"] = term
        return generate_search_uri(term), "search"
    elif state == "filter":
        term = ctx.get("search_term", random.choice(SEARCH_TERMS))
        return generate_search_uri(term), "filter"
    elif state == "category":
        cat = ctx.get("category", random.choice(CATEGORIES))
        ctx["category"] = cat
        return generate_category_uri(cat), "category"
    elif state == "product":
        pid = random.randint(100000, 9999999)
        ctx["last_product_id"] = pid
        return generate_product_uri(pid), "product"
    elif state == "review":
        pid = ctx.get("last_product_id", random.randint(100000, 999999))
        sid = random_session_id()
        return f"/reviews/product/{pid}?page=1&sort=newest&sid={sid}", "review"
    elif state == "compare":
        items = [str(ctx.get("last_product_id", random.randint(100000, 999999)))]
        items += [str(random.randint(100000, 999999)) for _ in range(random.randint(1, 3))]
        return f"/compare?items={','.join(items)}&sid={random_session_id()}", "compare"
    elif state == "cart":
        return generate_cart_uri("add"), "cart"
    elif state == "checkout":
        return generate_checkout_uri(), "checkout"
    elif state == "wishlist":
        pid = ctx.get("last_product_id", random.randint(100000, 999999))
        return f"/wishlist/add?item_id={pid}&sid={random_session_id()}&csrf={random_hex(16)}", "wishlist"
    elif state == "deals":
        cat = random.choice(CATEGORIES)
        return f"/deals/flash-sale?category={cat}&sid={random_session_id()}", "homepage"
    elif state == "orders":
        return f"/user/profile?sid={random_session_id()}&tab=orders", "misc"
    else:
        return generate_misc_uri(), "misc"


def generate_session_chain_algorithmic(session_type, session_id=None):
    """
    State-machine driven session chain generation.
    Returns a session dict with steps list.
    """
    if session_id is None:
        session_id = random_session_id()

    ua = random.choice(USER_AGENTS)
    ctx = {"search_term": random.choice(SEARCH_TERMS), "category": random.choice(CATEGORIES)}

    steps = []
    state = "start"
    max_steps = random.randint(4, 12)

    for step_num in range(max_steps):
        state = _pick_next_state(session_type, state)
        if state == "end":
            break

        uri, step_type = _state_to_uri(state, ctx)
        think_time = pick_think_time(step_type)

        steps.append({
            "uri": uri,
            "user_agent": ua,
            "headers": make_headers(ua, session_id),
            "think_time": think_time,
            "step_type": step_type,
            "step_num": step_num,
            "session_id": session_id,
        })

    return {
        "session_id": session_id,
        "session_type": session_type,
        "source": "algorithmic",
        "steps": steps,
    }


# ============================================================
# LLM session-chain generator
# ============================================================
def _call_llm_requests(messages):
    import requests

    url = LLM_BASE_URL.rstrip('/') + '/chat/completions'
    headers = {
        'Authorization': f'Bearer {LLM_API_KEY}',
        'Content-Type': 'application/json',
    }
    payload = {
        'model': LLM_MODEL,
        'messages': messages,
        'response_format': {'type': 'json_object'},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT_SEC)
    resp.raise_for_status()
    body = resp.json()
    content = body['choices'][0]['message']['content']
    if isinstance(content, dict):
        return content
    return json.loads(content)


def call_llm_with_retry(client, messages, max_retries=3):
    """Call LLM API with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            if LLM_TRANSPORT == 'requests':
                return _call_llm_requests(messages)

            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                timeout=LLM_TIMEOUT_SEC,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"      Retry {attempt+1}/{max_retries} after {wait:.1f}s: {e}")
                time.sleep(wait)
            else:
                raise


def generate_session_chain_llm(client, session_type, target_steps=50):
    """
    Use LLM to generate a complete browsing session chain with causal logic.
    Returns a session dict or None on failure.
    """
    prompt = f"""You are simulating a realistic e-commerce user browsing session of type: "{session_type}".

Generate a JSON object with key "steps" containing an array of exactly {target_steps} HTTP GET browsing actions forming a coherent session with causal logic.

Session type descriptions:
- shopping: systematic product search, compare a few, add to cart, checkout
- research: deep exploration without buying — many product views, reviews, comparisons
- impulse_buy: quick browsing, spotted deal, fast add-to-cart and checkout (shorter session)
- comparison: focused comparison shopping — search, view several products, compare, maybe buy
- return_customer: returning user checks orders, browses familiar categories, rebuys or explores new

Each step must have:
1. "uri": A realistic e-commerce URL path with complex query parameters (search filters, session IDs, tracking params, UTM codes).
   Use paths like: /search?q=..., /category/CATEGORY/..., /CATEGORY/product/ID/SLUG?..., /cart?action=..., /checkout?step=..., /reviews/product/ID?..., /compare?items=..., /wishlist/add?..., /, /deals/flash-sale?...
2. "user_agent": A real browser User-Agent string (Chrome/Firefox/Safari/Edge, desktop or mobile)
3. "think_time": Human reading pause in seconds (must be between 0.5 and 15.0). Use these ranges based on page type:
   - search page: 1-4s
   - product page: 3-12s
   - cart/checkout: 0.5-2s
   - review page: 4-15s
   - comparison: 3-10s
   - homepage/category: 1.5-6s
4. "step_type": one of "search", "product", "cart", "checkout", "review", "compare", "homepage", "category", "wishlist", "filter"

IMPORTANT: Steps must follow a logical browsing flow. The same session_id should appear in query params across steps. Make URIs extremely realistic with diverse parameters."""

    try:
        data = call_llm_with_retry(client, [{"role": "user", "content": prompt}])
        steps = data.get("steps", [])
        if not steps:
            for key in ["actions", "items", "session", "requests"]:
                if key in data and isinstance(data[key], list):
                    steps = data[key]
                    break

        if not steps:
            return None

        session_id = random_session_id()
        ua = random.choice(USER_AGENTS)

        cleaned_steps = []
        for i, step in enumerate(steps):
            uri = step.get("uri", generate_misc_uri())
            if not uri.startswith("/"):
                uri = "/" + uri
            step_ua = step.get("user_agent", ua)
            think = step.get("think_time", pick_think_time("misc"))
            if not isinstance(think, (int, float)):
                think = pick_think_time("misc")
            think = round(max(0.5, min(15.0, float(think))), 3)
            step_type = step.get("step_type", "misc")

            cleaned_steps.append({
                "uri": uri,
                "user_agent": step_ua,
                "headers": make_headers(step_ua, session_id),
                "think_time": think,
                "step_type": step_type,
                "step_num": i,
                "session_id": session_id,
            })

        return {
            "session_id": session_id,
            "session_type": session_type,
            "source": "llm",
            "steps": cleaned_steps,
        }
    except Exception as e:
        print(f"      LLM session failed: {e}")
        return None


# ============================================================
# Main generation pipeline
# ============================================================
def main():
    print("=" * 65)
    print("  Phase 1: LLM Session-Chain Payload Generation")
    print(f"  Target: {NUM_TOTAL_PAYLOADS} payloads in session chains -> {OUTPUT_FILE}")
    print("=" * 65)

    all_sessions = []
    total_payloads = 0

    # ---- Step 1: LLM session chains ----
    print(f"\n[1/4] Generating {NUM_LLM_SESSIONS} LLM session chains...")
    llm_sessions = 0
    llm_payloads = 0

    try:
        client = None

        if NUM_LLM_SESSIONS <= 0:
            print("  [!] NUM_LLM_SESSIONS <= 0, skipping LLM generation")
            raise RuntimeError('llm_disabled')
        if not LLM_API_KEY:
            print("  [!] Missing LLM_API_KEY/DEEPSEEK_API_KEY/OPENAI_API_KEY, using algorithmic fallback")
            if REQUIRE_REAL_LLM:
                raise RuntimeError('missing_api_key_hard')
            raise RuntimeError('missing_api_key')

        print(f"  [i] LLM transport: {LLM_TRANSPORT}")
        if LLM_TRANSPORT == 'openai':
            from openai import OpenAI
            client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

        type_counts = {t: 0 for t in SESSION_TYPES}
        for i in range(NUM_LLM_SESSIONS):
            # Weighted session type selection
            stype = random.choices(SESSION_TYPES,
                                   weights=[SESSION_TYPE_WEIGHTS[t] for t in SESSION_TYPES], k=1)[0]
            session = generate_session_chain_llm(client, stype, target_steps=LLM_TARGET_STEPS)
            if session and session["steps"]:
                all_sessions.append(session)
                llm_sessions += 1
                llm_payloads += len(session["steps"])
                type_counts[stype] += 1
                if (i + 1) % 10 == 0:
                    print(f"    [{i+1}/{NUM_LLM_SESSIONS}] LLM sessions: {llm_sessions}, payloads: {llm_payloads}")
            else:
                # Fallback to algorithmic for this session
                session = generate_session_chain_algorithmic(stype)
                all_sessions.append(session)
                llm_payloads += len(session["steps"])
                type_counts[stype] += 1

        print(f"  -> LLM generated: {llm_sessions} sessions, {llm_payloads} payloads")
        print(f"     Type distribution: {type_counts}")
    except ImportError:
        print("  [!] openai package not installed, using full algorithmic generation")
    except Exception as e:
        if str(e) == 'missing_api_key_hard':
            print("  [!] REQUIRE_REAL_LLM=1 and no API key found. Aborting.")
            sys.exit(2)
        if str(e) not in ('llm_disabled', 'missing_api_key'):
            print(f"  [!] LLM API error: {e}, falling back to algorithmic")

    total_payloads = sum(len(s["steps"]) for s in all_sessions)

    # ---- Step 2: Algorithmic session chains to fill remaining ----
    remaining = NUM_TOTAL_PAYLOADS - total_payloads
    print(f"\n[2/4] Generating algorithmic sessions for ~{remaining} remaining payloads...")

    algo_sessions = 0
    while total_payloads < NUM_TOTAL_PAYLOADS:
        stype = random.choices(SESSION_TYPES,
                               weights=[SESSION_TYPE_WEIGHTS[t] for t in SESSION_TYPES], k=1)[0]
        session = generate_session_chain_algorithmic(stype)
        if session["steps"]:
            all_sessions.append(session)
            total_payloads += len(session["steps"])
            algo_sessions += 1
            if algo_sessions % 50 == 0:
                print(f"    Algorithmic sessions: {algo_sessions}, total payloads: {total_payloads}")

    print(f"  -> Algorithmic: {algo_sessions} sessions, total payloads now: {total_payloads}")

    # ---- Step 3: Build flat_payloads for backward compatibility ----
    print(f"\n[3/4] Building flat payload list...")
    random.shuffle(all_sessions)

    flat_payloads = []
    for session in all_sessions:
        for step in session["steps"]:
            flat_payloads.append({
                "uri": step["uri"],
                "user_agent": step["user_agent"],
                "headers": step["headers"],
                "think_time": step["think_time"],
                "session_id": step.get("session_id"),
                "step_type": step.get("step_type", "misc"),
            })

    # ---- Step 4: Save ----
    output = {
        "sessions": all_sessions,
        "flat_payloads": flat_payloads,
        "metadata": {
            "total_payloads": len(flat_payloads),
            "total_sessions": len(all_sessions),
            "llm_sessions": llm_sessions if 'llm_sessions' in dir() else 0,
            "session_type_counts": {},
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "require_real_llm": REQUIRE_REAL_LLM,
        }
    }

    # Count session types
    for s in all_sessions:
        st = s["session_type"]
        output["metadata"]["session_type_counts"][st] = output["metadata"]["session_type_counts"].get(st, 0) + 1

    print(f"\n[4/4] Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False)

    file_size = os.path.getsize(OUTPUT_FILE)

    # ---- Statistics ----
    print(f"\n{'=' * 65}")
    print(f"  Generation Complete!")
    print(f"{'=' * 65}")
    print(f"  Total sessions : {len(all_sessions)}")
    print(f"  Total payloads : {len(flat_payloads)}")
    print(f"  File size      : {file_size / 1024:.1f} KB ({file_size / 1048576:.2f} MB)")

    # Session stats
    session_lens = [len(s["steps"]) for s in all_sessions]
    print(f"\n  Session length stats:")
    print(f"    Min: {min(session_lens)}, Max: {max(session_lens)}, "
          f"Avg: {sum(session_lens)/len(session_lens):.1f}")

    # Source breakdown
    llm_count = sum(1 for s in all_sessions if s["source"] == "llm")
    algo_count = sum(1 for s in all_sessions if s["source"] == "algorithmic")
    llm_payload_count = sum(len(s["steps"]) for s in all_sessions if s["source"] == "llm")

    if REQUIRE_REAL_LLM and llm_count == 0:
        print("\n  [ERROR] REQUIRE_REAL_LLM=1 but llm_sessions=0.")
        sys.exit(3)

    print(f"\n  Source breakdown:")
    llm_ratio = (llm_payload_count / len(flat_payloads) * 100.0) if flat_payloads else 0.0
    print(f"    LLM sessions       : {llm_count} ({llm_payload_count} payloads, {llm_ratio:.1f}%)")
    print(f"    Algorithmic sessions: {algo_count}")

    # Session type distribution
    print(f"\n  Session type distribution:")
    for st, count in sorted(output["metadata"]["session_type_counts"].items()):
        print(f"    {st:20s}: {count}")

    # URI uniqueness
    uris = [p["uri"] for p in flat_payloads]
    unique_uris = len(set(uris))
    avg_uri_len = sum(len(u) for u in uris) / len(uris)
    print(f"\n  URI stats:")
    print(f"    Unique URIs  : {unique_uris}/{len(uris)} ({unique_uris/len(uris)*100:.1f}%)")
    print(f"    Avg URI len  : {avg_uri_len:.0f} chars")

    think_times = [p["think_time"] for p in flat_payloads]
    print(f"    Think times  : min={min(think_times):.2f}s, max={max(think_times):.2f}s, "
          f"avg={sum(think_times)/len(think_times):.2f}s")

    print(f"\n  Sample session (first):")
    sample = all_sessions[0]
    print(f"    Type: {sample['session_type']}, Source: {sample['source']}, Steps: {len(sample['steps'])}")
    for i, step in enumerate(sample["steps"][:4]):
        print(f"      [{i}] {step['step_type']:10s} -> {step['uri'][:80]}...")

    print(f"\n[Done] Ready for Mininet arena!")


if __name__ == "__main__":
    main()
