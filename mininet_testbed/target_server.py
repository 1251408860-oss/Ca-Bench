#!/usr/bin/env python3
"""
Phase 1 - Custom Target HTTP Server (runs INSIDE Mininet on target host)
STDLIB ONLY - no pip dependencies required.

Returns different-sized HTML/JSON responses by URI path so pcap traffic
has realistic bidirectional volume variation.

Usage:
  python3 target_server.py [port]          # default port 80
"""
import sys
import random
import string
import hashlib
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 80

# ============================================================
# Pre-generate response bodies (once at startup for speed)
# ============================================================

def _random_words(n):
    words = ["product", "item", "sale", "deal", "new", "best", "top", "review",
             "price", "discount", "shipping", "free", "quality", "premium", "offer",
             "stock", "brand", "popular", "trending", "limited", "exclusive", "save",
             "compare", "rating", "star", "warranty", "delivery", "fast", "order",
             "category", "electronics", "clothing", "home", "garden", "sports"]
    return ' '.join(random.choices(words, k=n))


def _make_html(title, body_paragraphs, extra_kb=0):
    """Build a realistic HTML page of controllable size."""
    paras = "\n".join(
        f"<p>{_random_words(random.randint(20, 50))}</p>"
        for _ in range(body_paragraphs)
    )
    nav_items = "\n".join(
        f'<li><a href="/category/{c}">{c.title()}</a></li>'
        for c in ["electronics", "clothing", "home-garden", "sports", "books",
                   "toys", "automotive", "health-beauty", "jewelry", "office"]
    )
    # Pad with hidden divs to reach target size
    padding = ""
    if extra_kb > 0:
        padding = f'<div style="display:none">{"x" * (extra_kb * 1024)}</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} - MegaShop</title>
<meta name="description" content="{_random_words(15)}">
<link rel="stylesheet" href="/static/css/main.css">
<script src="/static/js/analytics.js" async></script>
</head>
<body>
<header>
<nav><ul>{nav_items}</ul></nav>
<div class="search-bar"><input type="text" placeholder="Search products..."></div>
</header>
<main>
<h1>{title}</h1>
{paras}
</main>
<footer>
<p>MegaShop Inc. All rights reserved. | <a href="/privacy">Privacy</a> | <a href="/terms">Terms</a></p>
<script>window._analytics={{page:"{title}",ts:{int(time.time())}}};</script>
</footer>
{padding}
</body>
</html>"""


def _make_product_grid(n_items):
    """Generate HTML for a grid of product cards."""
    items = []
    for _ in range(n_items):
        pid = random.randint(100000, 999999)
        price = round(random.uniform(9.99, 999.99), 2)
        rating = round(random.uniform(3.0, 5.0), 1)
        items.append(
            f'<div class="product-card" data-id="{pid}">'
            f'<img src="/img/product/{pid}.jpg" alt="{_random_words(4)}" loading="lazy">'
            f'<h3><a href="/product/{pid}">{_random_words(5).title()}</a></h3>'
            f'<span class="price">${price}</span>'
            f'<span class="rating">{rating}/5 ({random.randint(10, 5000)} reviews)</span>'
            f'<button class="add-cart" data-pid="{pid}">Add to Cart</button>'
            f'</div>'
        )
    return "\n".join(items)


# Pre-build response templates
HOMEPAGE_BODY = _make_html("Welcome to MegaShop", 15, extra_kb=10)         # ~20KB
SEARCH_BODY = _make_html("Search Results", 8, extra_kb=2)                  # ~15KB base
PRODUCT_BODY = _make_html("Product Details", 12, extra_kb=3)               # ~8KB
CATEGORY_BODY = _make_html("Category Browse", 10, extra_kb=3)              # ~12KB
CART_BODY = _make_html("Your Shopping Cart", 8, extra_kb=1)                # ~5KB
CHECKOUT_BODY = _make_html("Checkout", 4)                                  # ~4KB
NOTFOUND_BODY = _make_html("Page Not Found", 2)                           # ~2KB


class ShopHandler(BaseHTTPRequestHandler):
    """Custom HTTP handler — returns different-sized responses by URI."""

    # Suppress per-request logging for performance
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        path = self.path.split('?')[0].rstrip('/')

        if path == '' or path == '/':
            body = HOMEPAGE_BODY
            ctype = "text/html"
        elif path.startswith('/search'):
            # Dynamic: inject search query into page + product grid
            grid = _make_product_grid(random.randint(12, 24))
            body = SEARCH_BODY.replace("</main>", f"<div class='results'>{grid}</div></main>")
            ctype = "text/html"
        elif '/product/' in path or path.startswith('/product'):
            body = PRODUCT_BODY
            ctype = "text/html"
        elif path.startswith('/category'):
            grid = _make_product_grid(random.randint(12, 24))
            body = CATEGORY_BODY.replace("</main>", f"<div class='grid'>{grid}</div></main>")
            ctype = "text/html"
        elif path.startswith('/cart'):
            body = CART_BODY
            ctype = "text/html"
        elif path.startswith('/checkout'):
            body = CHECKOUT_BODY
            ctype = "text/html"
        elif path.startswith('/api/'):
            # Small JSON response
            body = '{"status":"ok","ts":%d,"data":{}}' % int(time.time())
            ctype = "application/json"
        elif path.startswith('/wishlist'):
            body = '{"status":"ok","action":"added"}'
            ctype = "application/json"
        elif path.startswith('/compare'):
            body = _make_html("Product Comparison", 8, extra_kb=2)
            ctype = "text/html"
        elif path.startswith('/reviews'):
            body = _make_html("Product Reviews", 15)
            ctype = "text/html"
        elif path.startswith('/deals'):
            grid = _make_product_grid(random.randint(8, 16))
            body = _make_html("Flash Deals", 6).replace("</main>", f"<div class='deals'>{grid}</div></main>")
            ctype = "text/html"
        elif path in ('/bestsellers', '/new-arrivals'):
            grid = _make_product_grid(random.randint(12, 20))
            body = _make_html(path.strip('/').replace('-', ' ').title(), 4).replace(
                "</main>", f"<div class='grid'>{grid}</div></main>")
            ctype = "text/html"
        elif path.startswith('/user') or path.startswith('/track'):
            body = _make_html("Account", 4)
            ctype = "text/html"
        elif path.startswith('/recommendations'):
            body = '{"items":[' + ','.join(
                f'{{"id":{random.randint(100000,999999)},"score":{round(random.uniform(0.5,1.0),3)}}}'
                for _ in range(10)
            ) + ']}'
            ctype = "application/json"
        else:
            body = NOTFOUND_BODY
            ctype = "text/html"

        encoded = body.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', f'{ctype}; charset=utf-8')
        self.send_header('Content-Length', str(len(encoded)))
        self.send_header('Server', 'MegaShop/2.1')
        self.send_header('X-Request-Id', hashlib.md5(
            f"{time.time()}{random.random()}".encode()).hexdigest()[:12])
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self):
        # Treat POST same as GET for simplicity (cart actions etc.)
        self.do_GET()


def main():
    server = HTTPServer(('0.0.0.0', PORT), ShopHandler)
    # Silently serve — no startup banner to avoid cluttering Mininet logs
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()


if __name__ == '__main__':
    main()
