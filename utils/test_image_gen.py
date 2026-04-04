"""
Synthetic grayscale test image generator.
Provides geometric patterns and simple object silhouettes usable as
pipeline input when no real image is available.
All generate() variants return a 2D uint8 numpy array of shape (size, size).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Pattern names shown in the GUI combo-box (order matters)
PATTERN_NAMES: list[str] = [
    # ── geometric ──────────────────────────────────────────────────────────
    "Random Noise",
    "Gradient (H)",
    "Gradient (V)",
    "Checkerboard",
    "Circles",
    "Stripes (H)",
    "Stripes (V)",
    "Rings",
    "Gaussian Blob",
    # ── objects ────────────────────────────────────────────────────────────
    "Airplane",
    "Bird in Flight",
    "House",
    "Tree",
    "Car (Side)",
    "Star",
    "Human Figure",
    "Rocket",
    # ── edge cases ─────────────────────────────────────────────────────────
    "All Black",
    "All White",
    "Uniform Gray",
    "Step Edge",
    "Single Bright Pixel",
    "Salt & Pepper (Sparse)",
    "High Freq Grid",
    "Low Contrast",
]


# ── geometric patterns ────────────────────────────────────────────────────────

def generate(pattern: str, size: int, seed: int = 42) -> np.ndarray:
    """Return a (size × size) uint8 numpy array for *pattern*."""
    rng = np.random.default_rng(seed)
    n = size

    if pattern == "Random Noise":
        return rng.integers(0, 256, (n, n), dtype=np.uint8)

    elif pattern == "Gradient (H)":
        row = np.linspace(0, 255, n, dtype=np.float32)
        return np.tile(row, (n, 1)).astype(np.uint8)

    elif pattern == "Gradient (V)":
        col = np.linspace(0, 255, n, dtype=np.float32).reshape(-1, 1)
        return np.tile(col, (1, n)).astype(np.uint8)

    elif pattern == "Checkerboard":
        tile = max(n // 16, 1)
        i = np.arange(n)
        I, J = np.meshgrid(i, i, indexing="ij")
        return (((I // tile) + (J // tile)) % 2 * 255).astype(np.uint8)

    elif pattern == "Circles":
        cy, cx = n / 2.0, n / 2.0
        Y, X = np.ogrid[:n, :n]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        spacing = max(n // 8, 1)
        thickness = max(n // 64, 1)
        img = np.zeros((n, n), dtype=np.uint8)
        for r in np.arange(spacing, n // 2 + spacing, spacing):
            img[np.abs(dist - r) < thickness] = 255
        return img

    elif pattern == "Stripes (H)":
        stripe = max(n // 16, 1)
        row = (np.arange(n) // stripe % 2 * 255).astype(np.uint8)
        return np.tile(row[:, None], (1, n))

    elif pattern == "Stripes (V)":
        stripe = max(n // 16, 1)
        col = (np.arange(n) // stripe % 2 * 255).astype(np.uint8)
        return np.tile(col[None, :], (n, 1))

    elif pattern == "Rings":
        cy, cx = n / 2.0, n / 2.0
        Y, X = np.ogrid[:n, :n]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        freq = max(n // 8, 1)
        return ((np.sin(dist * 2 * np.pi / freq) + 1) / 2 * 255).astype(np.uint8)

    elif pattern == "Gaussian Blob":
        cy, cx = n / 2.0, n / 2.0
        Y, X = np.ogrid[:n, :n]
        sigma = n / 4.0
        blob = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))
        return (blob * 255).astype(np.uint8)

    # ── object patterns ──────────────────────────────────────────────────────

    elif pattern == "Airplane":
        return _draw_airplane(n)

    elif pattern == "Bird in Flight":
        return _draw_bird(n)

    elif pattern == "House":
        return _draw_house(n)

    elif pattern == "Tree":
        return _draw_tree(n)

    elif pattern == "Car (Side)":
        return _draw_car(n)

    elif pattern == "Star":
        return _draw_star(n)

    elif pattern == "Human Figure":
        return _draw_human(n)

    elif pattern == "Rocket":
        return _draw_rocket(n)

    # ── edge cases ───────────────────────────────────────────────────────────

    elif pattern == "All Black":
        return np.zeros((n, n), dtype=np.uint8)

    elif pattern == "All White":
        return np.full((n, n), 255, dtype=np.uint8)

    elif pattern == "Uniform Gray":
        return np.full((n, n), 128, dtype=np.uint8)

    elif pattern == "Step Edge":
        # Left half black, right half white — classic edge-detection test case
        img = np.zeros((n, n), dtype=np.uint8)
        img[:, n // 2 :] = 255
        return img

    elif pattern == "Single Bright Pixel":
        # Black background with one 255-valued pixel dead-center
        img = np.zeros((n, n), dtype=np.uint8)
        img[n // 2, n // 2] = 255
        return img

    elif pattern == "Salt & Pepper (Sparse)":
        # Uniform gray (128) with ~2 % of pixels randomly set to 0 or 255
        img = np.full((n, n), 128, dtype=np.uint8)
        total = n * n
        n_salt = int(total * 0.01)
        n_pepper = int(total * 0.01)
        flat = img.ravel()
        salt_idx   = rng.choice(total, n_salt,   replace=False)
        pepper_idx = rng.choice(total, n_pepper, replace=False)
        flat[salt_idx]   = 255
        flat[pepper_idx] = 0
        return img

    elif pattern == "High Freq Grid":
        # Alternating 0/255 checkerboard at 1-pixel period (maximum frequency)
        I, J = np.ogrid[:n, :n]
        return ((I + J) % 2 * 255).astype(np.uint8)

    elif pattern == "Low Contrast":
        # Smooth gradient confined to a narrow band (120–135) — near-flat signal
        row = np.linspace(120, 135, n, dtype=np.float32)
        return np.tile(row, (n, 1)).astype(np.uint8)

    # Fallback: black image
    return np.zeros((n, n), dtype=np.uint8)


# ── object drawing helpers ────────────────────────────────────────────────────

def _canvas(n: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Return a black L-mode PIL image and its Draw object."""
    img = Image.new("L", (n, n), 0)
    return img, ImageDraw.Draw(img)


def _to_array(img: Image.Image) -> np.ndarray:
    return np.array(img, dtype=np.uint8)


def _draw_airplane(n: int) -> np.ndarray:
    """Top-down silhouette of a passenger jet."""
    img, d = _canvas(n)
    s = n / 512          # scale factor (1.0 at 512 px)
    cx, cy = n / 2, n / 2

    def sc(pts):
        return [(cx + x * s, cy + y * s) for x, y in pts]

    # Fuselage (long narrow ellipse)
    fw, fh = 24, 200
    d.ellipse([cx - fw * s, cy - fh * s, cx + fw * s, cy + fh * s], fill=220)

    # Main wings (swept-back trapezoid, bright)
    wing_l = sc([(-18, -20), (-200, 80), (-200, 110), (-18, 30)])
    wing_r = sc([(18, -20), (200, 80), (200, 110), (18, 30)])
    d.polygon(wing_l, fill=200)
    d.polygon(wing_r, fill=200)

    # Tail horizontal stabilizers
    stab_l = sc([(-10, 145), (-70, 185), (-70, 200), (-10, 175)])
    stab_r = sc([(10, 145), (70, 185), (70, 200), (10, 175)])
    d.polygon(stab_l, fill=180)
    d.polygon(stab_r, fill=180)

    # Cockpit bulge
    d.ellipse([cx - 10 * s, cy - 200 * s, cx + 10 * s, cy - 160 * s], fill=255)

    # Engines (circles on wings)
    for ex in (-130, 130):
        ey = 40
        d.ellipse([
            cx + (ex - 14) * s, cy + (ey - 18) * s,
            cx + (ex + 14) * s, cy + (ey + 18) * s,
        ], fill=160)

    return _to_array(img)


def _draw_bird(n: int) -> np.ndarray:
    """Two curved wings spread in flight — classic bird silhouette."""
    img, d = _canvas(n)
    s = n / 512
    cx, cy = n / 2, n * 0.48

    def sc(pts):
        return [(cx + x * s, cy + y * s) for x, y in pts]

    # Left wing (upward curve)
    left = sc([
        (0, 0), (-30, -20), (-80, -55), (-140, -65),
        (-200, -50), (-220, -30), (-200, -20),
        (-130, -40), (-70, -35), (-30, -5), (0, 10),
    ])
    # Right wing (mirror)
    right = sc([
        (0, 0), (30, -20), (80, -55), (140, -65),
        (200, -50), (220, -30), (200, -20),
        (130, -40), (70, -35), (30, -5), (0, 10),
    ])
    d.polygon(left, fill=230)
    d.polygon(right, fill=230)

    # Body
    body = sc([(-20, -5), (0, -30), (20, -5), (10, 30), (-10, 30)])
    d.polygon(body, fill=255)

    # Head
    d.ellipse([cx - 12 * s, cy - 55 * s, cx + 12 * s, cy - 30 * s], fill=255)

    # Tail feathers
    tail = sc([(-15, 28), (0, 65), (15, 28)])
    d.polygon(tail, fill=210)

    return _to_array(img)


def _draw_house(n: int) -> np.ndarray:
    """Simple front-view house: rectangle body + triangle roof + door + windows."""
    img, d = _canvas(n)
    s = n / 512
    cx, cy = n / 2, n / 2 + 30 * s

    def p(x, y):
        return (cx + x * s, cy + y * s)

    # Body
    bw, bh = 160, 130
    d.rectangle([p(-bw, -bh), p(bw, bh)], fill=180)

    # Roof (triangle)
    roof = [p(-180, -bh), p(0, -bh - 150), p(180, -bh)]
    d.polygon(roof, fill=230)

    # Door (centered, bottom half)
    dw, dh = 35, 75
    d.rectangle([p(-dw, bh - dh), p(dw, bh)], fill=100)
    # Door knob
    d.ellipse([p(22, bh - 35), p(30, bh - 27)], fill=220)

    # Left window
    d.rectangle([p(-130, -60), p(-60, 10)], fill=220)
    d.line([p(-130, -25), p(-60, -25)], fill=80, width=max(2, int(3 * s)))
    d.line([p(-95, -60), p(-95, 10)], fill=80, width=max(2, int(3 * s)))

    # Right window
    d.rectangle([p(60, -60), p(130, 10)], fill=220)
    d.line([p(60, -25), p(130, -25)], fill=80, width=max(2, int(3 * s)))
    d.line([p(95, -60), p(95, 10)], fill=80, width=max(2, int(3 * s)))

    # Chimney
    d.rectangle([p(80, -bh - 130), p(115, -bh - 10)], fill=200)

    return _to_array(img)


def _draw_tree(n: int) -> np.ndarray:
    """Layered canopy tree (three stacked triangles + trunk)."""
    img, d = _canvas(n)
    s = n / 512
    cx = n / 2
    top_y = n * 0.08

    def row(apex_y, half_w, ht, fill_v):
        apex = (cx, top_y + apex_y * s)
        left = (cx - half_w * s, top_y + (apex_y + ht) * s)
        right = (cx + half_w * s, top_y + (apex_y + ht) * s)
        d.polygon([apex, left, right], fill=fill_v)

    # Three layered triangles (gets darker toward bottom = depth)
    row(0,   170, 180, 255)
    row(130, 190, 180, 220)
    row(260, 210, 180, 190)

    # Trunk
    trunk_top = top_y + 420 * s
    trunk_bot = top_y + 500 * s if (top_y + 500 * s) < n else n - 4
    tw = 28 * s
    d.rectangle([cx - tw, trunk_top, cx + tw, trunk_bot], fill=160)

    return _to_array(img)


def _draw_car(n: int) -> np.ndarray:
    """Side-view silhouette of a sedan."""
    img, d = _canvas(n)
    s = n / 512
    cx, cy = n / 2, n / 2 + 50 * s

    def p(x, y):
        return (cx + x * s, cy + y * s)

    # Main body (lower box)
    d.rectangle([p(-210, -30), p(210, 70)], fill=200)

    # Cabin (upper polygon — tapered)
    cabin = [p(-120, -30), p(-150, -110), p(-60, -140),
             p(80, -140), p(140, -110), p(140, -30)]
    d.polygon(cabin, fill=220)

    # Windows (slightly darker)
    d.polygon([p(-130, -32), p(-138, -100), p(-55, -130),
               p(-10, -130), p(-10, -32)], fill=140)
    d.polygon([p(0, -32), p(0, -130), p(80, -130),
               p(128, -100), p(128, -32)], fill=140)
    # Window divider
    d.line([p(-10, -32), p(-10, -130)], fill=230, width=max(2, int(4 * s)))

    # Wheels (filled circles)
    wr = 48 * s
    for wx in (-120, 120):
        d.ellipse([p(wx - wr, 30), p(wx + wr, 30 + 2 * wr)], fill=50)
        # Hub
        d.ellipse([p(wx - wr * 0.45, 30 + wr * 0.55),
                   p(wx + wr * 0.45, 30 + wr * 1.45)], fill=160)

    # Headlight / taillight dots
    d.ellipse([p(195, 10), p(215, 30)], fill=255)
    d.ellipse([p(-215, 10), p(-195, 30)], fill=255)

    return _to_array(img)


def _draw_star(n: int) -> np.ndarray:
    """5-pointed star, filled, with slight glow."""
    img, d = _canvas(n)
    cx, cy = n / 2, n / 2
    outer = n * 0.42
    inner = outer * 0.42

    pts = []
    for i in range(10):
        angle = math.pi / 5 * i - math.pi / 2
        r = outer if i % 2 == 0 else inner
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))

    d.polygon(pts, fill=255)
    # Soft glow: blur, then composite
    glow = img.filter(ImageFilter.GaussianBlur(radius=n // 30))
    composite = Image.blend(glow, img, 0.75)
    return _to_array(composite)


def _draw_human(n: int) -> np.ndarray:
    """Simple stick-figure human silhouette."""
    img, d = _canvas(n)
    s = n / 512
    cx = n / 2
    top_y = n * 0.12
    lw = max(3, int(12 * s))

    def p(x, y):
        return (cx + x * s, top_y + y * s)

    # Head
    hr = 40 * s
    d.ellipse([p(-hr, 0), p(hr, 2 * hr)], fill=230)

    # Neck + torso
    d.line([p(0, 80), p(0, 230)], fill=230, width=lw)

    # Arms (spread wide)
    d.line([p(0, 120), p(-130, 200)], fill=230, width=lw)   # left upper arm
    d.line([p(-130, 200), p(-100, 290)], fill=230, width=lw) # left forearm
    d.line([p(0, 120), p(130, 200)], fill=230, width=lw)    # right upper arm
    d.line([p(130, 200), p(100, 290)], fill=230, width=lw)   # right forearm

    # Hips
    d.line([p(-50, 230), p(50, 230)], fill=230, width=lw)

    # Legs
    d.line([p(-40, 230), p(-60, 360)], fill=230, width=lw)   # left thigh
    d.line([p(-60, 360), p(-50, 460)], fill=230, width=lw)   # left shin
    d.line([p(40, 230), p(60, 360)], fill=230, width=lw)     # right thigh
    d.line([p(60, 360), p(50, 460)], fill=230, width=lw)     # right shin

    # Feet
    d.line([p(-50, 460), p(-85, 475)], fill=230, width=lw)
    d.line([p(50, 460), p(85, 475)], fill=230, width=lw)

    return _to_array(img)


def _draw_rocket(n: int) -> np.ndarray:
    """Upright rocket with nose cone, fins, and engine exhaust."""
    img, d = _canvas(n)
    s = n / 512
    cx, cy = n / 2, n / 2

    def p(x, y):
        return (cx + x * s, cy + y * s)

    # Body tube
    bw, bh = 55, 200
    d.rectangle([p(-bw, -bh), p(bw, bh)], fill=200)

    # Nose cone (pointed triangle)
    nose = [p(-bw, -bh), p(0, -bh - 160), p(bw, -bh)]
    d.polygon(nose, fill=240)

    # Porthole
    pr = 28 * s
    d.ellipse([p(-pr, -60 * s), p(pr, 60 * s - (60 * s - pr * 2))], fill=140)
    d.ellipse([cx - pr, cy - pr, cx + pr, cy + pr], fill=140)

    # Left fin
    fin_l = [p(-bw, 100), p(-bw - 80, 200), p(-bw, 200)]
    d.polygon(fin_l, fill=220)
    # Right fin
    fin_r = [p(bw, 100), p(bw + 80, 200), p(bw, 200)]
    d.polygon(fin_r, fill=220)

    # Engine bell
    bell = [p(-bw + 10, bh), p(-bw - 10, bh + 30),
            p(bw + 10, bh + 30), p(bw - 10, bh)]
    d.polygon(bell, fill=180)

    # Exhaust flame
    flame = [p(-40, bh + 30), p(0, bh + 130), p(40, bh + 30)]
    d.polygon(flame, fill=255)
    flame2 = [p(-25, bh + 30), p(0, bh + 90), p(25, bh + 30)]
    d.polygon(flame2, fill=200)

    return _to_array(img)


# ── public save helper ────────────────────────────────────────────────────────

def save_image(pattern: str, size: int, seed: int, output_path: Path) -> Path:
    """Generate *pattern* and save it as a grayscale PNG. Returns *output_path*."""
    arr = generate(pattern, size, seed)
    Image.fromarray(arr, mode="L").save(str(output_path))
    return output_path
