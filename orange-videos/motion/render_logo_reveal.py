"""Orange Videos logo reveal: 1920x1080, 30 fps, 10 s intro/outro.

Liquid drops drift in slowly and collect into the orange circle, the leaf
sprouts, the white play triangle pops in, the mark holds, then a fast zoom
into the triangle takes the frame to full white.

The logo geometry is measured from ../brand/orange-videos-logo.jpg (circle
fitted, triangle rebuilt as a vector with rounded corners, leaf traced as a
polar outline), so edges stay crisp even at 100x zoom.
Usage: python3 render_logo_reveal.py [output.mp4]
"""
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image

W, H, FPS = 1920, 1080, 30
DURATION = 10.0
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "brand", "orange-videos-logo.jpg")

# timeline (seconds): slow liquid build, quick zoom out of the hold
T_SETTLE = 3.4  # surface wobble starts to calm
T_LEAF = 4.3  # leaf sprouts
T_TRI = 4.8  # play triangle pops in
T_ZOOM = 7.8  # zoom starts
T_WHITE = 8.9  # frame fully white
ZOOM_MAX = 110.0

SCREEN_R = 250.0  # circle radius on screen, px


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def prog(t, start, dur):
    return clamp((t - start) / dur)


def ease_out(x):
    return 1 - (1 - x) ** 3


def ease_in(x):
    return x ** 3


def ease_in_out(x):
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out_back(x, s=2.2):
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


# ---------- measure the logo ----------
def bilinear(f, x, y):
    h, w = f.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return (f[y0, x0] * (1 - fx) * (1 - fy) + f[y0, x0 + 1] * fx * (1 - fy)
            + f[y0 + 1, x0] * (1 - fx) * fy + f[y0 + 1, x0 + 1] * fx * fy)


def polar_outline(field, cx, cy, n=1440, r_max=400, step=0.1, outermost=True):
    """Sub-pixel radius where `field` crosses 0.5 along n rays from (cx, cy).

    outermost=True takes the last inside sample, False the first outside one.
    """
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rs = np.arange(0, r_max, step)
    v = bilinear(field, cx + np.cos(th)[:, None] * rs, cy + np.sin(th)[:, None] * rs) > 0.5
    if outermost:
        idx = v.shape[1] - 1 - np.argmax(v[:, ::-1], axis=1)
    else:
        idx = np.argmax(~v, axis=1)
    return th, rs[idx]


def fit_circle(x, y):
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    (a, b, c), *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    return a, b, math.sqrt(c + a * a + b * b)


def fit_rounded_triangle(th, r, cx, cy):
    """Fit three straight edges to a polar outline; return sharp vertices + corner radius."""
    pts = np.c_[cx + r * np.cos(th), cy + r * np.sin(th)]
    n = len(r)
    # corners are the three well-separated maxima of r(θ)
    corners = []
    for i in np.argsort(-r):
        if all(min(abs(i - j), n - abs(i - j)) > n // 8 for j in corners):
            corners.append(i)
        if len(corners) == 3:
            break
    corners.sort()
    lines = []
    for k in range(3):
        a, b = corners[k], corners[(k + 1) % 3]
        span = (b - a) % n
        idx = (a + np.arange(int(span * 0.3), int(span * 0.7))) % n  # straight middle of the edge
        p = pts[idx]
        mu = p.mean(axis=0)
        normal = np.linalg.svd(p - mu)[2][1]
        lines.append((normal, normal @ mu))
    verts = np.array([np.linalg.solve(np.array([lines[k - 1][0], lines[k][0]]),
                                      np.array([lines[k - 1][1], lines[k][1]])) for k in range(3)])
    # a corner rounded with radius ρ sits ρ(1/sin(θ/2) - 1) inside the sharp vertex
    rhos = []
    for k in range(3):
        v = verts[k]
        rel = v - [cx, cy]
        ci = int(round(np.mod(math.atan2(rel[1], rel[0]), 2 * np.pi) / (2 * np.pi) * n)) % n
        gap = np.hypot(*rel) - r[ci]
        e1, e2 = verts[k - 1] - v, verts[(k + 1) % 3] - v
        ang = math.acos(np.dot(e1, e2) / np.linalg.norm(e1) / np.linalg.norm(e2))
        rhos.append(gap / (1 / math.sin(ang / 2) - 1))
    return verts, float(np.mean(rhos))


def measure_logo(path):
    im = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    rb = im[..., 0] - im[..., 2]
    orange_mask = rb > 150
    colour = np.median(im[orange_mask], axis=0)
    field = np.clip(rb / (colour[0] - colour[2]), 0, 1)  # 1 = orange, 0 = white
    ys, xs = np.nonzero(orange_mask)
    split_y = int(ys.min() + 0.19 * (ys.max() - ys.min()))  # gap between leaf and circle

    # body circle: everything below the leaf
    body = field.copy()
    body[:split_y] = 0
    by, bx = np.nonzero(body > 0.5)
    th, r = polar_outline(body, bx.mean(), by.mean())
    ccx, ccy, cr = fit_circle(bx.mean() + r * np.cos(th), by.mean() + r * np.sin(th))

    # play triangle: the white hole inside the circle
    yy, xx = np.mgrid[0:im.shape[0], 0:im.shape[1]]
    hole = ((xx - ccx) ** 2 + (yy - ccy) ** 2 < (0.85 * cr) ** 2) & (field < 0.5)
    ty, tx = np.nonzero(hole)
    tcx, tcy = tx.mean(), ty.mean()
    th, r = polar_outline(1 - field, tcx, tcy, r_max=0.8 * cr, outermost=False)
    tri = fit_rounded_triangle(th, r, tcx, tcy)

    # leaf: polar outline about its own centroid
    leaf = field.copy()
    leaf[split_y:] = 0
    ly, lx = np.nonzero(leaf > 0.5)
    lcx, lcy = lx.mean(), ly.mean()
    _, l_r = polar_outline(leaf, lcx, lcy, r_max=200)
    l_r = np.convolve(np.r_[l_r[-3:], l_r, l_r[:3]], np.ones(7) / 7, mode="valid")  # de-noise JPEG edge
    base = (lx.min() + 0.15 * (lx.max() - lx.min()), ly.max())

    return dict(colour=colour, circle=(ccx, ccy, cr), tri=tri,
                leaf=(lcx, lcy, l_r, base), bbox=(xs.min(), ys.min(), xs.max(), ys.max()))


LOGO = measure_logo(LOGO_PATH)
ORANGE = LOGO["colour"].astype(np.float32)
ORANGE_LIGHT = ORANGE + (255 - ORANGE) * 0.45
WHITE = np.array([255, 255, 255], np.float32)

# logo image -> screen mapping (whole mark centred, circle radius = SCREEN_R)
_ccx, _ccy, _cr = LOGO["circle"]
K = SCREEN_R / _cr
_x0, _y0, _x1, _y1 = LOGO["bbox"]
ANCHOR = np.array([(_x0 + _x1) / 2, (_y0 + _y1) / 2])
SCREEN_O = np.array([W / 2, H / 2])


def to_screen(p):
    return SCREEN_O + (np.asarray(p, np.float64) - ANCHOR) * K


C = to_screen((_ccx, _ccy))  # circle centre on screen
R = SCREEN_R
TRI_V = np.array([to_screen(v) for v in LOGO["tri"][0]])
TRI_RHO = LOGO["tri"][1] * K
_lcx, _lcy, _leaf_r, _lbase = LOGO["leaf"]
LEAF_C = to_screen((_lcx, _lcy))
LEAF_BASE = to_screen(_lbase)
LEAF_R = _leaf_r * K

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
XX += 0.5
YY += 0.5


# ---------- signed distance fields ----------
def smin(a, b, k):
    """Polynomial smooth minimum: the 'gooey' merge between liquid shapes."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1 - h)


def sd_ellipse_approx(px, py, cx, cy, a, b, ang):
    """Cheap stretched-circle distance, good enough for moving droplets."""
    ca, sa = math.cos(ang), math.sin(ang)
    dx, dy = px - cx, py - cy
    u = (dx * ca + dy * sa) / a
    v = (-dx * sa + dy * ca) / b
    return (np.sqrt(u * u + v * v) - 1.0) * min(a, b)


def sd_triangle(px, py, v):
    """Exact signed distance to a triangle with vertices v (3x2)."""
    d2 = None
    signs = []
    for i in range(3):
        p, e = v[i], v[(i + 1) % 3] - v[i]
        wx, wy = px - p[0], py - p[1]
        h = np.clip((wx * e[0] + wy * e[1]) / (e @ e), 0, 1)
        qx, qy = wx - e[0] * h, wy - e[1] * h
        dd = qx * qx + qy * qy
        d2 = dd if d2 is None else np.minimum(d2, dd)
        signs.append(wx * e[1] - wy * e[0] > 0)
    inside = (signs[0] == signs[1]) & (signs[1] == signs[2])
    d = np.sqrt(d2)
    return np.where(inside, -d, d)


def incenter(v):
    a, b, c = (np.linalg.norm(v[(i + 1) % 3] - v[(i + 2) % 3]) for i in range(3))
    inc = (a * v[0] + b * v[1] + c * v[2]) / (a + b + c)
    e1, e2 = v[1] - v[0], v[2] - v[0]
    area = abs(e1[0] * e2[1] - e1[1] * e2[0]) / 2
    return inc, area / ((a + b + c) / 2)


TRI_INC, TRI_INR = incenter(TRI_V)


def sd_play(px, py, scale):
    """Rounded play triangle, scaled about its incentre."""
    inner = TRI_INC + (TRI_V - TRI_INC) * scale * (TRI_INR - TRI_RHO) / TRI_INR
    return sd_triangle(px, py, inner) - TRI_RHO * scale


def sd_leaf(px, py, scale, rot):
    """Leaf outline, grown and rotated about its base point."""
    cr, sr = math.cos(-rot), math.sin(-rot)
    dx, dy = px - LEAF_BASE[0], py - LEAF_BASE[1]
    lx = LEAF_BASE[0] + (dx * cr - dy * sr) / scale - LEAF_C[0]
    ly = LEAF_BASE[1] + (dx * sr + dy * cr) / scale - LEAF_C[1]
    n = len(LEAF_R)
    f = np.mod(np.arctan2(ly, lx), 2 * np.pi) / (2 * np.pi) * n
    i0 = np.floor(f).astype(int) % n
    fr = f - np.floor(f)
    rr = LEAF_R[i0] * (1 - fr) + LEAF_R[(i0 + 1) % n] * fr
    return (np.sqrt(lx * lx + ly * ly) - rr) * scale


# ---------- liquid drops ----------
rng = np.random.default_rng(7)
N_DROPS = 11
DROPS = []
for i in range(N_DROPS):
    DROPS.append(dict(
        ang=i / N_DROPS * 2 * math.pi + rng.uniform(-0.25, 0.25),
        dist=rng.uniform(820, 1100),
        swirl=rng.uniform(0.5, 0.9),  # curved, vortex-like approach
        r=rng.uniform(55, 95),
        t0=0.1 + rng.uniform(0, 1.0),
        travel=rng.uniform(2.3, 2.7),
    ))
TOTAL_VOL = sum(d["r"] ** 2 for d in DROPS)


def drop_pos(dr, u):
    e = ease_in(u) * 0.85 + u * 0.15  # slow drift, then drawn into the centre
    dist = dr["dist"] * (1 - e)
    a = dr["ang"] + dr["swirl"] * (1 - e) ** 1.5
    return C[0] + dist * math.cos(a), C[1] + dist * math.sin(a)


def liquid_sdf(wx, wy, t):
    """Distance field of the liquid mass (drops + collected circle), in world px."""
    collected = 0.0
    d = None
    for dr in DROPS:
        u = prog(t, dr["t0"], dr["travel"])
        if u <= 0:
            continue
        cx, cy = drop_pos(dr, u)
        px0, py0 = drop_pos(dr, max(0.0, u - 0.02))
        vx, vy = cx - px0, cy - py0
        # merge: once arrived the drop shrinks into the pool
        absorb = prog(t, dr["t0"] + dr["travel"] - 0.15, 0.4)
        collected += dr["r"] ** 2 * ease_in_out(absorb)
        rr = dr["r"] * (1 - ease_in_out(absorb))
        if rr < 1:
            continue
        # stretch along the direction of travel, more when moving fast
        speed = math.hypot(vx, vy) / (0.02 * dr["dist"])
        stretch = 1 + 0.7 * clamp(speed / 2.5) * (1 - absorb)
        heading = math.atan2(vy, vx)
        di = sd_ellipse_approx(wx, wy, cx, cy, rr * stretch, rr / math.sqrt(stretch), heading)
        d = di if d is None else smin(d, di, 70.0)

    pool_r = R * math.sqrt(collected / TOTAL_VOL)
    if pool_r > 1:
        dx, dy = wx - C[0], wy - C[1]
        rad = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)
        # surface wobble that settles into a perfect circle
        settle = math.exp(-max(0.0, t - T_SETTLE) * 3.6) * clamp(t - 1.0)
        wob = settle * (0.06 * np.sin(3 * theta + 5 * t) + 0.04 * np.sin(5 * theta - 6.5 * t)
                        + 0.025 * np.sin(8 * theta + 9 * t))
        dp = rad - pool_r * (1 + wob)
        d = dp if d is None else smin(d, dp, 90.0)
    return d


def zoom_scale(t):
    if t < T_ZOOM:
        # gentle breathing during the hold, a small anticipation dip before the push
        s = 1 + 0.015 * math.sin(math.pi * prog(t, T_TRI + 0.6, T_ZOOM - T_TRI - 0.6))
        return s * (1 - 0.04 * math.sin(math.pi * prog(t, T_ZOOM - 0.45, 0.45)))
    # fast, accelerating push into the triangle
    return math.exp(math.log(ZOOM_MAX) * prog(t, T_ZOOM, T_WHITE - T_ZOOM) ** 1.8)


def over(img, col, alpha):
    img += (col - img) * alpha[..., None]


def render_frame(t):
    if t >= T_WHITE:
        return np.full((H, W, 3), 255, np.uint8)
    s = zoom_scale(t)
    # screen -> world: zoom about the triangle so it lands on screen centre
    F = TRI_INC
    k = 1 - 1 / s
    wx = SCREEN_O[0] + (XX - SCREEN_O[0]) / s + (F[0] - SCREEN_O[0]) * k
    wy = SCREEN_O[1] + (YY - SCREEN_O[1]) / s + (F[1] - SCREEN_O[1]) * k
    px = 1.0 / s  # one screen pixel in world units, for anti-aliasing

    img = np.empty((H, W, 3), np.float32)
    img[:] = WHITE

    # ripple ring when the triangle pops in
    rp = prog(t, T_TRI, 1.1)
    if 0 < rp < 1:
        rr = R * (1.02 + 0.55 * ease_out(rp))
        ring = np.abs(np.hypot(wx - C[0], wy - C[1]) - rr) - 5 * (1 - rp)
        over(img, ORANGE, np.clip(0.5 - ring / px, 0, 1) * (1 - rp) * 0.6)

    d = liquid_sdf(wx, wy, t)

    # leaf sprouts from its base with a little overshoot and sway
    lp = prog(t, T_LEAF, 0.8)
    leaf_scale = ease_out_back(lp, 1.6)
    if leaf_scale > 0.01:
        sway = 0.25 * (1 - ease_out(lp)) * math.sin(lp * 9)
        dl = sd_leaf(wx, wy, leaf_scale, sway)
        d = dl if d is None else np.minimum(d, dl)

    if d is not None:
        alpha = np.clip(0.5 - d / px, 0, 1)
        # glossy highlight while it is still liquid, fading to flat brand orange
        gloss = 0.5 * (1 - ease_in_out(prog(t, T_SETTLE + 0.3, 0.8)))
        if gloss > 0:
            hx, hy = wx - (C[0] - 0.35 * R), wy - (C[1] - 0.4 * R)
            g = np.exp(-(hx * hx + hy * hy) / (0.55 * R) ** 2) * gloss
            col = ORANGE + (ORANGE_LIGHT - ORANGE) * g[..., None]
        else:
            col = ORANGE
        img += (col - img) * alpha[..., None]

    tp = prog(t, T_TRI, 0.6)
    sc = ease_out_back(tp) if tp > 0 else 0
    if sc > 0.01:
        over(img, WHITE, np.clip(0.5 - sd_play(wx, wy, sc) / px, 0, 1))

    return img.astype(np.uint8)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "orange-videos-logo-reveal.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE,
    )
    n = int(DURATION * FPS)
    for i in range(n):
        ff.stdin.write(render_frame(i / FPS).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"wrote {out} ({n} frames)")


if __name__ == "__main__":
    main()
