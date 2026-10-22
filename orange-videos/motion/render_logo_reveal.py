"""Orange Videos logo reveal: 1920x1080, 30 fps, 10 s intro/outro.

Liquid drops flow in and collect into the orange circle, the white play
triangle pops in, the mark holds, then the camera zooms into the triangle
until the whole frame is white.

Everything is drawn with signed distance fields in numpy, so edges stay
crisp at any zoom. Usage: python3 render_logo_reveal.py [output.mp4]
"""
import math
import subprocess
import sys

import numpy as np

W, H, FPS = 1920, 1080, 30
DURATION = 10.0

ORANGE = np.array([255, 106, 0], np.float32)  # placeholder until the official HEX is confirmed
ORANGE_LIGHT = np.array([255, 170, 90], np.float32)
WHITE = np.array([255, 255, 255], np.float32)

C = np.array([W / 2, H / 2], np.float32)  # logo centre on screen
R = 240.0  # final circle radius
TRI_R = 0.36 * R  # half side length of the play triangle
TRI_DX = 0.07 * R  # optical offset to the right

# timeline (seconds)
T_TRI = 4.0  # play triangle pops in
T_ZOOM = 7.0  # zoom starts
T_WHITE = 9.2  # frame fully white
ZOOM_MAX = 80.0

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
XX += 0.5
YY += 0.5


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


def sd_play_triangle(px, py, cx, cy, r, round_r):
    """Equilateral triangle pointing right, centroid at (cx, cy), rounded corners."""
    r = r - round_r * math.sqrt(3)
    k = math.sqrt(3.0)
    # iq's sdEquilateralTriangle points 'up'; map its up axis to screen +x
    x = np.abs(py - cy) - r
    y = (px - cx) + r / k  # apex points right
    m = x + k * y > 0
    x2 = np.where(m, (x - k * y) / 2, x)
    y2 = np.where(m, (-k * x - y) / 2, y)
    x2 = x2 - np.clip(x2, -2 * r, 0.0)
    return -np.sqrt(x2 * x2 + y2 * y2) * np.sign(y2) - round_r


# ---------- liquid drops ----------
rng = np.random.default_rng(7)
N_DROPS = 11
DROPS = []
for i in range(N_DROPS):
    ang = i / N_DROPS * 2 * math.pi + rng.uniform(-0.25, 0.25)
    DROPS.append(dict(
        ang=ang,
        dist=rng.uniform(900, 1250),
        swirl=rng.uniform(0.5, 0.9),  # curved, vortex-like approach
        r=rng.uniform(55, 95),
        t0=0.15 + rng.uniform(0, 0.9),
        travel=rng.uniform(1.5, 2.1),
    ))
TOTAL_VOL = sum(d["r"] ** 2 for d in DROPS)


def drop_pos(dr, u):
    e = ease_in(u) * 0.85 + u * 0.15  # accelerate into the centre
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
        # merge: once arrived the drop shrinks into the pool over 0.35 s
        absorb = prog(t, dr["t0"] + dr["travel"] - 0.15, 0.35)
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
        settle = math.exp(-max(0.0, t - 2.6) * 2.4) * clamp(t - 0.8)
        wob = settle * (0.06 * np.sin(3 * theta + 6 * t) + 0.04 * np.sin(5 * theta - 8 * t)
                        + 0.025 * np.sin(8 * theta + 11 * t))
        dp = rad - pool_r * (1 + wob)
        d = dp if d is None else smin(d, dp, 90.0)
    return d, pool_r


def render_frame(t):
    # breathing during the hold, a small anticipation dip, then the zoom
    if t < T_ZOOM:
        s = 1 + 0.015 * math.sin(math.pi * prog(t, 4.6, 2.0))
        s *= 1 - 0.035 * math.sin(math.pi * prog(t, T_ZOOM - 0.5, 0.5)) ** 2 * (t > T_ZOOM - 0.25)
    else:
        s = math.exp(math.log(ZOOM_MAX) * prog(t, T_ZOOM, T_WHITE - T_ZOOM) ** 2)
    if t >= T_WHITE:
        return np.full((H, W, 3), 255, np.uint8)

    # screen -> world: zoom about the triangle centroid so it lands on screen centre
    F = C + np.array([TRI_DX, 0], np.float32)
    k = 1 - 1 / s
    wx = C[0] + (XX - C[0]) / s + (F[0] - C[0]) * k
    wy = C[1] + (YY - C[1]) / s + (F[1] - C[1]) * k
    px = 1.0 / s  # one screen pixel in world units, for anti-aliasing

    img = np.empty((H, W, 3), np.float32)
    img[:] = WHITE

    # ripple ring when the triangle pops in
    rp = prog(t, T_TRI, 1.1)
    if 0 < rp < 1:
        rr = R * (1.02 + 0.55 * ease_out(rp))
        ring = np.abs(np.hypot(wx - C[0], wy - C[1]) - rr) - 5 * (1 - rp)
        a = np.clip(0.5 - ring / px, 0, 1) * (1 - rp) * 0.6
        img += (ORANGE - img) * a[..., None]

    d, _ = liquid_sdf(wx, wy, t)
    if d is not None:
        alpha = np.clip(0.5 - d / px, 0, 1)
        # glossy highlight while it is still liquid, fading to flat brand orange
        gloss = 0.45 * (1 - ease_in_out(prog(t, 3.4, 0.8)))
        if gloss > 0:
            hx, hy = wx - (C[0] - 0.35 * R), wy - (C[1] - 0.4 * R)
            g = np.exp(-(hx * hx + hy * hy) / (0.55 * R) ** 2) * gloss
            col = ORANGE + (ORANGE_LIGHT - ORANGE) * g[..., None]
        else:
            col = ORANGE
        img += (col - img) * alpha[..., None]

    tp = prog(t, T_TRI, 0.6)
    if tp > 0:
        sc = max(0.0, ease_out_back(tp))
        if sc > 0.01:
            dt = sd_play_triangle(wx, wy, F[0], F[1], TRI_R * sc, 0.05 * R * sc)
            a = np.clip(0.5 - dt / px, 0, 1)
            img += (WHITE - img) * a[..., None]

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
