"""Orange Videos 3D animation intro clip: 1920x1080, 30 fps, 10 s.

Same timeline, liquid motion and zoom as the 2D intro clip
(render_2d_animation_intro_clip.py), but every shape is lit as a 3D volume:
the liquid and circle become a glossy orange sphere, the leaf a puffy 3D
leaf, and the play triangle a raised white bevel that casts a shadow.
Height maps are derived from the 2D distance fields, then shaded with
diffuse + specular + rim light.
Usage: python3 render_3d_animation_intro_clip.py [output.mp4]
"""
import math
import subprocess
import sys
from multiprocessing import Pool

import numpy as np

import render_2d_animation_intro_clip as base
from render_2d_animation_intro_clip import (C, FPS, H, ORANGE, R, T_LEAF, T_TRI, T_WHITE, T_ZOOM,
                                            DURATION, TRI_INC, TRI_INR, W, WHITE, XX, YY, SCREEN_O,
                                            ease_in_out, ease_out, ease_out_back, prog)

LIGHT = np.array([-0.45, -0.6, 0.66])
LIGHT /= np.linalg.norm(LIGHT)
HALF = LIGHT + [0, 0, 1]
HALF /= np.linalg.norm(HALF)
LEAF_RHO = 45.0  # bevel radius of the leaf, world px
TRI_RHO_3D = 0.45 * TRI_INR  # bevel radius of the play triangle (flat top)


def height(d, rho):
    """Rounded profile: a full hemisphere for a circle of radius rho."""
    delta = np.clip(-d, 0, rho)
    return np.sqrt(np.maximum(rho * rho - (rho - delta) ** 2, 0))


def box_blur(a, r):
    """Separable box blur, applied twice (close to a Gaussian)."""
    if r < 1:
        return a
    for _ in range(2):
        for axis in (0, 1):
            pad = [(0, 0), (0, 0)]
            pad[axis] = (r + 1, r)
            c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
            hi = np.take(c, np.arange(2 * r + 1, c.shape[axis]), axis=axis)
            lo = np.take(c, np.arange(0, c.shape[axis] - 2 * r - 1), axis=axis)
            a = ((hi - lo) / (2 * r + 1)).astype(np.float32)
    return a


def normals(z, s):
    """Surface normals from a height map in world units, sampled per screen pixel."""
    gy, gx = np.gradient(z)
    nx, ny = -gx * s, -gy * s
    inv = 1 / np.sqrt(nx * nx + ny * ny + 1)
    return nx * inv, ny * inv, inv


def shade(albedo, n, ambient, diffuse, spec, shininess, rim):
    nx, ny, nz = n
    ndl = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)
    ndh = np.clip(nx * HALF[0] + ny * HALF[1] + nz * HALF[2], 0, 1)
    col = albedo * (ambient + diffuse * ndl)[..., None]
    col += 255 * (spec * ndh ** shininess)[..., None]
    col += (255 - albedo) * (rim * (1 - nz) ** 3)[..., None]
    return col


def shift(a, dx, dy, fill):
    """Translate a screen-space array by whole pixels (for cast shadows)."""
    out = np.full_like(a, fill)
    dx, dy = int(round(dx)), int(round(dy))
    ys = slice(max(dy, 0), H + min(dy, 0))
    xs = slice(max(dx, 0), W + min(dx, 0))
    yd = slice(max(-dy, 0), H + min(-dy, 0))
    xd = slice(max(-dx, 0), W + min(-dx, 0))
    out[ys, xs] = a[yd, xd]
    return out


def pool_radius(t):
    """Current radius of the collected pool (mirrors base.liquid_sdf)."""
    collected = sum(dr["r"] ** 2 * ease_in_out(prog(t, dr["t0"] + dr["travel"] - 0.15, 0.4))
                    for dr in base.DROPS)
    return R * math.sqrt(collected / base.TOTAL_VOL)


def render_frame(t):
    if t >= T_WHITE:
        return np.full((H, W, 3), 255, np.uint8)
    s = base.zoom_scale(t)
    F = TRI_INC
    k = 1 - 1 / s
    wx = SCREEN_O[0] + (XX - SCREEN_O[0]) / s + (F[0] - SCREEN_O[0]) * k
    wy = SCREEN_O[1] + (YY - SCREEN_O[1]) / s + (F[1] - SCREEN_O[1]) * k
    px = 1.0 / s
    zp = prog(t, T_ZOOM, T_WHITE - T_ZOOM)  # zoom progress
    fx = 1 - ease_in_out(clamp_(zp * 1.6))  # shadows/shading fade while zooming in
    sh_s = min(s, 2.5)

    img = np.empty((H, W, 3), np.float32)
    img[:] = WHITE

    # ripple ring when the triangle pops in
    rp = prog(t, T_TRI, 1.1)
    if 0 < rp < 1:
        rr = R * (1.02 + 0.55 * ease_out(rp))
        ring = np.abs(np.hypot(wx - C[0], wy - C[1]) - rr) - 5 * (1 - rp)
        base.over(img, ORANGE, np.clip(0.5 - ring / px, 0, 1) * (1 - rp) * 0.6)

    d_liq = base.liquid_sdf(wx, wy, t)
    lp = prog(t, T_LEAF, 0.8)
    leaf_scale = ease_out_back(lp, 1.6)
    d_leaf = None
    if leaf_scale > 0.01:
        sway = 0.25 * (1 - ease_out(lp)) * math.sin(lp * 9)
        d_leaf = base.sd_leaf(wx, wy, leaf_scale, sway)

    parts = [d for d in (d_liq, d_leaf) if d is not None]
    if parts:
        d = parts[0] if len(parts) == 1 else np.minimum(*parts)

        # soft shadow cast down-right onto the white background
        ds = shift(d, 18 * sh_s, 34 * sh_s, 1e6) / px
        sh = np.clip((40 - ds) / 110, 0, 1) ** 1.5 * 0.28 * fx
        img *= (1 - sh)[..., None]

        # extra gloss while it is liquid, settling to a satin finish
        wet = 1 - ease_in_out(prog(t, base.T_SETTLE + 0.3, 0.8))
        z = np.zeros((H, W), np.float32)
        if d_liq is not None:
            # bevel follows the pool's current size, so it is always a smooth dome
            z = np.maximum(z, height(d_liq, max(pool_radius(t), 90.0)))
        if d_leaf is not None:
            z = np.maximum(z, height(d_leaf, LEAF_RHO * max(leaf_scale, 0.05)))
        # smooth out distance-field creases so the surface reads as liquid
        n = normals(box_blur(z, 6 + int(6 * wet)), s)
        col = shade(ORANGE, n, ambient=0.58, diffuse=0.55, spec=0.45 + 0.35 * wet,
                    shininess=40 + 40 * wet, rim=0.18)
        col = np.clip(col, 0, 255)
        base.over(img, col, np.clip(0.5 - d / px, 0, 1))

    tp = prog(t, T_TRI, 0.6)
    sc = ease_out_back(tp) if tp > 0 else 0
    if sc > 0.01:
        dt = base.sd_play(wx, wy, sc)
        # the raised triangle shadows the sphere just below-right of it
        dts = shift(dt, 7 * sh_s, 12 * sh_s, 1e6) / px
        tsh = np.clip((6 - dts) / 30, 0, 1) * 0.35 * fx
        img *= (1 - tsh)[..., None]

        zt = box_blur(height(dt, TRI_RHO_3D * sc), 3)
        tcol = shade(np.array([252, 248, 244], np.float32), normals(zt, s),
                     ambient=0.8, diffuse=0.25, spec=0.35, shininess=60, rim=0.0)
        # flatten to pure white as the camera dives in, so the clip ends on white
        flat = ease_in_out(clamp_((zp - 0.35) / 0.45))
        tcol = np.clip(tcol + (WHITE - tcol) * flat, 0, 255)
        base.over(img, tcol, np.clip(0.5 - dt / px, 0, 1))

    return np.clip(img, 0, 255).astype(np.uint8)


def clamp_(x):
    return max(0.0, min(1.0, x))


def frame_bytes(i):
    return render_frame(i / FPS).tobytes()


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "orange-videos-3d-animation-intro-clip.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE,
    )
    n = int(DURATION * FPS)
    with Pool(4) as pool:
        for buf in pool.imap(frame_bytes, range(n), chunksize=2):
            ff.stdin.write(buf)
    ff.stdin.close()
    ff.wait()
    print(f"wrote {out} ({n} frames)")


if __name__ == "__main__":
    main()
