"""Orange Videos 3D intro clip with a globe spin: 1920x1080, 30 fps, 10 s.

Same as render_3d_animation_intro_clip.py, but before the zoom the sphere
turns a full 360 degrees about its vertical axis: the play triangle (a
raised decal on the sphere's surface) slides round the side, passes behind
and returns to the front, the leaf turns with it, and the camera then dives
into the triangle to full white.
Usage: python3 render_3d_spin_intro_clip.py [output.mp4]
"""
import math
import subprocess
import sys
from multiprocessing import Pool

import numpy as np

import render_2d_animation_intro_clip as base
from render_2d_animation_intro_clip import (C, FPS, H, LEAF_C, ORANGE, R, T_LEAF, T_TRI, T_WHITE,
                                            T_ZOOM, DURATION, TRI_INC, W, WHITE, XX, YY, SCREEN_O,
                                            ease_in_out, ease_out, ease_out_back, prog)
from render_3d_animation_intro_clip import (LEAF_RHO, TRI_RHO_3D, box_blur, clamp_, height, normals,
                                            pool_radius, shade, shift)

T_SPIN = 6.1  # sphere starts turning
SPIN_DUR = 1.85  # one full turn, landing just as the zoom picks up speed
TRI_ALBEDO = np.array([252, 248, 244], np.float32)


def spin_angle(t):
    return 2 * math.pi * ease_in_out(prog(t, T_SPIN, SPIN_DUR))


def aa_alpha(d):
    """Anti-aliased coverage of d<0, using d's own screen-space gradient."""
    gy, gx = np.gradient(d)
    g = np.maximum(np.hypot(gx, gy), 1e-4)
    return np.clip(0.5 - d / g, 0, 1)


def render_frame(t):
    if t >= T_WHITE:
        return np.full((H, W, 3), 255, np.uint8)
    s = base.zoom_scale(t)
    F = TRI_INC
    k = 1 - 1 / s
    wx = SCREEN_O[0] + (XX - SCREEN_O[0]) / s + (F[0] - SCREEN_O[0]) * k
    wy = SCREEN_O[1] + (YY - SCREEN_O[1]) / s + (F[1] - SCREEN_O[1]) * k
    px = 1.0 / s
    zp = prog(t, T_ZOOM, T_WHITE - T_ZOOM)
    fx = 1 - ease_in_out(clamp_(zp * 1.6))
    sh_s = min(s, 2.5)
    th = spin_angle(t)
    ct, st = math.cos(th), math.sin(th)

    img = np.empty((H, W, 3), np.float32)
    img[:] = WHITE

    rp = prog(t, T_TRI, 1.1)
    if 0 < rp < 1:
        rr = R * (1.02 + 0.55 * ease_out(rp))
        ring = np.abs(np.hypot(wx - C[0], wy - C[1]) - rr) - 5 * (1 - rp)
        base.over(img, ORANGE, np.clip(0.5 - ring / px, 0, 1) * (1 - rp) * 0.6)

    d_liq = base.liquid_sdf(wx, wy, t)

    # leaf: a thin object on top of the sphere, turning about the same vertical axis
    lp = prog(t, T_LEAF, 0.8)
    leaf_scale = ease_out_back(lp, 1.6)
    d_leaf = None
    if leaf_scale > 0.01 and abs(ct) > 0.04:
        sway = 0.25 * (1 - ease_out(lp)) * math.sin(lp * 9)
        lwx = C[0] + (wx - C[0]) / ct  # undo the horizontal foreshortening (mirrors when facing away)
        d_leaf = base.sd_leaf(lwx, wy, leaf_scale, sway)
    leaf_behind = -(LEAF_C[0] - C[0]) * st < 0

    def draw_leaf():
        z = height(d_leaf, LEAF_RHO * max(leaf_scale, 0.05))
        col = shade(ORANGE * (0.85 if ct < 0 else 1.0), normals(box_blur(z, 2), s),
                    ambient=0.58, diffuse=0.55, spec=0.45, shininess=40, rim=0.18)
        base.over(img, np.clip(col, 0, 255), aa_alpha(d_leaf))

    parts = [d for d in (d_liq, d_leaf) if d is not None]
    if parts:
        d_all = parts[0] if len(parts) == 1 else np.minimum(*parts)
        ds = shift(d_all, 18 * sh_s, 34 * sh_s, 1e6) / px
        img *= (1 - np.clip((40 - ds) / 110, 0, 1) ** 1.5 * 0.28 * fx)[..., None]

    if d_leaf is not None and leaf_behind:
        draw_leaf()

    if d_liq is not None:
        wet = 1 - ease_in_out(prog(t, base.T_SETTLE + 0.3, 0.8))
        z = height(d_liq, max(pool_radius(t), 90.0))
        n_body = normals(box_blur(z, 6 + int(6 * wet)), s)
        col = shade(ORANGE, n_body, ambient=0.58, diffuse=0.55, spec=0.45 + 0.35 * wet,
                    shininess=40 + 40 * wet, rim=0.18)
        base.over(img, np.clip(col, 0, 255), np.clip(0.5 - d_liq / px, 0, 1))

    if d_leaf is not None and not leaf_behind:
        draw_leaf()

    tp = prog(t, T_TRI, 0.6)
    sc = ease_out_back(tp) if tp > 0 else 0
    if sc > 0.01:
        # ray-cast the sphere, rotate the hit point back into the decal's frame
        x, y = wx - C[0], wy - C[1]
        zz = np.sqrt(np.maximum(R * R - x * x - y * y, 0))
        on_sphere = x * x + y * y < R * R
        u = x * ct - zz * st
        wdep = x * st + zz * ct  # > 0 on the decal's (front) hemisphere
        visible = on_sphere & (wdep > 0)
        dt_raw = base.sd_play(u + C[0], wy, sc).astype(np.float32)
        dt = np.where(visible, dt_raw, 1e6)

        # triangle shadow on the sphere
        dts = shift(dt, 7 * sh_s, 12 * sh_s, 1e6) / px
        img *= (1 - np.clip((6 - dts) / 30, 0, 1) * 0.35 * fx * on_sphere)[..., None]

        # bevel: sphere normal tilted by the slope of the raised edge
        rho = TRI_RHO_3D * sc
        delta = np.clip(-dt, 0, rho)
        hgt = np.sqrt(np.maximum(rho * rho - (rho - delta) ** 2, 1e-3))
        slope = np.where((dt < 0) & (delta < rho), np.minimum((rho - delta) / hgt, 4.0), 0.0)
        eps = 0.5
        du = (base.sd_play(u + eps + C[0], wy, sc) - base.sd_play(u - eps + C[0], wy, sc)) / (2 * eps)
        dv = (base.sd_play(u + C[0], wy + eps, sc) - base.sd_play(u + C[0], wy - eps, sc)) / (2 * eps)
        tx, ty = slope * du, slope * dv  # outward tilt in the decal plane
        nx = x / R + tx * ct
        ny = y / R + ty
        nz = zz / R - tx * st
        inv = 1 / np.maximum(np.sqrt(nx * nx + ny * ny + nz * nz), 1e-6)
        tcol = shade(TRI_ALBEDO, (nx * inv, ny * inv, nz * inv),
                     ambient=0.8, diffuse=0.25, spec=0.35, shininess=60, rim=0.0)
        flat = ease_in_out(clamp_((zp - 0.35) / 0.45))
        tcol = np.clip(tcol + (WHITE - tcol) * flat, 0, 255)
        base.over(img, tcol, aa_alpha(dt_raw) * visible)

    return np.clip(img, 0, 255).astype(np.uint8)


def frame_bytes(i):
    return render_frame(i / FPS).tobytes()


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "orange-videos-3d-spin-intro-clip.mp4"
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
