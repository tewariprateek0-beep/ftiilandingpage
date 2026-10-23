"""Orange Videos brand sizzle: 1920x1080, 30 fps, ~22 s motion graphic.

Renders frames with Pillow and pipes them to ffmpeg.
Usage: python3 render_brand_sizzle.py [output.mp4]

The wordmark is a typographic placeholder. Swap in the real logo by setting
LOGO_PATH to a transparent PNG.
"""
import math
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 30
DURATION = 22.0
LOGO_PATH = None  # e.g. "../brand/orange-videos-logo.png"

# Brand palette (placeholder orange until the official HEX is confirmed)
ORANGE = (255, 106, 0)
DARK = (17, 17, 19)
CREAM = (255, 245, 236)
WHITE = (255, 255, 255)
GREY = (150, 150, 155)

FONT_DIR = "/usr/share/fonts/opentype/inter/"
_fonts = {}


def font(weight, size):
    key = (weight, size)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(f"{FONT_DIR}Inter-{weight}.otf", size)
    return _fonts[key]


# ---------- easing / timing helpers ----------
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def prog(t, start, dur):
    return clamp((t - start) / dur)


def ease_out(x):
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out_back(x):
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


# ---------- drawing helpers ----------
def text_size(text, f):
    l, t, r, b = f.getbbox(text)
    asc, desc = f.getmetrics()
    return r, asc + desc


def reveal(img, text, f, x, y, offset, color, anchor="l"):
    """Draw text clipped to its own line box.

    offset 0 = fully visible, 1 = hidden below, -1 = hidden above.
    """
    w, h = text_size(text, f)
    if anchor == "c":
        x -= w // 2
    elif anchor == "r":
        x -= w
    if abs(offset) >= 1:
        return w
    layer = Image.new("RGBA", (w + 8, h), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((0, offset * h), text, font=f, fill=color)
    img.paste(layer, (int(x), int(y)), layer)
    return w


def in_out_offset(t, t_in, t_out, d_in=0.6, d_out=0.45):
    """Offset for a line that rises in at t_in and rises out at t_out."""
    if t < t_out:
        return 1 - ease_out(prog(t, t_in, d_in))
    return -ease_in_out(prog(t, t_out, d_out))


def fade_text(img, text, f, x, y, alpha, color, anchor="l", dy=0):
    if alpha <= 0:
        return
    w, h = text_size(text, f)
    if anchor == "c":
        x -= w // 2
    layer = Image.new("RGBA", (w + 8, h), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((0, 0), text, font=f, fill=color + (int(255 * alpha),))
    img.paste(layer, (int(x), int(y + dy)), layer)


def wordmark(img, cx, cy, t0, t, scale=1.0, on_orange=True):
    """Typographic placeholder logo: ● orange videos, letters staggered in."""
    if LOGO_PATH:
        logo = Image.open(LOGO_PATH).convert("RGBA")
        k = ease_out(prog(t, t0, 0.8))
        lw = int(900 * scale)
        logo = logo.resize((lw, int(logo.height * lw / logo.width)))
        logo.putalpha(logo.getchannel("A").point(lambda a: int(a * k)))
        img.paste(logo, (cx - lw // 2, int(cy - logo.height // 2 + (1 - k) * 40)), logo)
        return
    fb, fl = font("Black", int(150 * scale)), font("Light", int(150 * scale))
    dot_r = int(34 * scale)
    gap = int(30 * scale)
    w1, h = text_size("orange", fb)
    w2, _ = text_size("videos", fl)
    total = dot_r * 2 + gap + w1 + int(28 * scale) + w2
    x = cx - total // 2
    y = cy - h // 2
    d = ImageDraw.Draw(img)
    k = ease_out_back(prog(t, t0, 0.5))
    r = max(0, dot_r * k)
    dot_col = WHITE if on_orange else ORANGE
    d.ellipse([x + dot_r - r, cy - r + 8 * scale, x + dot_r + r, cy + r + 8 * scale], fill=dot_col)
    x += dot_r * 2 + gap
    c1 = WHITE
    c2 = DARK if on_orange else ORANGE
    for i, ch in enumerate("orange"):
        off = 1 - ease_out(prog(t, t0 + 0.15 + i * 0.05, 0.5))
        x += reveal(img, ch, fb, x, y, off, c1)
    x += int(28 * scale)
    for i, ch in enumerate("videos"):
        off = 1 - ease_out(prog(t, t0 + 0.45 + i * 0.05, 0.5))
        x += reveal(img, ch, fl, x, y, off, c2)


# ---------- scenes ----------
def scene_intro(img, t):
    d = ImageDraw.Draw(img)
    img.paste(DARK, [0, 0, W, H])
    r = ease_in_out(prog(t, 0.1, 1.0)) * math.hypot(W, H) / 2 * 1.05
    d.ellipse([W / 2 - r, H / 2 - r, W / 2 + r, H / 2 + r], fill=ORANGE)
    wordmark(img, W // 2, H // 2, 0.9, t)
    fade_text(img, "VIDEO PRODUCTION  ·  MUMBAI, INDIA", font("SemiBold", 30), W // 2, H // 2 + 130,
              ease_out(prog(t, 1.8, 0.6)), DARK, anchor="c")


def scene_tagline(img, t):
    img.paste(DARK, [0, 0, W, H])
    d = ImageDraw.Draw(img)
    f = font("Black", 130)
    x, y = 180, 360
    reveal(img, "Videos That Tell", f, x, y, in_out_offset(t, 0.15, 3.0), WHITE)
    reveal(img, "Powerful Brand Stories.", f, x, y + 165, in_out_offset(t, 0.35, 3.05), ORANGE)
    bar = ease_out(prog(t, 0.7, 0.8)) * (1 - ease_in_out(prog(t, 3.0, 0.4)))
    d.rectangle([x, y - 50, x + 220 * bar, y - 38], fill=ORANGE)


def scene_stats(img, t):
    img.paste(CREAM, [0, 0, W, H])
    d = ImageDraw.Draw(img)
    stats = [(13, "+", "Years of\nstorytelling"), (1000, "+", "Projects\ndelivered"), (600, "+", "Happy\nclients")]
    fn, fl = font("Black", 150), font("Medium", 40)
    col_w = W // 3
    for i, (val, suf, label) in enumerate(stats):
        st = 0.2 + i * 0.25
        k = ease_out(prog(t, st, 1.4))
        cx = col_w * i + col_w // 2
        out = -ease_in_out(prog(t, 3.4, 0.45))
        reveal(img, f"{int(round(val * k)):,}{suf}", fn, cx, 330, (1 - ease_out(prog(t, st, 0.5))) if out == 0 else out,
               ORANGE, anchor="c")
        for j, line in enumerate(label.split("\n")):
            fade_text(img, line, fl, cx, 560 + j * 52, ease_out(prog(t, st + 0.4, 0.5)) * (1 + out), DARK, anchor="c")
        if i:
            h = ease_out(prog(t, 0.4, 0.8)) * (1 + out)
            d.rectangle([col_w * i - 1, 540 - 160 * h, col_w * i + 1, 540 + 160 * h], fill=(220, 205, 190))


SERVICES = ["Brand Films", "Ad Films", "Testimonials", "Documentaries", "Product Videos",
            "2D & 3D Animation", "AI Videos", "Video Podcasts", "Drone Films"]


def scene_services(img, t):
    img.paste(DARK, [0, 0, W, H])
    d = ImageDraw.Draw(img)
    f_small, f_big = font("Medium", 54), font("Black", 140)
    x, y = 180, 300
    reveal(img, "We make", f_small, x, y, in_out_offset(t, 0.1, 4.7), GREY)
    step = 0.5
    for i, word in enumerate(SERVICES):
        t_in = 0.3 + i * step
        last = i == len(SERVICES) - 1
        t_out = 4.7 if last else t_in + step
        if t_in - 0.1 <= t <= t_out + 0.5:
            off = in_out_offset(t, t_in, t_out, d_in=0.35, d_out=0.3)
            reveal(img, word, f_big, x, y + 90, off, ORANGE)
    # progress ticks
    for i in range(len(SERVICES)):
        active = t >= 0.3 + i * step
        col = ORANGE if active else (60, 60, 64)
        d.rectangle([x + i * 70, 760, x + i * 70 + 54, 766], fill=col)
    current = min(len(SERVICES), max(1, int((t - 0.3) / step) + 1))
    fade_text(img, f"{current:02d} / {len(SERVICES):02d}", font("Medium", 30), x, 790, 1, GREY)


CLIENTS = ["Tata AIG", "Tata Communications", "Razorpay", "Schindler",
           "Hapag-Lloyd", "Medtronic", "TotalEnergies", "SLB",
           "IDEX", "Nilkamal", "CoinDCX", "EatSure"]


def scene_clients(img, t):
    img.paste(ORANGE, [0, 0, W, H])
    reveal(img, "Trusted by brands like", font("SemiBold", 50), W // 2, 260, in_out_offset(t, 0.1, 3.0), DARK,
           anchor="c")
    f = font("Bold", 44)
    cols, cell_w, cell_h = 4, 440, 150
    x0, y0 = (W - cols * cell_w) // 2, 440
    d = ImageDraw.Draw(img)
    for i, name in enumerate(CLIENTS):
        r, c = divmod(i, cols)
        st = 0.3 + (r + c) * 0.08
        a = ease_out(prog(t, st, 0.5)) * (1 - ease_in_out(prog(t, 3.0, 0.4)))
        cx = x0 + c * cell_w + cell_w // 2
        cy = y0 + r * cell_h
        fade_text(img, name, f, cx, cy, a, WHITE, anchor="c", dy=(1 - a) * 30)
        if a > 0:
            lw = int(60 * a)
            d.rectangle([cx - lw // 2, cy + 82, cx + lw // 2, cy + 85], fill=DARK)


def scene_end(img, t):
    img.paste(DARK, [0, 0, W, H])
    wordmark(img, W // 2, H // 2 - 40, 0.15, t, scale=0.85, on_orange=False)
    fade_text(img, "Videos That Tell Powerful Brand Stories", font("Medium", 40), W // 2, H // 2 + 90,
              ease_out(prog(t, 1.0, 0.6)), GREY, anchor="c")
    fade_text(img, "orangevideos.in", font("Bold", 46), W // 2, H // 2 + 170,
              ease_out(prog(t, 1.3, 0.6)), ORANGE, anchor="c")


# (start, end, scene, sweep colour used for the transition into the next scene)
TIMELINE = [
    (0.0, 3.2, scene_intro, DARK),
    (3.2, 6.8, scene_tagline, ORANGE),
    (6.8, 10.8, scene_stats, ORANGE),
    (10.8, 16.2, scene_services, DARK),
    (16.2, 19.6, scene_clients, DARK),
    (19.6, DURATION, scene_end, None),
]
SWEEP = 0.55  # seconds for a full wipe-on / wipe-off


def render_frame(t):
    img = Image.new("RGB", (W, H), DARK)
    for start, end, fn, _ in TIMELINE:
        if start <= t < end or (fn is TIMELINE[-1][2] and t >= start):
            fn(img, t - start)
            break
    # transitions: a colour bar sweeps across each cut
    for start, end, _, col in TIMELINE:
        if col is None:
            continue
        p = (t - (end - SWEEP / 2)) / SWEEP
        if 0 <= p <= 1:
            d = ImageDraw.Draw(img)
            if p < 0.5:
                x1 = ease_in_out(p * 2) * W
                d.rectangle([0, 0, x1, H], fill=col)
            else:
                x0 = ease_in_out((p - 0.5) * 2) * W
                d.rectangle([x0, 0, W, H], fill=col)
    # final fade to black
    fo = prog(t, DURATION - 0.5, 0.5)
    if fo > 0:
        img = Image.blend(img, Image.new("RGB", (W, H), (0, 0, 0)), fo)
    return img


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "orange-videos-brand-sizzle.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
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
