"""
Offcut animated scene graphics.

Each scene of a short gets a "visual". The script writer can set one explicitly:

  "visual": {"type": "title",     "text": "...", "icon": "💰", "highlight": "terms"}
  "visual": {"type": "statement", "text": "...", "icon": "⏰", "highlight": "late"}
  "visual": {"type": "list",      "title": "...", "items": ["...", "...", "..."]}
  "visual": {"type": "stat",      "value": "30%", "label": "...", "icon": "📈"}
  "visual": {"type": "compare",   "left_label": "DON'T", "left": "...",
                                  "right_label": "DO", "right": "..."}
  "visual": {"type": "footage",   "search": "plumber at work"}

If a scene has no "visual", plan_visuals() works one out from the scene text.
Animated scenes are drawn with Pillow and piped straight into FFmpeg.
"""

import math
import re
import subprocess
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30

# ---------------------------------------------------------------- brand

BG_TOP = (18, 21, 27)
BG_BOTTOM = (30, 34, 43)
CARD = (37, 42, 52)
CARD_EDGE = (58, 64, 76)
ORANGE = (255, 122, 26)
WHITE = (245, 246, 248)
MUTED = (165, 172, 184)
GREEN = (52, 199, 120)
RED = (235, 84, 72)

# graphics live between these lines; captions sit below, Shorts UI below that
TOP, BOTTOM = 230, 1290

HERE = Path(__file__).parent
EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
BOLD_FONTS = [HERE / "fonts/Montserrat-ExtraBold.ttf",
              Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")]
SEMI_FONTS = [HERE / "fonts/Montserrat-SemiBold.ttf",
              Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")]


def _font_ok(p) -> bool:
    try:
        return Path(p).exists() and Path(p).stat().st_size > 10000
    except OSError:
        return False


def _first_existing(paths):
    for p in paths:
        if _font_ok(p):
            return str(p)
    raise FileNotFoundError(f"none of these fonts exist: {paths}")


@lru_cache(maxsize=None)
def bold(size: int):
    return ImageFont.truetype(_first_existing(BOLD_FONTS), size)


@lru_cache(maxsize=None)
def semi(size: int):
    return ImageFont.truetype(_first_existing(SEMI_FONTS), size)


def caption_font_name() -> str:
    """Font family name for the burned-in captions (matches the graphics)."""
    return "Montserrat ExtraBold" if _font_ok(BOLD_FONTS[0]) else "DejaVu Sans"


# ---------------------------------------------------------------- icons

ICON_WORDS = [
    (r"cash ?flow|cashflow", "💸"), (r"cash|money|paid|pay\b|payment|deposit|price|pric|charge|\$", "💰"),
    (r"invoice|bill|receipt", "🧾"), (r"late|overdue|time|deadline|due|wait", "⏰"),
    (r"contract|writing|written|terms|agree|sign", "📝"), (r"client|customer|homeowner", "🤝"),
    (r"profit|margin|grow|increase|more", "📈"), (r"loss|lose|drop|cost|expens", "📉"),
    (r"tax|gst|vat|irs|ird", "🏛️"), (r"bank|loan|debt|interest", "🏦"),
    (r"quote|estimate|calculat|number|math", "🧮"), (r"tool|plumb|tile|tiling|build|job|site|trade", "🛠️"),
    (r"house|home|bathroom|kitchen|renovat|refit", "🏠"), (r"phone|call|text|email|message", "📱"),
    (r"calendar|week|month|schedule|book", "📅"), (r"never|stop|don't|avoid|mistake|wrong", "🚫"),
    (r"automat|ai\b|app|software|system", "🤖"), (r"hire|staff|team|apprentice|worker", "👷"),
    (r"protect|safe|insur|secure", "🛡️"), (r"idea|tip|secret|trick", "💡"), (r"goal|target|aim", "🎯"),
]
DEFAULT_ICON = "💡"


def pick_icon(text: str) -> str:
    t = text.lower()
    for pattern, icon in ICON_WORDS:
        if re.search(pattern, t):
            return icon
    return DEFAULT_ICON


@lru_cache(maxsize=None)
def emoji_image(char: str, size: int):
    try:
        f = ImageFont.truetype(EMOJI_FONT, 109)
        im = Image.new("RGBA", (180, 160), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((10, 10), char, font=f, embedded_color=True)
        box = im.getbbox()
        if not box:
            return None
        im = im.crop(box)
        k = size / max(im.size)
        return im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
    except Exception:
        return None


# ---------------------------------------------------------------- easing / helpers

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out_cubic(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_out_back(x):
    x = clamp(x)
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def progress(t, start, length):
    return clamp((t - start) / length) if length > 0 else (1.0 if t >= start else 0.0)


def with_alpha(img: Image.Image, a: float) -> Image.Image:
    if a >= 0.999:
        return img
    out = img.copy()
    alpha = out.getchannel("A").point(lambda v: int(v * a))
    out.putalpha(alpha)
    return out


def paste_center(frame: Image.Image, img: Image.Image, cx: float, cy: float, alpha=1.0, scale=1.0):
    if img is None or alpha <= 0.01 or scale <= 0.01:
        return
    if abs(scale - 1) > 0.01:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BILINEAR)
    img = with_alpha(img, alpha)
    x, y = int(cx - img.width / 2), int(cy - img.height / 2)
    # clip to frame
    sx, sy = max(0, -x), max(0, -y)
    ex, ey = min(img.width, W - x), min(img.height, H - y)
    if ex <= sx or ey <= sy:
        return
    if sx or sy or ex < img.width or ey < img.height:
        img = img.crop((sx, sy, ex, ey))
        x, y = x + sx, y + sy
    frame.alpha_composite(img, dest=(x, y))


def norm_words(s: str):
    return re.findall(r"[a-z0-9$%']+", s.lower())


def word_times(full_text: str, words: list, speak: float):
    """Estimate when each of the given words is spoken inside the full scene text."""
    lower = full_text.lower()
    n = max(1, len(lower))
    pos, times = 0, []
    for w in words:
        key = re.sub(r"[^a-z0-9$%']", "", w.lower())
        i = lower.find(key, pos) if key else -1
        if i < 0:
            times.append(times[-1] + 0.15 if times else 0.0)
            continue
        times.append(speak * i / n)
        pos = i + len(key)
    return times


def phrase_time(full_text: str, phrase: str, speak: float) -> float:
    ws = phrase.split()
    if not ws:
        return 0.0
    return word_times(full_text, ws[:1], speak)[0]


# ---------------------------------------------------------------- text layout

def layout_words(text, font_fn, max_w, max_lines, size_max, size_min, step=4, line_gap=1.15):
    """Greedy wrap, shrinking the font until it fits. Returns (font, lines[[(word, x)]], line_h, block_w)."""
    words = text.split()
    size = size_max
    while True:
        f = font_fn(size)
        space = f.getlength(" ")
        lines, cur, cur_w = [], [], 0.0
        for w in words:
            ww = f.getlength(w)
            add = ww if not cur else cur_w + space + ww
            if cur and add > max_w:
                lines.append(cur)
                cur, cur_w = [w], ww
            else:
                cur.append(w)
                cur_w = add
        if cur:
            lines.append(cur)
        widest = max((f.getlength(" ".join(l)) for l in lines), default=0)
        if (len(lines) <= max_lines and widest <= max_w) or size <= size_min:
            break
        size -= step
    asc, desc = f.getmetrics()
    line_h = int((asc + desc) * line_gap)
    placed = []
    for l in lines:
        lw = f.getlength(" ".join(l))
        x = -lw / 2
        row = []
        for w in l:
            row.append((w, x))
            x += f.getlength(w) + space
        placed.append(row)
    return f, placed, line_h, widest


def is_highlight(word: str, highlights) -> bool:
    k = re.sub(r"[^a-z0-9$%']", "", word.lower())
    return any(k == h or (len(h) > 3 and k.startswith(h)) for h in highlights)


def draw_words(frame, text, cx, cy, t, times, font_fn, max_w, max_lines, size_max, size_min,
               highlights=(), color=WHITE, accent=ORANGE, rise=36, fade=0.22, upper=True):
    """Words fade/slide in one by one at the given times. Returns the block's bottom y."""
    if upper:
        text = text.upper()
    f, lines, lh, bw = layout_words(text, font_fn, max_w, max_lines, size_max, size_min)
    total_h = lh * len(lines)
    top = cy - total_h / 2
    layer = Image.new("RGBA", (W, int(total_h + rise + 40)), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    i = 0
    for li, row in enumerate(lines):
        for w, x in row:
            st = times[i] if i < len(times) else 0
            p = progress(t, st, fade)
            if p > 0:
                a = int(255 * ease_out_cubic(p))
                dy = rise * (1 - ease_out_cubic(p))
                col = accent if is_highlight(w, highlights) else color
                d.text((W / 2 + x, li * lh + dy + 10), w, font=f, fill=col + (a,),
                       stroke_width=0)
            i += 1
    frame.alpha_composite(layer, dest=(0, int(top - 10)))
    return top + total_h


def rounded_card(w, h, fill=CARD, edge=CARD_EDGE, radius=34, accent_bar=None):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=fill + (255,), outline=edge + (255,), width=3)
    if accent_bar:
        d.rounded_rectangle((0, 0, 14, h - 1), radius=7, fill=accent_bar + (255,))
    return im


# ---------------------------------------------------------------- background

@lru_cache(maxsize=1)
def _bg_canvas():
    pad = 120
    cw, ch = W + pad, H + pad
    im = Image.new("RGBA", (cw, ch))
    d = ImageDraw.Draw(im)
    for y in range(ch):
        k = y / ch
        c = tuple(int(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * k) for i in range(3))
        d.line((0, y, cw, y), fill=c + (255,))
    grid = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    g = ImageDraw.Draw(grid)
    for x in range(0, cw, 60):
        g.line((x, 0, x, ch), fill=(255, 255, 255, 9), width=1)
    for y in range(0, ch, 60):
        g.line((0, y, cw, y), fill=(255, 255, 255, 9), width=1)
    for x in range(0, cw, 240):
        g.line((x, 0, x, ch), fill=(255, 255, 255, 16), width=2)
    for y in range(0, ch, 240):
        g.line((0, y, cw, y), fill=(255, 255, 255, 16), width=2)
    im.alpha_composite(grid)
    # soft orange glow near the top
    glow = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for r in range(520, 0, -20):
        a = int(26 * (1 - r / 520))
        gd.ellipse((cw / 2 - r * 1.6, 380 - r, cw / 2 + r * 1.6, 380 + r), fill=ORANGE + (a,))
    im.alpha_composite(glow)
    return im


def background(t: float) -> Image.Image:
    off = int((t * 18) % 240)  # slow drifting blueprint grid; 240 keeps major lines seamless
    return _bg_canvas().crop((off // 2, off // 2, off // 2 + W, off // 2 + H)).copy()


# ---------------------------------------------------------------- scene types

def scene_title(v, sc):
    text, speak = v["text"], sc["duration"]
    icon = emoji_image(v.get("icon") or pick_icon(text), 250)
    hl = [h.lower() for h in v.get("highlight", [])] if isinstance(v.get("highlight"), list) \
        else [v["highlight"].lower()] if v.get("highlight") else auto_highlight(text)
    words = text.upper().split()
    times = [max(0.05, x) for x in word_times(sc["text"], words, speak * 0.95)]

    _, _lines, _lh, _ = layout_words(text.upper(), bold, 940, 4, 120, 64)
    block_h = _lh * len(_lines)
    total = 260 + block_h
    icon_y = TOP + max(0, ((BOTTOM - TOP) - total) / 2) + 125
    text_cy = icon_y + 150 + block_h / 2

    def frame(t):
        fr = background(t)
        p = progress(t, 0, 0.45)
        paste_center(fr, icon, W / 2, icon_y + 6 * math.sin(t * 2.2), alpha=clamp(p * 2), scale=0.4 + 0.6 * ease_out_back(p))
        bottom = draw_words(fr, text, W / 2, text_cy, t, times, bold, 940, 4, 120, 64, hl, rise=50)
        # orange underline grows after the last word
        q = ease_out_cubic(progress(t, (times[-1] if times else 0) + 0.15, 0.4))
        if q > 0:
            d = ImageDraw.Draw(fr)
            half = 230 * q
            d.rounded_rectangle((W / 2 - half, bottom + 34, W / 2 + half, bottom + 50), radius=8, fill=ORANGE)
        return fr
    return frame


def scene_statement(v, sc):
    text, speak = v["text"], sc["duration"]
    icon = emoji_image(v.get("icon") or pick_icon(sc["text"]), 190)
    hl = [v["highlight"].lower()] if v.get("highlight") else auto_highlight(text)
    words = text.upper().split()
    times = [max(0.25, x) for x in word_times(sc["text"], words, speak * 0.9)]
    _, _lines, _lh, _ = layout_words(text.upper(), bold, 820, 4, 92, 54)
    block_h = _lh * len(_lines)
    card_h = int(70 + 190 + 40 + block_h + 70)
    card = rounded_card(960, card_h, accent_bar=ORANGE)
    card_cy = (TOP + BOTTOM) / 2
    card_top = card_cy - card_h / 2
    icon_y = card_top + 70 + 95
    text_cy = icon_y + 95 + 40 + block_h / 2

    def frame(t):
        fr = background(t)
        p = ease_out_cubic(progress(t, 0, 0.4))
        paste_center(fr, card, W / 2, card_cy + 90 * (1 - p), alpha=p)
        q = progress(t, 0.12, 0.45)
        paste_center(fr, icon, W / 2, icon_y + 5 * math.sin(t * 2.4), alpha=clamp(q * 2), scale=0.3 + 0.7 * ease_out_back(q))
        draw_words(fr, text, W / 2 + 6, text_cy, t, times, bold, 820, 4, 92, 54, hl, rise=34)
        return fr
    return frame


def _check_badge(size=74, color=ORANGE, mark="check"):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((0, 0, size - 1, size - 1), fill=color + (255,))
    s = size
    if mark == "check":
        d.line([(s * .27, s * .52), (s * .44, s * .68), (s * .74, s * .34)], fill=(20, 20, 24, 255), width=int(s * .12), joint="curve")
    else:
        d.text((s / 2, s / 2), mark, font=bold(int(s * .55)), fill=(20, 20, 24, 255), anchor="mm")
    return im


def scene_list(v, sc):
    title = v.get("title") or ""
    items = [i for i in v["items"] if i.strip()][:5]
    speak = sc["duration"]
    n = len(items)
    card_h = {2: 190, 3: 170, 4: 150}.get(n, 132)
    gap = 26
    numbered = v.get("numbered", False)
    title_h = 0
    title_layout = None
    if title:
        f, lines, lh, _ = layout_words(title.upper(), bold, 920, 3, 72, 48)
        title_h = lh * len(lines) + 40
    block_h = title_h + n * card_h + (n - 1) * gap
    top = TOP + max(0, ((BOTTOM - TOP) - block_h) / 2)
    title_cy = top + (title_h - 40) / 2
    title_words = title.upper().split()
    title_times = [min(0.1 + i * 0.07, 0.6) for i in range(len(title_words))]
    item_times = []
    for k, it in enumerate(items):
        st = phrase_time(sc["text"], it, speak)
        st = max(st - 0.15, 0.35 + 0.25 * k)
        item_times.append(st)
    cards = []
    for k, it in enumerate(items):
        c = rounded_card(940, card_h)
        d = ImageDraw.Draw(c)
        badge = _check_badge(76, mark=str(k + 1) if numbered else "check")
        c.alpha_composite(badge, dest=(36, int(card_h / 2 - 38)))
        f, lines, lh, _ = layout_words(it.upper(), bold, 760, 2, 64, 40)
        ty = card_h / 2 - lh * len(lines) / 2
        for li, row in enumerate(lines):
            x0 = 140
            d.text((x0, ty + li * lh), " ".join(w for w, _ in row), font=f, fill=WHITE)
        cards.append(c)

    def frame(t):
        fr = background(t)
        if title:
            draw_words(fr, title, W / 2, title_cy, t, title_times, bold, 920, 3, 72, 48, auto_highlight(title), color=MUTED)
        y = top + title_h
        for k, c in enumerate(cards):
            p = progress(t, item_times[k], 0.35)
            if p > 0:
                e = ease_out_back(p)
                paste_center(fr, c, W / 2 + 160 * (1 - e), y + card_h / 2, alpha=clamp(p * 1.8))
            y += card_h + gap
        return fr
    return frame


def _parse_value(value: str):
    m = re.search(r"([^\d\-]*)(-?\d[\d,]*\.?\d*)(.*)", value.strip())
    if not m:
        return "", 0.0, 0, value
    pre, num, post = m.groups()
    num_clean = num.replace(",", "")
    decimals = len(num_clean.split(".")[1]) if "." in num_clean else 0
    return pre, float(num_clean), decimals, post


def scene_stat(v, sc):
    value, label = v["value"], v.get("label", "")
    speak = sc["duration"]
    pre, num, dec, post = _parse_value(value)
    icon = emoji_image(v.get("icon") or pick_icon(sc["text"]), 150)
    start = max(0.2, phrase_time(sc["text"], re.sub(r"[^\d]", "", value)[:2] or value, speak) - 0.4)
    label_words = label.upper().split()
    label_times = [start + 0.5 + 0.06 * i for i in range(len(label_words))]
    pct = num / 100 if post.strip().startswith("%") and 0 < num <= 100 else 1.0
    comma = "," in value

    def fmt(x):
        s = f"{x:,.{dec}f}" if comma else f"{x:.{dec}f}"
        return f"{pre}{s}{post}"

    big_size = 250 if len(value) <= 5 else 200 if len(value) <= 7 else 160

    def frame(t):
        fr = background(t)
        p = progress(t, 0, 0.4)
        paste_center(fr, icon, W / 2, 400, alpha=clamp(p * 2), scale=0.4 + 0.6 * ease_out_back(p))
        c = ease_out_cubic(progress(t, start, 1.1))
        if t >= start:
            s = fmt(num * c)
            d = ImageDraw.Draw(fr)
            pop = 1 + 0.08 * math.sin(math.pi * clamp((t - start - 1.1) / 0.25)) if t > start + 1.1 else 1
            f = bold(int(big_size * pop))
            d.text((W / 2, 700), s, font=f, fill=ORANGE, anchor="mm")
            # bar
            bw = 820
            d.rounded_rectangle((W / 2 - bw / 2, 870, W / 2 + bw / 2, 900), radius=15, fill=CARD_EDGE)
            fill_w = bw * pct * c
            if fill_w > 30:
                d.rounded_rectangle((W / 2 - bw / 2, 870, W / 2 - bw / 2 + fill_w, 900), radius=15, fill=ORANGE)
        if label:
            draw_words(fr, label, W / 2, 1080, t, label_times, bold, 900, 3, 76, 48, auto_highlight(label))
        return fr
    return frame


def scene_compare(v, sc):
    speak = sc["duration"]
    lt, rt = v.get("left", ""), v.get("right", "")
    ll, rl = (v.get("left_label") or "DON'T").upper(), (v.get("right_label") or "DO").upper()
    t_left = max(0.2, phrase_time(sc["text"], lt, speak) - 0.2)
    t_right = max(t_left + 0.6, phrase_time(sc["text"], rt, speak) - 0.2)
    cw, chh = 470, 760

    def make(label, body, col, mark):
        c = rounded_card(cw, chh, edge=col)
        d = ImageDraw.Draw(c)
        d.rounded_rectangle((0, 0, cw - 1, 150), radius=34, fill=col + (255,))
        d.rectangle((0, 110, cw - 1, 150), fill=col + (255,))
        d.text((cw / 2, 76), f"{label}", font=bold(60), fill=(20, 20, 24), anchor="mm")
        badge = _check_badge(120, color=col, mark=mark)
        c.alpha_composite(badge, dest=(int(cw / 2 - 60), 190))
        f, lines, lh, _ = layout_words(body.upper(), bold, cw - 70, 6, 58, 34)
        y = 350
        for row in lines:
            d.text((cw / 2, y), " ".join(w for w, _ in row), font=f, fill=WHITE, anchor="ma")
            y += lh
        return c

    left = make(ll, lt, RED, "✕")
    right = make(rl, rt, GREEN, "check")

    def frame(t):
        fr = background(t)
        p = progress(t, t_left, 0.4)
        if p > 0:
            e = ease_out_back(p)
            paste_center(fr, left, 285 - 300 * (1 - e), 780, alpha=clamp(p * 2))
        q = progress(t, t_right, 0.4)
        if q > 0:
            e = ease_out_back(q)
            paste_center(fr, right, 795 + 300 * (1 - e), 780, alpha=clamp(q * 2))
        return fr
    return frame


SCENES = {"title": scene_title, "statement": scene_statement, "list": scene_list,
          "stat": scene_stat, "compare": scene_compare}


def render_animated(visual: dict, sc: dict, clip_seconds: float, dest: Path):
    """sc needs 'text' and 'duration' (spoken seconds). Writes an H.264 clip."""
    frame_fn = SCENES[visual["type"]](visual, sc)
    n = max(1, round(clip_seconds * FPS))
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "20", "-pix_fmt", "yuv420p", str(dest)],
        stdin=subprocess.PIPE)
    try:
        for i in range(n):
            proc.stdin.write(frame_fn(i / FPS).convert("RGB").tobytes())
    finally:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed while rendering {dest}")


# ---------------------------------------------------------------- overlay (brand tag)

def brand_overlay(path: Path):
    """Transparent PNG with the OFFCUT tag, laid over every frame of the final video."""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = bold(40)
    label = "OFFCUT"
    tw = f.getlength(label)
    x, y = 60, 120
    d.rounded_rectangle((x, y, x + tw + 56, y + 70), radius=16, fill=ORANGE + (235,))
    d.text((x + 28, y + 35), label, font=f, fill=(20, 20, 24, 255), anchor="lm")
    im.save(path)


# ---------------------------------------------------------------- automatic planning

STOP = set("""a an the and or but if so to of in on at for with by from as is are was were be been it its
this that these those you your you're youre i we our they them their he she his her my me do does did not no
can will just than then there here what when where which who how all any some more most very really get got
up out about into over plain simple say like make sure""".split())


def auto_highlight(text: str):
    """One or two words worth colouring orange."""
    words = [re.sub(r"[^a-z0-9$%']", "", w.lower()) for w in text.split()]
    for pattern, _ in ICON_WORDS:
        for w in words:
            if w and w not in STOP and re.fullmatch(f"(?:{pattern}).*", w):
                return [w]
    cands = [w for w in words if w not in STOP and len(w) > 4]
    return [max(cands, key=len)] if cands else []


LEAD_VERBS = r"^(put|add|get|use|set|make|include|offer|keep|send|write|spell out|list|try|ask for|charge|consider)\s+"
CUT_AT = r"\s+(in writing|before|after|so|so that|because|when|which|to make|to keep|to stop)\b.*$"


def _clean_item(s: str) -> str:
    s = s.strip(" .;:!?")
    s = re.sub(r"^(and|or)\s+", "", s, flags=re.I)
    s = re.sub(LEAD_VERBS, "", s, flags=re.I)
    s = re.sub(CUT_AT, "", s, flags=re.I)
    s = re.sub(r"^(a|an|the|your)\s+", "", s, flags=re.I)
    words = s.split()
    if len(words) > 5:
        words = words[:5]
    s = " ".join(words)
    return s[:1].upper() + s[1:]


def find_list(text: str):
    """'... : a, b, and c' or 'a, b and c' -> (title, [a, b, c]) or None."""
    head, body = "", text
    if ":" in text:
        head, body = text.split(":", 1)
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        if sentence.count(",") < (1 if head else 2):
            continue
        parts = re.split(r",\s*(?:and|or)?\s*|\s+(?:and|or)\s+", sentence.strip())
        parts = [p for p in parts if p.strip()]
        if len(parts) < 3:
            continue
        items = [_clean_item(p) for p in parts[:5]]
        if all(1 <= len(i.split()) <= 5 for i in items):
            title = head.strip()
            if len(title.split()) > 9:
                title = ""
            if not title and sentence is not body.strip() and ":" not in text:
                title = ""
            return title, items
    return None


VALUE_RE = re.compile(r"(\$\s?\d[\d,]*(?:\.\d+)?\s?(?:k|K|m|M)?|\d[\d,]*(?:\.\d+)?\s?%|\d[\d,]*(?:\.\d+)?\s?(?:per cent|percent))")


def find_stat(text: str):
    m = VALUE_RE.search(text)
    if not m:
        return None
    value = m.group(1).replace(" ", "").replace("percent", "%").replace("per cent", "%")
    rest = (text[:m.start()] + " " + text[m.end():]).strip()
    clause = re.split(r"[.;:!?]", rest)[0]
    label = " ".join(clause.split()[:9])
    return {"type": "stat", "value": value, "label": label}


def find_compare(text: str):
    m = re.search(r"(.+?)\s+(?:vs\.?|versus|instead of|rather than)\s+(.+)", text, flags=re.I)
    if not m:
        return None
    a, b = m.group(1), m.group(2)
    if "instead of" in text.lower() or "rather than" in text.lower():
        a, b = b, a  # "do X instead of Y" -> left (don't) = Y, right (do) = X
    return {"type": "compare", "left": " ".join(a.split()[-7:]).strip(" ,."),
            "right": " ".join(b.split()[:7]).strip(" ,."), "left_label": "DON'T", "right_label": "DO"}


def headline(text: str) -> str:
    """A short on-screen headline: the first punchy clause, trimmed at a natural break."""
    clauses = [c.strip() for c in re.split(r"[.;:!?]|,\s", text) if c.strip()]
    best = clauses[0] if clauses else text
    for c in clauses:
        if 3 <= len(c.split()) <= 9:
            best = c
            break
    words = best.split()
    if len(words) > 9:
        cut = None
        for k, w in enumerate(words[:10]):
            if k >= 3 and w.lower() in ("so", "because", "which", "before", "after", "when", "if", "to", "that", "while", "but"):
                cut = k
                break
        words = words[:cut] if cut else words[:8]
    return " ".join(words)


def plan_visuals(scenes: list) -> list:
    """Choose a visual per scene; about one in three becomes real footage."""
    plans = []
    footage_left = max(1, len(scenes) // 3)
    prev = None
    for i, sc in enumerate(scenes):
        v = sc.get("visual")
        if isinstance(v, dict) and v.get("type") in list(SCENES) + ["footage"]:
            v = dict(v)
            if v["type"] in ("title", "statement") and not v.get("text"):
                v["text"] = headline(sc["text"])
            if v["type"] == "footage":
                v.setdefault("search", sc.get("search") or "tradesman working")
                footage_left -= 1
            plans.append(v)
            prev = v["type"]
            continue
        text = sc["text"]
        if i == 0:
            v = {"type": "title", "text": headline(text)}
        elif (lst := find_list(text)):
            v = {"type": "list", "title": lst[0], "items": lst[1]}
        elif (st := find_stat(text)):
            v = st
        elif (cp := find_compare(text)):
            v = cp
        elif footage_left > 0 and prev not in ("footage", None) and sc.get("search") and i >= 1:
            v = {"type": "footage", "search": sc["search"]}
            footage_left -= 1
        else:
            v = {"type": "statement", "text": headline(text)}
        plans.append(v)
        prev = v["type"]
    return plans
