"""Ginseng tea UGC ad renderer.

python3 -I render.py <workdir> <main|cut> [--preview t1,t2,...]
"""
import json
import math
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio  # noqa: E402

FPS = 24
OW, OH = 1080, 1920
SW, SH = 720, 1280
WORK = sys.argv[1]
MODE = sys.argv[2]
PREVIEW = None
if "--preview" in sys.argv:
    PREVIEW = [float(x) for x in sys.argv[sys.argv.index("--preview") + 1].split(",")]

FONT_BLACK = "/usr/share/fonts/opentype/inter/Inter-Black.otf"
FONT_XB = "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"
FONT_B = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
YELLOW = (255, 212, 0, 255)
WHITE = (255, 255, 255, 255)
RED = (196, 22, 36, 255)
GOLD = (232, 180, 72, 255)
CREAM = (255, 248, 231, 255)

# ---------------------------------------------------------------- framing
# center (cx, cy) and width of the 9:16 crop window in source pixels
FRAMES = {
    "hook": dict(c=(380, 482), w=470, f=(380, 470)),
    "A": dict(c=(335, 640), w=600, f=(380, 460)),
    "Acup": dict(c=(315, 650), w=600, f=(300, 640)),
    "P": dict(c=(310, 643), w=600, f=(210, 850), box=True),
    "Pbox": dict(c=(310, 643), w=600, f=(200, 850), box=True),
    "E": dict(c=(335, 650), w=600, f=(380, 460)),
    "Ebox": dict(c=(330, 660), w=600, f=(200, 840), box=True),
    "C": dict(c=(330, 693), w=660, f=(330, 693), box=True),
}
# caption block placement: ("center"|"bottom", y)
CAP_POS = {"hook": ("center", 1440), "A": ("center", 1440), "Acup": ("center", 1440),
           "P": ("bottom", 1622), "Pbox": ("bottom", 1622), "C": ("bottom", 1462),
           "E": ("center", 1440), "Ebox": ("center", 1440)}

MAIN = [
    dict(clip="A", s=0.0, e=1.64, fr="hook", z=1.0, caps=[["Après", "une", "longue", "*journée*,"]]),
    dict(clip="A", s=1.68, e=3.64, fr="A", z=1.0, caps=[["Ta", "*famille*", "mérite", "encore"], ["le", "*meilleur*", "de", "toi."]]),
    dict(clip="A", s=3.96, e=6.30, fr="A", z=1.12, caps=[["Moi,", "je", "rentrais", "*vidé*,"], ["sans", "force", "pour", "personne."]]),
    dict(clip="A", s=6.60, e=8.32, fr="A", z=1.0, caps=[["Maintenant,", "j'ai"], ["mon", "*rituel*", "du", "soir."]]),
    dict(clip="A", s=8.42, e=10.10, fr="Acup", z=1.12, caps=[["Un", "thé", "au", "*ginseng*,"], ["bien", "*chaud*."]]),
    dict(clip="B", s=0.42, e=2.24, fr="P", z=1.0, caps=[["Le", "thé", "traditionnel"], ["au", "*ginseng*."]], callouts=[("Instantané", 0.25)]),
    dict(clip="B", s=2.98, e=3.86, fr="Pbox", z=1.12, caps=[["*2*", "*minutes*"]], callouts=[("Prêt en 2 min", 0.08)]),
    dict(clip="B", s=4.16, e=5.50, fr="P", z=1.0, caps=[["et", "un", "vrai"], ["*moment*", "pour", "toi."]], callouts=[("12 sachets", 0.25)]),
    dict(clip="A", s=10.08, e=11.35, fr="E", z=1.06, caps=[]),
    dict(clip="A", s=12.00, e=13.04, fr="Ebox", z=1.0, caps=[]),
    dict(clip="B", s=6.72, e=8.78, fr="C", z=1.0, caps=[["Clique", "sur", "le", "*bouton*"], ["en", "bas", "de", "la", "vidéo"]], cta=True),
    dict(clip="B", s=8.92, e=9.72, fr="C", z=1.06, caps=[["pour", "*commander*."]], cta=True),
    dict(clip="B", s=10.30, e=12.04, fr="C", z=1.0, caps=[["Paiement", "à", "la", "*livraison*."]], cta=True),
]
CUT = [MAIN[0], MAIN[3], MAIN[4], MAIN[5], MAIN[6], MAIN[7], MAIN[10], MAIN[11], MAIN[12]]
EDL = MAIN if MODE == "main" else CUT

HOOK = ("Mon rituel du soir", "en 2 minutes")
HOOK_END = 2.5

# ---------------------------------------------------------------- source
GRADE = ("hqdn3d=1.2:1.0:2.5:2.5,"
         "curves=master='0/0 0.06/0.045 0.5/0.5 0.8/0.79 1/0.925',"
         "eq=contrast=1.04:saturation=1.06,"
         "colorbalance=rs=0.01:bs=-0.02:rm=0.025:gm=0.005:bm=-0.03:rh=0.015:bh=-0.035")


def load_clip(name):
    cmd = ["ffmpeg", "-v", "error", "-i", f"{WORK}/src/{name}.mp4", "-vf", GRADE,
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, SH, SW, 3)
    # graduated "ND" over the overexposed foreground sleeve (right / bottom)
    yy, xx = np.mgrid[0:SH, 0:SW].astype(np.float32)
    m = np.clip((xx - 330) / 260, 0, 1) * np.clip((yy - 520) / 380, 0, 1)
    m = np.maximum(m, np.clip((yy - 1050) / 230, 0, 1) * 0.8)
    out = np.empty_like(fr)
    for i in range(len(fr)):
        f = fr[i].astype(np.float32) / 255
        lum = f.mean(2, keepdims=True)
        hl = np.clip((lum - 0.68) / 0.3, 0, 1) ** 1.5
        f = f * (1 - 0.13 * m[..., None] * hl)
        out[i] = np.clip(f * 255 + 0.5, 0, 255).astype(np.uint8)
    return out


def track_box(frames):
    """Bounding box of the red product box (ignores background boxes)."""
    res = []
    for f in frames:
        sub = f[700:1180].astype(np.int16)
        r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
        mask = (r > 140) & (r - g > 85) & (r - b > 70)
        ys, xs = np.nonzero(mask)
        if len(xs) > 3000:
            res.append([np.percentile(xs, 1), np.percentile(ys, 1) + 700,
                        np.percentile(xs, 99), np.percentile(ys, 99) + 700])
        else:
            res.append(None)
    # fill gaps + smooth
    last = None
    for i in range(len(res)):
        if res[i] is None:
            res[i] = last
        last = res[i]
    arr = np.array([r if r is not None else [0, 0, 0, 0] for r in res], np.float32)
    k = np.ones(5) / 5
    for j in range(4):
        arr[:, j] = np.convolve(np.pad(arr[:, j], 2, mode="edge"), k, "valid")
    return arr


# ---------------------------------------------------------------- timing
def voiced_profile(clip, s, e):
    y, sr = sf.read(f"{WORK}/src/{clip}_voice.wav")
    seg = y[int(s * sr):int(e * sr)]
    hop = int(0.01 * sr)
    rms = np.array([np.sqrt(np.mean(seg[i:i + hop] ** 2) + 1e-12) for i in range(0, len(seg) - hop, hop)])
    db = 20 * np.log10(rms)
    thr = max(db.max() - 28, -45)
    return db > thr


def syllables(w):
    w = w.lower().strip("*,.")
    n, prev = 0, False
    for ch in w:
        v = ch in "aeiouyéèêàâîôûùë0123456789"
        if v and not prev:
            n += 1
        prev = v
    return max(n, 1) + 0.4


def word_times(seg, t0):
    words = [w for g in seg["caps"] for w in g]
    if not words:
        return []
    voiced = voiced_profile(seg["clip"], seg["s"], seg["e"])
    idx = np.nonzero(voiced)[0]
    cum = np.cumsum(voiced)
    total_v = cum[-1]
    wts = np.array([syllables(w) for w in words])
    starts = np.concatenate([[0], np.cumsum(wts)[:-1]]) / wts.sum() * total_v
    out = []
    for st in starts:
        k = int(np.searchsorted(cum, st + 1))
        k = min(max(k, idx[0]), len(voiced) - 1)
        out.append(t0 + k * 0.01)
    return out


# ---------------------------------------------------------------- graphics
_fonts = {}


def font(path, size):
    key = (path, int(size))
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(path, int(size))
    return _fonts[key]


def ease_out_back(x):
    x = min(max(x, 0), 1)
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def ease_out(x):
    x = min(max(x, 0), 1)
    return 1 - (1 - x) ** 3


def layout_caption(words, size=74, maxw=900):
    f = font(FONT_BLACK, size)
    space = size * 0.28
    lines, cur, curw = [], [], 0
    for w in words:
        txt = w.replace("*", "")
        ww = f.getlength(txt)
        if cur and curw + space + ww > maxw:
            lines.append(cur)
            cur, curw = [], 0
        cur.append((w, txt, ww))
        curw += (space if len(cur) > 1 else 0) + ww
    lines.append(cur)
    return lines[:2], space


def draw_caption(layer, words, wtimes, t, pos, size=74):
    lines, space = layout_caption(words, size)
    lh = size * 1.18
    mode, y = pos
    block_h = lh * len(lines)
    top = y - block_h / 2 if mode == "center" else y - block_h
    d = ImageDraw.Draw(layer)
    k = 0
    for li, line in enumerate(lines):
        lw = sum(x[2] for x in line) + space * (len(line) - 1)
        x = OW / 2 - lw / 2
        cy = top + li * lh + lh / 2
        for raw, txt, ww in line:
            wt = wtimes[k]
            k += 1
            if t + 1e-6 < wt:
                x += ww + space
                continue
            p = ease_out_back((t - wt) / 0.16)
            sc = 0.7 + 0.3 * p
            fs = size * sc
            col = YELLOW if raw.startswith("*") else WHITE
            cx = x + ww / 2
            fo = font(FONT_BLACK, fs)
            d.text((cx + 3, cy + 5), txt, font=fo, fill=(0, 0, 0, 110), anchor="mm",
                   stroke_width=int(fs * 0.11), stroke_fill=(0, 0, 0, 110))
            d.text((cx, cy), txt, font=fo, fill=col, anchor="mm",
                   stroke_width=max(2, int(fs * 0.095)), stroke_fill=(0, 0, 0, 255))
            x += ww + space


def pill(d, box, fill, radius=None, outline=None, width=0):
    x0, y0, x1, y1 = box
    r = radius if radius is not None else (y1 - y0) / 2
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def draw_hook(layer, t):
    if t > HOOK_END:
        return
    a = min(1, (HOOK_END - t) / 0.15)
    d = ImageDraw.Draw(layer)
    for i, (txt, bg, fg, size) in enumerate([(HOOK[0], WHITE, (20, 20, 20, 255), 62),
                                             (HOOK[1], YELLOW, (20, 20, 20, 255), 66)]):
        p = ease_out_back((t - 0.05 - i * 0.18) / 0.28)
        if p <= 0:
            continue
        fs = size * (0.6 + 0.4 * p)
        f = font(FONT_BLACK, fs)
        tw = f.getlength(txt)
        cy = 262 + i * 104
        pad_x, h = 30 * (0.6 + 0.4 * p), fs * 1.42
        box = (OW / 2 - tw / 2 - pad_x, cy - h / 2, OW / 2 + tw / 2 + pad_x, cy + h / 2)
        sh = tuple(v + o for v, o in zip(box, (0, 8, 0, 8)))
        pill(d, sh, (0, 0, 0, int(70 * a)), radius=22)
        pill(d, box, bg[:3] + (int(255 * a),), radius=22)
        d.text((OW / 2, cy + 2), txt, font=f, fill=fg[:3] + (int(255 * a),), anchor="mm")


def draw_callouts(layer, active, t, boxo):
    """active: list of (text, t_appear, slot). boxo: product box in output coords."""
    d = ImageDraw.Draw(layer)
    bx0, by0, bx1, by1 = boxo
    for text, ta, slot in active:
        if t < ta:
            continue
        p = ease_out_back((t - ta) / 0.3)
        f = font(FONT_XB, 50 * (0.5 + 0.5 * p))
        tw = f.getlength(text)
        h = 86 * (0.5 + 0.5 * p)
        bob = 5 * math.sin(2 * math.pi * 0.8 * (t - ta) + slot)
        cx = min(bx1 + 40 + (tw + 90) / 2, OW - 30 - (tw + 90) / 2)
        cy = by0 + 40 + slot * ((by1 - by0 - 80) / 2) + bob
        box = (cx - tw / 2 - 45, cy - h / 2, cx + tw / 2 + 45, cy + h / 2)
        pill(d, tuple(v + o for v, o in zip(box, (0, 6, 0, 6))), (0, 0, 0, 80))
        pill(d, box, RED, outline=GOLD, width=4)
        # gold dot = "check" marker
        r = h * 0.17
        dx = box[0] + 30
        d.ellipse((dx - r, cy - r, dx + r, cy + r), fill=GOLD)
        d.text((cx + 12, cy + 1), text, font=f, fill=WHITE, anchor="mm")
        # leader tick toward the box
        d.line((box[0] - 4, cy, max(bx1 - 10, box[0] - 34), cy), fill=GOLD, width=4)


def draw_cta(layer, t, ta):
    if t < ta:
        return
    d = ImageDraw.Draw(layer)
    p = ease_out_back((t - ta) / 0.35)
    pulse = 1 + 0.035 * max(0, math.sin(2 * math.pi * 1.6 * (t - ta - 0.35))) if t - ta > 0.35 else 1
    s = (0.5 + 0.5 * p) * pulse
    cx, cy = OW / 2, 1548
    w, h = 940 * s, 136 * s
    box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    pill(d, tuple(v + o for v, o in zip(box, (0, 9, 0, 9))), (0, 0, 0, 95), radius=h / 2)
    pill(d, box, RED, radius=h / 2, outline=GOLD, width=max(2, int(5 * s)))
    d.text((cx - 46 * s, cy - 20 * s), "COMMANDER MAINTENANT", font=font(FONT_BLACK, 47 * s), fill=WHITE, anchor="mm")
    d.text((cx - 46 * s, cy + 30 * s), "Paiement à la livraison", font=font(FONT_XB, 33 * s), fill=YELLOW, anchor="mm")
    # bouncing down-arrow badge on the right of the button
    bounce = 9 * abs(math.sin(2 * math.pi * 1.4 * (t - ta)))
    ax, ay = box[2] - 58 * s, cy + bounce * s - 4 * s
    r = 36 * s
    d.ellipse((ax - r, ay - r, ax + r, ay + r), fill=WHITE)
    aw = 20 * s
    d.polygon([(ax - aw, ay - 6 * s), (ax + aw, ay - 6 * s), (ax, ay + 16 * s)], fill=RED)
    d.rectangle((ax - 7 * s, ay - 22 * s, ax + 7 * s, ay - 4 * s), fill=RED)


# ---------------------------------------------------------------- plan
def build_plan():
    plan, t = [], 0.0
    for seg in EDL:
        sf_ = round(seg["s"] * FPS)
        ef = round(seg["e"] * FPS)
        n = ef - sf_
        seg = dict(seg, sf=sf_, n=n, t0=t)
        seg["wt"] = word_times(seg, t)
        plan.append(seg)
        t += n / FPS
    return plan, t


def caption_state(plan, t):
    for si, seg in enumerate(plan):
        if seg["t0"] <= t < seg["t0"] + seg["n"] / FPS:
            break
    caps, wt = seg["caps"], seg["wt"]
    k, groups = 0, []
    for g in caps:
        groups.append((g, wt[k:k + len(g)]))
        k += len(g)
    cur = None
    for gi, (g, w) in enumerate(groups):
        if t >= (w[0] if gi else min(w[0], seg["t0"] + 0.02)) - 0.02:
            cur = (g, w)
    return si, seg, cur


def crop_for(seg, k, t, boxes):
    F = FRAMES[seg["fr"]]
    cx, cy = F["c"]
    w = F["w"]
    fx, fy = F["f"]
    z = seg["z"]
    if seg["fr"] == "hook":
        z *= 1 + 0.10 * ease_out(t / 0.3)
    z *= 1 + 0.045 * (1 - ease_out(k / 3.0))  # 0.1 s punch-settle on each cut
    z *= 1 + 0.004 * math.sin(2 * math.pi * 0.22 * t)  # breathing
    cx = fx + (cx - fx) / z
    cy = fy + (cy - fy) / z
    w = w / z
    h = w * 16 / 9
    sc = OW / w
    # handheld micro-shake (output px -> source px)
    dx = 1.3 * math.sin(2 * math.pi * 0.71 * t + 0.3) + 0.6 * math.sin(2 * math.pi * 1.93 * t + 1.1)
    dy = 1.1 * math.sin(2 * math.pi * 0.53 * t + 2.0) + 0.5 * math.sin(2 * math.pi * 2.31 * t)
    cx += dx / sc
    cy += dy / sc
    x0, y0 = cx - w / 2, cy - h / 2
    if F.get("box") and boxes is not None:
        bx0, by0, bx1, by1 = boxes[seg["clip"]][seg["sf"] + k]
        m = 14
        if bx1 > 0:
            x0 = min(x0, bx0 - m)
            x0 = max(x0, bx1 + m - w)
            y0 = max(y0, by1 + m - h)
    x0 = min(max(x0, 0), SW - w)
    y0 = min(max(y0, 0), SH - h)
    return (x0, y0, x0 + w, y0 + h)


VIG = None
GRAIN_RNG = np.random.default_rng(3)


def vignette():
    yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
    r = np.sqrt(((xx - OW / 2) / (OW / 2)) ** 2 + ((yy - OH * 0.45) / (OH / 2)) ** 2)
    return (1 - 0.22 * np.clip((r - 0.55) / 0.75, 0, 1) ** 1.6)[..., None]


SRC = {}
BOXES = {}
PLAN = None


def render_frame(j):
    global VIG
    if VIG is None:
        VIG = vignette()
    t = j / FPS
    si, seg, cap = caption_state(PLAN, t)
    k = j - round(seg["t0"] * FPS)
    k = min(max(k, 0), seg["n"] - 1)
    src = SRC[seg["clip"]][seg["sf"] + k]
    box = crop_for(seg, k, t, BOXES)
    im = Image.fromarray(src).resize((OW, OH), Image.LANCZOS, box=box)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.8, percent=65, threshold=2))
    im = im.filter(ImageFilter.UnsharpMask(radius=0.8, percent=35, threshold=1))
    a = np.asarray(im).astype(np.float32) * VIG
    rng = np.random.default_rng(1000 + j)
    g = rng.standard_normal((OH // 2, OW // 2)).astype(np.float32)
    g = np.repeat(np.repeat(g, 2, 0), 2, 1)[..., None]
    lum = a.mean(2, keepdims=True) / 255
    a += g * 3.2 * (1.2 - lum)
    frame = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGBA")

    layer = Image.new("RGBA", (OW, OH), (0, 0, 0, 0))
    # callouts (product segments)
    if seg["fr"] in ("P", "Pbox"):
        x0, y0, x1, y1 = box
        sc = OW / (x1 - x0)
        b = BOXES[seg["clip"]][seg["sf"] + k]
        boxo = ((b[0] - x0) * sc, (b[1] - y0) * sc, (b[2] - x0) * sc, (b[3] - y0) * sc)
        active, slot = [], 0
        for s2 in PLAN:
            if s2["fr"] in ("P", "Pbox") and abs(s2["t0"] - seg["t0"]) < 8 and s2["t0"] <= seg["t0"]:
                # include callouts from earlier contiguous product segments
                contiguous = all(PLAN[q]["fr"] in ("P", "Pbox") for q in range(PLAN.index(s2), si + 1))
                if contiguous:
                    for text, off in s2.get("callouts", []):
                        active.append((text, s2["t0"] + off, slot))
                        slot += 1
        draw_callouts(layer, active, t, boxo)
    if seg.get("cta"):
        first = next(s2 for s2 in PLAN if s2.get("cta"))
        draw_cta(layer, t, first["t0"] + 0.3)
    if cap:
        draw_caption(layer, cap[0], cap[1], t, CAP_POS[seg["fr"]])
    draw_hook(layer, t)
    frame.alpha_composite(layer)
    return np.asarray(frame.convert("RGB")).tobytes()


def init_worker(src, boxes, plan):
    global SRC, BOXES, PLAN
    SRC, BOXES, PLAN = src, boxes, plan


# ---------------------------------------------------------------- audio
def build_audio(plan, total):
    sr = audio.SR
    n = int(round(total * sr))
    voice = np.zeros(n)
    src = {c: sf.read(f"{WORK}/src/{c}_voice.wav")[0] for c in ("A", "B")}
    fade = int(0.006 * sr)
    cuts = []
    for seg in plan:
        a0 = int(round(seg["sf"] / FPS * sr))
        ln = int(round(seg["n"] / FPS * sr))
        piece = src[seg["clip"]][a0:a0 + ln].copy()
        piece[:fade] *= np.linspace(0, 1, fade)
        piece[-fade:] *= np.linspace(1, 0, fade)
        o = int(round(seg["t0"] * sr))
        voice[o:o + len(piece)] += piece[: n - o]
        cuts.append(seg)
    voice2 = np.stack([voice, voice], 1)

    cta_t = next(s["t0"] for s in plan if s.get("cta"))
    emo = [s for s in plan if s["fr"] in ("E", "Ebox")]
    swell = (emo[0]["t0"], emo[-1]["t0"] + emo[-1]["n"] / FPS) if emo else None
    mus = audio.music(total, cta_t, swell)
    mus = np.pad(mus, ((0, max(0, n - len(mus))), (0, 0)))[:n]

    def lufs_like(x):  # rough K-weight-free loudness of the active part
        x = x if x.ndim == 1 else x.mean(1)
        blk = int(0.4 * sr)
        e = np.array([np.mean(x[i:i + blk] ** 2) for i in range(0, len(x) - blk, blk // 4)])
        e = e[e > 1e-7]
        return 10 * np.log10(np.mean(e[e > np.percentile(e, 30)]))

    target = lufs_like(voice) - 22
    mus *= 10 ** ((target - lufs_like(mus)) / 20)

    sfx = np.zeros((n, 2))
    for i, seg in enumerate(plan[1:], 1):
        big = seg["clip"] != plan[i - 1]["clip"]
        w = audio.whoosh(0.30 if big else 0.2, 0.06 if big else 0.025)
        o = int((seg["t0"] - (0.15 if big else 0.1)) * sr)
        sfx[o:o + len(w)] += w[: n - o]
    for seg in plan:
        for _, off in seg.get("callouts", []):
            p = audio.pop(0.05)
            o = int((seg["t0"] + off) * sr)
            sfx[o:o + len(p)] += p[: n - o]
    p = audio.pop(0.06)
    o = int((cta_t + 0.3) * sr)
    sfx[o:o + len(p)] += p[: n - o]
    room = audio.room_tone(total)[:n]
    mix = voice2 + mus + sfx + room
    path = f"{WORK}/out/{MODE}_mix.wav"
    sf.write(path, mix.astype(np.float32), sr, subtype="FLOAT")
    return path


def main():
    global SRC, BOXES, PLAN
    plan, total = build_plan()
    PLAN = plan
    nfr = sum(s["n"] for s in plan)
    print(f"{MODE}: {len(plan)} segments, {nfr} frames, {total:.2f}s")
    SRC = {c: load_clip(c) for c in ("A", "B")}
    BOXES = {c: track_box(SRC[c]) for c in ("A", "B")}
    json.dump([{k: v for k, v in s.items() if k != "wt"} | {"wt": [round(x, 2) for x in s["wt"]]} for s in plan],
              open(f"{WORK}/out/{MODE}_plan.json", "w"), ensure_ascii=False, indent=1)
    if PREVIEW:
        init_worker(SRC, BOXES, plan)
        for tp in PREVIEW:
            j = int(tp * FPS)
            Image.frombytes("RGB", (OW, OH), render_frame(j)).save(f"{WORK}/out/prev_{MODE}_{tp:05.2f}.png")
        return
    apath = build_audio(plan, total)
    vpath = f"{WORK}/out/{MODE}_video.mp4"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
                            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                            vpath], stdin=subprocess.PIPE)
    with Pool(4, initializer=init_worker, initargs=(SRC, BOXES, plan)) as pool:
        for i, buf in enumerate(pool.imap(render_frame, range(nfr), chunksize=4)):
            enc.stdin.write(buf)
            if i % 48 == 0:
                print(f"  frame {i}/{nfr}", flush=True)
    enc.stdin.close()
    enc.wait()
    print("video:", vpath, "audio:", apath)


if __name__ == "__main__":
    main()
