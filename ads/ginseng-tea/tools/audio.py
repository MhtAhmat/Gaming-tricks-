"""Procedural music bed + sound design (no stock assets)."""
import numpy as np

SR = 48000
rng = np.random.default_rng(7)


def onepole_lp(x, fc):
    a = np.exp(-2 * np.pi * fc / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):  # small buffers only
        acc = (1 - a) * x[i] + a * acc
        y[i] = acc
    return y


def fft_filter(x, lo=None, hi=None):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    g = np.ones_like(f)
    if lo:
        g *= 1 / np.sqrt(1 + (lo / np.maximum(f, 1)) ** 4)
    if hi:
        g *= 1 / np.sqrt(1 + (f / hi) ** 4)
    return np.fft.irfft(X * g, len(x))


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def epiano(freq, dur, vel=1.0):
    t = np.arange(int(dur * SR)) / SR
    env = np.exp(-t * 2.2) * (1 - np.exp(-t * 300))
    tone = (np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * 2 * freq * t) * np.exp(-t * 4)
            + 0.12 * np.sin(2 * np.pi * 3 * freq * t) * np.exp(-t * 7)
            + 0.5 * np.sin(2 * np.pi * freq * 1.003 * t))
    trem = 1 + 0.08 * np.sin(2 * np.pi * 4.5 * t)
    return vel * tone * env * trem


def kick():
    t = np.arange(int(0.35 * SR)) / SR
    f = 45 + 70 * np.exp(-t * 30)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t * 9)


def hat(dur=0.05):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return fft_filter(rng.standard_normal(n), lo=7000) * np.exp(-t * 80)


def snap():
    n = int(0.18 * SR)
    t = np.arange(n) / SR
    return fft_filter(rng.standard_normal(n), lo=1200, hi=6000) * np.exp(-t * 28)


def add(buf, sig, at, gain=1.0):
    i = int(at * SR)
    if i >= len(buf):
        return
    sig = sig[: len(buf) - i]
    buf[i:i + len(sig)] += gain * sig


def music(total, lift_at, swell=None, bpm=88):
    """Warm lo-fi bed. lift_at: seconds where the CTA lift starts."""
    n = int((total + 1.5) * SR)
    L = np.zeros(n)
    R = np.zeros(n)
    beat = 60 / bpm
    # Fmaj9 - Em7 - Dm9 - Cmaj7 voicings (warm, unresolved)
    chords = [[53, 60, 64, 67, 69], [52, 59, 62, 67, 71], [50, 57, 60, 64, 65], [48, 55, 59, 64, 67]]
    bars = int(total / (4 * beat)) + 2
    for b in range(bars):
        t0 = b * 4 * beat
        ch = chords[b % 4]
        lifted = t0 + 4 * beat > lift_at
        for k, note in enumerate(ch):
            sig = epiano(midi(note), 4 * beat + 0.6, 0.18)
            pan = 0.35 + 0.3 * (k / 4)
            add(L, sig, t0 + 0.012 * k, 1 - pan + 0.5)
            add(R, sig, t0 + 0.012 * k, pan + 0.5)
        # soft re-strike on beat 3
        for k, note in enumerate(ch[1:]):
            sig = epiano(midi(note + 12), 2 * beat, 0.06)
            add(L, sig, t0 + 2 * beat + 0.5 * beat, 0.8)
            add(R, sig, t0 + 2 * beat + 0.5 * beat, 1.0)
        # bass
        bt = np.arange(int(1.6 * beat * SR)) / SR
        bass = np.sin(2 * np.pi * midi(ch[0] - 12) * bt) * np.exp(-bt * 2.5) * (1 - np.exp(-bt * 200))
        for at in (0, 2.5 * beat):
            add(L, bass, t0 + at, 0.55)
            add(R, bass, t0 + at, 0.55)
        for q in range(4):
            tb = t0 + q * beat
            if q in (0, 2) or (lifted and q == 3):
                add(L, kick(), tb, 0.55)
                add(R, kick(), tb, 0.55)
            if q in (1, 3) and b > 0:
                s = snap()
                add(L, s, tb, 0.10)
                add(R, s, tb, 0.13)
            subdiv = 4 if lifted else 2
            for h in range(subdiv):
                th = tb + h * beat / subdiv
                g = (0.05 if h % 2 else 0.035) * (1.4 if lifted else 1.0)
                add(L, hat(), th, g * 0.8)
                add(R, hat(), th, g)
    # riser into CTA
    rn = int(1.0 * SR)
    rt = np.arange(rn) / SR
    riser = fft_filter(rng.standard_normal(rn), lo=2000, hi=9000) * (rt / 1.0) ** 2.5 * 0.12
    add(L, riser, lift_at - 1.0)
    add(R, riser, lift_at - 1.0)
    t = np.arange(n) / SR
    # overall dynamics: gentle intro fade, +3 dB lift at CTA, optional swell, fade out
    gain = np.clip(t / 0.6, 0, 1) * 0.35 + 0.65
    gain *= np.where(t >= lift_at, 10 ** (3 / 20), 1.0)
    if swell:
        s0, s1 = swell
        gain *= 1 + (10 ** (4 / 20) - 1) * np.clip(np.minimum((t - s0) / 0.3, (s1 - t) / 0.3), 0, 1)
    gain *= np.clip((total + 0.2 - t) / 0.6, 0, 1)
    # warmth: roll off top
    L = fft_filter(L * gain, lo=35, hi=9000)
    R = fft_filter(R * gain, lo=35, hi=9000)
    return np.stack([L, R], 1)[: int(total * SR)]


def whoosh(dur=0.32, gain=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    # swept band: process in short blocks
    blk = 512
    for i in range(0, n, blk):
        frac = i / n
        fc = 600 + 4000 * np.sin(np.pi * frac)
        seg = noise[max(0, i - 2048): i + blk]
        f = fft_filter(seg, lo=fc * 0.6, hi=fc * 1.6)[-min(blk, n - i):]
        out[i:i + len(f)] = f
    env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 2
    s = out * env
    s /= np.abs(s).max() + 1e-9
    pan = np.linspace(-0.6, 0.6, n)
    return np.stack([s * (1 - pan) / 1.6, s * (1 + pan) / 1.6], 1) * gain


def pop(gain=1.0):
    n = int(0.09 * SR)
    t = np.arange(n) / SR
    f = 380 + 900 * np.exp(-t * 60)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-t * 45) * (1 - np.exp(-t * 2000))
    s /= np.abs(s).max()
    return np.stack([s, s], 1) * gain


def room_tone(total, level_db=-56):
    n = int(total * SR)
    w = np.cumsum(rng.standard_normal(n))
    w = fft_filter(w - np.linspace(w[0], w[-1], n), lo=60, hi=1800)
    w /= np.sqrt(np.mean(w ** 2)) + 1e-9
    w *= 10 ** (level_db / 20)
    return np.stack([w, w * 0.97], 1)
