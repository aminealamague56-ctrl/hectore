#!/usr/bin/env python3
"""GTA 6 YouTube Short (Moroccan Darija) - 1080x1920 9:16, 30 fps.

Usage:  python3 build.py            -> gta6_short_darija.mp4

Voice : your own recording in ./voice/seg0..seg5.(wav|mp3|m4a) if present (see SCRIPT_DARIJA.md),
        otherwise Piper neural TTS (male Arabic voice "kareem") reading the Darija script.
Music : synthwave/trap beat + SFX generated in numpy (no third-party audio).
Video : original GTA-inspired neon visuals rendered with Pillow.

Optional footage: drop your own licensed clips/images in ./clips named after a
scene (hook, date, city, heroes, live, cta) e.g. clips/city.mp4 or
clips/heroes.jpg - they replace that scene's generated background
(auto-cropped to 9:16, zoomed, darkened so subtitles stay readable).
"""
import os, sys, math, random, subprocess, shutil, glob, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
CLIPS = os.path.join(HERE, "clips")
OUT = os.path.join(HERE, "gta6_short_darija.mp4")
W, H, FPS, SR = 1080, 1920, 30, 48000
K = 1.25                      # backgrounds rendered larger for zoom headroom
BW, BH = int(W * K), int(H * K)
AR = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LAT = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
VOICE_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "tts-models/vits-piper-ar_JO-kareem-medium.tar.bz2")
VOICE = os.path.join(CACHE, "vits-piper-ar_JO-kareem-medium", "ar_JO-kareem-medium.onnx")

PINK, CYAN, YEL, WHITE, RED = (255, 45, 149), (45, 226, 255), (255, 210, 63), (255, 255, 255), (230, 0, 35)

# ----------------------------------------------------------------- script ---
# tts: what the voice says (chunks separated by |, spelled for pronunciation)
# sub: on-screen subtitle for each chunk (same number of chunks)
SEGS = [
    dict(scene="hook",
         tts="سْمْعُو مزيان الخوت!|اللعبة اللي بْقِينَا كَنْتْسَنَّاوْ سنين...|هِيَّا وْصْلَاتْ!",
         sub="سمعوا مزيان الخوت!|اللعبة اللي بقينا كنتسناو سنين|هيا وصلات!"),
    dict(scene="date",
         tts="صافي، جي تي إي سيكس خارجة|نهار الخميس،|تسعطاش نونبر!",
         sub="صافي، GTA 6 خارجة|نهار الخميس|19 نونبر!"),
    dict(scene="city",
         tts="غادي نْرْجْعُو لفايس سيتي!|الليل، الضو، البحر،|والطوموبيلات كَيْطِيرُو!|حاجة خطيرة!",
         sub="غادي نرجعو لـ Vice City!|الليل، الضو، البحر|والطوموبيلات كيطيرو!|حاجة خطيرة!"),
    dict(scene="heroes",
         tts="جوج ديال الأبطال،|جيسون و لوسيا،|و واحد القصة غادي تْشْدّْكُم|من اللول حتى للخر!",
         sub="جوج ديال الأبطال|جيسون و لوسيا|و واحد القصة غادي تشدكم|من اللول حتى للخر!"),
    dict(scene="live",
         tts="و نهار الخروج،|عندنا لايف هنا فالقناة،|غادي نْلْعْبُو مع بعضياتنا|من الدقيقة اللولة!",
         sub="و نهار الخروج|عندنا لايف هنا فالقناة|غادي نلعبو مع بعضياتنا|من الدقيقة اللولة!"),
    dict(scene="cta",
         tts="ما تنساوش!|دِيرُو أبوني فالقناة،|و كليكيو على الجرس|باش تْوْصْلْكُم النوتيفيكاسيون.|نتلاقاو فاللايف، يالاه!",
         sub="ما تنساوش!|ديرو أبوني فالقناة|و كليكيو على الجرس|باش توصلكم النوتيفيكاسيون|نتلاقاو فاللايف، يالاه!"),
]
INTRO, GAP, OUTRO = 0.5, 0.22, 1.3


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


# ------------------------------------------------------------------ voice ---
def ensure_voice():
    if os.path.exists(VOICE):
        return
    os.makedirs(CACHE, exist_ok=True)
    print("downloading voice model ...")
    run(f"curl -sSL {VOICE_URL} | tar xj -C {CACHE}", shell=True)


def load_audio(path, filters=None):
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if filters:
        cmd += ["-af", filters]
    cmd += ["-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(run(cmd, capture_output=True).stdout, np.float32).copy()


def trim(a, thr=0.01):
    idx = np.where(np.abs(a) > thr)[0]
    if len(idx) == 0:
        return a
    return a[max(0, idx[0] - int(0.03 * SR)): idx[-1] + int(0.08 * SR)]


# phone / WhatsApp recordings: denoise, remove rumble, tame harshness, even out volume
REC_FILTER = ("highpass=f=80,lowpass=f=12000,afftdn=nf=-25:nr=12,"
              "equalizer=f=250:t=q:w=1:g=-2,equalizer=f=3500:t=q:w=1.2:g=3,equalizer=f=7000:t=q:w=2:g=-2,"
              "acompressor=threshold=-24dB:ratio=3.5:attack=5:release=120:makeup=6dB,"
              "loudnorm=I=-16:TP=-2:LRA=7")
MAX_LEN = 45.0


def fit_length(clips):
    """If narration is too long for a 45 s Short, speed it up (pitch preserved, max x1.15)."""
    need = INTRO + OUTRO + GAP * (len(clips) - 1) + sum(len(c) for c in clips) / SR
    if need <= MAX_LEN - 0.2:
        return clips
    voice_len = sum(len(c) for c in clips) / SR
    tempo = min(1.15, voice_len / (MAX_LEN - 0.3 - INTRO - OUTRO - GAP * (len(clips) - 1)))
    print(f"  narration {need:.1f}s > {MAX_LEN}s -> speeding up x{tempo:.3f}")
    out = []
    for c in clips:
        r = run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                 "-af", f"atempo={tempo:.4f}", "-f", "f32le", "-"], input=c.tobytes(), capture_output=True)
        out.append(np.frombuffer(r.stdout, np.float32).copy())
    return out


def make_voice():
    ensure_voice()
    vdir = os.path.join(CACHE, "voice"); os.makedirs(vdir, exist_ok=True)
    clips = []
    vf = ("highpass=f=85,equalizer=f=180:t=q:w=1:g=2,equalizer=f=3200:t=q:w=1.2:g=4,"
          "acompressor=threshold=-22dB:ratio=4:attack=4:release=90:makeup=5dB")
    for i, s in enumerate(SEGS):
        rec = sorted(p for p in glob.glob(os.path.join(HERE, "voice", f"seg{i}.*")) if not p.endswith(".gitkeep"))
        if rec:                                   # real Moroccan recording wins over TTS
            print(f"  using recorded voice {rec[0]}")
            a = load_audio(rec[0], REC_FILTER)
            a = trim(a, 0.08 * np.abs(a).max())
            clips.append(a); continue
        text = " ".join(s["tts"].split("|"))
        wav = os.path.join(vdir, f"seg{i}.wav")
        run([sys.executable, "-m", "piper", "-m", VOICE, "--length-scale", "0.74",
             "--noise-scale", "0.7", "--noise-w-scale", "0.9", "--sentence-silence", "0.12", "-f", wav],
            input=text.encode(), capture_output=True)
        a = trim(load_audio(wav, vf))
        clips.append(a)
    return clips


# --------------------------------------------------------------- timeline ---
def build_timeline(vclips):
    t = INTRO; segs = []
    for s, a in zip(SEGS, vclips):
        d = len(a) / SR
        tts_chunks = s["tts"].split("|"); subs = s["sub"].split("|")
        assert len(tts_chunks) == len(subs)
        lens = np.array([max(4, len(c)) for c in tts_chunks], float)
        bounds = t + d * np.concatenate([[0], np.cumsum(lens) / lens.sum()])
        chunks = [(subs[j], bounds[j], bounds[j + 1]) for j in range(len(subs))]
        segs.append(dict(scene=s["scene"], start=t, end=t + d, chunks=chunks))
        t += d + GAP
    total = segs[-1]["end"] + OUTRO
    scenes = []
    for i, s in enumerate(segs):
        a = 0.0 if i == 0 else s["start"] - 0.15
        b = total if i == len(segs) - 1 else segs[i + 1]["start"] - 0.15
        scenes.append(dict(name=s["scene"], a=a, b=b))
    return segs, scenes, total


# ------------------------------------------------------------------ music ---
def synth_music(total, cuts, sub_time):
    n = int(total * SR); t = np.arange(n) / SR
    rng = np.random.default_rng(7)
    BPM = 128; beat = 60 / BPM; bar = 4 * beat
    music = np.zeros((n, 2), np.float32)
    start = INTRO - 0.02          # beat drops on the hook boom

    def add(sig, at, gain=1.0, pan=0.0):
        i = int(at * SR)
        if i >= n or i + len(sig) <= 0:
            return
        sig = sig[: n - i]
        l, r = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        music[i:i + len(sig), 0] += sig * gain * l * 1.414
        music[i:i + len(sig), 1] += sig * gain * r * 1.414

    tk = np.arange(int(0.4 * SR)) / SR
    kick = np.sin(2 * np.pi * (45 * tk + 110 / 28 * (1 - np.exp(-28 * tk)))) * np.exp(-6 * tk)
    kick += 0.3 * rng.standard_normal(len(tk)) * np.exp(-120 * tk)
    tc = np.arange(int(0.25 * SR)) / SR
    noise = rng.standard_normal(len(tc))
    clap = np.diff(np.concatenate([[0], noise])) * np.exp(-16 * tc) * 0.5
    th = np.arange(int(0.06 * SR)) / SR
    hat = np.diff(np.concatenate([[0], rng.standard_normal(len(th))])) * np.exp(-70 * th) * 0.25

    roots = [55.0, 43.65, 65.41, 49.0]        # Am  F  C  G
    chords = [[220, 261.6, 329.6], [174.6, 220, 261.6], [261.6, 329.6, 392], [196, 246.9, 293.7]]
    nb = int((total - start) / beat) + 2
    for b in range(nb):
        bt = start + b * beat
        if bt > total - 1.0:
            break
        add(kick, bt, 0.9)
        if b % 2 == 1:
            add(clap, bt, 0.5)
        for h in range(2):
            add(hat, bt + h * beat / 2 + beat / 4, 0.7 if h else 0.5, pan=0.35 if h else -0.35)
        ci = int((bt - start) / bar) % 4
        # bass (8ths, pumped)
        for e in range(2):
            ln = int(beat / 2 * SR); te = np.arange(ln) / SR
            f = roots[ci]
            saw = sum(np.sin(2 * np.pi * f * k * te) / k for k in range(1, 6))
            env = (1 - np.exp(-60 * te)) * np.exp(-3 * te)
            add(saw * env * 0.28, bt + e * beat / 2)
        # arp (16ths)
        for q in range(4):
            ln = int(beat / 4 * SR); tq = np.arange(ln) / SR
            f = chords[ci][(b * 4 + q) % 3] * 2
            sq = np.tanh(3 * np.sin(2 * np.pi * f * tq)) * np.exp(-9 * tq)
            add(sq * 0.05, bt + q * beat / 4, pan=0.5 * math.sin(b + q))
    # pad
    pad = np.zeros(n)
    for ci_bar in range(int(total / bar) + 1):
        a = int((start + ci_bar * bar) * SR); bb = min(n, a + int(bar * SR))
        if a >= n or a < 0:
            continue
        tt = t[a:bb] - t[a]
        for f in chords[ci_bar % 4]:
            pad[a:bb] += np.sin(2 * np.pi * f * tt) * 0.018 + np.sin(2 * np.pi * f * 1.004 * tt) * 0.018
    music += pad[:, None].astype(np.float32)
    fade = np.clip((total - 1.2 - t) / 1.2, 0, 1)
    music *= fade[:, None]

    sfx = np.zeros((n, 2), np.float32)

    def add_sfx(sig, at, g=1.0):
        i = int(at * SR)
        if i < 0:
            sig = sig[-i:]; i = 0
        sig = sig[: n - i]
        sfx[i:i + len(sig)] += (sig * g)[:, None]

    def whoosh(d=0.5):
        m = int(d * SR); x = rng.standard_normal(m)
        y = np.zeros(m); s = 0.0
        cut = np.linspace(0.02, 0.6, m) ** 2
        for k in range(m):
            s += cut[k] * (x[k] - s); y[k] = s
        env = np.sin(np.linspace(0, np.pi, m)) ** 2
        return y * env * 0.9

    def impact(d=1.2):
        m = int(d * SR); tt = np.arange(m) / SR
        return (np.sin(2 * np.pi * (38 * tt + 60 / 10 * (1 - np.exp(-10 * tt)))) * np.exp(-3.5 * tt)
                + 0.5 * rng.standard_normal(m) * np.exp(-30 * tt)) * 0.9

    def riser(d=0.55):
        m = int(d * SR); tt = np.arange(m) / SR
        f = 200 + 1800 * (tt / d) ** 2
        return (np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.25 + rng.standard_normal(m) * 0.15) * (tt / d) ** 2

    def glitch():
        m = int(0.09 * SR); tt = np.arange(m) / SR
        return np.sign(np.sin(2 * np.pi * 900 * tt)) * 0.18 * (tt < 0.03) + np.sign(np.sin(2 * np.pi * 1500 * tt)) * 0.12 * (tt > 0.05)

    def click():
        m = int(0.05 * SR); tt = np.arange(m) / SR
        return rng.standard_normal(m) * np.exp(-200 * tt) * 0.6

    def bell():
        m = int(1.6 * SR); tt = np.arange(m) / SR
        return sum(a * np.sin(2 * np.pi * f * tt) * np.exp(-d * tt)
                   for f, a, d in [(1318, .35, 3), (2637, .2, 4.5), (3951, .12, 6), (5274, .06, 8)])

    add_sfx(riser(), INTRO - 0.55, 0.9)
    add_sfx(impact(), INTRO - 0.02, 1.0)
    for c in cuts:
        add_sfx(whoosh(), c - 0.3, 0.55)
        add_sfx(glitch(), c, 0.5)
    for c in cuts[-2:]:
        add_sfx(impact(), c, 0.6)
    add_sfx(click(), sub_time, 1.0)
    add_sfx(bell(), sub_time + 0.45, 0.8)
    return music, sfx


def mix_audio(vclips, segs, total, scenes, sub_time):
    n = int(total * SR)
    voice = np.zeros(n, np.float32)
    for s, a in zip(segs, vclips):
        i = int(s["start"] * SR); voice[i:i + len(a)] += a[: n - i]
    voice *= 0.16 / (np.sqrt(np.mean(voice[voice != 0] ** 2)) + 1e-9)   # speech RMS ~ -16 dBFS
    cuts = [sc["a"] for sc in scenes[1:]]
    music, sfx = synth_music(total, cuts, sub_time)
    # sidechain ducking: music drops under the narration
    env = np.abs(voice)
    k = int(0.25 * SR)
    env = np.convolve(env, np.ones(k) / k, mode="same")
    duck = 1 - 0.55 * np.clip(env / 0.05, 0, 1)
    music_rms = np.sqrt(np.mean(music ** 2)) + 1e-9
    music *= (0.16 * 10 ** (-10 / 20) / music_rms)                    # music bed ~10 dB under voice
    music *= duck[:, None]
    sfx *= 0.16 / 0.25
    mix = music + sfx + voice[:, None]
    mix /= max(1.0, np.abs(mix).max() / 0.95)
    stems = dict(voice=voice, music=music.mean(1), sfx=sfx.mean(1))
    return mix.astype(np.float32), stems


def write_wav(path, x):
    x = np.atleast_2d(x.T).T if x.ndim == 1 else x
    ch = 1 if x.ndim == 1 else x.shape[1]
    run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", str(ch), "-i", "-", path],
        input=x.astype(np.float32).tobytes())


# ---------------------------------------------------------------- visuals ---
def font(p, s):
    return ImageFont.truetype(p, s)


def vgrad(w, h, stops):
    ys = np.linspace(0, 1, h)
    pos = [p for p, _ in stops]; cols = np.array([c for _, c in stops], float)
    img = np.stack([np.interp(ys, pos, cols[:, k]) for k in range(3)], 1)
    return Image.fromarray(np.repeat(img[:, None, :], w, 1).astype(np.uint8), "RGB")


def palm(d, x, y, h, col, lean=1, rng=None):
    pts = []
    for i in range(30):
        u = i / 29
        pts.append((x + lean * 90 * u ** 2, y - h * u))
    for i in range(len(pts) - 1):
        wdt = int(34 - 18 * i / 29)
        d.line([pts[i], pts[i + 1]], fill=col, width=wdt)
    tx, ty = pts[-1]
    for ang in range(-170, 20, 24):
        a = math.radians(ang + (rng.uniform(-8, 8) if rng else 0))
        L = h * 0.42
        poly_top, poly_bot = [], []
        for j in range(16):
            u = j / 15
            px = tx + math.cos(a) * L * u
            py = ty + math.sin(a) * L * u + 120 * u ** 2
            wdt = 26 * math.sin(math.pi * min(1, u * 1.2))
            poly_top.append((px, py - wdt)); poly_bot.append((px, py + wdt * 0.4))
        d.polygon(poly_top + poly_bot[::-1], fill=col)


def skyline(d, rng, y0, w, col, hmin, hmax, win_cols, win_p=0.35, x0=0, x1=None):
    x = x0; x1 = w if x1 is None else x1
    while x < x1:
        bw = rng.integers(60, 170); bh = rng.integers(hmin, hmax)
        d.rectangle([x, y0 - bh, x + bw, y0], fill=col)
        if rng.random() < 0.3:
            d.rectangle([x + bw * 0.4, y0 - bh - 60, x + bw * 0.45, y0 - bh], fill=col)
        for wy in range(int(y0 - bh + 20), int(y0 - 10), 26):
            for wx in range(int(x + 12), int(x + bw - 16), 22):
                if rng.random() < win_p:
                    d.rectangle([wx, wy, wx + 9, wy + 13], fill=tuple(win_cols[rng.integers(len(win_cols))]))
        x += bw + rng.integers(2, 14)


def glow(img, radius=18, strength=1.6):
    g = img.filter(ImageFilter.GaussianBlur(radius))
    return ImageChops.add(img, ImageChops.multiply(g, Image.new("RGB", img.size, tuple([int(255 * min(1, strength / 2))] * 3))))


def bg_sunset(rng, blur=0):
    img = vgrad(BW, BH, [(0, (20, 4, 48)), (0.3, (90, 20, 110)), (0.5, (255, 70, 120)), (0.6, (255, 170, 80)), (0.6001, (30, 10, 60)), (1, (8, 4, 25))])
    d = ImageDraw.Draw(img, "RGBA")
    cx, cy, r = BW // 2, int(BH * 0.47), 360
    sun = vgrad(2 * r, 2 * r, [(0, (255, 235, 120)), (1, (255, 60, 140))])
    m = Image.new("L", (2 * r, 2 * r), 0); md = ImageDraw.Draw(m); md.ellipse([0, 0, 2 * r - 1, 2 * r - 1], fill=255)
    for k in range(9):
        yy = int(r + k * k * 6 + k * 18); md.rectangle([0, yy, 2 * r, yy + 4 + k * 3], fill=0)
    img.paste(sun, (cx - r, cy - r), m)
    hy = int(BH * 0.6)
    skyline(d, rng, hy, BW, (30, 8, 50), 120, 520, [(255, 120, 200), (255, 200, 120)], 0.12)
    for k in range(70):
        yy = hy + 10 + k * 12 + rng.integers(0, 6)
        ww = int((380 - k * 4) * rng.uniform(0.4, 1.1))
        if ww > 10:
            d.rectangle([cx - ww // 2 + rng.integers(-30, 30), yy, cx + ww // 2, yy + 3], fill=(255, 120 + rng.integers(0, 100), 140, 140))
    palm(d, 120, BH + 40, 1500, (10, 3, 22), 1, rng)
    palm(d, BW - 160, BH + 40, 1250, (10, 3, 22), -1, rng)
    img = glow(img, 25, 1.2)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return img


def bg_city(rng):
    img = vgrad(BW, BH, [(0, (4, 2, 18)), (0.5, (35, 10, 70)), (0.72, (80, 20, 90)), (0.72, (12, 8, 22)), (1, (5, 3, 12))])
    d = ImageDraw.Draw(img, "RGBA")
    for _ in range(160):
        x, y = rng.integers(0, BW), rng.integers(0, int(BH * 0.4))
        d.point((x, y), fill=(255, 255, 255, int(rng.integers(80, 255))))
    hy = int(BH * 0.72)
    skyline(d, rng, hy, BW, (40, 18, 80), 500, 1150, [(120, 90, 200)], 0.15)
    skyline(d, rng, hy, BW, (14, 6, 30), 250, 800, [(255, 210, 120), (45, 226, 255), (255, 45, 149)], 0.4)
    for _ in range(9):
        x, y = rng.integers(40, BW - 240), rng.integers(int(hy - 700), hy - 200)
        c = [PINK, CYAN, YEL][rng.integers(3)]
        d.rounded_rectangle([x, y, x + rng.integers(90, 220), y + 46], 10, outline=c + (255,), width=7)
    # road
    vx = BW // 2
    for k in range(-6, 7):
        d.line([(vx, hy), (vx + k * 420, BH)], fill=(60, 40, 90, 160), width=4)
    img = glow(img, 16, 1.5)
    return img


def bg_hook(rng):
    img = vgrad(BW, BH, [(0, (0, 0, 0)), (0.55, (40, 6, 60)), (1, (8, 2, 20))])
    d = ImageDraw.Draw(img, "RGBA")
    skyline(d, rng, int(BH * 0.55), BW, (25, 6, 45), 100, 500, [(255, 45, 149)], 0.08)
    return glow(img, 20, 1.0)


def bg_heroes(rng):
    a = np.array(vgrad(BW, BH, [(0, (5, 30, 60)), (1, (2, 8, 20))])).astype(float)
    b = np.array(vgrad(BW, BH, [(0, (70, 8, 50)), (1, (20, 2, 15))])).astype(float)
    yy, xx = np.mgrid[0:BH, 0:BW]
    m = (xx + (yy - BH / 2) * 0.35 > BW / 2)[..., None]
    img = Image.fromarray(np.where(m, b, a).astype(np.uint8))
    d = ImageDraw.Draw(img, "RGBA")
    for k in range(7):
        x = rng.integers(0, BW)
        d.polygon([(x, 0), (x + 120, 0), (x - 500, BH), (x - 650, BH)], fill=(255, 255, 255, 10))
    palm(d, 60, BH + 40, 1100, (2, 2, 10), 1, rng)
    palm(d, BW - 80, BH + 40, 1350, (2, 2, 10), -1, rng)
    return img


def bg_live(rng):
    img = vgrad(BW, BH, [(0, (25, 0, 10)), (0.5, (90, 5, 40)), (1, (10, 0, 15))])
    d = ImageDraw.Draw(img, "RGBA")
    for _ in range(40):
        x, y, r = rng.integers(0, BW), rng.integers(0, BH), rng.integers(30, 160)
        c = [PINK, RED, CYAN][rng.integers(3)]
        d.ellipse([x - r, y - r, x + r, y + r], fill=c + (int(rng.integers(20, 60)),))
    return img.filter(ImageFilter.GaussianBlur(14))


def clip_frames(name, dur):
    """Use user footage clips/<name>.(mp4|mov|webm|jpg|png) if present."""
    for ext in ("mp4", "mov", "webm", "mkv", "jpg", "jpeg", "png"):
        p = os.path.join(CLIPS, f"{name}.{ext}")
        if os.path.exists(p):
            break
    else:
        return None
    out = os.path.join(CACHE, "clip_" + name); shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    vf = f"scale={BW}:{BH}:force_original_aspect_ratio=increase,crop={BW}:{BH},eq=brightness=-0.06:saturation=1.15"
    if ext in ("jpg", "jpeg", "png"):
        run(["ffmpeg", "-v", "error", "-y", "-i", p, "-vf", vf, "-frames:v", "1", f"{out}/f00001.jpg"])
    else:
        run(["ffmpeg", "-v", "error", "-y", "-stream_loop", "-1", "-i", p, "-t", f"{dur + 0.5:.2f}", "-an",
             "-vf", f"fps={FPS}," + vf, "-q:v", "3", f"{out}/f%05d.jpg"])
    files = sorted(glob.glob(out + "/*.jpg"))
    print(f"  using footage {p} ({len(files)} frames)")
    return files


# text sprites ----------------------------------------------------------------
_cache = {}


def text_sprite(txt, size, color, fontp=AR, stroke=10, glow_r=22, rtl=True, grad=None):
    key = (txt, size, color, fontp, stroke, glow_r, grad)
    if key in _cache:
        return _cache[key]
    f = font(fontp, size)
    kw = dict(direction="rtl", language="ar") if rtl else {}
    bb = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), txt, font=f, stroke_width=stroke, **kw)
    pad = glow_r * 3 + 10
    w, h = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
    pos = (pad - bb[0], pad - bb[1])
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).text(pos, txt, font=f, fill=255, **kw)
    smask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(smask).text(pos, txt, font=f, fill=255, stroke_width=stroke, **kw)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if glow_r:
        g = smask.filter(ImageFilter.GaussianBlur(glow_r)).point(lambda v: min(255, int(v * 1.8)))
        out.paste(Image.new("RGBA", (w, h), color + (255,)), (0, 0), g)
    out.paste(Image.new("RGBA", (w, h), (0, 0, 0, 255)), (0, 0), smask)
    if grad:
        fill = vgrad(w, h, [(0.25, grad[0]), (0.75, grad[1])]).convert("RGBA")
    else:
        fill = Image.new("RGBA", (w, h), color + (255,))
    out.paste(fill, (0, 0), mask)
    _cache[key] = out
    return out


def sub_size(txt, size=76, max_w=W - 120):
    """Largest subtitle font size (<= size) whose text fits inside the safe width."""
    while size > 40:
        bb = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox(
            (0, 0), txt, font=font(AR, size), stroke_width=9, direction="rtl", language="ar")
        if bb[2] - bb[0] <= max_w:
            return size
        size -= 2
    return size


def ease_out_back(x):
    c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def place(frame, spr, cx, cy, t0, t, scale_from=1.5, dur=0.35, fade_out=None):
    if t < t0:
        return
    u = min(1, (t - t0) / dur)
    s = scale_from + (1 - scale_from) * ease_out_back(u)
    a = min(1, (t - t0) / 0.12)
    if fade_out is not None and t > fade_out - 0.2:
        a *= max(0, (fade_out - t) / 0.2)
    if a <= 0:
        return
    sp = spr
    if abs(s - 1) > 0.01:
        sp = spr.resize((max(1, int(spr.width * s)), max(1, int(spr.height * s))), Image.BILINEAR)
    if a < 1:
        sp = sp.copy(); sp.putalpha(sp.getchannel("A").point(lambda v: int(v * a)))
    frame.alpha_composite(sp, (int(cx - sp.width / 2), int(cy - sp.height / 2)))


# ------------------------------------------------------------ frame render ---
G = {}


def setup(segs, scenes, total, sub_time):
    rng = np.random.default_rng(3)
    makers = dict(hook=bg_hook, date=bg_sunset, city=bg_city, heroes=bg_heroes, live=bg_live,
                  cta=lambda r: bg_sunset(r, blur=10))
    G["bg"] = {}; G["clip"] = {}
    for sc in scenes:
        G["bg"][sc["name"]] = makers[sc["name"]](rng)
        G["clip"][sc["name"]] = clip_frames(sc["name"], sc["b"] - sc["a"])
    G.update(segs=segs, scenes=scenes, total=total, sub_time=sub_time)
    # vignette
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    v = np.clip((r - 0.65) / 0.75, 0, 1) ** 1.5 * 220
    G["vig"] = Image.fromarray(np.dstack([np.zeros((H, W, 3), np.uint8), v.astype(np.uint8)]), "RGBA")
    sl = np.zeros((H, W, 4), np.uint8); sl[::4, :, 3] = 18
    G["scan"] = Image.fromarray(sl, "RGBA")
    G["rng_streak"] = [(np.random.default_rng(i).random(4)) for i in range(26)]


def scene_at(t):
    for i, sc in enumerate(G["scenes"]):
        if sc["a"] <= t < sc["b"]:
            return i, sc
    return len(G["scenes"]) - 1, G["scenes"][-1]


def camera(bg, u, d, seed, shake=0.0):
    z = 1.0 + 0.16 * (u / d) + 0.14 * math.exp(-u * 7)
    cw, ch = BW / (z * K) * 1.0, BH / (z * K)
    cw, ch = W * K / z / 1.0, H * K / z
    cw, ch = min(BW, cw), min(BH, ch)
    px = (BW - cw) / 2 + math.sin(u * 0.6 + seed) * (BW - cw) * 0.3
    py = (BH - ch) / 2 + math.cos(u * 0.5 + seed) * (BH - ch) * 0.25
    if shake:
        px += math.sin(u * 91) * shake; py += math.cos(u * 77) * shake
    px = min(max(0, px), BW - cw); py = min(max(0, py), BH - ch)
    return bg.resize((W, H), Image.BILINEAR, box=(px, py, px + cw, py + ch))


def render(fi):
    t = fi / FPS
    idx, sc = scene_at(t)
    name = sc["name"]; u = t - sc["a"]; d = sc["b"] - sc["a"]
    shake = 0
    if name == "hook":
        shake = 28 * math.exp(-max(0, t - INTRO) * 5) if t >= INTRO else 6 * math.sin(t * 40)
    clip = G["clip"][name]
    if clip:
        src = Image.open(clip[min(len(clip) - 1, int(u * FPS))]).convert("RGB")
        base = camera(src, u, d, idx, shake)
    else:
        base = camera(G["bg"][name], u, d, idx, shake)
    fr = base.convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); od = ImageDraw.Draw(ov)

    # ---- scene specific animated layers
    if name == "hook" and not clip:
        hy = 1080; off = (t * 2.2) % 1
        for k in range(-12, 13):
            od.line([(W / 2, hy), (W / 2 + k * 260, H)], fill=PINK + (150,), width=3)
        for k in range(14):
            y = hy + (H - hy) * ((k + off) / 14) ** 2.2
            od.line([(0, y), (W, y)], fill=PINK + (int(60 + 150 * (k + off) / 14),), width=3)
    if name == "city":
        for i, r in enumerate(G["rng_streak"]):
            y = 1380 + r[0] * 520
            spd = 900 + r[1] * 1600
            dirn = 1 if i % 2 else -1
            L = 120 + r[2] * 380
            x = ((t * spd + r[3] * 3000) % (W + L * 2)) - L
            if dirn < 0:
                x = W - x
            c = (255, 240, 200) if dirn > 0 else (255, 40, 60)
            od.line([(x, y), (x - dirn * L, y)], fill=c + (230,), width=int(4 + r[2] * 6))
    if name == "date" or name == "cta":
        for k in range(40):
            rr = np.random.default_rng(k).random(3)
            y = H - ((t * (60 + rr[1] * 120) + rr[2] * H) % H)
            x = rr[0] * W + math.sin(t * 2 + k) * 20
            od.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 220, 160, 150))
    if name == "heroes":
        x = (u * 900) % (W * 2) - W / 2
        od.polygon([(x, 0), (x + 140, 0), (x - 260, H), (x - 400, H)], fill=(255, 255, 255, 28))
    fr.alpha_composite(ov)
    if clip:
        fr.alpha_composite(Image.new("RGBA", (W, H), (10, 0, 25, 70)))

    # ---- titles
    seg = G["segs"][idx]
    if name == "hook":
        if t >= INTRO:
            spr = text_sprite("GTA 6", 300, PINK, LAT, 12, 34, False, grad=((255, 230, 120), (255, 45, 149)))
            jit = 10 * math.exp(-(t - INTRO) * 4)
            place(fr, spr, W / 2 + jit * math.sin(t * 70), 640, INTRO, t, 2.2, 0.4)
            place(fr, text_sprite("قربات بزاف", 70, CYAN, AR, 6, 14), W / 2, 880, INTRO + 0.5, t, 1.4)
    elif name == "date":
        place(fr, text_sprite("الخميس", 90, WHITE, AR, 7, 16), W / 2, 420, sc["a"], t)
        place(fr, text_sprite("19", 330, YEL, LAT, 12, 30, False, grad=((255, 240, 150), (255, 120, 60))), W / 2, 660, sc["a"] + 0.15, t, 1.8)
        place(fr, text_sprite("نونبر 2026", 110, PINK, AR, 8, 20), W / 2, 900, sc["a"] + 0.3, t)
        place(fr, text_sprite("PS5  •  XBOX SERIES X|S", 52, WHITE, LAT, 5, 10, False), W / 2, 1030, sc["a"] + 0.5, t, 1.2)
    elif name == "city":
        spr = text_sprite("VICE CITY", 170, PINK, LAT, 10, 30, False, grad=((255, 120, 220), (255, 45, 149)))
        flick = 1 if (int(t * 12) % 9) else 0.5
        if flick > 0.6:
            place(fr, spr, W / 2, 560, sc["a"] + 0.1, t, 1.6)
        place(fr, text_sprite("رجعنا ليها", 80, CYAN, AR, 6, 14), W / 2, 760, sc["a"] + 0.4, t)
    elif name == "heroes":
        place(fr, text_sprite("JASON", 190, CYAN, LAT, 10, 28, False), W / 2 - 40, 470, sc["a"] + 0.05, t, 1.7)
        place(fr, text_sprite("×", 120, WHITE, LAT, 6, 10, False), W / 2, 640, sc["a"] + 0.25, t, 2.0)
        place(fr, text_sprite("LUCIA", 190, PINK, LAT, 10, 28, False), W / 2 + 40, 800, sc["a"] + 0.45, t, 1.7)
    elif name == "live":
        pulse = 1 + 0.06 * math.sin(t * 9)
        badge = text_sprite("● LIVE", 170, RED, LAT, 10, 40, False, grad=((255, 255, 255), (255, 220, 220)))
        if t >= sc["a"]:
            b2 = badge.resize((int(badge.width * pulse), int(badge.height * pulse)), Image.BILINEAR)
            box = Image.new("RGBA", (b2.width - 60, int(b2.height * 0.62)), (0, 0, 0, 0))
            ImageDraw.Draw(box).rounded_rectangle([0, 0, box.width - 1, box.height - 1], 40, fill=RED + (235,))
            place(fr, box, W / 2, 560, sc["a"], t, 1.6)
            place(fr, b2, W / 2, 560, sc["a"], t, 1.6)
        place(fr, text_sprite("لايف فالقناة", 120, WHITE, AR, 8, 20), W / 2, 800, sc["a"] + 0.3, t)
        place(fr, text_sprite("نهار الخروج • 19 نونبر", 70, YEL, AR, 6, 14), W / 2, 960, sc["a"] + 0.5, t)
    elif name == "cta":
        st = G["sub_time"]
        place(fr, text_sprite("@esjine", 150, CYAN, LAT, 10, 28, False), W / 2, 430, sc["a"] + 0.05, t, 1.6)
        clicked = t >= st
        bw_, bh_ = 640, 170
        btn = Image.new("RGBA", (bw_ + 40, bh_ + 40), (0, 0, 0, 0)); bd = ImageDraw.Draw(btn)
        bd.rounded_rectangle([20, 20, 20 + bw_, 20 + bh_], 85, fill=(70, 70, 80, 255) if clicked else (230, 0, 35, 255))
        lab = text_sprite("صافي ✓" if clicked else "دير أبوني", 80, WHITE, AR, 0, 0)
        btn.alpha_composite(lab, ((btn.width - lab.width) // 2, (btn.height - lab.height) // 2))
        press = 0.9 if st <= t < st + 0.12 else 1.0
        if press != 1.0:
            btn = btn.resize((int(btn.width * press), int(btn.height * press)))
        place(fr, btn, W / 2 - 90, 720, sc["a"] + 0.3, t, 1.5)
        # bell
        bell = Image.new("RGBA", (200, 200), (0, 0, 0, 0)); bl = ImageDraw.Draw(bell)
        bcol = YEL if clicked else WHITE
        bl.pieslice([40, 30, 160, 150], 180, 360, fill=bcol); bl.rectangle([40, 90, 160, 135], fill=bcol)
        bl.polygon([(28, 150), (172, 150), (160, 128), (40, 128)], fill=bcol); bl.ellipse([85, 145, 115, 175], fill=bcol)
        bl.ellipse([88, 15, 112, 39], fill=bcol)
        if t >= st + 0.45:
            ang = 25 * math.sin((t - st) * 30) * math.exp(-(t - st - 0.45) * 2.5)
            bell = bell.rotate(ang, Image.BICUBIC)
        place(fr, bell, W / 2 + 330, 720, sc["a"] + 0.5, t, 1.5)
        # cursor
        if sc["a"] + 0.8 <= t < st + 0.9:
            p = min(1, (t - sc["a"] - 0.8) / max(0.1, st - sc["a"] - 0.8))
            p = 1 - (1 - p) ** 3
            cx, cy = W * 0.85 + (W / 2 - 40 - W * 0.85) * p, 1250 + (760 - 1250) * p
            cur = [(cx, cy), (cx, cy + 90), (cx + 24, cy + 68), (cx + 44, cy + 110), (cx + 60, cy + 102), (cx + 40, cy + 62), (cx + 72, cy + 60)]
            cd = ImageDraw.Draw(fr); cd.polygon(cur, fill=WHITE, outline=(0, 0, 0), width=4)
            if st <= t < st + 0.3:
                rr = 30 + (t - st) * 400
                cd.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(255, 255, 255), width=5)
        place(fr, text_sprite("كليكيو على الجرس", 70, YEL, AR, 6, 14), W / 2, 930, sc["a"] + 0.7, t)

    # ---- subtitles (synced to narration chunks)
    for s in G["segs"]:
        for txt, a, b in s["chunks"]:
            if a <= t < b + 0.05:
                spr = text_sprite(txt, sub_size(txt), WHITE, AR, 9, 0)
                pill = Image.new("RGBA", (spr.width + 20, spr.height - 10), (0, 0, 0, 0))
                ImageDraw.Draw(pill).rounded_rectangle([0, 0, pill.width - 1, pill.height - 1], 36, fill=(0, 0, 0, 140))
                place(fr, pill, W / 2, 1340, a, t, 1.12, 0.12)
                hl = text_sprite(txt, sub_size(txt), YEL if (s["chunks"].index((txt, a, b)) % 2) else WHITE, AR, 9, 0)
                place(fr, hl, W / 2, 1340, a, t, 1.25, 0.15)

    # ---- global overlays
    fr.alpha_composite(G["scan"]); fr.alpha_composite(G["vig"])
    fr.alpha_composite(text_sprite("@esjine", 44, WHITE, LAT, 3, 0, False), (60, 70))
    # transitions: flash + RGB split around each cut
    for c in [s["a"] for s in G["scenes"][1:]] + [INTRO]:
        dt = t - c
        if -0.1 < dt < 0.25:
            k = math.exp(-abs(dt) * 18)
            arr = np.array(fr.convert("RGB")).astype(np.int16)
            sh = int(40 * k)
            if sh:
                arr[..., 0] = np.roll(arr[..., 0], sh, 1); arr[..., 2] = np.roll(arr[..., 2], -sh, 1)
            arr = arr + int(230 * k * (1 if dt >= 0 else 0.5))
            fr = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")
    # fade in/out
    tot = G["total"]
    f = min(1, t / 0.25, (tot - t) / 0.6)
    out = np.array(fr.convert("RGB"))
    if f < 1:
        out = (out * max(0, f)).astype(np.uint8)
    return out.tobytes()


# ------------------------------------------------------------------- main ---
def main():
    os.makedirs(CACHE, exist_ok=True)
    print("1/4 voice (Darija TTS) ...")
    vclips = fit_length(make_voice())
    segs, scenes, total = build_timeline(vclips)
    cta = segs[-1]
    sub_time = cta["chunks"][1][1] + 0.35            # click "subscribe" while it is said
    print(f"    duration {total:.2f}s")
    if not 30 <= total <= 45:
        print("WARNING: duration outside 30-45 s")
    print("2/4 music + sfx + mix ...")
    mix, stems = mix_audio(vclips, segs, total, scenes, sub_time)
    wav = os.path.join(CACHE, "mix.wav"); write_wav(wav, mix)
    write_wav(os.path.join(CACHE, "voice_only.wav"), stems["voice"])
    write_wav(os.path.join(CACHE, "music_only.wav"), stems["music"])
    json.dump(dict(total=total, segs=segs, scenes=scenes), open(os.path.join(CACHE, "timeline.json"), "w"),
              ensure_ascii=False, indent=1)
    print("3/4 frames + encode ...")
    setup(segs, scenes, total, sub_time)
    nf = int(round(total * FPS))
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-i", wav, "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-profile:v", "high", "-pix_fmt", "yuv420p",
         "-g", str(FPS), "-bf", "2", "-r", str(FPS),
         "-af", "loudnorm=I=-14:TP=-1.5:LRA=9,aresample=48000",
         "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-ac", "2",
         "-movflags", "+faststart", "-shortest", OUT], stdin=subprocess.PIPE)
    with mp.get_context("fork").Pool(os.cpu_count()) as pool:
        for i, buf in enumerate(pool.imap(render, range(nf), chunksize=4)):
            enc.stdin.write(buf)
            if i % 150 == 0:
                print(f"    frame {i}/{nf}")
    enc.stdin.close(); enc.wait()
    if enc.returncode:
        sys.exit("encode failed")
    print("4/4 done ->", OUT)


if __name__ == "__main__":
    main()
