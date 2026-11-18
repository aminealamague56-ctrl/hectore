"""Synthesize an original 150 BPM gaming/trap track + speed/wind SFX, timed to the edit."""
import numpy as np, wave, sys
from scipy.signal import butter, sosfilt

SR = 44100
BPM = 150
BEAT = 60 / BPM          # 0.4 s
BAR = BEAT * 4           # 1.6 s
DUR = 16 * BAR           # 25.6 s
N = int(DUR * SR)
rng = np.random.default_rng(7)

def t_(sec): return np.arange(int(sec * SR)) / SR
def lp(x, f, o=2): return sosfilt(butter(o, f, 'low', fs=SR, output='sos'), x)
def hp(x, f, o=2): return sosfilt(butter(o, f, 'high', fs=SR, output='sos'), x)
def bp(x, lo, hi, o=2): return sosfilt(butter(o, [lo, hi], 'band', fs=SR, output='sos'), x)

def add(bus, sig, at, gain=1.0, pan=0.0):
    i = int(at * SR)
    if i >= N: return
    sig = sig[: N - i]
    if sig.ndim == 1:
        l, r = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
        sig = np.stack([sig * l, sig * r], 1) * 1.414
    bus[i:i + len(sig)] += sig * gain

def kick():
    t = t_(0.45)
    f = 45 + 110 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.tanh(2.2 * np.sin(ph) * np.exp(-t * 7))

def snare():
    t = t_(0.3)
    n = bp(rng.standard_normal(len(t)), 900, 7000) * np.exp(-t * 16)
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30)
    return 0.8 * n + 0.5 * tone

def hat(open_=False):
    t = t_(0.25 if open_ else 0.06)
    return hp(rng.standard_normal(len(t)), 7000) * np.exp(-t * (12 if open_ else 70)) * 0.45

def bass808(freq, length, glide_from=None):
    t = t_(length)
    f = np.full_like(t, freq)
    if glide_from:
        f = freq + (glide_from - freq) * np.exp(-t * 18)
    ph = 2 * np.pi * np.cumsum(f) / SR
    env = np.minimum(1, t * 200) * np.exp(-t * 1.6)
    return np.tanh(1.8 * np.sin(ph) * env) * 0.9

def saw_lead(freq, length):
    t = t_(length)
    s = sum(((t * freq * d) % 1) * 2 - 1 for d in (1, 1.006, 0.994)) / 3
    env = np.minimum(1, t * 120) * np.exp(-t * 5)
    return lp(s * env, 3200)

def riser(length):
    t = t_(length)
    prog = t / length
    n = rng.standard_normal(len(t))
    out = np.zeros_like(t)
    seg = 2048
    for i in range(0, len(t), seg):
        c = 300 + 9000 * prog[i] ** 2
        out[i:i + seg] = bp(n[i:i + seg], c * 0.7, min(c * 1.4, 20000))
    f = 200 + 1400 * prog ** 2
    tone = (((np.cumsum(f) / SR) % 1) * 2 - 1) * 0.25
    return (out * 0.8 + lp(tone, 4000)) * prog ** 1.5

def impact():
    t = t_(1.6)
    f = 30 + 90 * np.exp(-t * 10)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.5)
    crash = hp(rng.standard_normal(len(t)), 3000) * np.exp(-t * 3) * 0.35
    return np.tanh(1.5 * boom) + crash

def whoosh(length=0.5, up=True):
    t = t_(length)
    p = t / length
    n = rng.standard_normal(len(t))
    out = np.zeros_like(t)
    seg = 1024
    for i in range(0, len(t), seg):
        c = 500 + 5000 * (p[i] if up else 1 - p[i])
        out[i:i + seg] = bp(n[i:i + seg], c * 0.6, c * 1.5)
    env = np.sin(np.pi * p) ** 2
    mono = out * env * 1.4
    pan = np.linspace(-0.9, 0.9, len(t))
    return np.stack([mono * np.sqrt(0.5 * (1 - pan)), mono * np.sqrt(0.5 * (1 + pan))], 1) * 1.414

def wind(length, intensity):
    t = t_(length)
    l = lp(rng.standard_normal(len(t)), 700, 1)
    r = lp(rng.standard_normal(len(t)), 700, 1)
    mod = 0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t + rng.random() * 6)
    env = intensity(t) * mod
    return np.stack([l * env, r * env], 1) * 0.9

def motor(length, speed):
    t = t_(length)
    s = speed(t)
    f = 220 + 620 * s
    ph = np.cumsum(f) / SR
    tone = np.sin(2 * np.pi * ph) + 0.3 * np.sin(4 * np.pi * ph) + 0.15 * (((3 * ph) % 1) * 2 - 1)
    return lp(tone, 2500) * (0.08 + 0.12 * s)

def ding():
    t = t_(1.2)
    return sum(np.sin(2 * np.pi * f * t) * a for f, a in ((1318, .5), (1975, .3), (2637, .15))) * np.exp(-t * 4)

def chime():  # original console-like boot chord
    t = t_(2.5)
    out = np.zeros_like(t)
    for k, f in enumerate((349.2, 523.3, 698.5, 880.0, 1046.5)):
        d = int(k * 0.06 * SR)
        tt = t[: len(t) - d]
        out[d:] += np.sin(2 * np.pi * f * tt) * np.exp(-tt * 1.6) * np.minimum(1, tt * 40)
    return lp(out, 6000) * 0.35

def click():
    t = t_(0.03)
    return hp(rng.standard_normal(len(t)), 2000) * np.exp(-t * 300)

music = np.zeros((N, 2))
sfx = np.zeros((N, 2))

# F minor progression (root freqs for 808): F, Db, Eb, C
roots = [43.65, 34.65, 38.89, 32.70]
arp_notes = [[349.2, 415.3, 523.3, 622.3], [277.2, 349.2, 415.3, 523.3],
             [311.1, 392.0, 466.2, 622.3], [261.6, 311.1, 392.0, 523.3]]

def bar_beat(bar, beat): return bar * BAR + beat * BEAT

for bar in range(16):
    chord = bar % 4
    intro = bar < 2
    build1 = bar == 5               # GTA card interlude stutter (8.0-9.6)
    approach = 7 <= bar < 11        # intense
    prebuild = bar == 11            # riser before home
    home = bar >= 12
    # arp lead (16ths)
    for s in range(16):
        f = arp_notes[chord][[0, 1, 2, 3, 2, 1, 3, 2][s % 8]] * (2 if (approach and s % 4 == 3) else 1)
        g = 0.10 if intro else 0.16
        if prebuild: g = 0.12
        add(music, saw_lead(f, BEAT / 2), bar * BAR + s * BEAT / 4, g, pan=0.35 * np.sin(s))
    if intro:
        for b in range(4):
            add(music, hat(), bar_beat(bar, b + 0.5), 0.5)
        continue
    if prebuild:
        continue
    # 808 + kick pattern
    kpos = [0, 1.5, 2.5] if not approach else [0, 0.75, 1.5, 2.5, 3.25]
    for k in kpos:
        add(music, kick(), bar_beat(bar, k), 0.95)
    add(music, bass808(roots[chord] * 2, BAR * 0.9, glide_from=roots[chord] * 3 if bar % 2 else None), bar * BAR, 0.75)
    for b in (1, 3):
        add(music, snare(), bar_beat(bar, b), 0.7)
    # hats: 8ths, with triplet rolls during approach
    step = BEAT / (4 if approach else 2)
    for i in range(int(BAR / step)):
        add(music, hat(), bar * BAR + i * step, 0.55, pan=0.3)
    if approach or home:
        add(music, hat(True), bar_beat(bar, 3.5), 0.4, pan=-0.3)
    if build1:  # stutter: gate the bar later
        pass

# Stutter-gate during GTA interlude bar (bar 5): 16th chops on second half
i0, i1 = int(bar_beat(5, 2) * SR), int(bar_beat(6, 0) * SR)
gate = np.ones(i1 - i0)
seg = int(BEAT / 4 * SR)
for k in range(0, len(gate), seg):
    if (k // seg) % 2: gate[k:k + seg] = 0.15
music[i0:i1] *= gate[:, None]

# Tape-stop on music at arrival (bar 11 end -> 18.8..19.2)
ts0, ts1 = int(18.6 * SR), int(19.2 * SR)
seg_len = ts1 - ts0
rate = np.linspace(1, 0.05, seg_len)
pos = ts0 + np.cumsum(rate)
for ch in range(2):
    music[ts0:ts1, ch] = np.interp(pos, np.arange(N), music[:, ch]) * np.linspace(1, 0, seg_len)

# Lowpass the intro (filtered "radio" feel) and open it up
cut = int(2 * BAR * SR)
for ch in range(2):
    music[:cut, ch] = lp(music[:cut, ch], 900)

# ---- SFX ----
add(sfx, riser(3.0), 0.2, 0.55)                 # into first drop
add(sfx, impact(), 3.2, 0.9)
add(sfx, ding(), 0.25, 0.35)                     # notification at start
add(sfx, riser(1.6), 7.6, 0.45)                  # into GTA card
add(sfx, impact(), 9.6, 0.6)
add(sfx, riser(2.4), 16.4, 0.7)                  # pre-home build
add(sfx, impact(), 19.2, 1.0)
add(sfx, chime(), 19.35, 0.9)
for tt in (21.6, 22.0):
    add(sfx, click(), tt, 0.6)
add(sfx, impact(), 22.4, 0.7)

# whooshes at cuts
for tt in [3.1, 4.0, 4.8, 5.6, 6.4, 7.2, 8.0, 9.5, 11.1, 12.0, 12.8, 13.6, 14.4, 15.2, 16.0, 18.9, 22.3]:
    add(sfx, whoosh(0.45, up=bool(int(tt * 10) % 2)), tt - 0.2, 0.55)

# wind + motor following "speed"
def speed_curve(t):
    tt = t + 3.2
    s = np.interp(tt, [3.2, 4.0, 8.0, 9.6, 11.2, 16.0, 18.4, 19.2], [0.3, 0.6, 0.75, 0.4, 0.8, 1.0, 1.0, 0.0])
    return s
add(sfx, wind(16.0, lambda t: speed_curve(t) * 0.5), 3.2, 1.0)
add(sfx, motor(16.0, speed_curve), 3.2, 1.0)

mix = music * 0.8 + sfx * 0.9
# sidechain-ish pump from kicks is implicit; simple master: soft clip + normalize
mix = np.tanh(mix * 1.1)
mix /= np.max(np.abs(mix)) / 0.95
fade = int(0.6 * SR)
mix[-fade:] *= np.linspace(1, 0, fade)[:, None]

out = sys.argv[1]
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype('<i2').tobytes())
print('wrote', out, DUR)
