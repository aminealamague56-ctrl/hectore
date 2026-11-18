"""Render the 9:16 cinematic edit frame by frame and pipe to ffmpeg."""
import math, sys, subprocess, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = sys.argv[1]
AUDIO = sys.argv[2]
OUT = sys.argv[3]
W, H, FPS = 1080, 1920, 30
BEAT = 0.4
DUR = 25.6
NFR = int(DUR * FPS)
FONT = os.path.join(HERE, 'fonts/package/Cairo_900Black.ttf')
FONT_B = os.path.join(HERE, 'fonts/package/Cairo_700Bold.ttf')
EMOJI = '/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf'

# ---------------------------------------------------------------- source grade
def grade(img):
    a = np.asarray(img).astype(np.float32) / 255
    # lift contrast (fog is flat)
    a = np.clip((a - 0.5) * 1.45 + 0.47, 0, 1)
    a = a * a * (3 - 2 * a) * 0.55 + a * 0.45          # s-curve blend
    lum = a.mean(2, keepdims=True)
    # teal shadows, warm highlights
    shadow = np.clip(1 - lum * 1.6, 0, 1)
    high = np.clip((lum - 0.55) * 2.2, 0, 1)
    a += shadow * np.array([-0.04, 0.03, 0.07]) + high * np.array([0.06, 0.02, -0.05])
    # saturation boost
    a = lum + (a - lum) * 1.5
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))

SRC = grade(Image.open(SRC_PATH).convert('RGB'))
SRC2 = SRC.resize((SRC.width * 2, SRC.height * 2), Image.LANCZOS)
SW, SH = SRC.size

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIGN = (1 - 0.55 * (((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)).clip(0.25, 1)[..., None]
EDGE = (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2).clip(0, 1.4)[..., None] / 1.4
GRAIN = [np.random.default_rng(k).normal(0, 7, (H // 2, W // 2, 1)).astype(np.float32) for k in range(6)]

def ease_out(x): x = min(max(x, 0), 1); return 1 - (1 - x) ** 3
def ease_in(x): x = min(max(x, 0), 1); return x ** 3
def ease_io(x): x = min(max(x, 0), 1); return x * x * (3 - 2 * x)
def back_out(x):
    x = min(max(x, 0), 1); c = 2.2
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2
def noise1(t, seed=0):
    return sum(math.sin(t * f + seed * p) / k for k, (f, p) in enumerate([(13.1, 1.3), (27.7, 2.1), (41.3, 0.7)], 1))

# ---------------------------------------------------------------- photo camera
def shot(cx, cy, h, rot=0.0):
    """Render crop of source centred at (cx,cy) showing h source px vertically."""
    w = h * W / H
    cx = min(max(cx, w / 2 + 2), SW - w / 2 - 2) if w < SW else SW / 2
    cy = min(max(cy, h / 2 + 2), SH - h / 2 - 2) if h < SH else SH / 2
    s = (h * 2) / H
    ca, sa = math.cos(rot), math.sin(rot)
    a, b, d, e = s * ca, -s * sa, s * sa, s * ca
    c = cx * 2 - a * W / 2 - b * H / 2
    f = cy * 2 - d * W / 2 - e * H / 2
    return np.asarray(SRC2.transform((W, H), Image.AFFINE, (a, b, c, d, e, f), Image.BICUBIC)).astype(np.float32)

def blurred_shot(cam_fn, t, spread=0.0, samples=1):
    """Motion blur over shutter + optional radial zoom spread."""
    if samples <= 1 and spread <= 0:
        return shot(*cam_fn(t))
    n = max(samples, 4 if spread > 0 else 1)
    acc = np.zeros((H, W, 3), np.float32)
    for k in range(n):
        cx, cy, h, rot = cam_fn(t - k * (0.5 / FPS) / n)
        acc += shot(cx, cy, h * (1 - spread * k / n), rot)
    return acc / n

# Shot library (source coords): centre x, centre y, height
SHOTS = {
    'wide':   (450, 600, 1170),
    'scoot':  (360, 760, 620),
    'helmet': (548, 470, 330),
    'wheel':  (455, 900, 300),
    'feet':   (590, 950, 270),
    'bars':   (425, 600, 380),
    'road':   (200, 520, 560),
    'bag':    (555, 640, 300),
    'body':   (580, 650, 650),
    'gta':    (450, 600, 1170),
    'gta2':   (450, 600, 1170),
}

def make_cam(name, t0, dur, zoom_from=1.25, zoom_to=1.0, rot=0.0, drift=(0, 0), shake=0.0, punch=True, ramp=True):
    cx0, cy0, h0 = SHOTS[name]
    def cam(t):
        p = (t - t0) / dur
        # speed ramp: fast camera move at start, then slow-mo drift
        q = ease_out(p * 2.2) * 0.85 + p * 0.15 if ramp else p
        z = zoom_from + (zoom_to - zoom_from) * q
        h = h0 / z
        # beat punch
        if punch:
            bp = ((t + 1e-4) % BEAT) / BEAT
            h *= 1 - 0.06 * math.exp(-bp * 9)
        sx = noise1(t, 1) * shake * h * 0.01
        sy = noise1(t, 2) * shake * h * 0.01
        r = rot + noise1(t * 0.7, 3) * shake * 0.004
        return cx0 + drift[0] * q + sx, cy0 + drift[1] * q + sy, h, r
    return cam

# ---------------------------------------------------------------- text
_font_cache = {}
def font(path, size):
    k = (path, size)
    if k not in _font_cache: _font_cache[k] = ImageFont.truetype(path, size)
    return _font_cache[k]

_emoji_cache = {}
def emoji_img(ch, size):
    k = (ch, size)
    if k not in _emoji_cache:
        f = ImageFont.truetype(EMOJI, 109)
        im = Image.new('RGBA', (160, 160))
        ImageDraw.Draw(im).text((10, 10), ch, font=f, embedded_color=True)
        im = im.crop(im.getbbox())
        _emoji_cache[k] = im.resize((int(im.width * size / im.height), size), Image.LANCZOS)
    return _emoji_cache[k]

_text_cache = {}
def text_img(text, size, fill=(255, 255, 255), stroke=(0, 0, 0), emoji=None, path=FONT, box=None):
    k = (text, size, fill, stroke, emoji, path, box)
    if k in _text_cache: return _text_cache[k]
    f = font(path, size)
    sw = max(3, size // 14)
    import re
    kw = {'direction': 'rtl'} if re.search('[\u0600-\u06FF]', text) else {}
    bb = f.getbbox(text, stroke_width=sw, **kw)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    em = emoji_img(emoji, int(size * 0.95)) if emoji else None
    gap = int(size * 0.2) if em else 0
    ew = em.width if em else 0
    pad = int(size * 0.35)
    Wt = tw + ew + gap + pad * 2
    Ht = max(th, em.height if em else 0) + pad * 2
    im = Image.new('RGBA', (Wt, Ht))
    d = ImageDraw.Draw(im)
    if box:
        d.rounded_rectangle((pad // 3, pad // 2, Wt - pad // 3, Ht - pad // 2), radius=size // 4, fill=box)
    # shadow
    sh = Image.new('RGBA', im.size)
    ImageDraw.Draw(sh).text((pad + ew + gap - bb[0] + 6, pad - bb[1] + (Ht - 2 * pad - th) // 2 + 8), text, font=f,
                            fill=(0, 0, 0, 160), stroke_width=sw, stroke_fill=(0, 0, 0, 160), **kw)
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(8)))
    d.text((pad + ew + gap - bb[0], pad - bb[1] + (Ht - 2 * pad - th) // 2), text, font=f, fill=fill,
           stroke_width=sw if stroke else 0, stroke_fill=stroke, **kw)
    if em:  # RTL: trailing emoji sits on the left
        im.alpha_composite(em, (pad, (Ht - em.height) // 2))
    _text_cache[k] = im
    return im

def paste_text(frame, timg, cx, cy, t, t_in, t_out=None, rot=0.0, pop=True):
    if t < t_in or (t_out is not None and t > t_out): return
    p = (t - t_in) / 0.28
    s = back_out(p) if pop else 1.0
    a = min(1.0, (t - t_in) / 0.08)
    if t_out is not None and t > t_out - 0.12:
        a *= max(0, (t_out - t) / 0.12)
        s *= 1 + (0.12 - (t_out - t)) * 2
    if s <= 0.02 or a <= 0: return
    im = timg
    wiggle = rot + (math.sin((t - t_in) * 30) * 0.03 * max(0, 1 - p) if pop else 0)
    if abs(s - 1) > 0.01 or abs(wiggle) > 0.001:
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BILINEAR)
        if abs(wiggle) > 0.001: im = im.rotate(math.degrees(wiggle), Image.BILINEAR, expand=True)
    if a < 1:
        im = im.copy(); im.putalpha(im.getchannel('A').point(lambda v: int(v * a)))
    frame.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))

# ---------------------------------------------------------------- generated art (original, GTA-inspired vibe)
def sunset_card(w, h, t=0.0, title=True):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    v = yy / h
    horizon = 0.62
    top = np.array([40, 10, 80]); mid = np.array([255, 70, 120]); low = np.array([255, 170, 60])
    sky = np.where((v < horizon * 0.6)[..., None], top + (mid - top) * (v / (horizon * 0.6))[..., None],
                   mid + (low - mid) * ((v - horizon * 0.6) / (horizon * 0.4)).clip(0, 1)[..., None])
    img = sky.astype(np.float32)
    # striped sun
    r = w * 0.28
    cy = h * horizon - r * 0.35 + math.sin(t * 2) * 4
    dist = np.sqrt((xx - w / 2) ** 2 + (yy - cy) ** 2)
    sun_col = np.array([255, 230, 120]) * (1 - (yy - (cy - r)) / (2 * r)).clip(0, 1)[..., None] + \
              np.array([255, 90, 140]) * ((yy - (cy - r)) / (2 * r)).clip(0, 1)[..., None]
    stripe = ((yy - cy) > 0) & (((yy - cy) / (r * 0.12) + t * 2) % 1 < 0.28 + (yy - cy) / r * 0.3)
    m = (dist < r) & ~stripe
    img[m] = sun_col[m]
    glow = np.exp(-((dist - r).clip(0) / (r * 0.6)) ** 2)[..., None] * 60
    img += glow * np.array([1.0, 0.5, 0.6])
    # sea
    sea = v > horizon
    sea_col = np.array([60, 20, 90]) + (np.array([20, 10, 40]) - np.array([60, 20, 90])) * ((v - horizon) / (1 - horizon)).clip(0, 1)[..., None]
    img[sea] = sea_col[sea]
    # sun reflection streaks
    refl = sea & (np.abs(xx - w / 2) < r * (1 - (v - horizon) * 1.5)) & ((np.sin(yy * 0.35 + t * 8) > 0.55))
    img[refl] = img[refl] * 0.4 + np.array([255, 150, 120]) * 0.6
    im = Image.fromarray(img.clip(0, 255).astype(np.uint8)).convert('RGBA')
    d = ImageDraw.Draw(im)
    # palm silhouettes
    def palm(x, base_y, height, lean, scale):
        col = (25, 5, 35)
        top_x = x + lean * height
        cl = [(x + (top_x - x) * q ** 1.5, base_y - height * q) for q in np.linspace(0, 1, 16)]
        wid = [(30 - 16 * q) * scale for q in np.linspace(0, 1, 16)]
        left = [(px - wd / 2, py) for (px, py), wd in zip(cl, wid)]
        right = [(px + wd / 2, py) for (px, py), wd in zip(cl, wid)]
        d.polygon(left + right[::-1], fill=col)
        tx, ty = cl[-1]
        for ang in (-200, -165, -135, -105, -75, -45, -15, 20):
            a = math.radians(ang + math.sin(t * 3 + x) * 3)
            L = height * 0.36
            pts = []
            for q in np.linspace(0, 1, 14):
                droop = q * q * L * (0.55 if abs(math.cos(a)) > 0.4 else 0.2)
                pts.append((tx + math.cos(a) * L * q, ty + math.sin(a) * L * q + droop))
            side1, side2 = [], []
            for k, q in enumerate(np.linspace(0, 1, 14)):
                wv = math.sin(math.pi * min(1, q * 1.15)) * 26 * scale + 2
                if k < 13:
                    dx, dy = pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]
                nl = math.hypot(dx, dy) or 1
                nx, ny = -dy / nl, dx / nl
                side1.append((pts[k][0] + nx * wv, pts[k][1] + ny * wv))
                side2.append((pts[k][0] - nx * wv * 0.4, pts[k][1] - ny * wv * 0.4))
            d.polygon(side1 + side2[::-1], fill=col)
            # serrations
            for k in range(2, 13, 2):
                px, py = side1[k]
                d.line((px, py, px + (side1[k][0] - pts[k][0]) * 0.5, py + 12 * scale), fill=col, width=max(2, int(4 * scale)))
        d.ellipse((tx - 16 * scale, ty - 10 * scale, tx + 16 * scale, ty + 18 * scale), fill=col)
    palm(w * 0.12, h * 1.02, h * 0.62, 0.18, w / 1080)
    palm(w * 0.88, h * 1.02, h * 0.55, -0.22, w / 1080)
    palm(w * 0.24, h * 1.02, h * 0.38, 0.05, w / 1080 * 0.7)
    if title:
        size = int(min(w, h * 0.9) * 0.3)
        f = font(FONT, size)
        txt = 'GTA VI'
        bb = f.getbbox(txt)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        mask = Image.new('L', (tw + 40, th + 40))
        ImageDraw.Draw(mask).text((20 - bb[0], 20 - bb[1]), txt, font=f, fill=255)
        grad = np.zeros((th + 40, tw + 40, 3), np.uint8)
        gv = np.linspace(0, 1, th + 40)[:, None, None]
        grad[:] = (np.array([255, 240, 140]) * (1 - gv) + np.array([255, 60, 160]) * gv).astype(np.uint8)
        gimg = Image.fromarray(grad).convert('RGBA'); gimg.putalpha(mask)
        outline = Image.new('RGBA', mask.size)
        ImageDraw.Draw(outline).text((20 - bb[0], 20 - bb[1]), txt, font=f, fill=(30, 0, 50), stroke_width=size // 14, stroke_fill=(30, 0, 50))
        ty = int(h * 0.16)
        im.alpha_composite(outline.filter(ImageFilter.GaussianBlur(2)), ((w - mask.width) // 2, ty))
        im.alpha_composite(gimg, ((w - mask.width) // 2, ty))
    return im

def night_card(w, h, t=0.0, progress=0.0):
    rng = np.random.default_rng(3)
    v = np.linspace(0, 1, h)[:, None, None]
    img = (np.array([10, 5, 30]) * (1 - v) + np.array([80, 20, 90]) * v).repeat(w, 1).astype(np.uint8)
    im = Image.fromarray(img).convert('RGBA')
    d = ImageDraw.Draw(im)
    base = int(h * 0.72)
    x = 0
    while x < w:
        bw = int(rng.integers(50, 140) * w / 1080)
        bh = int(rng.integers(200, 900) * h / 1920)
        d.rectangle((x, base - bh, x + bw, base), fill=(15, 8, 28))
        for wy in range(base - bh + 20, base - 10, int(28 * h / 1920)):
            for wx in range(x + 10, x + bw - 10, int(22 * w / 1080)):
                if rng.random() < 0.35:
                    d.rectangle((wx, wy, wx + 8 * w // 1080 + 1, wy + 12 * h // 1920 + 1),
                                fill=(255, 200, 90) if rng.random() < 0.7 else (90, 220, 255))
        x += bw + int(rng.integers(4, 20))
    d.rectangle((0, base, w, h), fill=(12, 6, 24))
    glow = Image.new('RGBA', (w, h))
    gd = ImageDraw.Draw(glow)
    gd.rectangle((0, base - 4, w, base + 4), fill=(255, 40, 200, 255))
    im.alpha_composite(glow.filter(ImageFilter.GaussianBlur(14)))
    im.alpha_composite(glow)
    # reflections
    for k in range(40):
        yy = base + 20 + k * (h - base) // 40
        xs = int(rng.integers(0, w))
        d.line((xs, yy, xs + int(rng.integers(30, 160)), yy), fill=(255, 60, 200, 90), width=3)
    # loading bar
    bx0, bx1, by = int(w * 0.15), int(w * 0.85), int(h * 0.84)
    d.rounded_rectangle((bx0, by, bx1, by + 26), radius=13, outline=(255, 255, 255), width=4)
    d.rounded_rectangle((bx0 + 6, by + 6, bx0 + 6 + int((bx1 - bx0 - 12) * progress), by + 20), radius=7, fill=(255, 60, 200))
    f = font(FONT, int(w * 0.07))
    d.text((w / 2, by - 50), 'LOADING... %d%%' % int(progress * 100), font=f, anchor='ms', fill=(255, 255, 255))
    return im

# Pre-render static art
CARD_SUN = None
def get_sun():
    global CARD_SUN
    if CARD_SUN is None: CARD_SUN = sunset_card(W, H, 0)
    return CARD_SUN
_room = None

def build_room():
    """Living room with TV + console + LED glow, 1080x1920 at 1.0 zoom."""
    im = Image.new('RGBA', (W, H), (12, 8, 22, 255))
    a = np.asarray(im).astype(np.float32)
    v = np.linspace(0, 1, H)[:, None]
    wall = (np.array([22, 14, 40]) * (1 - v) + np.array([35, 20, 55]) * v)[:, None, :].repeat(W, 1)
    a[..., :3] = wall
    floor_y = 1250
    fv = ((np.arange(H) - floor_y) / (H - floor_y)).clip(0, 1)[:, None]
    floor = np.array([45, 28, 30]) * (1 - fv) + np.array([18, 10, 14]) * fv
    a[floor_y:, :, :3] = floor[floor_y:, None, :]
    # floor planks
    for k in range(0, W, 120):
        a[floor_y:, k:k + 3, :3] *= 0.7
    im = Image.fromarray(a.astype(np.uint8))
    # LED glow
    glow = Image.new('RGBA', (W, H))
    gd = ImageDraw.Draw(glow)
    tv = (60, 560, 1020, 1100)
    gd.rounded_rectangle((tv[0] - 40, tv[1] - 40, tv[2] + 40, tv[3] + 40), radius=40, fill=(160, 40, 255, 255))
    im.alpha_composite(glow.filter(ImageFilter.GaussianBlur(70)))
    g2 = Image.new('RGBA', (W, H)); ImageDraw.Draw(g2).rounded_rectangle((tv[0] - 10, tv[1] - 10, tv[2] + 10, tv[3] + 10), radius=20, fill=(255, 60, 200, 200))
    im.alpha_composite(g2.filter(ImageFilter.GaussianBlur(25)))
    d = ImageDraw.Draw(im)
    # cabinet
    d.rounded_rectangle((40, 1180, 1040, 1330), radius=14, fill=(30, 22, 30))
    d.rectangle((40, 1180, 1040, 1192), fill=(60, 45, 55))
    d.line((540, 1200, 540, 1320), fill=(15, 10, 15), width=4)
    # TV bezel
    d.rounded_rectangle((tv[0] - 14, tv[1] - 14, tv[2] + 14, tv[3] + 14), radius=12, fill=(8, 8, 10))
    d.rectangle((500, tv[3] + 14, 580, 1180), fill=(15, 15, 18))
    # console (generic white tower) on the right of cabinet
    cx0 = 860
    d.rounded_rectangle((cx0, 900, cx0 + 70, 1180), radius=22, fill=(240, 240, 245))
    d.rounded_rectangle((cx0 + 26, 905, cx0 + 44, 1175), radius=8, fill=(20, 20, 26))
    lg = Image.new('RGBA', (W, H)); ImageDraw.Draw(lg).line((cx0 + 35, 920, cx0 + 35, 1160), fill=(60, 140, 255, 255), width=8)
    im.alpha_composite(lg.filter(ImageFilter.GaussianBlur(10))); im.alpha_composite(lg.filter(ImageFilter.GaussianBlur(2)))
    d = ImageDraw.Draw(im)
    d.text((cx0 + 35, 1190 + 70), 'PS5', font=font(FONT, 40), anchor='mm', fill=(200, 200, 220))
    # controller on cabinet front
    ccx, ccy = 330, 1150
    d.ellipse((ccx - 110, ccy - 30, ccx - 30, ccy + 50), fill=(235, 235, 240))
    d.ellipse((ccx + 30, ccy - 30, ccx + 110, ccy + 50), fill=(235, 235, 240))
    d.rounded_rectangle((ccx - 80, ccy - 40, ccx + 80, ccy + 20), radius=26, fill=(235, 235, 240))
    d.rounded_rectangle((ccx - 40, ccy - 36, ccx + 40, ccy - 5), radius=8, fill=(25, 25, 30))
    for dx, col in ((-70, (30, 30, 35)), (70, (30, 30, 35))):
        d.ellipse((ccx + dx - 12, ccy - 2, ccx + dx + 12, ccy + 22), fill=col)
    # plant + sofa hint at bottom
    d.rounded_rectangle((-50, 1560, 1130, 1960), radius=60, fill=(40, 25, 45))
    d.rounded_rectangle((-50, 1500, 1130, 1600), radius=50, fill=(55, 35, 60))
    return im, tv

def room_frame(t, t0):
    global _room
    if _room is None: _room = build_room()
    room, tv = _room
    im = room.copy()
    tw, th = tv[2] - tv[0], tv[3] - tv[1]
    screen = sunset_card(tw, th, t, title=True).convert('RGBA')
    if (t - t0) < 0.5:  # TV boot: black -> image
        k = ease_io((t - t0) / 0.5)
        screen = Image.blend(Image.new('RGBA', screen.size, (0, 0, 0, 255)), screen, k)
    sd = ImageDraw.Draw(screen)
    if int(t * 2.5) % 2 == 0:
        sd.text((tw / 2, th * 0.88), 'اضغط للبدء', font=font(FONT, 46), anchor='mm', direction='rtl', fill=(255, 255, 255), stroke_width=4, stroke_fill=(40, 0, 60))
    im.alpha_composite(screen, (tv[0], tv[1]))
    # screen glare
    gl = Image.new('RGBA', (tw, th)); ImageDraw.Draw(gl).polygon([(0, 0), (tw * 0.35, 0), (tw * 0.1, th), (0, th)], fill=(255, 255, 255, 18))
    im.alpha_composite(gl, (tv[0], tv[1]))
    return im, tv

# ---------------------------------------------------------------- HUD (original design)
def speed_at(t):
    return float(np.interp(t, [0, 3.2, 4.0, 8.0, 9.6, 11.2, 16.0, 18.4, 19.2], [0, 18, 38, 46, 30, 52, 68, 72, 0]))

def draw_hud(frame, t):
    if not (2.2 <= t < 19.2): return
    d = ImageDraw.Draw(frame)
    appear = ease_out((t - 2.2) / 0.4)
    # countdown pill
    remain = max(0, 19.2 - t)
    urgent = remain < 6
    col = (255, 50, 70) if urgent and int(t * 5) % 2 == 0 else (255, 255, 255)
    yo = int(-120 * (1 - appear))
    d.rounded_rectangle((340, 210 + yo, 740, 300 + yo), radius=45, fill=(0, 0, 0, 150), outline=col, width=4)
    d.text((540, 255 + yo), '00:%02d:%02d' % (int(remain), int((remain % 1) * 100)), font=font(FONT, 52), anchor='mm', fill=col)
    # wanted-style stars (rise with speed)
    nstars = min(5, int(speed_at(t) / 14))
    for k in range(5):
        cx = 540 - 2 * 46 + k * 46; cy = 335 + yo
        pts = [(cx + math.cos(math.radians(-90 + 36 * i)) * (18 if i % 2 == 0 else 8),
                cy + math.sin(math.radians(-90 + 36 * i)) * (18 if i % 2 == 0 else 8)) for i in range(10)]
        d.polygon(pts, fill=(255, 210, 60) if k < nstars else (60, 60, 60, 180), outline=(0, 0, 0))
    # speedometer (top-right)
    sp = speed_at(t) + noise1(t * 3, 5) * 1.5
    cx, cy, r = 935, 470, 95
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(0, 0, 0, 150), outline=(255, 255, 255, 200), width=3)
    frac = min(sp / 80, 1)
    d.arc((cx - r + 12, cy - r + 12, cx + r - 12, cy + r - 12), 135, 135 + 270 * frac, fill=(255, 60, 200) if frac > 0.75 else (90, 220, 255), width=12)
    d.text((cx, cy - 8), '%d' % max(0, sp), font=font(FONT, 64), anchor='mm', fill=(255, 255, 255))
    d.text((cx, cy + 46), 'KM/H', font=font(FONT_B, 24), anchor='mm', fill=(200, 200, 200))
    # minimap (top-left)
    mx, my, mr = 150, 470, 100
    mm = Image.new('RGBA', (2 * mr, 2 * mr))
    md = ImageDraw.Draw(mm)
    md.rectangle((0, 0, 2 * mr, 2 * mr), fill=(40, 60, 50, 220))
    off = (t * 40) % 60
    for k in range(-2, 6):
        md.line((0, k * 60 + off, 2 * mr, k * 60 + off), fill=(70, 95, 80), width=10)
        md.line((k * 60 - off * 0.3, 0, k * 60 - off * 0.3, 2 * mr), fill=(70, 95, 80), width=10)
    prog = min(1, max(0, (t - 1.4) / (19.2 - 1.4)))
    hy = mr - 160 * (1 - prog) - 10
    md.line((mr, mr, mr, hy), fill=(255, 60, 200), width=8)
    md.ellipse((mr - 14, hy - 14, mr + 14, hy + 14), fill=(255, 210, 60))
    md.polygon([(mr, mr - 16), (mr - 12, mr + 12), (mr + 12, mr + 12)], fill=(255, 255, 255))
    mask = Image.new('L', mm.size); ImageDraw.Draw(mask).ellipse((0, 0, 2 * mr, 2 * mr), fill=255)
    mm.putalpha(mask)
    frame.alpha_composite(mm, (mx - mr, my - mr))
    d.ellipse((mx - mr, my - mr, mx + mr, my + mr), outline=(255, 255, 255, 200), width=4)
    dist = int(np.interp(t, [2.2, 12.8, 14.4, 16.0, 19.0], [1200, 500, 200, 50, 0]))
    d.text((mx, my + mr + 30), '%d m' % dist, font=font(FONT_B, 34), anchor='mm', fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))

def speed_lines(frame, t, strength):
    if strength <= 0: return
    rng = np.random.default_rng(int(t * FPS) + 99)
    lay = Image.new('RGBA', (W, H))
    d = ImageDraw.Draw(lay)
    for _ in range(int(60 * strength)):
        ang = rng.uniform(0, 2 * math.pi)
        r0 = rng.uniform(380, 700)
        r1 = r0 + rng.uniform(250, 900)
        d.line((W / 2 + math.cos(ang) * r0, H / 2 + math.sin(ang) * r0 * 1.4,
                W / 2 + math.cos(ang) * r1, H / 2 + math.sin(ang) * r1 * 1.4),
               fill=(255, 255, 255, int(rng.uniform(60, 170) * min(1, strength))), width=int(rng.uniform(2, 6)))
    frame.alpha_composite(lay)

# ---------------------------------------------------------------- timeline
def cuts(t0, names, step, **kw):
    out = []
    for k, n in enumerate(names):
        a = t0 + k * step
        opts = dict(kw)
        opts.setdefault('rot', (0.06 if k % 2 else -0.05))
        out.append((a, t0 + (k + 1) * step, n, make_cam(n, a, step, **opts)))
    return out

SEG_B = cuts(3.2, ['wide', 'scoot', 'helmet', 'wheel', 'feet', 'bars'], 0.8, zoom_from=1.5, zoom_to=1.05, shake=1.2)
SEG_D = [(9.6, 11.2, 'wide', make_cam('wide', 9.6, 1.6, zoom_from=1.0, zoom_to=1.35, shake=1.5, rot=0.03, ramp=False))]
SEG_E = cuts(11.2, ['road', 'helmet', 'wheel', 'scoot', 'bars', 'body', 'gta', 'feet',
                    'helmet', 'wheel', 'bag', 'scoot', 'gta2', 'bars', 'wheel', 'helmet'],
             0.4, zoom_from=1.6, zoom_to=1.1, shake=2.2)
def segF_cam(t):
    p = (t - 17.6) / 1.6
    # speed ramp: slow-mo then violent zoom into helmet at the end
    z = 1.0 + 0.25 * p + 2.4 * ease_in(max(0, (p - 0.55) / 0.45))
    cx = 450 + (548 - 450) * ease_io(p); cy = 600 + (470 - 600) * ease_io(p)
    return cx, cy, 1170 / z, noise1(t, 4) * 0.004
PHOTO_SEGS = SEG_B + SEG_D + SEG_E

def photo_frame(t):
    """Returns float32 array for photo-based segments or None."""
    if t < 3.2:
        # intro: slow-mo push in on wide, ramping fast at the end
        p = (t - 0.9) / 2.3
        z = 1.0 + 0.12 * max(0, p) + 0.6 * ease_in(max(0, (t - 2.7) / 0.5))
        def cam(tt):
            pp = max(0, (tt - 0.9) / 2.3)
            zz = 1.0 + 0.12 * pp + 0.6 * ease_in(max(0, (tt - 2.7) / 0.5))
            return 450, 610, 1170 / zz, 0.0
        return blurred_shot(cam, t, spread=0.08 if t > 2.8 else 0, samples=4 if t > 2.8 else 1), 'intro'
    if 17.6 <= t < 19.2:
        fast = t > 18.5
        return blurred_shot(segF_cam, t, spread=0.1 if fast else 0, samples=5 if fast else 1), 'F'
    for a, b, name, cam in PHOTO_SEGS:
        if a <= t < b:
            p = (t - a) / (b - a)
            if name in ('gta', 'gta2'):
                return None, name
            spread = 0.12 * max(0, 1 - p * 2.5) + (0.05 if a >= 11.2 else 0.02)
            if a == 9.6: spread = 0.09
            arr = blurred_shot(cam, t, spread=spread, samples=5)
            # whip transition: horizontal smear in first frames of cut
            if p < 0.18 and a > 3.3:
                k = int((1 - p / 0.18) * 90) * (1 if int(a * 10) % 2 else -1)
                acc = np.zeros_like(arr)
                for j in range(6):
                    acc += np.roll(arr, int(k * j / 5), axis=1)
                arr = acc / 6
            return arr, name
    return None, None

def post(arr, t, flash=0.0, ca=0, tint=None, desat=0.0):
    if desat > 0:
        l = arr.mean(2, keepdims=True); arr = arr * (1 - desat) + l * desat
    if tint is not None:
        arr = arr * (1 - tint[3]) + np.array(tint[:3], np.float32) * tint[3] * (EDGE if len(tint) > 4 else 1)
    arr = arr * VIGN
    g = GRAIN[int(t * FPS) % 6]
    arr = arr + np.repeat(np.repeat(g, 2, 0), 2, 1)
    if ca:
        arr[..., 0] = np.roll(arr[..., 0], ca, 1); arr[..., 2] = np.roll(arr[..., 2], -ca, 1)
    if flash > 0:
        arr = arr * (1 - flash) + 255 * flash
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert('RGBA')

def beat_flash(t, cuts_at, length=0.1, amt=0.6):
    f = 0
    for c in cuts_at:
        if c <= t < c + length: f = max(f, amt * (1 - (t - c) / length))
    return f

def render(i):
    t = i / FPS
    frame = None
    # ---------- A: intro notification on black (0 - 1.0), then photo
    if t < 1.0:
        frame = Image.new('RGBA', (W, H), (0, 0, 0, 255))
        arr, _ = photo_frame(t)
        dim = 0.25 * ease_io(t / 1.0)
        frame = post(arr * dim, t)
    elif t < 19.2:
        arr, name = photo_frame(t)
        if name in ('gta', 'gta2') or (8.0 <= t < 9.6):
            # GTA-vibe interlude cards
            if 8.0 <= t < 8.8 or name == 'gta':
                card = sunset_card(W, H, t)
                z = 1.0 + 0.08 * ((t - 8.0) / 0.8 if 8.0 <= t < 8.8 else 0.5)
            else:
                prog = min(1, (t - 8.8) / 0.8) if 8.8 <= t < 9.6 else 0.97
                card = night_card(W, H, t, prog)
                z = 1.04
            arr = np.asarray(card.convert('RGB')).astype(np.float32)
            if z > 1:
                c = Image.fromarray(arr.astype(np.uint8)).resize((int(W * z), int(H * z)), Image.BILINEAR)
                arr = np.asarray(c.crop(((c.width - W) // 2, (c.height - H) // 2, (c.width - W) // 2 + W, (c.height - H) // 2 + H))).astype(np.float32)
            # stutter: flash back to photo on 16ths in second half of the bar
            if 8.8 <= t < 9.6 and int((t - 8.8) / 0.1) % 2 == 1:
                arr = blurred_shot(make_cam('helmet', 8.8, 0.8, zoom_from=1.3, zoom_to=1.0), t, spread=0.06, samples=4)
        cut_points = [3.2] + [a for a, *_ in PHOTO_SEGS] + [8.0, 8.8]
        fl = beat_flash(t, [3.2], 0.25, 0.9) + beat_flash(t, [8.0, 9.6], 0.15, 0.7) + beat_flash(t, cut_points, 0.07, 0.25)
        ca = 0
        for c in cut_points:
            if c <= t < c + 0.12: ca = 10
        approach = 11.2 <= t < 17.6
        tint = None
        if approach:
            pulse = 0.5 + 0.5 * math.cos(((t % BEAT) / BEAT) * 2 * math.pi)
            tint = (255, 20, 40, 0.35 * pulse * min(1, (t - 11.2) / 3), 'edge')
        desat = 0.0
        if 17.6 <= t < 18.5: desat = 0.5
        if t >= 18.8: fl = max(fl, ease_in((t - 18.8) / 0.4))
        frame = post(arr, t, flash=min(fl, 1), ca=ca, tint=tint, desat=desat)
        # speed lines
        sl = 0
        if 3.2 <= t < 8.0: sl = 0.7
        if 9.6 <= t < 11.2: sl = 1.0
        if approach: sl = 1.3
        if 18.4 <= t < 19.2: sl = 1.6
        speed_lines(frame, t, sl)
    else:
        # ---------- G: home. room push-in, then full-screen game card
        if t < 22.4:
            room, tv = room_frame(t, 19.2)
            p = (t - 19.2) / 3.2
            z = 1.0 + 0.25 * ease_io(p) + 0.9 * ease_in(max(0, (t - 22.0) / 0.4))
            tvc = ((tv[0] + tv[2]) / 2, (tv[1] + tv[3]) / 2)
            fx = W / 2 + (tvc[0] - W / 2) * min(1, p * 1.5)
            fy = H / 2 + (tvc[1] - H / 2) * min(1, p * 1.5)
            sh = noise1(t, 7) * (6 if t < 19.8 else 1.5)
            box = (fx - W / 2 / z + sh, fy - H / 2 / z + sh, fx + W / 2 / z + sh, fy + H / 2 / z + sh)
            img = room.transform((W, H), Image.EXTENT, box, Image.BICUBIC)
            arr = np.asarray(img.convert('RGB')).astype(np.float32)
            fl = beat_flash(t, [19.2], 0.5, 1.0) + beat_flash(t, [21.6, 22.0], 0.08, 0.3)
            if t > 22.2: fl = max(fl, ease_in((t - 22.2) / 0.2))
            frame = post(arr, t, flash=min(1, fl), ca=6 if 19.2 <= t < 19.35 else 0)
        else:
            card = sunset_card(W, H, t)
            z = 1.15 - 0.15 * ease_out((t - 22.4) / 1.2)
            c = card.resize((int(W * z), int(H * z)), Image.BILINEAR)
            card = c.crop(((c.width - W) // 2, (c.height - H) // 2, (c.width - W) // 2 + W, (c.height - H) // 2 + H))
            arr = np.asarray(card.convert('RGB')).astype(np.float32)
            fl = beat_flash(t, [22.4], 0.4, 1.0)
            if t > 25.0: arr *= max(0, 1 - (t - 25.0) / 0.6)
            frame = post(arr, t, flash=fl)
            # player card with the rider photo
            pa = ease_out((t - 23.0) / 0.4)
            if pa > 0:
                pc = SRC.crop((300, 330, 760, 1010)).resize((360, 532), Image.LANCZOS).convert('RGBA')
                card_im = Image.new('RGBA', (400, 640), (0, 0, 0, 0))
                cd = ImageDraw.Draw(card_im)
                cd.rounded_rectangle((0, 0, 399, 639), radius=28, fill=(20, 5, 40, 230), outline=(255, 210, 60), width=6)
                card_im.alpha_composite(pc, (20, 20))
                cd.text((200, 598), 'PLAYER 1', font=font(FONT, 44), anchor='mm', fill=(255, 210, 60))
                card_im = card_im.rotate(-4, Image.BICUBIC, expand=True)
                y = int(1080 + 900 * (1 - pa))
                frame.alpha_composite(card_im, ((W - card_im.width) // 2, y - card_im.height // 2))
    # ---------- HUD + captions
    if not (8.0 <= t < 9.6 or 13.6 <= t < 14.0 or 16.0 <= t < 16.4): draw_hud(frame, t)
    T = text_img
    # intro notification
    if t < 2.2:
        sy = int(-260 + 520 * ease_out(t / 0.35)) if t < 1.85 else int(260 - 520 * ease_in((t - 1.85) / 0.35))
        n = Image.new('RGBA', (960, 210))
        nd = ImageDraw.Draw(n)
        nd.rounded_rectangle((0, 0, 959, 209), radius=40, fill=(30, 30, 38, 235))
        n.alpha_composite(emoji_img('🎮', 90), (830, 30))
        nd.text((800, 60), 'GTA 6 • الآن', font=font(FONT_B, 34), anchor='rm', direction='rtl', fill=(170, 170, 185))
        nd.text((800, 135), 'التحميل اكتمل 100% - جاهز للعب', font=font(FONT, 46), anchor='rm', direction='rtl', fill=(255, 255, 255))
        frame.alpha_composite(n, (60, sy))
    paste_text(frame, T('الطريق إلى المنزل', 92, emoji='🚀'), 540, 1200, t, 1.3, 3.15)
    paste_text(frame, T('ما عندي حتى دقيقة!', 78, emoji='⏱️', fill=(255, 230, 80)), 540, 1250, t, 4.8, 5.55, rot=0.05)
    paste_text(frame, T('ما كاينش الوقت نلبس الصبّاط', 64, emoji='😂'), 540, 1180, t, 6.4, 7.15, rot=-0.04)
    paste_text(frame, T('GTA 6 كيتسناني!', 88, emoji='🎮', fill=(255, 110, 220)), 540, 1230, t, 9.7, 11.15)
    paste_text(frame, T('قربت نوصل!!', 100, emoji='🏠', fill=(255, 255, 255), box=(230, 20, 50, 230), stroke=None), 540, 1150, t, 11.25, 12.75)
    for k, (tt, txt) in enumerate([(12.8, '500m'), (14.4, '200m'), (16.0, '50m')]):
        paste_text(frame, T(txt, 170, fill=(255, 220, 60)), 540, 1200, t, tt, tt + 1.55, rot=(-0.06 if k % 2 else 0.06))
    paste_text(frame, T('وصلت!!', 160, fill=(255, 255, 255)), 540, 1150, t, 18.2, 19.05)
    paste_text(frame, T('أخيراً… GTA 6', 100, emoji='🔥', fill=(255, 220, 80)), 540, 360, t, 19.9, 22.3)
    if t >= 22.6:
        paste_text(frame, T('المهمة: نجحت', 88, emoji='✅', fill=(255, 255, 255), box=(10, 10, 20, 200), stroke=None), 540, 1720, t, 22.6, 25.3)
        paste_text(frame, T('أخيراً… GTA 6', 104, emoji='🔥', fill=(255, 220, 80)), 540, 1560, t, 22.7, 25.3)
    if t >= 25.0:
        a = (t - 25.0) / 0.6
        frame = Image.blend(frame, Image.new('RGBA', (W, H), (0, 0, 0, 255)), min(1, a))
    return frame.convert('RGB').tobytes()

if __name__ == '__main__':
    only = os.environ.get('ONLY')
    if only:  # preview stills
        for s in only.split(','):
            i = int(float(s) * FPS)
            Image.frombytes('RGB', (W, H), render(i)).save(os.path.join(HERE, 'prev_%s.jpg' % s), quality=85)
        sys.exit()
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
           '-i', AUDIO, '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-maxrate', '14M', '-bufsize', '28M', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
           '-c:a', 'aac', '-b:a', '256k', '-movflags', '+faststart', '-shortest', OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(4) as pool:
        for k, fr in enumerate(pool.imap(render, range(NFR), chunksize=4)):
            p.stdin.write(fr)
            if k % 60 == 0: print('frame', k, flush=True)
    p.stdin.close(); p.wait()
    print('done', OUT)
