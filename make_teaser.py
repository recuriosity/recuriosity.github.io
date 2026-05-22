#!/usr/bin/env python3
"""
Teaser structure:
  [text]  ~6s   : "Let's be curious... wait... Where are we?"
  [blink] ~2s   : eye opens
  [video] ~51s  :
      0–10s    : 1.3× slow center
      10–13s   : ramp up 1.3× → 2.1×
      13–P3    : INTERLUDE — semi-transparent overlay on running video:
                   typewriter text 1 → hobbit world video (11s) → typewriter text 2
      P3–P3+3s : ramp down 2.1× → 1.3×
      –P5      : zoom-out 1.5s
      –P6      : full 3×3 grid
  [blink] ~1.5s : eye closes
  [text]  ~10s  : paper title
"""
import subprocess, os, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── paths ──────────────────────────────────────────────────────────────────────
video_dir  = "/Users/lily.goli/Desktop/recurious/assets/videos/teaser"
SRC        = [os.path.join(video_dir, f"{i}.mp4") for i in range(1, 13)]
HOBBIT_SRC   = os.path.join(video_dir, "hobbit_world_ppo_gpu.mp4")
HIGHRES_OBS  = os.path.join(video_dir, "highres.mp4")          # 1920×1080, 12fps
HIGHRES_MAP  = os.path.join(video_dir, "highres_topdown.mp4")  # 360×360,   12fps
VID_ONLY   = os.path.join(video_dir, "_teaser_vid.mp4")
AUD_WAV    = os.path.join(video_dir, "_teaser_aud.wav")
OUT        = os.path.join(video_dir, "teaser.mp4")
_audio_dir = os.path.join(os.path.dirname(os.path.dirname(video_dir)), "audio")

# ── video config ───────────────────────────────────────────────────────────────
SRC_FPS = 12
OUT_FPS = 24
SRC_W, SRC_H = 1648, 368
OUT_W, OUT_H = 1920, 1080
CW, CH   = 640, 360
MAP_HQ   = 360
MAP_PIP      = 120   # pip size for full-screen center view
MAP_PIP_GRID = 160   # larger pip for each cell in the 3×3 grid

V_SLOW = 1.3
HIGHRES_DUR = 85.416   # duration of highres.mp4 (from ffprobe)

# ── terminal text config ───────────────────────────────────────────────────────
TERM_GREEN = (0, 230, 65)
TERM_DIM   = (0, 150, 45)
MONO_PATH  = "/System/Library/Fonts/Courier.ttc"

OPEN_LINE1  = "Let's be curious... wait, where are we?"
OPEN_LINE2  = "> explore"
CLOSE_L1    = "Remember to be Curious:"
CLOSE_L2    = "Episodic Context and Persistent Worlds"
CLOSE_L3    = "for 3D Exploration"
INTER_TEXT1 = "...kind of dull here. I wish this was a Lord of the Rings world. brb."
INTER_TEXT2 = "...that was fun, back to our world!"

OPEN_HOLD  = 1.0
CLOSE_HOLD = 0.0   # end immediately after last character

# ── blink configs ──────────────────────────────────────────────────────────────
BLINK_DUR = 2.0
BLINK_KF  = [
    (0.00, 0.00), (0.12, 0.00), (0.20, 0.28), (0.28, 0.00),
    (0.44, 0.00), (0.54, 0.55), (0.66, 0.00), (0.80, 0.00),
    (0.90, 0.06), (2.00, 1.00),
]
CLOSE_BLINK_DUR = 1.5
CLOSE_BLINK_KF  = [
    (0.00, 1.00), (0.25, 0.85), (0.55, 0.35),
    (0.85, 0.05), (1.00, 0.00), (1.50, 0.00),
]

# ── audio config ───────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100

def _load_sound(path, gain=0.80):
    raw = subprocess.check_output(
        ["ffmpeg", "-i", path, "-f", "f32le", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "pipe:1"], stderr=subprocess.DEVNULL)
    s = np.frombuffer(raw, dtype=np.float32).copy()
    peak = np.abs(s).max()
    if peak > 0: s *= gain / peak
    return s

_SND_CLICK  = _load_sound(os.path.join(_audio_dir, "one_click.mov"), gain=0.15)
_SND_BELL   = _load_sound(os.path.join(_audio_dir, "bell.mov"),      gain=0.55)
_SND_BACH   = _load_sound(os.path.join(_audio_dir, "chopin.mp3"),    gain=0.30)
_SND_HOBBIT = _load_sound(os.path.join(_audio_dir, "hobbit.mov"),    gain=0.55)

all_click_times: list[float] = []
all_bell_times:  list[float] = []

# ── typewriter event builder ───────────────────────────────────────────────────
def build_typewriter_events(text, base_interval=0.09, start_t=0.0,
                             dot_extra=0.07, ellipsis_extra=0.38):
    """Pure function — builds timing data with no side effects."""
    events, clicks = [], []
    t = start_t
    for i, ch in enumerate(text):
        events.append((t, i + 1))
        clicks.append(t)
        delay = base_interval
        if ch == ' ':     delay *= 0.65
        elif ch in '.!?': delay += dot_extra
        if i >= 2 and text[i-2:i+1] == '...': delay += ellipsis_extra
        t += delay
    return events, clicks, t   # t = time after last char

# ── interlude timing (computed before phase boundaries need it) ────────────────
HOBBIT_W, HOBBIT_H = 1280, 720
HOBBIT_FPS  = 12     # actual fps of the hobbit video
HOBBIT_DUR  = 10.833
HOBBIT_HOLD = 0.65   # seconds to freeze on last hobbit frame before it disappears
INTER_ALPHA_TEXT = 0.85  # overlay alpha during text phases (lighter)
INTER_ALPHA      = 0.92  # overlay alpha during hobbit video (dark)
INTER_HOB_TRANS  = 0.5   # seconds to cross-fade between text and hobbit alpha
INTER_FADE  = 0.5    # intro fade-in duration
INTER_OUTRO = 0.6    # outro fade-out duration

# Text 1: original pace with natural pauses at "..."
_TEXT1_START = INTER_FADE
_ie1, _ic1, _it1_end = build_typewriter_events(INTER_TEXT1, 0.09,
                                                start_t=_TEXT1_START)
_hobbit_rel_s = _it1_end + 0.4                       # pause before hobbit appears
_hobbit_rel_e = _hobbit_rel_s + HOBBIT_DUR           # hobbit playback ends
_hobbit_rel_gone = _hobbit_rel_e + HOBBIT_HOLD       # hobbit disappears (after hold)

# Text 2: type in ~1.5s (fast intervals, reduced pauses)
_ie2, _ic2, _it2_end = build_typewriter_events(INTER_TEXT2, 0.09,
                                                start_t=_hobbit_rel_gone + 0.4,
                                                dot_extra=0.02, ellipsis_extra=0.05)
_inter_fo_s = _it2_end + 0.4         # outro blink start
INTER_DUR   = _inter_fo_s + INTER_OUTRO

# ── phase boundaries ───────────────────────────────────────────────────────────
P1 = 10.0
P2 = 21.5   # slow center ends, interlude begins
P3 = P2 + INTER_DUR    # end of interlude (dynamic)
P4 = P3 + 3.0          # zoom-out start   (3s slow after interlude before zoom)
P5 = P4 + 1.5          # zoom-out end
P6 = P5 + 6.5          # full grid end
N   = int(P6 * OUT_FPS)
dt  = 1.0 / OUT_FPS

# ── speed profile ──────────────────────────────────────────────────────────────
# V_SLOW before and after interlude.
# V_INTER under the black overlay: computed so highres.mp4 is consumed exactly.
POST_P3_DUR = P6 - P3   # seconds of slow video after interlude
V_INTER = (HIGHRES_DUR - P2 * V_SLOW - POST_P3_DUR * V_SLOW) / INTER_DUR
print(f"V_INTER = {V_INTER:.3f}x  (auto-computed to consume highres.mp4 exactly)")

def speed_at(t):
    if t < P2:   return V_SLOW
    elif t < P3: return V_INTER   # hidden under overlay
    else:         return V_SLOW

src_times = np.zeros(N)
s = 0.0
for i in range(N):
    src_times[i] = s
    s += speed_at(i * dt) * dt

print(f"Interlude: {INTER_DUR:.2f}s  "
      f"(text1→{_it1_end:.1f}s | hobbit {_hobbit_rel_s:.1f}–{_hobbit_rel_e:.1f}s "
      f"| text2→{_it2_end:.1f}s)")
print(f"Main video: {P6:.1f}s  ({N} frames)   "
      f"Source consumed: {src_times[-1]:.1f}s / 85.5s")

# ── eyelid helpers ─────────────────────────────────────────────────────────────
def _interp_kf(kf, t):
    if t <= kf[0][0]:  return kf[0][1]
    if t >= kf[-1][0]: return kf[-1][1]
    for i in range(len(kf)-1):
        t0, v0 = kf[i]; t1, v1 = kf[i+1]
        if t0 <= t <= t1:
            u = (t-t0)/(t1-t0); u = u*u*(3-2*u)
            return v0 + (v1-v0)*u
    return 1.0

def apply_eye_mask(frame, open_amt):
    open_amt = float(np.clip(open_amt, 0, 1))
    if open_amt <= 0: return np.zeros_like(frame)
    vh = int(open_amt * OUT_H); ys = (OUT_H - vh)//2; ye = ys + vh
    out = np.zeros_like(frame); out[ys:ye] = frame[ys:ye]
    return out

# ── image helpers ──────────────────────────────────────────────────────────────
def smoothstep(t):
    t = float(np.clip(t, 0, 1)); return t*t*(3-2*t)

def upscale_3x(img):
    return np.repeat(np.repeat(img, 3, axis=0), 3, axis=1)

def downscale_3x(img):
    h, w, c = img.shape
    r = img.reshape(h//3, 3, w, c).mean(1)
    return r.reshape(h//3, w//3, 3, c).mean(2).astype(np.uint8)

def resize_bilinear(img, oh, ow):
    sh, sw = img.shape[:2]
    yf = np.linspace(0, sh-1, oh); xf = np.linspace(0, sw-1, ow)
    y0 = np.floor(yf).astype(np.int32).clip(0,sh-1); y1=(y0+1).clip(0,sh-1)
    x0 = np.floor(xf).astype(np.int32).clip(0,sw-1); x1=(x0+1).clip(0,sw-1)
    dy = (yf-y0)[:,None,None].astype(np.float32)
    dx = (xf-x0)[None,:,None].astype(np.float32)
    f  = img.astype(np.float32)
    return (f[y0[:,None],x0[None,:]]*(1-dy)*(1-dx) +
            f[y1[:,None],x0[None,:]]*dy*(1-dx) +
            f[y0[:,None],x1[None,:]]*(1-dy)*dx +
            f[y1[:,None],x1[None,:]]*dy*dx).clip(0,255).astype(np.uint8)

# ── text rendering ─────────────────────────────────────────────────────────────
def load_font(size): return ImageFont.truetype(MONO_PATH, size)
def _char_width(f):  bb = f.getbbox("M");  return bb[2]-bb[0]
def _line_height(f): bb = f.getbbox("Ay"); return bb[3]-bb[1]
def _text_width(f, t):
    if not t: return 0
    bb = f.getbbox(t); return bb[2]-bb[0]

_font_open  = load_font(52)
_font_title = load_font(62)
_font_sub   = load_font(44)
_font_inter = load_font(38)

# Map label: auto-size to 85% of MAP_HQ width
MAP_LABEL = "bird's eye view"
_lbl_sz = 10
while True:
    _f = load_font(_lbl_sz + 1)
    if _text_width(_f, MAP_LABEL) > int(MAP_HQ * 0.85):
        break
    _lbl_sz += 1
_font_lbl = load_font(_lbl_sz)
_lbl_w    = _text_width(_font_lbl, MAP_LABEL)
_lbl_h    = _line_height(_font_lbl)
# Position: centered in map area, sitting in the bottom 5% of the map
# position within the 360×360 map frame itself — anchor to 80% down, never overflow
_lbl_map_x = (MAP_HQ - _lbl_w) // 2
_lbl_map_y = min(int(MAP_HQ * 0.88), MAP_HQ - _lbl_h - 4)

_open1_full_w = _text_width(_font_open,  OPEN_LINE1)
_open2_full_w = _text_width(_font_open,  OPEN_LINE2)
_cl1_full_w   = _text_width(_font_title, CLOSE_L1)
_cl2_full_w   = _text_width(_font_sub,   CLOSE_L2)
_cl3_full_w   = _text_width(_font_sub,   CLOSE_L3)
_it1_full_w   = _text_width(_font_inter, INTER_TEXT1)
_it2_full_w   = _text_width(_font_inter, INTER_TEXT2)

_open1_x = (OUT_W - _open1_full_w) // 2
_open2_x = (OUT_W - _open2_full_w) // 2
_cl1_x   = (OUT_W - _cl1_full_w)   // 2
_cl2_x   = (OUT_W - _cl2_full_w)   // 2
_cl3_x   = (OUT_W - _cl3_full_w)   // 2
_it1_x   = max(20, (OUT_W - _it1_full_w) // 2)
_it2_x   = (OUT_W - _it2_full_w)   // 2

_lh_open  = _line_height(_font_open)
_lh_title = _line_height(_font_title)
_lh_sub   = _line_height(_font_sub)
_lh_inter = _line_height(_font_inter)
_GAP = 22
_close_block_h = _lh_title + _GAP + _lh_sub + _GAP + _lh_sub
_cl1_y   = (OUT_H - _close_block_h) // 2
_cl2_y   = _cl1_y + _lh_title + _GAP
_cl3_y   = _cl2_y + _lh_sub   + _GAP
# Two-line opening block, vertically centred
_open_block_h = _lh_open + _GAP + _lh_open
_open1_y = (OUT_H - _open_block_h) // 2
_open2_y = _open1_y + _lh_open + _GAP
_inter_y = OUT_H * 2 // 3      # lower-third for interlude text

# Hobbit video: 1280×720 centered on 1920×1080
HOBBIT_PX = (OUT_W - HOBBIT_W) // 2   # 320
HOBBIT_PY = (OUT_H - HOBBIT_H) // 2   # 180


def render_open_frame(n1, n2=0, cursor_on=True):
    """n1 = chars of line1 revealed, n2 = chars of line2 revealed."""
    img  = Image.new('RGB', (OUT_W, OUT_H), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    p1 = OPEN_LINE1[:n1]
    if p1: draw.text((_open1_x, _open1_y), p1, font=_font_open, fill=TERM_GREEN)
    if n2 > 0:
        p2 = OPEN_LINE2[:n2]
        if p2: draw.text((_open2_x, _open2_y), p2, font=_font_open, fill=TERM_GREEN)
        if cursor_on:
            cx = _open2_x + _text_width(_font_open, p2)
            draw.rectangle([cx, _open2_y, cx+_char_width(_font_open)-1,
                            _open2_y+_lh_open-1], fill=TERM_GREEN)
    else:
        if cursor_on:
            cx = _open1_x + _text_width(_font_open, p1)
            draw.rectangle([cx, _open1_y, cx+_char_width(_font_open)-1,
                            _open1_y+_lh_open-1], fill=TERM_GREEN)
    return np.array(img)


def render_close_frame(n1, n2, n3, cursor_on=True):
    img = Image.new('RGB', (OUT_W, OUT_H), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    specs = [
        (CLOSE_L1[:n1], _font_title, TERM_GREEN, _cl1_x, _cl1_y, _lh_title),
        (CLOSE_L2[:n2], _font_sub,   TERM_DIM,   _cl2_x, _cl2_y, _lh_sub),
        (CLOSE_L3[:n3], _font_sub,   TERM_DIM,   _cl3_x, _cl3_y, _lh_sub),
    ]
    for li, (txt, font, col, x, y, lh) in enumerate(specs):
        if txt: draw.text((x, y), txt, font=font, fill=col)
        active = (li == 0 and n2 == 0) or \
                 (li == 1 and n2 < len(CLOSE_L2) and n1 == len(CLOSE_L1)) or \
                 (li == 2 and n3 < len(CLOSE_L3) and n2 == len(CLOSE_L2))
        if cursor_on and active:
            cx = x + _text_width(font, txt); cw = _char_width(font)
            draw.rectangle([cx, y, cx+cw-1, y+lh-1], fill=col)
    return np.array(img)


def paint_text_on(base_np, text_full, n_chars, font, x, y, lh, cursor_on, color=None):
    """Draw typewriter text onto an existing numpy frame; return new numpy array."""
    if color is None: color = TERM_GREEN
    img  = Image.fromarray(base_np)
    draw = ImageDraw.Draw(img)
    p = text_full[:n_chars]
    if p: draw.text((x, y), p, font=font, fill=color)
    if cursor_on:
        cx = x + _text_width(font, p); cw = _char_width(font)
        draw.rectangle([cx, y, cx+cw-1, y+lh-1], fill=color)
    return np.array(img)


# ── typewriter section renderers ───────────────────────────────────────────────
def typewriter_frames_open(enc, offset):
    # Line 1 — original pace with natural pauses at "..."
    ev1, cl1, t1_end = build_typewriter_events(OPEN_LINE1, 0.09)
    for ct in cl1: all_click_times.append(offset + ct)
    # no bell in the opening
    # Line 2 starts after a short pause
    LINE2_DELAY = 0.35
    ev2, cl2, t2_end = build_typewriter_events(OPEN_LINE2, 0.07,
                                                start_t=t1_end + LINE2_DELAY)
    for ct in cl2: all_click_times.append(offset + ct)
    typing_end = t2_end
    total_dur  = typing_end + OPEN_HOLD
    n_frames   = int(np.ceil(total_dur * OUT_FPS))
    ei1 = 0; ei2 = 0; nc1 = 0; nc2 = 0
    for fi in range(n_frames):
        ft = fi / OUT_FPS
        while ei1 < len(ev1) and ev1[ei1][0] <= ft: nc1 = ev1[ei1][1]; ei1 += 1
        while ei2 < len(ev2) and ev2[ei2][0] <= ft: nc2 = ev2[ei2][1]; ei2 += 1
        hold  = ft >= typing_end
        blink = (fi // 12) % 2 == 0
        frame = render_open_frame(nc1, nc2, cursor_on=not hold or blink)
        enc.stdin.write(frame.tobytes())
    return total_dur


def typewriter_frames_close(enc, offset):
    line_pauses = [0.4, 0.4, 0.4]
    lines       = [CLOSE_L1, CLOSE_L2, CLOSE_L3]
    events_all  = []; t = 0.0; counts = [0, 0, 0]
    for li, line in enumerate(lines):
        evts, clicks, t = build_typewriter_events(line, 0.075, start_t=t)
        for ct in clicks: all_click_times.append(offset + ct)
        for at, n in evts:
            c = list(counts); c[li] = n; events_all.append((at, c[0], c[1], c[2]))
        counts[li] = len(line); t += line_pauses[li]
    typing_end = t; total_dur = typing_end + CLOSE_HOLD
    n_frames = int(np.ceil(total_dur * OUT_FPS))
    ei = 0; n1, n2, n3 = 0, 0, 0
    for fi in range(n_frames):
        ft = fi / OUT_FPS
        while ei < len(events_all) and events_all[ei][0] <= ft:
            _, n1, n2, n3 = events_all[ei]; ei += 1
        hold = ft >= typing_end
        frame = render_close_frame(n1, n2, n3, cursor_on=not hold or (fi//12)%2==0)
        enc.stdin.write(frame.tobytes())
    return total_dur


# ── source streams ─────────────────────────────────────────────────────────────
class SrcStream:
    FS = SRC_W * SRC_H * 3
    def __init__(self, path, start_t=0.0):
        self.start_t = start_t; self.idx = -1; self.frame = None
        self.proc = subprocess.Popen(
            ["ffmpeg", "-ss", str(start_t), "-i", path,
             "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._next()
    def _next(self):
        raw = self.proc.stdout.read(self.FS)
        if len(raw) >= self.FS:
            self.frame = np.frombuffer(raw, np.uint8).reshape(SRC_H, SRC_W, 3).copy()
            self.idx += 1; return True
        return False
    def get(self, src_t):
        tgt = max(0, int((src_t - self.start_t) * SRC_FPS))
        while self.idx < tgt:
            if not self._next(): break
        return self.frame
    def close(self):
        try: self.proc.stdout.close(); self.proc.terminate(); self.proc.wait()
        except: pass


class VideoStream:
    """Generic video stream (arbitrary resolution / FPS)."""
    def __init__(self, path, w, h, fps, start_t=0.0):
        self.w = w; self.h = h; self.fps = fps
        self.FS = w * h * 3; self.start_t = start_t; self.idx = -1; self.frame = None
        self.proc = subprocess.Popen(
            ["ffmpeg", "-ss", str(start_t), "-i", path,
             "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._next()
    def _next(self):
        raw = self.proc.stdout.read(self.FS)
        if len(raw) >= self.FS:
            self.frame = np.frombuffer(raw, np.uint8).reshape(self.h, self.w, 3).copy()
            self.idx += 1; return True
        return False
    def get(self, t):
        tgt = max(0, int((t - self.start_t) * self.fps))
        while self.idx < tgt:
            if not self._next(): break
        return self.frame
    def close(self):
        try: self.proc.stdout.close(); self.proc.terminate(); self.proc.wait()
        except: pass


# ── composition ────────────────────────────────────────────────────────────────
GRID_ORDER = [1, 2, 3, 4, 0, 7, 9, 6, 11]  # mid-right=8.mp4, bottom-left=10.mp4, bottom-right=12.mp4

def compose_center(sf):
    """Fallback: upscale 640×360 obs from combined source (used for grid cells only)."""
    out = upscale_3x(sf[:CH, :CW])
    out[:MAP_HQ, OUT_W-MAP_HQ:] = sf[:CH, 648:1008]
    return out

def draw_map_label(map_frame):
    """Stamp 'bird's eye view' in black directly onto the 360×360 map frame."""
    img  = Image.fromarray(map_frame)
    draw = ImageDraw.Draw(img)
    draw.text((_lbl_map_x, _lbl_map_y), MAP_LABEL, font=_font_lbl, fill=(0, 0, 0))
    return np.array(img)


def compose_center_highres(obs_frame, map_frame):
    """Center video: obs already 1920×1080 — just overlay 360×360 map, no upscaling."""
    out = obs_frame.copy()
    out[:MAP_HQ, OUT_W-MAP_HQ:] = map_frame
    return out

def area_downscale_3x(img):
    """1920×1080 → 640×360 via exact 3× area averaging."""
    h, w, c = img.shape
    r = img.reshape(h//3, 3, w, c).mean(axis=1)
    r = r.reshape(h//3, w//3, 3, c).mean(axis=2)
    return r.astype(np.uint8)

def crop_map_content(map_src, pad=4):
    """Crop a map frame to its non-white content bounding box, with a small pad."""
    gray = map_src.mean(axis=2)
    mask = gray < 240          # non-white pixels
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return map_src         # fully white — return as-is
    r0 = max(rows[0]  - pad, 0)
    r1 = min(rows[-1] + pad + 1, map_src.shape[0])
    c0 = max(cols[0]  - pad, 0)
    c1 = min(cols[-1] + pad + 1, map_src.shape[1])
    return map_src[r0:r1, c0:c1]

def compose_grid(frames, center_hq_obs=None, center_hq_map=None):
    """center_hq_obs/map: optional highres arrays for the grid centre cell (position 4)."""
    cells = []
    for idx, vi in enumerate(GRID_ORDER):
        if idx == 4 and center_hq_obs is not None:
            obs = area_downscale_3x(center_hq_obs)      # 1920×1080 → 640×360
            pip = resize_bilinear(center_hq_map, MAP_PIP_GRID, MAP_PIP_GRID)
            obs[:MAP_PIP_GRID, CW-MAP_PIP_GRID:] = pip
        else:
            obs = frames[vi][:CH, :CW].copy()
            # Crop map to its content bounding box so sparse maps fill the PIP
            map_src = frames[vi][:CH, 648:1008]         # 360×360 crop
            if vi in (6, 7):  # sparse map videos — crop to content, preserve aspect ratio
                map_cropped = crop_map_content(map_src)
                ch, cw = map_cropped.shape[:2]
                scale = MAP_PIP_GRID / max(ch, cw)
                th, tw = int(round(ch * scale)), int(round(cw * scale))
                resized = resize_bilinear(map_cropped, th, tw)
                pip = np.full((MAP_PIP_GRID, MAP_PIP_GRID, 3), 255, dtype=np.uint8)
                yo = (MAP_PIP_GRID - th) // 2
                xo = (MAP_PIP_GRID - tw) // 2
                pip[yo:yo+th, xo:xo+tw] = resized
            else:
                pip = resize_bilinear(map_src, MAP_PIP_GRID, MAP_PIP_GRID)
            obs[:MAP_PIP_GRID, CW-MAP_PIP_GRID:] = pip
        cells.append(obs)
    rows = [np.concatenate(cells[r*3:(r+1)*3], axis=1) for r in range(3)]
    return np.concatenate(rows, axis=0)

def interlude_alpha(ti):
    """Smooth alpha envelope: light during text, dark during hobbit video."""
    if ti < INTER_FADE:
        return smoothstep(ti / INTER_FADE) * INTER_ALPHA_TEXT
    t_ramp_start = _hobbit_rel_s - INTER_HOB_TRANS
    t_ramp_back  = _hobbit_rel_gone
    t_ramp_done  = _hobbit_rel_gone + INTER_HOB_TRANS
    if ti < t_ramp_start:
        return INTER_ALPHA_TEXT
    elif ti < _hobbit_rel_s:
        u = (ti - t_ramp_start) / INTER_HOB_TRANS
        return INTER_ALPHA_TEXT + (INTER_ALPHA - INTER_ALPHA_TEXT) * smoothstep(u)
    elif ti < t_ramp_back:
        return INTER_ALPHA
    elif ti < t_ramp_done:
        u = (ti - t_ramp_back) / INTER_HOB_TRANS
        return INTER_ALPHA + (INTER_ALPHA_TEXT - INTER_ALPHA) * smoothstep(u)
    elif ti < _inter_fo_s:
        return INTER_ALPHA_TEXT
    else:
        outro_t = ti - _inter_fo_s
        return INTER_ALPHA_TEXT * smoothstep(1.0 - outro_t / INTER_OUTRO)


def zoom_frame(grid, s):
    cx = int(round(CW*(1-s))); cy = int(round(CH*(1-s)))
    cw = min(int(round(CW+(OUT_W-CW)*s)), OUT_W-cx)
    ch = min(int(round(CH+(OUT_H-CH)*s)), OUT_H-cy)
    return resize_bilinear(grid[cy:cy+ch, cx:cx+cw], OUT_H, OUT_W)


# ── encoder setup ──────────────────────────────────────────────────────────────
enc = subprocess.Popen(
    ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{OUT_W}x{OUT_H}", "-r", str(OUT_FPS), "-i", "pipe:0",
     "-c:v", "libx264", "-crf", "18", "-preset", "fast",
     "-pix_fmt", "yuv420p", VID_ONLY],
    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

# ── PHASE 1: opening typewriter ────────────────────────────────────────────────
print("Rendering opening typewriter …")
t_cursor = 0.0
t_cursor += typewriter_frames_open(enc, offset=t_cursor)
print(f"  → cursor at {t_cursor:.2f}s")

# ── PHASE 2: opening blink ─────────────────────────────────────────────────────
print("Opening highres center streams …")
hr_obs    = VideoStream(HIGHRES_OBS, OUT_W, OUT_H, SRC_FPS, start_t=0.0)
hr_map    = VideoStream(HIGHRES_MAP, MAP_HQ, MAP_HQ, SRC_FPS, start_t=0.0)
cstream   = SrcStream(SRC[0], start_t=0.0)   # still needed for grid cells
first_frame = compose_center_highres(hr_obs.get(0.0), hr_map.get(0.0))
for i in range(int(BLINK_DUR * OUT_FPS)):
    enc.stdin.write(apply_eye_mask(first_frame, _interp_kf(BLINK_KF, i*dt)).tobytes())
t_cursor += BLINK_DUR
print(f"  → cursor at {t_cursor:.2f}s")

# ── PHASE 3: main video (with interlude embedded) ──────────────────────────────
# Register interlude clicks now that we know the absolute offset
inter_abs = t_cursor + P2
for ct in _ic1: all_click_times.append(inter_abs + ct)
for ct in _ic2: all_click_times.append(inter_abs + ct)

print(f"Rendering main video ({P6:.1f}s) …")
gstreams       = None
hstream        = None
last_vid_frame = first_frame.copy()
P4F, P5F       = int(P4 * OUT_FPS), int(P5 * OUT_FPS)

# Per-event indices for interlude text (advanced incrementally)
_ei1 = 0;  _nc1 = 0
_ei2 = 0;  _nc2 = 0

for i in range(N):
    t     = i * dt
    src_t = src_times[i]

    # ── center video (slow / ramp-up) ─────────────────────────────────────────
    if t < P2:
        out = compose_center_highres(hr_obs.get(src_t), draw_map_label(hr_map.get(src_t)))

    # ── interlude ─────────────────────────────────────────────────────────────
    elif t < P3:
        ti   = t - P2
        base = compose_center_highres(hr_obs.get(src_t), draw_map_label(hr_map.get(src_t)))

        # single alpha drives all interlude phases
        alpha = interlude_alpha(ti)
        dark  = (base.astype(np.float32) * (1 - alpha)).clip(0, 255).astype(np.uint8)
        blink = (int(ti * OUT_FPS) // 12) % 2 == 0

        # ── hobbit video (+ hold on last frame) ───────────────────────────────
        if _hobbit_rel_s <= ti < _hobbit_rel_gone:
            if hstream is None:
                print(f"\n  Opening hobbit stream …")
                hstream = VideoStream(HOBBIT_SRC, HOBBIT_W, HOBBIT_H,
                                      HOBBIT_FPS, start_t=0.0)
            hobbit_t = min(ti - _hobbit_rel_s, HOBBIT_DUR - 1.0/HOBBIT_FPS)
            hf = hstream.get(hobbit_t)
            out = dark.copy()
            out[HOBBIT_PY:HOBBIT_PY+HOBBIT_H, HOBBIT_PX:HOBBIT_PX+HOBBIT_W] = hf

        # ── text 1 ────────────────────────────────────────────────────────────
        elif _TEXT1_START <= ti < _hobbit_rel_s:
            while _ei1 < len(_ie1) and _ie1[_ei1][0] <= ti:
                _nc1 = _ie1[_ei1][1]; _ei1 += 1
            hold = ti >= _it1_end
            out = paint_text_on(dark, INTER_TEXT1, _nc1,
                                 _font_inter, _it1_x, _inter_y, _lh_inter,
                                 cursor_on=not hold or blink)

        # ── text 2 ────────────────────────────────────────────────────────────
        elif _hobbit_rel_gone <= ti < _inter_fo_s:
            while _ei2 < len(_ie2) and _ie2[_ei2][0] <= ti:
                _nc2 = _ie2[_ei2][1]; _ei2 += 1
            hold = ti >= _it2_end
            out = paint_text_on(dark, INTER_TEXT2, _nc2,
                                 _font_inter, _it2_x, _inter_y, _lh_inter,
                                 cursor_on=not hold or blink)

        # ── fade-in / gap / outro ─────────────────────────────────────────────
        else:
            out = dark

    # ── ramp-down / zoom / grid ───────────────────────────────────────────────
    elif t < P4:
        out = compose_center_highres(hr_obs.get(src_t), draw_map_label(hr_map.get(src_t)))
    else:
        if gstreams is None:
            print(f"\n  Opening 9 grid streams at src_t={src_t:.1f}s …")
            gstreams = [SrcStream(SRC[k], start_t=src_t) for k in range(12)]
        frames = [gs.get(src_t) for gs in gstreams]
        grid   = compose_grid(frames,
                              center_hq_obs=hr_obs.get(src_t),
                              center_hq_map=hr_map.get(src_t))
        out    = zoom_frame(grid, smoothstep((t-P4)/(P5-P4))) if t < P5 else grid

    last_vid_frame = out
    enc.stdin.write(out.tobytes())
    if i % (OUT_FPS * 5) == 0:
        print(f"  {i}/{N}  t={t:.1f}s  src={src_t:.1f}s", end="\r", flush=True)

t_cursor += P6
print(f"\n  → cursor at {t_cursor:.2f}s")

# ── PHASE 4: closing blink ─────────────────────────────────────────────────────
print("Rendering closing blink …")
for i in range(int(CLOSE_BLINK_DUR * OUT_FPS)):
    enc.stdin.write(
        apply_eye_mask(last_vid_frame, _interp_kf(CLOSE_BLINK_KF, i*dt)).tobytes())
t_cursor += CLOSE_BLINK_DUR

# ── PHASE 5: closing typewriter ────────────────────────────────────────────────
print("Rendering closing typewriter …")
close_abs = t_cursor   # absolute start of closing title sequence
t_cursor += typewriter_frames_close(enc, offset=t_cursor)
print(f"  → cursor at {t_cursor:.2f}s")

# ── finalise video ─────────────────────────────────────────────────────────────
print("Finalising video …")
enc.stdin.close(); enc.wait()
hr_obs.close(); hr_map.close(); cstream.close()
if hstream:  hstream.close()
if gstreams: [gs.close() for gs in gstreams]

total_video_dur = t_cursor
print(f"Video-only done → {total_video_dur:.1f}s total")

# ── generate audio ─────────────────────────────────────────────────────────────
print(f"Generating audio: {len(all_click_times)} clicks, {len(all_bell_times)} bells …")
n_audio = int(total_video_dur * SAMPLE_RATE) + SAMPLE_RATE
audio   = np.zeros(n_audio, dtype=np.float32)

def _mix(buf, snd, t_s):
    s = int(t_s * SAMPLE_RATE); e = min(s + len(snd), len(buf))
    buf[s:e] += snd[:e-s]

hobbit_abs      = inter_abs + _hobbit_rel_s
hobbit_end_abs  = inter_abs + _hobbit_rel_gone
main_vid_start  = inter_abs - P2          # absolute time main video starts
open_blink_end  = main_vid_start          # blink finishes, main video begins
open_blink_start= main_vid_start - BLINK_DUR

# ── duck envelope: per-click dip so clicks cut through clearly ────────────────
DUCK_LEVEL   = 0.18
DUCK_ATTACK  = int(0.02 * SAMPLE_RATE)
DUCK_RELEASE = int(0.18 * SAMPLE_RATE)
duck_env = np.ones(n_audio, dtype=np.float32)
for ct in sorted(all_click_times + all_bell_times):
    h = int(ct * SAMPLE_RATE)
    a = max(0, h - DUCK_ATTACK)
    r = min(n_audio, h + DUCK_RELEASE)
    if h > a:
        duck_env[a:h] = np.minimum(duck_env[a:h],
                                    np.linspace(1.0, DUCK_LEVEL, h - a, dtype=np.float32))
    if r > h:
        duck_env[h:r] = np.minimum(duck_env[h:r],
                                    np.linspace(DUCK_LEVEL, 1.0, r - h, dtype=np.float32))
# Opening: bach_vol already holds the low level — suppress per-click flutter.
duck_env[:int(open_blink_end * SAMPLE_RATE)] = 1.0
# Closing title: keep music steadily low rather than bouncing on each keystroke.
duck_env[int(close_abs * SAMPLE_RATE):] = 1.0

# ── bach: loops throughout with a smooth volume envelope ──────────────────────
bach_buf = np.zeros(n_audio, dtype=np.float32)
pos = 0
while pos < n_audio:
    end = min(pos + len(_SND_BACH), n_audio)
    bach_buf[pos:end] += _SND_BACH[:end - pos]
    pos += len(_SND_BACH)

bach_vol = np.ones(n_audio, dtype=np.float32)

# 1. Duck during opening typewriter (t=0 → blink start), fade back up during blink
OPEN_DUCK = 0.12   # how low bach goes during opening typing
s0 = 0
e_down = int(0.25 * SAMPLE_RATE)          # reach low level quickly (250ms)
e_low  = int(open_blink_start * SAMPLE_RATE)
e_rise = int(open_blink_end   * SAMPLE_RATE)
if e_down > s0:
    bach_vol[s0:e_down] = np.linspace(1.0, OPEN_DUCK, e_down - s0, dtype=np.float32)
if e_low > e_down:
    bach_vol[e_down:e_low] = OPEN_DUCK
if e_rise > e_low:
    bach_vol[e_low:e_rise] = np.linspace(OPEN_DUCK, 1.0, e_rise - e_low, dtype=np.float32)

# 3. Closing title: fade down before title starts, stay low through writing
CLOSE_DUCK   = 0.12
CLOSE_FADE_IN = 0.5   # seconds to fade down before close_abs
s_close_down = max(0, int((close_abs - CLOSE_FADE_IN) * SAMPLE_RATE))
e_close_down = int(close_abs * SAMPLE_RATE)
if e_close_down > s_close_down:
    bach_vol[s_close_down:e_close_down] = np.linspace(1.0, CLOSE_DUCK,
                                                        e_close_down - s_close_down,
                                                        dtype=np.float32)
bach_vol[e_close_down:n_audio] = CLOSE_DUCK

# 2. Interlude: fade bach out gradually as overlay appears → reaches 0 at hobbit start;
#    stays 0 through hobbit and text2; fades back in during interlude outro.
s_inter    = int(inter_abs              * SAMPLE_RATE)
e_inter_lo = int(hobbit_abs             * SAMPLE_RATE)
e_inter_0  = int((inter_abs + _it2_end) * SAMPLE_RATE)
e_inter_up = int((inter_abs + INTER_DUR)* SAMPLE_RATE)
if e_inter_lo > s_inter:
    bach_vol[s_inter:e_inter_lo] = np.minimum(
        bach_vol[s_inter:e_inter_lo],
        np.linspace(1.0, 0.0, e_inter_lo - s_inter, dtype=np.float32))
if e_inter_0 > e_inter_lo:
    bach_vol[e_inter_lo:e_inter_0] = 0.0
if e_inter_up > e_inter_0:
    bach_vol[e_inter_0:min(e_inter_up, n_audio)] = np.linspace(
        0.0, 1.0, min(e_inter_up, n_audio) - e_inter_0, dtype=np.float32)

audio += bach_buf * bach_vol * duck_env

# ── hobbit music: fade in, play, fade out ────────────────────────────────────
HOBBIT_FADE = 0.6   # seconds to fade in and out
hobbit_samps = len(_SND_HOBBIT)
hobbit_env   = np.ones(hobbit_samps, dtype=np.float32)
fade_n = int(HOBBIT_FADE * SAMPLE_RATE)
if fade_n > 0:
    hobbit_env[:fade_n]              = np.linspace(0.0, 1.0, fade_n)
    hobbit_env[max(0, hobbit_samps - fade_n):] = np.linspace(1.0, 0.0, min(fade_n, hobbit_samps))
hobbit_faded = (_SND_HOBBIT * hobbit_env)
hobbit_buf = np.zeros(n_audio, dtype=np.float32)
hs = int(hobbit_abs * SAMPLE_RATE)
he = min(hs + hobbit_samps, n_audio)
hobbit_buf[hs:he] = hobbit_faded[:he - hs]
audio += hobbit_buf * duck_env

# ── typewriter sound effects (on top, not ducked) ────────────────────────────
for ct in all_click_times: _mix(audio, _SND_CLICK, ct)
for bt in all_bell_times:  _mix(audio, _SND_BELL,  bt)

peak = np.abs(audio).max()
if peak > 0: audio *= 0.80 / peak

audio_i16 = (audio * 32767).astype(np.int16)
with wave.open(AUD_WAV, 'w') as wf:
    wf.setnchannels(1); wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE); wf.writeframes(audio_i16.tobytes())

# ── mux ────────────────────────────────────────────────────────────────────────
print("Muxing …")
subprocess.run(["ffmpeg", "-y", "-i", VID_ONLY, "-i", AUD_WAV,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", OUT],
               check=True, stderr=subprocess.DEVNULL)
for f in [VID_ONLY, AUD_WAV]:
    try: os.remove(f)
    except: pass

print(f"\nDone! → {OUT}  ({total_video_dur:.1f}s total)")
