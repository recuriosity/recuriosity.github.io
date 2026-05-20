#!/usr/bin/env python3
"""
Produce training_labeled.mp4 from training.mp4.

Layout (both rows are 1288px wide):
  top row:    [RGB Obs (640)] [gap] [3DGS Rendering (640)]
  bottom row: [Actions (316)] [gap] [Reward Signal (640)] [gap] [BEV (316)]
  Total: 1288 × 728
"""

import subprocess, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIDEO_IN   = "assets/videos/training.mp4"
VIDEO_MAIN = "assets/videos/training_labeled.mp4"

SRC_W, SRC_H = 2572, 360
FPS  = 12
GAP  = 8

# ── Source column layout ─────────────────────────────────────────────────────
C1_X, C1_W = 0,    640   # RGB observation
C2_X, C2_W = 648,  640   # 3DGS rendering
C3_X, C3_W = 1296, 640   # Reward signal
C4_X, C4_W = 1944, 360   # Bird's eye view
C5_X, C5_W = 2312, 260   # Action distribution chart (prob sampling only)

# ── Output dimensions ─────────────────────────────────────────────────────────
TOP_W  = C1_W + GAP + C2_W   # 1288  (top row)
BOT_W  = TOP_W                # 1288  (bottom row same width)

# Bottom row column widths: Actions | gap | Reward | gap | BEV
# Reward stays 640 centered → flanks are (1288-640-2*GAP)//2 = 300px each
FLANK  = (TOP_W - C3_W - 2 * GAP) // 2   # 300
ACT_W  = FLANK   # 300  (action panel)
BEV_W  = FLANK   # 300  (BEV panel)

# Bottom row x-offsets
BOT_ACT_X  = 0
BOT_REW_X  = ACT_W + GAP               # 308
BOT_BEV_X  = ACT_W + GAP + C3_W + GAP  # 956

OUT_W = TOP_W   # 1288
OUT_H = SRC_H + GAP + SRC_H            # 728

LABEL_H = 32

# ── Reward indicator position (top-right of reward cell in output frame) ─────
CIRCLE_R     = 36
CIRCLE_ALPHA = 180
CIRCLE_CX    = BOT_REW_X + C3_W - CIRCLE_R - 8
CIRCLE_CY    = SRC_H + GAP + CIRCLE_R + 8

# ── Fonts ────────────────────────────────────────────────────────────────────
def load_font(size, bold=False):
    candidates = []
    if bold:
        candidates += ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                       "/System/Library/Fonts/Supplemental/Verdana Bold.ttf"]
    candidates += ["/System/Library/Fonts/Helvetica.ttc",
                   "/System/Library/Fonts/Arial.ttf",
                   "/Library/Fonts/Arial.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for p in candidates:
        if os.path.exists(p):
            try:   return ImageFont.truetype(p, size)
            except: pass
    return ImageFont.load_default()

label_font = load_font(20, bold=True)

# ── Reward indicator ──────────────────────────────────────────────────────────
def make_indicator(png_path, size):
    icon = Image.open(png_path).convert("RGBA").resize((size, size), Image.LANCZOS)
    out  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d    = ImageDraw.Draw(out)
    d.ellipse([0, 0, size-1, size-1], fill=(255, 255, 255, CIRCLE_ALPHA))
    pad   = size // 7
    small = icon.resize((size-pad*2, size-pad*2), Image.LANCZOS)
    out.paste(small, (pad, pad), small)
    return np.array(out)

_IS = CIRCLE_R * 2
_ind_coins = make_indicator("assets/images/coins.png", _IS)
_ind_cross = make_indicator("assets/images/cross.png", _IS)

def is_green_reward(src):
    region = src[335:360, C3_X:C3_X+C3_W]
    green  = ((region[:,:,1].astype(int) - region[:,:,0].astype(int)) > 60) & (region[:,:,1] > 100)
    return green.sum() > 10

def paste_indicator(arr, ind_rgba):
    x0, y0 = CIRCLE_CX - CIRCLE_R, CIRCLE_CY - CIRCLE_R
    x1, y1 = x0 + _IS, y0 + _IS
    roi   = arr[y0:y1, x0:x1].astype(float)
    alpha = ind_rgba[:,:,3:4].astype(float) / 255.0
    arr[y0:y1, x0:x1] = np.clip(
        roi*(1-alpha) + ind_rgba[:,:,:3].astype(float)*alpha, 0, 255
    ).astype(np.uint8)

# ── BEV scaling (360×360 → BEV_W×BEV_W, centered vertically in SRC_H cell) ─
_BEV_SZ   = BEV_W                            # 300 px square
_BEV_PAD_Y = (SRC_H - _BEV_SZ) // 2          # 30 px top/bottom pad

def scaled_bev(src):
    col  = src[:, C4_X:C4_X+C4_W]             # 360×360
    img  = Image.fromarray(col).resize((_BEV_SZ, _BEV_SZ), Image.LANCZOS)
    cell = np.full((SRC_H, BEV_W, 3), 255, dtype=np.uint8)
    cell[_BEV_PAD_Y:_BEV_PAD_Y+_BEV_SZ, :] = np.array(img)
    return cell

# ── Gamepad rendering ─────────────────────────────────────────────────────────
_CHART_SAMPLES = {'F': (130,75), 'L': (75,130), 'R': (185,130), 'S': (130,185)}
_SAMPLE_R = 10

def read_action_probs(src, c5x):
    vals = {}
    for action, (dx, dy) in _CHART_SAMPLES.items():
        patch = src[dy-_SAMPLE_R:dy+_SAMPLE_R, c5x+dx-_SAMPLE_R:c5x+dx+_SAMPLE_R]
        vals[action] = float(patch.mean())
    return vals

_BTN_SZ = 64
_BTN_R  = 11
_BTN_ACTIVE   = (59, 130, 246)
_BTN_INACTIVE = (210, 210, 210)
_ARR_ACTIVE   = (255, 255, 255)
_ARR_INACTIVE = (130, 130, 130)

# Button centers in a ACT_W × SRC_H panel (SRC_H=360, LABEL_H=32 → 328 usable)
_GP_CX    = ACT_W // 2   # 150
_GP_UP_Y  = 110
_GP_LR_Y  = 230
_GP_LX    = ACT_W // 4           # 75
_GP_RX    = ACT_W * 3 // 4       # 225

def _arrow_pts(cx, cy, direction, size=24):
    h, w = size * 0.85, size * 0.65
    if direction == 'up':
        return [(cx, cy-h/2), (cx-w/2, cy+h/2), (cx+w/2, cy+h/2)]
    if direction == 'left':
        return [(cx-h/2, cy), (cx+h/2, cy-w/2), (cx+h/2, cy+w/2)]
    if direction == 'right':
        return [(cx+h/2, cy), (cx-h/2, cy-w/2), (cx-h/2, cy+w/2)]

def render_gamepad(probs):
    """Return ACT_W × SRC_H numpy RGB array with D-pad buttons + label."""
    img  = Image.new("RGB", (ACT_W, SRC_H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    max_act = max(('F','L','R'), key=lambda a: probs[a])
    for action, direction, (bcx, bcy) in [
        ('F', 'up',    (_GP_CX, _GP_UP_Y)),
        ('L', 'left',  (_GP_LX, _GP_LR_Y)),
        ('R', 'right', (_GP_RX, _GP_LR_Y)),
    ]:
        active = (action == max_act)
        bx0, by0 = bcx - _BTN_SZ//2, bcy - _BTN_SZ//2
        bx1, by1 = bx0 + _BTN_SZ,     by0 + _BTN_SZ
        draw.rounded_rectangle([bx0, by0, bx1, by1], radius=_BTN_R,
                               fill=(_BTN_ACTIVE if active else _BTN_INACTIVE))
        pts = _arrow_pts(bcx, bcy, direction, size=24)
        draw.polygon([(int(x), int(y)) for x, y in pts],
                     fill=(_ARR_ACTIVE if active else _ARR_INACTIVE))
    return np.array(img)

# ── Label helper ─────────────────────────────────────────────────────────────
_probe = ImageDraw.Draw(Image.new("RGB",(1,1)))

def cell_label(draw, text, x0, y0, w, h):
    bb = _probe.textbbox((0,0), text, font=label_font)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    sy = y0 + h - LABEL_H
    draw.rectangle([x0, sy, x0+w-1, y0+h-1], fill=(255,255,255))
    draw.text((x0 + w//2 - tw//2 - bb[0], sy + (LABEL_H-th)//2 - bb[1]),
              text, font=label_font, fill=(10,10,10))

# ── Pipes ─────────────────────────────────────────────────────────────────────
reader = subprocess.Popen(
    ["ffmpeg", "-i", VIDEO_IN, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

writer_main = subprocess.Popen(
    ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", VIDEO_MAIN],
    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

frame_bytes = SRC_W * SRC_H * 3
processed   = 0

while True:
    raw = reader.stdout.read(frame_bytes)
    if len(raw) < frame_bytes:
        break

    src   = np.frombuffer(raw, dtype=np.uint8).reshape(SRC_H, SRC_W, 3)
    green = is_green_reward(src)

    main = np.full((OUT_H, OUT_W, 3), 255, dtype=np.uint8)

    # Top row
    main[0:SRC_H, 0:C1_W]            = src[:, C1_X:C1_X+C1_W]
    main[0:SRC_H, C1_W+GAP:TOP_W]    = src[:, C2_X:C2_X+C2_W]

    # Bottom row
    R = SRC_H + GAP
    main[R:R+SRC_H, BOT_ACT_X:BOT_ACT_X+ACT_W] = render_gamepad(read_action_probs(src, C5_X))
    main[R:R+SRC_H, BOT_REW_X:BOT_REW_X+C3_W]  = src[:, C3_X:C3_X+C3_W]
    main[R:R+SRC_H, BOT_BEV_X:BOT_BEV_X+BEV_W] = scaled_bev(src)

    paste_indicator(main, _ind_coins if green else _ind_cross)

    img  = Image.fromarray(main)
    draw = ImageDraw.Draw(img)
    cell_label(draw, "RGB Observation",  0,         0, C1_W,  SRC_H)
    cell_label(draw, "3DGS Rendering",   C1_W+GAP,  0, C2_W,  SRC_H)
    cell_label(draw, "Actions",          BOT_ACT_X, R, ACT_W, SRC_H)
    cell_label(draw, "Reward Signal",    BOT_REW_X, R, C3_W,  SRC_H)
    cell_label(draw, "Bird's Eye View",  BOT_BEV_X, R, BEV_W, SRC_H)
    writer_main.stdin.write(np.array(img).tobytes())

    processed += 1
    if processed % 120 == 0:
        print(f"  {processed} frames ({processed/FPS:.0f}s)...", flush=True)

reader.stdout.close(); reader.wait()
writer_main.stdin.close(); writer_main.wait()
print(f"Done. {processed} frames → {VIDEO_MAIN} ({OUT_W}×{OUT_H})")
