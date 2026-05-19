#!/usr/bin/env python3
"""
Produce two videos from training.mp4:
  training_labeled.mp4  — 4-column main (RGB Obs, 3DGS, Reward, Gamepad)
  training_bev.mp4      — col4 only (Bird's Eye View, 360×360)
"""

import subprocess, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIDEO_IN   = "assets/videos/training.mp4"
VIDEO_MAIN = "assets/videos/training_labeled.mp4"
VIDEO_ACT  = "assets/videos/training_bev.mp4"   # Bird's Eye View panel

SRC_W, SRC_H = 2572, 360
FPS = 12
GAP = 8

# Source column layout
C1_X, C1_W = 0,    640
C2_X, C2_W = 648,  640
C3_X, C3_W = 1296, 640
C4_X, C4_W = 1944, 360
C5_X, C5_W = 2312, 260

# Main output: cols 1, 2, 3, 5 (Action Distribution moves to 4th slot)
OUT_W = C1_W + GAP + C2_W + GAP + C3_W + GAP + C5_W   # 1940
OUT_H = SRC_H

D_C1 = 0
D_C2 = C1_W + GAP
D_C3 = C1_W + GAP + C2_W + GAP
D_C5 = C1_W + GAP + C2_W + GAP + C3_W + GAP   # action dist in 4th slot

# S→P — col5 is now at D_C5 in main video
S_ERASE   = (D_C5 + 127, 206, D_C5 + 146, 226)
S_CENTER  = (D_C5 + 136, 216)
BG_SAMPLE = (D_C5 + 144, 211, D_C5 + 156, 221)

# Reward indicator (in output frame, top-right of col3)
CIRCLE_R     = 36
CIRCLE_ALPHA = 180
CIRCLE_CX    = D_C3 + C3_W - CIRCLE_R - 8
CIRCLE_CY    = CIRCLE_R + 8

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
    arr[y0:y1, x0:x1] = np.clip(roi*(1-alpha) + ind_rgba[:,:,:3].astype(float)*alpha,
                                  0, 255).astype(np.uint8)

LABEL_H = 32

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
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()

label_font = load_font(20, bold=True)
p_font     = load_font(12)

# ── Gamepad rendering ────────────────────────────────────────────────────────
# Sample centers for F/L/R/S in source col-5 (260×260 chart, center≈(130,130))
_CHART_SAMPLES = {
    'F': (130, 75),
    'L': (75,  130),
    'R': (185, 130),
    'S': (130, 185),
}
_SAMPLE_R = 10   # half-side of averaging patch

def read_action_probs(src, c5x):
    """Return (F,L,R,S) brightness values from source col-5 quadrant chart."""
    vals = {}
    for action, (dx, dy) in _CHART_SAMPLES.items():
        patch = src[dy-_SAMPLE_R:dy+_SAMPLE_R, c5x+dx-_SAMPLE_R:c5x+dx+_SAMPLE_R]
        vals[action] = float(patch.mean())
    return vals

_BTN_W, _BTN_H = 72, 72
_BTN_R = 12       # corner radius
_BTN_COL_ACTIVE   = (59, 130, 246)    # blue
_BTN_COL_INACTIVE = (210, 210, 210)   # light gray
_ARROW_ACTIVE     = (255, 255, 255)
_ARROW_INACTIVE   = (130, 130, 130)

# Button centers inside the 260×(SRC_H-LABEL_H) panel
_GAP_H = SRC_H - LABEL_H   # 328 px
_CX    = C5_W // 2          # 130
_BTN_UP    = (_CX,           85)
_BTN_LEFT  = (_CX - 84,     175)
_BTN_RIGHT = (_CX + 84,     175)

def _arrow_polygon(cx, cy, direction, size=26):
    h = size * 0.85
    w = size * 0.65
    if direction == 'up':
        return [(cx, cy-h/2), (cx-w/2, cy+h/2), (cx+w/2, cy+h/2)]
    if direction == 'left':
        return [(cx-h/2, cy), (cx+h/2, cy-w/2), (cx+h/2, cy+w/2)]
    if direction == 'right':
        return [(cx+h/2, cy), (cx-h/2, cy-w/2), (cx-h/2, cy+w/2)]

def render_gamepad(probs):
    """Return a 260×SRC_H numpy RGB array with D-pad buttons."""
    img  = Image.new("RGB", (C5_W, SRC_H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    max_action = max(('F','L','R'), key=lambda a: probs[a])

    buttons = [
        ('F', 'up',    _BTN_UP),
        ('L', 'left',  _BTN_LEFT),
        ('R', 'right', _BTN_RIGHT),
    ]
    for action, direction, (bcx, bcy) in buttons:
        active = (action == max_action)
        bx0, by0 = bcx - _BTN_W//2, bcy - _BTN_H//2
        bx1, by1 = bx0 + _BTN_W,     by0 + _BTN_H
        draw.rounded_rectangle([bx0, by0, bx1, by1],
                               radius=_BTN_R,
                               fill=(_BTN_COL_ACTIVE if active else _BTN_COL_INACTIVE))
        pts = _arrow_polygon(bcx, bcy, direction, size=30)
        draw.polygon([(int(x), int(y)) for x, y in pts],
                     fill=(_ARROW_ACTIVE if active else _ARROW_INACTIVE))

    return np.array(img)

# Pre-compute label positions for 4-col main video (cols 1,2,3,5)
MAIN_COLS = [
    ("RGB Observation",    D_C1, C1_W),
    ("3DGS Rendering",     D_C2, C2_W),
    ("Reward Signal",      D_C3, C3_W),
    ("Actions",            D_C5, C5_W),
]
_probe = ImageDraw.Draw(Image.new("RGB", (1,1)))
main_labels = []
for name, dx, dw in MAIN_COLS:
    bb = _probe.textbbox((0,0), name, font=label_font)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    sy = SRC_H - LABEL_H
    tx = dx + dw//2 - tw//2 - bb[0]
    ty = sy + (LABEL_H - th)//2 - bb[1]
    main_labels.append((name, dx, dx+dw, sy, tx, ty))

# Label for BEV panel video
bb = _probe.textbbox((0,0), "Bird's Eye View", font=label_font)
tw, th = bb[2]-bb[0], bb[3]-bb[1]
act_label = ("Bird's Eye View",
             C4_W//2 - tw//2 - bb[0],
             SRC_H - LABEL_H + (LABEL_H-th)//2 - bb[1])

# ── Pipes ────────────────────────────────────────────────────────────────────
reader = subprocess.Popen(
    ["ffmpeg", "-i", VIDEO_IN, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

writer_main = subprocess.Popen(
    ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", VIDEO_MAIN],
    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

writer_act = subprocess.Popen(
    ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{C4_W}x{SRC_H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", VIDEO_ACT],
    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

frame_bytes = SRC_W * SRC_H * 3
processed = 0

while True:
    raw = reader.stdout.read(frame_bytes)
    if len(raw) < frame_bytes:
        break

    src   = np.frombuffer(raw, dtype=np.uint8).reshape(SRC_H, SRC_W, 3)
    green = is_green_reward(src)

    # ── Main video (cols 1,2,3 + gamepad panel) ───────────────────────────────
    main = np.full((OUT_H, OUT_W, 3), 255, dtype=np.uint8)
    main[:, D_C1:D_C1+C1_W] = src[:, C1_X:C1_X+C1_W]
    main[:, D_C2:D_C2+C2_W] = src[:, C2_X:C2_X+C2_W]
    main[:, D_C3:D_C3+C3_W] = src[:, C3_X:C3_X+C3_W]

    probs = read_action_probs(src, C5_X)
    main[:, D_C5:D_C5+C5_W] = render_gamepad(probs)

    paste_indicator(main, _ind_coins if green else _ind_cross)

    img = Image.fromarray(main)
    draw = ImageDraw.Draw(img)
    for name, rx0, rx1, sy, tx, ty in main_labels:
        draw.rectangle([rx0, sy, rx1-1, OUT_H-1], fill=(255,255,255))
        draw.text((tx, ty), name, font=label_font, fill=(10,10,10))
    writer_main.stdin.write(np.array(img).tobytes())

    # ── BEV panel video (col4 only) ───────────────────────────────────────────
    bev = src[:, C4_X:C4_X+C4_W].copy()
    bev_img = Image.fromarray(bev)
    bdraw = ImageDraw.Draw(bev_img)
    bdraw.rectangle([0, SRC_H-LABEL_H, C4_W-1, SRC_H-1], fill=(255,255,255))
    bdraw.text((act_label[1], act_label[2]), act_label[0], font=label_font, fill=(10,10,10))
    writer_act.stdin.write(np.array(bev_img).tobytes())

    processed += 1
    if processed % 120 == 0:
        print(f"  {processed} frames ({processed/FPS:.0f}s)...", flush=True)

reader.stdout.close(); reader.wait()
writer_main.stdin.close(); writer_main.wait()
writer_act.stdin.close(); writer_act.wait()
print(f"Done. {processed} frames → {VIDEO_MAIN}, {VIDEO_ACT}")
