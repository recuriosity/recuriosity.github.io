"""
Process image-goal.mp4 into a single image-goal-main.mp4:
  Source layout: C1(640) + 8 + C2(640) + 8 + C3(364) = 1660 wide, 368 tall
    C1: agent RGB observation
    C2: target image (goal)
    C3: bird's eye view map

  Output at 2x resolution (2024 × 736):
    LEFT  (1280px): RGB observation with target image PIP in top-right corner
    GAP   (16px)
    RIGHT (728px):  Bird's eye view
    Bottom labels on each column.
"""
import subprocess, os, numpy as np
from PIL import Image, ImageDraw, ImageFont

VIDEO_IN   = "assets/videos/image-goal.mp4"
VIDEO_MAIN = "assets/videos/image-goal-main.mp4"

SRC_W, SRC_H = 1660, 368
FPS  = 12
GAP  = 8

C1_X, C1_W = 0,    640
C2_X, C2_W = 648,  640   # target image
C3_X, C3_W = 1296, 364   # BEV

# ── 2x output dimensions ─────────────────────────────────────────────────────
S = 2                                    # scale factor
OUT_W = (C1_W + GAP + C3_W) * S         # (640+8+364)*2 = 2024
OUT_H = SRC_H * S                        # 368*2 = 736
OC1_W = C1_W * S                         # 1280
OC3_W = C3_W * S                         # 728
OGAP  = GAP  * S                         # 16

LABEL_H = 36 * S   # 72px at 2x (visually ~36pt)

# ── Target image PIP ──────────────────────────────────────────────────────────
PIP_W  = 150 * S     # 300px at 2x
PIP_H  = int(PIP_W * SRC_H / C2_W)   # keep aspect of col2: ≈ 276
PIP_LABEL_H = 32 * S                  # strip below PIP image: 64px
PIP_BOX_H = PIP_H + PIP_LABEL_H
PIP_MARGIN = 8 * S   # 16px margin from edges
PIP_BORDER = 3 * S   # 6px white border
PIP_X = OC1_W - PIP_W - PIP_MARGIN   # right-aligned
PIP_Y = PIP_MARGIN                    # top-aligned

# ── Fonts ─────────────────────────────────────────────────────────────────────
def load_font(size, bold=False):
    candidates = []
    if bold:
        candidates += ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                       "/System/Library/Fonts/Supplemental/Verdana Bold.ttf"]
    candidates += ["/System/Library/Fonts/Helvetica.ttc",
                   "/System/Library/Fonts/Arial.ttf",
                   "/Library/Fonts/Arial.ttf"]
    for p in candidates:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except: pass
    return ImageFont.load_default()

label_font     = load_font(20 * S, bold=True)   # 40pt
pip_label_font = load_font(14 * S, bold=True)   # 28pt

_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))

def bottom_label(draw, text, x0, col_w, font):
    bb = _probe.textbbox((0, 0), text, font=font)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    sy = OUT_H - LABEL_H
    draw.rectangle([x0, sy, x0+col_w-1, OUT_H-1], fill=(255, 255, 255))
    tx = x0 + col_w//2 - tw//2 - bb[0]
    ty = sy + (LABEL_H - th)//2 - bb[1]
    draw.text((tx, ty), text, font=font, fill=(10, 10, 10))

# ── Pipes ─────────────────────────────────────────────────────────────────────
reader = subprocess.Popen(
    ["ffmpeg", "-i", VIDEO_IN, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

writer = subprocess.Popen(
    ["ffmpeg", "-y",
     "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS),
     "-i", "pipe:0",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", VIDEO_MAIN],
    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

frame_bytes = SRC_W * SRC_H * 3
processed   = 0

while True:
    raw = reader.stdout.read(frame_bytes)
    if len(raw) < frame_bytes:
        break

    src = np.frombuffer(raw, dtype=np.uint8).reshape(SRC_H, SRC_W, 3)

    # Build 2x output canvas
    out = np.full((OUT_H, OUT_W, 3), 255, dtype=np.uint8)

    # LEFT: RGB obs upscaled 2x
    obs2x = np.repeat(np.repeat(src[:, C1_X:C1_X+C1_W], S, axis=0), S, axis=1)
    out[:, :OC1_W] = obs2x

    # RIGHT: BEV upscaled 2x
    bev2x = np.repeat(np.repeat(src[:, C3_X:C3_X+C3_W], S, axis=0), S, axis=1)
    out[:, OC1_W+OGAP:] = bev2x

    img  = Image.fromarray(out)
    draw = ImageDraw.Draw(img)

    # ── Target image PIP ──────────────────────────────────────────────────────
    target_src = src[:, C2_X:C2_X+C2_W]
    target_2x  = np.repeat(np.repeat(target_src, S, axis=0), S, axis=1)
    pip_img    = Image.fromarray(target_2x).resize((PIP_W, PIP_H), Image.LANCZOS)

    # White border box
    box_x0, box_y0 = PIP_X - PIP_BORDER, PIP_Y - PIP_BORDER
    box_x1, box_y1 = PIP_X + PIP_W + PIP_BORDER, PIP_Y + PIP_BOX_H + PIP_BORDER
    draw.rectangle([box_x0, box_y0, box_x1, box_y1], fill=(255, 255, 255))

    # Paste PIP image
    img.paste(pip_img, (PIP_X, PIP_Y))

    # "Target image" label strip below PIP
    lbl_y = PIP_Y + PIP_H
    draw.rectangle([PIP_X, lbl_y, PIP_X+PIP_W-1, lbl_y+PIP_LABEL_H-1], fill=(255, 255, 255))
    bb  = _probe.textbbox((0,0), "Target image", font=pip_label_font)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    draw.text((PIP_X + PIP_W//2 - tw//2 - bb[0],
               lbl_y + (PIP_LABEL_H - th)//2 - bb[1]),
              "Target image", font=pip_label_font, fill=(10, 10, 10))

    # Bottom column labels
    bottom_label(draw, "RGB observation", 0,          OC1_W, label_font)
    bottom_label(draw, "Bird's eye view", OC1_W+OGAP, OC3_W, label_font)

    writer.stdin.write(np.array(img).tobytes())
    processed += 1
    if processed % 60 == 0:
        print(f"  {processed} frames", flush=True)

reader.stdout.close(); reader.wait()
writer.stdin.close();  writer.wait()
print(f"Done — {processed} frames → {VIDEO_MAIN} ({OUT_W}×{OUT_H})")
