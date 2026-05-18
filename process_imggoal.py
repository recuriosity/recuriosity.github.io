"""
Process image-goal.mp4:
  - Source layout: C1(640) + 8 + C2(640) + 8 + C3(364) = 1660 wide, 368 tall
  - C1: agent observation
  - C2: target image
  - C3: bird's eye view map

Outputs at 2x resolution:
  image-goal-main.mp4  — cols 1+2 (2576x736) with "Target image" label on col2
  image-goal-bev.mp4   — col3 only (728x736)
"""
import subprocess, numpy as np
from PIL import Image, ImageDraw, ImageFont

VIDEO_IN   = "assets/videos/image-goal.mp4"
VIDEO_MAIN = "assets/videos/image-goal-main.mp4"
VIDEO_BEV  = "assets/videos/image-goal-bev.mp4"

SRC_W, SRC_H = 1660, 368
FPS  = 12
GAP  = 8

C1_X, C1_W = 0,   640
C2_X, C2_W = 648, 640
C3_X, C3_W = 1296, 364   # 640+8+640+8 = 1296; 1660-1296 = 364

# Main output: cols 1 + 2 (with gap preserved)
MAIN_W = C1_W + GAP + C2_W   # 1288
MAIN_H = SRC_H
OUT_W  = MAIN_W * 2          # 2576
OUT_H  = SRC_H  * 2          # 736
BEV_W  = C3_W * 2            # 728

LABEL_H = 36

def load_font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
    ] if bold else [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

label_font = load_font(26, bold=True)

# "Target image" label position in 2x output (on col2)
LBL_X0 = (C1_W + GAP) * 2   # 1296
LBL_X1 = LBL_X0 + C2_W * 2 - 1  # 2575

reader = subprocess.Popen(
    ["ffmpeg", "-i", VIDEO_IN, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
)

def make_writer(path, w, h):
    return subprocess.Popen(
        ["ffmpeg", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
         "-i", "pipe:0",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", path],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

writer_main = make_writer(VIDEO_MAIN, OUT_W, OUT_H)
writer_bev  = make_writer(VIDEO_BEV,  BEV_W, OUT_H)

frame_bytes = SRC_W * SRC_H * 3
processed   = 0

while True:
    raw = reader.stdout.read(frame_bytes)
    if len(raw) < frame_bytes:
        break

    src = np.frombuffer(raw, dtype=np.uint8).reshape(SRC_H, SRC_W, 3)

    # ── Main: cols 1 + 2 ─────────────────────────────────────────────────────
    main_src = np.zeros((SRC_H, MAIN_W, 3), dtype=np.uint8)
    main_src[:, :C1_W]           = src[:, C1_X:C1_X+C1_W]
    main_src[:, C1_W+GAP:]       = src[:, C2_X:C2_X+C2_W]
    # 2x upscale
    main2x = np.repeat(np.repeat(main_src, 2, axis=0), 2, axis=1).copy()
    img = Image.fromarray(main2x)
    draw = ImageDraw.Draw(img)
    # White label strip on col2
    draw.rectangle([LBL_X0, OUT_H - LABEL_H, LBL_X1, OUT_H - 1], fill=(255, 255, 255))
    text = "Target image"
    bb   = draw.textbbox((0, 0), text, font=label_font)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    tx = LBL_X0 + (C2_W * 2 - tw) // 2
    ty = OUT_H - LABEL_H + (LABEL_H - th) // 2
    draw.text((tx, ty), text, font=label_font, fill=(10, 10, 10))
    writer_main.stdin.write(np.array(img).tobytes())

    # ── BEV: col 3 ───────────────────────────────────────────────────────────
    bev_src = src[:, C3_X:C3_X+C3_W].copy()
    bev2x   = np.repeat(np.repeat(bev_src, 2, axis=0), 2, axis=1).copy()
    writer_bev.stdin.write(bev2x.tobytes())

    processed += 1
    if processed % 60 == 0:
        print(f"  {processed} frames", flush=True)

reader.stdout.close(); reader.wait()
writer_main.stdin.close(); writer_main.wait()
writer_bev.stdin.close();  writer_bev.wait()
print(f"Done — {processed} frames → {VIDEO_MAIN}, {VIDEO_BEV}")
