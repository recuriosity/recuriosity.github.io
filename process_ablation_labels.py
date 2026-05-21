"""
Add bottom labels to all ablation videos (1376x368):
  "RGB observation" | "Bird's eye view trajectory" | "Bird's eye view 3D completion"
Overwrites files in place.
"""
import os, subprocess, glob, numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS     = 12
SRC_W   = 1376
SRC_H   = 368
LABEL_H = 36

LABELS = [
    ("RGB observation",               0,   640),
    ("Bird's eye view trajectory",    648, 360),
    ("Bird's eye view 3D completion", 1016, 360),
]

def load_font(size, bold=False):
    candidates = []
    if bold:
        candidates += [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
        ]
    candidates += ["/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"]
    for p in candidates:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except: pass
    return ImageFont.load_default()

label_font = load_font(18, bold=True)
_probe     = ImageDraw.Draw(Image.new("RGB", (1, 1)))

def make_label_info():
    info = []
    for name, cx, cw in LABELS:
        bb = _probe.textbbox((0, 0), name, font=label_font)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        sy = SRC_H - LABEL_H
        tx = cx + cw//2 - tw//2 - bb[0]
        ty = sy + (LABEL_H - th)//2 - bb[1]
        info.append((name, cx, cx+cw, sy, tx, ty))
    return info

label_info  = make_label_info()
frame_bytes = SRC_W * SRC_H * 3

videos = sorted(glob.glob("assets/videos/ablation/*.mp4"))
print(f"Processing {len(videos)} ablation videos…")

for path in videos:
    tmp = path + ".tmp.mp4"
    reader = subprocess.Popen(
        ["ffmpeg", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    writer = subprocess.Popen(
        ["ffmpeg", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{SRC_W}x{SRC_H}", "-r", str(FPS),
         "-i", "pipe:0",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    processed = 0
    while True:
        raw = reader.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(SRC_H, SRC_W, 3).copy()
        img   = Image.fromarray(frame)
        draw  = ImageDraw.Draw(img)
        for name, lx0, lx1, sy, tx, ty in label_info:
            draw.rectangle([lx0, sy, lx1, SRC_H-1], fill=(255, 255, 255))
            draw.text((tx, ty), name, font=label_font, fill=(10, 10, 10))
        writer.stdin.write(np.array(img).tobytes())
        processed += 1

    reader.stdout.close(); reader.wait()
    writer.stdin.close(); writer.wait()
    os.replace(tmp, path)
    print(f"  {processed} frames → {path}")

print("All done.")
