"""
Process scene1 comparison videos:
  1. Cols 2 & 3 downsized by 6% (94%), centered with white background
  2. Labels at bottom: "RGB observation", "Bird's eye view trajectory",
     "Bird's eye view 3D completion"
  3. Crop last column → output 1376px wide
Overwrites files in place.

Source layouts:
  ours    : 1648x368  C1(640)+8+C2(360)+8+C3(360)+8+C4(264)
  others  : 1728x368  C1(640)+8+C2(360)+8+C3(360)+8+C4(344)
"""
import os, subprocess, glob, numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS    = 12
SCALE  = 0.94          # downsize cols 2 & 3 by 6%
OUT_W  = 1376          # keep first 3 cols: 640+8+360+8+360
LABEL_H = 36
LABELS = [
    ("RGB observation",               0,   640),
    ("Bird's eye view trajectory",    648, 360),
    ("Bird's eye view 3D completion", 1016, 360),
]
C2_X, C3_X, COL_W, SRC_H_ALL = 648, 1016, 360, 368

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
_probe = ImageDraw.Draw(Image.new("RGB", (1,1)))

def make_label_info(labels, src_h):
    info = []
    for name, cx, cw in labels:
        bb = _probe.textbbox((0,0), name, font=label_font)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        sy = src_h - LABEL_H
        tx = cx + cw//2 - tw//2 - bb[0]
        ty = sy + (LABEL_H - th)//2 - bb[1]
        info.append((name, cx, cx+cw, sy, tx, ty))
    return info

def zoom_col(src, cx, cw, src_h):
    """Downscale column by SCALE, center on white background."""
    col = src[:, cx:cx+cw]
    new_w = int(cw * SCALE)
    new_h = int(src_h * SCALE)
    small = np.array(Image.fromarray(col).resize((new_w, new_h), Image.LANCZOS))
    canvas = np.full((src_h, cw, 3), 255, dtype=np.uint8)
    ox = (cw - new_w) // 2
    oy = (src_h - new_h) // 2
    canvas[oy:oy+new_h, ox:ox+new_w] = small
    src[:, cx:cx+cw] = canvas

def process_file(path, src_w, src_h):
    tmp = path + ".tmp.mp4"
    reader = subprocess.Popen(
        ["ffmpeg", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    writer = subprocess.Popen(
        ["ffmpeg", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OUT_W}x{src_h}", "-r", str(FPS),
         "-i", "pipe:0",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    label_info = make_label_info(LABELS, src_h)
    frame_bytes = src_w * src_h * 3
    processed = 0

    while True:
        raw = reader.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break

        src = np.frombuffer(raw, dtype=np.uint8).reshape(src_h, src_w, 3).copy()

        # Downsize cols 2 & 3 with white padding
        zoom_col(src, C2_X, COL_W, src_h)
        zoom_col(src, C3_X, COL_W, src_h)

        # Crop to first 3 cols
        out = src[:, :OUT_W].copy()

        # Add labels
        img  = Image.fromarray(out)
        draw = ImageDraw.Draw(img)
        for name, lx0, lx1, sy, tx, ty in label_info:
            draw.rectangle([lx0, sy, lx1, src_h-1], fill=(255,255,255))
            draw.text((tx, ty), name, font=label_font, fill=(10,10,10))

        writer.stdin.write(np.array(img).tobytes())
        processed += 1
        if processed % 120 == 0:
            print(f"  {processed} frames", flush=True)

    reader.stdout.close(); reader.wait()
    writer.stdin.close(); writer.wait()
    os.replace(tmp, path)
    print(f"  Done — {processed} frames → {path}")


configs = {
    "scene1_ours.mp4":     (1648, 368),
    "scene1_ansdepth.mp4": (1728, 368),
    "scene1_ansrgb.mp4":   (1728, 368),
    "scene1_occargb.mp4":  (1728, 368),
    "scene1_occargbd.mp4": (1728, 368),
}

for fname, (w, h) in configs.items():
    path = f"assets/videos/comparison/{fname}"
    print(f"\n{path}")
    process_file(path, w, h)

print("\nAll done.")
