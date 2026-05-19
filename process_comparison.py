"""
Process all comparison videos: crop out the last column, keeping only
the first 3 columns (RGB obs + BEV trajectory + BEV 3D completion).
First 3 cols: 640 + 8 + 360 + 8 + 360 = 1376px wide.
Overwrites files in place.
"""
import subprocess, glob

KEEP_W = 1376   # 640 + 8 + 360 + 8 + 360
FPS    = 12

videos = sorted(glob.glob("assets/videos/comparison/scene*.mp4"))
print(f"Processing {len(videos)} files...")

for path in videos:
    tmp = path + ".tmp.mp4"
    ret = subprocess.run(
        ["ffmpeg", "-y", "-i", path,
         "-vf", f"crop={KEEP_W}:ih:0:0",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", tmp],
        stderr=subprocess.DEVNULL
    )
    if ret.returncode == 0:
        import os; os.replace(tmp, path)
        print(f"  done: {path}")
    else:
        print(f"  FAILED: {path}")

print("All done.")
