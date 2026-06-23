from pathlib import Path

from PIL import Image

SRC_DIR = Path("/home/sx2022/Diff-IP2D/docs/architecture_images")
OUT = Path("/home/sx2022/Diff-IP2D/docs/final_project_images/slide2_egomotion_overlay.png")

# t-2 (earliest) and t (latest) frames from the slide-2 EGTEA clip
frame_tm2 = Image.open(SRC_DIR / "01a_frame_18958.png").convert("RGB")
frame_t = Image.open(SRC_DIR / "01c_frame_18978.png").convert("RGB")

# Up-sample for projection clarity (originals are only 341 x 256)
SCALE = 4
w, h = frame_tm2.size
frame_tm2 = frame_tm2.resize((w * SCALE, h * SCALE), Image.LANCZOS)
frame_t = frame_t.resize((w * SCALE, h * SCALE), Image.LANCZOS)

# Symmetric 50/50 alpha blend — both frames equally visible, no colour shift
overlay = Image.blend(frame_tm2, frame_t, alpha=0.5)
overlay.save(OUT, "PNG")
print(f"Wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, {overlay.size[0]} x {overlay.size[1]})")
