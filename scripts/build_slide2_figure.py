from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from PIL import Image

IMG_DIR = Path("/home/sx2022/Diff-IP2D/docs/architecture_images")
OUT = Path("/home/sx2022/Diff-IP2D/docs/final_project_images/slide2_challenges.png")

f1 = np.array(Image.open(IMG_DIR / "01a_frame_18958.png"))
f2 = np.array(Image.open(IMG_DIR / "01b_frame_18968.png"))
f3 = np.array(Image.open(IMG_DIR / "01c_frame_18978.png"))

H, W = f1.shape[:2]

IMP_BLUE = "#003E74"
IMP_ACCENT = "#C8102E"
IMP_DARK = "#1A1A1A"
IMP_GREY = "#5A5A5A"
TRAJ_COLORS = ["#C8102E", "#0099CC", "#28A745"]

fig = plt.figure(figsize=(8.5, 11.0), dpi=160, facecolor="white")
gs = fig.add_gridspec(
    3,
    3,
    height_ratios=[1.05, 1.55, 1.55],
    width_ratios=[1, 1, 1],
    hspace=0.32,
    wspace=0.08,
    left=0.04,
    right=0.96,
    top=0.95,
    bottom=0.03,
)

# ---------------------------------------------------------------------------
# Panel A — Egomotion: three sequential frames with drift annotation
# ---------------------------------------------------------------------------
# Light-switch position (approximately) in each frame — pixel coords (x, y)
ls_positions = [(195, 70), (170, 70), (155, 95)]
plate_centres = [(120, 145), (95, 165), (120, 155)]
hand_positions = [(220, 250), (135, 245), (200, 240)]

frames = [f1, f2, f3]
time_labels = ["frame  t − 2", "frame  t − 1", "frame  t"]
for i, (img, lab) in enumerate(zip(frames, time_labels)):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(img)
    # Highlight one static landmark (the light switch) so the audience can see drift
    sx, sy = ls_positions[i]
    ax.add_patch(Circle((sx, sy), 16, fill=False, edgecolor="#FFD400", linewidth=2.2))
    # Tiny hand marker
    hx, hy = hand_positions[i]
    ax.plot(hx, hy, "o", color=IMP_ACCENT, markersize=7, markeredgecolor="white", markeredgewidth=1.2)
    ax.set_title(lab, fontsize=11, color=IMP_BLUE, fontweight="bold", pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# Section title at the very top
fig.text(
    0.04,
    0.975,
    "1.  Egomotion entanglement  —  the same static landmark drifts across the frame",
    fontsize=13,
    color=IMP_BLUE,
    fontweight="bold",
    ha="left",
)
fig.text(
    0.04,
    0.955,
    "Yellow circle = light switch (physically stationary).   Red dot = hand.",
    fontsize=9,
    color=IMP_GREY,
    ha="left",
    style="italic",
)

# ---------------------------------------------------------------------------
# Panel B — Spatial ambiguity: candidate objects with question marks
# ---------------------------------------------------------------------------
ax_amb = fig.add_subplot(gs[1, :])
ax_amb.imshow(f3)

candidates = [
    (95, 150, 34),
    (240, 140, 32),
    (35, 145, 28),
]
amb_outline = [pe.Stroke(linewidth=4.5, foreground="white"), pe.Normal()]
for x, y, r in candidates:
    c = Circle((x, y), r, fill=False, edgecolor=IMP_ACCENT, linewidth=3.0, linestyle="--")
    c.set_path_effects(amb_outline)
    ax_amb.add_patch(c)
    t = ax_amb.text(
        x,
        y,
        "?",
        fontsize=26,
        color=IMP_ACCENT,
        ha="center",
        va="center",
        fontweight="bold",
    )
    t.set_path_effects(amb_outline)

# Hand marker
hx, hy = 200, 235
ax_amb.plot(
    hx,
    hy,
    "o",
    color=IMP_DARK,
    markersize=14,
    markeredgecolor="white",
    markeredgewidth=2.5,
    zorder=5,
)
ht = ax_amb.text(hx + 14, hy - 6, "hand", fontsize=11, color=IMP_DARK, fontweight="bold")
ht.set_path_effects(amb_outline)

ax_amb.set_title(
    "2.  Spatial ambiguity  —  multiple plausible target objects along the same hand vector",
    fontsize=13,
    color=IMP_BLUE,
    fontweight="bold",
    loc="left",
    pad=6,
)
ax_amb.set_xticks([])
ax_amb.set_yticks([])
for spine in ax_amb.spines.values():
    spine.set_visible(False)

# ---------------------------------------------------------------------------
# Panel C — Multimodal future: diverging trajectories from current hand
# ---------------------------------------------------------------------------
ax_mm = fig.add_subplot(gs[2, :])
ax_mm.imshow(f3)

# Hand starting position (just above the small plate the wearer is holding)
hand_pos = (200, 235)
targets = [(95, 150), (240, 140), (35, 145)]
labels = ["reach big plate", "reach utensils", "reach jars"]
outline = [pe.Stroke(linewidth=5.5, foreground="white"), pe.Normal()]
for (tx, ty), color, lab in zip(targets, TRAJ_COLORS, labels):
    # Quadratic Bezier through a control point above the midpoint for a natural reach arc
    mx = (hand_pos[0] + tx) / 2 + (tx - hand_pos[0]) * 0.20
    my = min(hand_pos[1], ty) - 55
    t_arr = np.linspace(0, 1, 80)
    xs = (1 - t_arr) ** 2 * hand_pos[0] + 2 * (1 - t_arr) * t_arr * mx + t_arr**2 * tx
    ys = (1 - t_arr) ** 2 * hand_pos[1] + 2 * (1 - t_arr) * t_arr * my + t_arr**2 * ty
    line = ax_mm.plot(xs, ys, linestyle="--", color=color, linewidth=3.2, alpha=0.98, dash_capstyle="round")
    line[0].set_path_effects(outline)
    arrow = FancyArrowPatch(
        (xs[-4], ys[-4]),
        (tx, ty),
        arrowstyle="-|>",
        color=color,
        mutation_scale=22,
        linewidth=3.2,
    )
    arrow.set_path_effects(outline)
    ax_mm.add_patch(arrow)
    # Solid target dot with outline
    ax_mm.plot(
        tx,
        ty,
        "o",
        color=color,
        markersize=12,
        markeredgecolor="white",
        markeredgewidth=2.0,
        zorder=5,
    )

# Hand marker on top (drawn last so it stays visible)
ax_mm.plot(
    *hand_pos,
    "o",
    color=IMP_DARK,
    markersize=14,
    markeredgecolor="white",
    markeredgewidth=2.5,
    zorder=6,
)
hand_text = ax_mm.text(
    hand_pos[0] + 14,
    hand_pos[1] - 6,
    "hand  t",
    fontsize=11,
    color=IMP_DARK,
    fontweight="bold",
)
hand_text.set_path_effects(outline)

ax_mm.set_title(
    "3.  Multimodal future  —  the same past plausibly precedes several different reaches",
    fontsize=13,
    color=IMP_BLUE,
    fontweight="bold",
    loc="left",
    pad=6,
)
ax_mm.set_xticks([])
ax_mm.set_yticks([])
for spine in ax_mm.spines.values():
    spine.set_visible(False)

plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white", pad_inches=0.05)
print(f"Wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
