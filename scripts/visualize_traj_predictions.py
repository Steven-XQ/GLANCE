import argparse
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator


RAW_SIZE = {
    "egtea": (1280, 960),     # (width, height)
    "meccano": (1920, 1080),
}

COLOR_GT = "#2ca02c"        # green
COLOR_BASELINE = "#d62728"  # red
COLOR_GAZE = "#1f77b4"      # blue
COLOR_ANCHOR = "#7f7f7f"    # grey — shared starting point (last-observation hand position)
HAND_LABEL = {0: "right", 1: "left"}
HAND_MARKER = {0: "o", 1: "s"}


def to_pixels(traj_norm, raw_w, raw_h):
    out = np.empty_like(traj_norm)
    out[..., 0] = traj_norm[..., 0] * raw_w
    out[..., 1] = traj_norm[..., 1] * raw_h
    return out


def arc_length_resample(waypoints_px, n_points=30, n_oversample=500):
    # Drop consecutive coincident points so the cumulative chord-length param is
    # strictly monotone (cubic spline requires this).
    diffs = np.linalg.norm(np.diff(waypoints_px, axis=0), axis=1)
    keep = np.concatenate(([True], diffs > 1e-9))
    waypoints_px = waypoints_px[keep]
    W = waypoints_px.shape[0]
    if W <= 1:
        return np.tile(waypoints_px, (n_points, 1))
    chord = np.linalg.norm(np.diff(waypoints_px, axis=0), axis=1)
    t = np.concatenate(([0.0], np.cumsum(chord)))
    if t[-1] <= 1e-9:
        return np.tile(waypoints_px[0:1], (n_points, 1))
    t = t / t[-1]
    if W >= 3:
        # PCHIP: smooth (C^1), shape-preserving, no overshoot on irregular waypoints.
        cs_x = PchipInterpolator(t, waypoints_px[:, 0])
        cs_y = PchipInterpolator(t, waypoints_px[:, 1])
        t_dense = np.linspace(0.0, 1.0, n_oversample)
        dense = np.stack([cs_x(t_dense), cs_y(t_dense)], axis=1)
    else:
        t_dense = np.linspace(0.0, 1.0, n_oversample)
        dense = np.stack([
            np.interp(t_dense, t, waypoints_px[:, 0]),
            np.interp(t_dense, t, waypoints_px[:, 1]),
        ], axis=1)
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    targets = np.linspace(0.0, cum[-1], n_points)
    out_x = np.interp(targets, cum, dense[:, 0])
    out_y = np.interp(targets, cum, dense[:, 1])
    return np.stack([out_x, out_y], axis=1)


def plot_traj(ax, waypoints_px, color, label, marker, n_points=30):
    interp = arc_length_resample(waypoints_px, n_points=n_points)
    ax.plot(interp[:, 0], interp[:, 1], "-", color=color, linewidth=2.2,
            alpha=0.85, label=label, zorder=4)
    ax.scatter(interp[:, 0], interp[:, 1], s=14, c=color, marker=marker,
               edgecolors="none", alpha=0.95, zorder=5)
    # Final waypoint emphasised. (Anchor is drawn separately, in grey, since the same
    # last-observation point is shared by GT/baseline/gaze.)
    ax.scatter(waypoints_px[-1, 0], waypoints_px[-1, 1], s=120, c=color, marker=marker,
               edgecolors="white", linewidths=1.6, zorder=7)


def render_one(entry, dataset, frames_root, out_path, raw_w, raw_h):
    img_path = os.path.join(frames_root, entry["last_obs_name"])
    if not os.path.exists(img_path):
        raise FileNotFoundError(
            f"RGB frame missing: {img_path}\n"
            f"  Expected layout: {frames_root}/<video_id>/<frame>.jpg"
        )
    img = Image.open(img_path).convert("RGB")
    # Resize to the canonical raw size in case the disk file is a different resolution
    # (e.g. video re-extracted at different scale). Our pixel coords are relative to RAW_SIZE.
    if img.size != (raw_w, raw_h):
        img = img.resize((raw_w, raw_h), Image.BILINEAR)

    gt = entry["gt"]                         # (2, T, 2)
    pred_b = entry["pred_baseline"]          # (2, T, 2)
    pred_g = entry["pred_gaze"]              # (2, T, 2)
    valid = entry["valid"]                   # (2,)
    last_obs_hand = entry.get("last_obs_hand")  # (2, 2) anchor at the last-obs frame, optional

    fig, ax = plt.subplots(figsize=(raw_w / 200, raw_h / 200), dpi=200)
    ax.imshow(np.asarray(img))
    ax.set_xlim(0, raw_w)
    ax.set_ylim(raw_h, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    for h in range(2):
        if valid[h] <= 0:
            continue
        if last_obs_hand is not None:
            anchor = last_obs_hand[h:h+1]              # (1, 2)
            gt_pts = np.concatenate([anchor, gt[h]], axis=0)
            b_pts  = np.concatenate([anchor, pred_b[h]], axis=0)
            g_pts  = np.concatenate([anchor, pred_g[h]], axis=0)
        else:
            gt_pts, b_pts, g_pts = gt[h], pred_b[h], pred_g[h]
        plot_traj(ax, to_pixels(gt_pts, raw_w, raw_h),
                  COLOR_GT, f"GT ({HAND_LABEL[h]})", HAND_MARKER[h])
        plot_traj(ax, to_pixels(b_pts, raw_w, raw_h),
                  COLOR_BASELINE, f"Diff-IP2D ({HAND_LABEL[h]})", HAND_MARKER[h])
        plot_traj(ax, to_pixels(g_pts, raw_w, raw_h),
                  COLOR_GAZE, f"Diff-IP2D + Gaze ({HAND_LABEL[h]})", HAND_MARKER[h])
        # Shared anchor (last-observation hand position) — drawn once in grey.
        if last_obs_hand is not None:
            anchor_px = to_pixels(last_obs_hand[h], raw_w, raw_h)
            ax.scatter(anchor_px[0], anchor_px[1], s=100, c=COLOR_ANCHOR,
                       marker=HAND_MARKER[h], edgecolors="white", linewidths=1.4,
                       zorder=8, label=f"start ({HAND_LABEL[h]})")

    # Deduped legend.
    by_label = {}
    for h, l in zip(*ax.get_legend_handles_labels()):
        # Collapse (right)/(left) variants into one source label for the legend.
        if l.startswith("GT"):
            key = "Ground Truth"
        elif l.startswith("Diff-IP2D + Gaze"):
            key = "Diff-IP2D + Gaze"
        elif l.startswith("start"):
            key = "Start (last obs)"
        else:
            key = "Diff-IP2D"
        by_label.setdefault(key, h)
    order = ["Ground Truth", "Diff-IP2D", "Diff-IP2D + Gaze", "Start (last obs)"]
    order = [k for k in order if k in by_label]
    leg = ax.legend([by_label[k] for k in order], order, loc="lower right",
                    framealpha=0.85, fontsize=10)
    leg_color_map = {"Ground Truth": COLOR_GT, "Diff-IP2D": COLOR_BASELINE,
                     "Diff-IP2D + Gaze": COLOR_GAZE, "Start (last obs)": COLOR_ANCHOR}
    for txt, key in zip(leg.get_texts(), order):
        txt.set_color(leg_color_map[key])

    title = (f"{dataset.upper()} | uid={entry['uid']} | "
             f"WDE: base={entry['wde_baseline']:.3f}, gaze={entry['wde_gaze']:.3f} "
             f"(gain {entry['gain']:+.3f})")
    ax.set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["egtea", "meccano"])
    ap.add_argument("--frames_root", required=True,
                    help="root containing <video_id>/<frame>.jpg files")
    ap.add_argument("--picked_pkl", default=None,
                    help="path to picked.pkl (default: viz_data/<dataset>/picked.pkl)")
    ap.add_argument("--out_dir", default=None,
                    help="directory for output PNGs (default: viz_data/<dataset>/figs)")
    args = ap.parse_args()

    picked_pkl = args.picked_pkl or os.path.join("viz_data", args.dataset, "picked.pkl")
    out_dir = args.out_dir or os.path.join("viz_data", args.dataset, "figs")
    os.makedirs(out_dir, exist_ok=True)

    with open(picked_pkl, "rb") as f:
        rows = pickle.load(f)
    print(f"[{args.dataset}] rendering {len(rows)} panels -> {out_dir}")
    raw_w, raw_h = RAW_SIZE[args.dataset]

    for i, entry in enumerate(rows):
        out_path = os.path.join(out_dir, f"{i+1:02d}_uid{entry['uid']}.png")
        path = render_one(entry, args.dataset, args.frames_root, out_path, raw_w, raw_h)
        print(f"  [{i+1}/{len(rows)}] uid={entry['uid']}  gain={entry['gain']:+.3f}  ->  {path}")


if __name__ == "__main__":
    main()
