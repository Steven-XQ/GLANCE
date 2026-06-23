import argparse
import glob
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator


RAW_SIZE = {
    "egtea": (1280, 960),
    "meccano": (1920, 1080),
}
FRAMES_ROOT = {
    "egtea": "data/raw_images/EGTEA",
    "meccano": "data/raw_images/MECCANO",
}
GAZE_ROOT = {
    "egtea":  "/data/datasets/EGTEA/Gaze_Data/gaze_data/gaze_data",
    "meccano": "/data/datasets/MECCANO/annotations/MECCANO_Gaze_data",
}
# Native fps used for the index gap between the T=10 observation frames.
# (ori_fps // fps with fps=6.)
GAZE_GAP = {"egtea": 5, "meccano": 2}
SEQ_LEN_OBS = 10
COLOR_GT = "#2ca02c"
COLOR_BASELINE = "#d62728"
COLOR_GAZE = "#1f77b4"
COLOR_ANCHOR = "#7f7f7f"
COLOR_SACCADE = "#ff7f0e"  # orange — distinct from GT/baseline/gaze/anchor.
HAND_MARKER = {0: "o", 1: "s"}


# Per-video gaze caches so we don't reparse the same .txt/.csv multiple times.
_GAZE_CACHE = {}


def load_egtea_gaze(video_id):
    key = ("egtea", video_id)
    if key in _GAZE_CACHE:
        return _GAZE_CACHE[key]
    gaze_file = os.path.join(GAZE_ROOT["egtea"], f"{video_id}.txt")
    if not os.path.exists(gaze_file):
        _GAZE_CACHE[key] = None
        return None
    frame_gaze = {}
    with open(gaze_file, "r") as f:
        for line in f:
            if line.startswith("#") or line.startswith("Time"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 8 or parts[1] != "SMP":
                continue
            try:
                x_px = float(parts[3]); y_px = float(parts[4])
                frame_num = int(parts[5]); event_type = parts[7].strip()
            except (ValueError, IndexError):
                continue
            if event_type == "Blink":
                continue
            x_n = max(0.0, min(1.0, x_px / 1280.0))
            y_n = max(0.0, min(1.0, y_px / 960.0))
            frame_gaze.setdefault(frame_num, []).append((x_n, y_n))
    averaged = {n: (sum(p[0] for p in pts) / len(pts),
                    sum(p[1] for p in pts) / len(pts))
                for n, pts in frame_gaze.items()}
    _GAZE_CACHE[key] = averaged
    return averaged


def load_meccano_gaze(video_id):
    key = ("meccano", video_id)
    if key in _GAZE_CACHE:
        return _GAZE_CACHE[key]
    gaze_file = None
    for sub in ("Train", "Val", "Test"):
        cand = os.path.join(GAZE_ROOT["meccano"], sub, f"{video_id}_gaze-data.csv")
        if os.path.exists(cand):
            gaze_file = cand
            break
    if gaze_file is None:
        _GAZE_CACHE[key] = None
        return None
    frame_gaze = {}
    with open(gaze_file, "r") as f:
        header = True
        for line in f:
            if header:
                header = False; continue
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                frame_num = int(parts[0].replace(".jpg", ""))
                x_px = float(parts[2]); y_px = float(parts[3])
            except ValueError:
                continue
            x_n = max(0.0, min(1.0, x_px / 1920.0))
            y_n = max(0.0, min(1.0, y_px / 1080.0))
            frame_gaze.setdefault(frame_num, []).append((x_n, y_n))
    averaged = {n: (sum(p[0] for p in pts) / len(pts),
                    sum(p[1] for p in pts) / len(pts))
                for n, pts in frame_gaze.items()}
    _GAZE_CACHE[key] = averaged
    return averaged


def gaze_saccade(dataset, video_id, last_obs_frame_idx, max_points=12):
    if dataset == "egtea":
        gaze = load_egtea_gaze(video_id)
    elif dataset == "meccano":
        gaze = load_meccano_gaze(video_id)
    else:
        return None
    if gaze is None:
        return None
    gap = GAZE_GAP[dataset]
    obs_start = last_obs_frame_idx - (SEQ_LEN_OBS - 1) * gap
    keys_in_window = sorted(k for k in gaze if obs_start <= k <= last_obs_frame_idx)
    if not keys_in_window:
        return None
    if len(keys_in_window) > max_points:
        idx = np.linspace(0, len(keys_in_window) - 1, max_points).astype(int)
        keys_in_window = [keys_in_window[i] for i in idx]
    pts = [gaze[k] for k in keys_in_window]
    return np.asarray(pts, dtype=np.float32)


def load_dump(prefix):
    files = sorted(glob.glob(f"{prefix}.rank*.pkl"))
    merged = {}
    for f in files:
        with open(f, "rb") as fh:
            shard = pickle.load(fh)
        for u, e in shard.items():
            merged.setdefault(u, e)
    return merged


def per_sample_wde(preds_S, gt, T):
    err = np.linalg.norm(preds_S - gt[None, ...], axis=-1)
    weights = np.arange(1, T + 1, dtype=np.float32) / float(T)
    return (err * weights[None, None, :]).sum(axis=-1)


def best_joint_sample(entry):
    preds = entry["preds"]
    gt = entry["gt"]
    valid = entry["valid"]
    T = preds.shape[2]
    wde = per_sample_wde(preds, gt, T)
    if not (valid > 0).any():
        return 0
    return int(np.argmin((wde * valid[None, :]).sum(axis=1)))


def arc_length_resample(waypoints_px, n_points=30, n_oversample=500):
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


def to_pixels(traj_norm, raw_w, raw_h):
    out = np.empty_like(traj_norm)
    out[..., 0] = traj_norm[..., 0] * raw_w
    out[..., 1] = traj_norm[..., 1] * raw_h
    return out


def plot_traj_on_ax(ax, waypoints_px, color, marker, n_points=30):
    interp = arc_length_resample(waypoints_px, n_points=n_points)
    ax.plot(interp[:, 0], interp[:, 1], "-", color=color, linewidth=2.2, alpha=0.85,
            zorder=4)
    ax.scatter(interp[:, 0], interp[:, 1], s=14, c=color, marker=marker,
               edgecolors="none", alpha=0.95, zorder=5)
    ax.scatter(waypoints_px[-1, 0], waypoints_px[-1, 1], s=120, c=color, marker=marker,
               edgecolors="white", linewidths=1.6, zorder=7)


def draw_panel(ax, dataset, entry, bs_b, bs_g, show_saccade=True):
    raw_w, raw_h = RAW_SIZE[dataset]
    img_path = os.path.join(FRAMES_ROOT[dataset], entry["last_obs_name"])
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"missing RGB frame: {img_path}")
    img = Image.open(img_path).convert("RGB")
    if img.size != (raw_w, raw_h):
        img = img.resize((raw_w, raw_h), Image.BILINEAR)
    ax.imshow(np.asarray(img))
    ax.set_xlim(0, raw_w)
    ax.set_ylim(raw_h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    gt = entry["gt"]
    pred_b = entry["preds_baseline_best"] if "preds_baseline_best" in entry else None  # filled by caller
    pred_g = entry["preds_gaze_best"] if "preds_gaze_best" in entry else None
    valid = entry["valid"]
    last_obs_hand = entry.get("last_obs_hand")
    for h in range(2):
        if valid[h] <= 0:
            continue
        if last_obs_hand is not None:
            anchor = last_obs_hand[h:h + 1]
            gt_pts = np.concatenate([anchor, gt[h]], axis=0)
            b_pts = np.concatenate([anchor, pred_b[h]], axis=0)
            g_pts = np.concatenate([anchor, pred_g[h]], axis=0)
        else:
            gt_pts, b_pts, g_pts = gt[h], pred_b[h], pred_g[h]
        plot_traj_on_ax(ax, to_pixels(gt_pts, raw_w, raw_h), COLOR_GT, HAND_MARKER[h])
        plot_traj_on_ax(ax, to_pixels(b_pts, raw_w, raw_h), COLOR_BASELINE, HAND_MARKER[h])
        plot_traj_on_ax(ax, to_pixels(g_pts, raw_w, raw_h), COLOR_GAZE, HAND_MARKER[h])
        if last_obs_hand is not None:
            anchor_px = to_pixels(last_obs_hand[h], raw_w, raw_h)
            ax.scatter(anchor_px[0], anchor_px[1], s=100, c=COLOR_ANCHOR,
                       marker=HAND_MARKER[h], edgecolors="none", zorder=8)

    if show_saccade:
        # Resolve video_id from last_obs_name. EGTEA: '<video_id>/frame_xxx.jpg';
        # MECCANO: '<video_id>/<frame>.jpg'.
        video_id = entry["last_obs_name"].split("/")[0]
        saccade_norm = gaze_saccade(dataset, video_id, int(entry["last_obs_frame_idx"]))
        if saccade_norm is not None and len(saccade_norm) >= 2:
            saccade_px = to_pixels(saccade_norm, raw_w, raw_h)
            # Path: dashed line so it reads differently from the solid hand curves.
            ax.plot(saccade_px[:, 0], saccade_px[:, 1], "--", color=COLOR_SACCADE,
                    linewidth=2.0, alpha=0.85, zorder=3.5)
            # Fixations sized down for "earlier" steps and up for later.
            n = saccade_px.shape[0]
            sizes = np.linspace(28, 95, n)
            ax.scatter(saccade_px[:, 0], saccade_px[:, 1], s=sizes, c=COLOR_SACCADE,
                       marker="^", edgecolors="none", alpha=0.9, zorder=6)
            # Emphasise the most recent gaze (the one closest in time to last obs).
            ax.scatter(saccade_px[-1, 0], saccade_px[-1, 1], s=140, c=COLOR_SACCADE,
                       marker="^", edgecolors="white", linewidths=1.6, zorder=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="cells in row-major order, each as '<dataset>:<uid>'")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell_width_in", type=float, default=6.0,
                    help="width of each panel in inches (default 6.0)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--no_saccade", action="store_true",
                    help="omit the orange gaze-saccade overlay")
    args = ap.parse_args()

    assert len(args.pairs) == args.rows * args.cols, (
        f"expected {args.rows * args.cols} pairs, got {len(args.pairs)}")

    # Load all baseline + gaze dumps once per dataset.
    datasets_used = {p.split(":")[0] for p in args.pairs}
    base_dumps, gaze_dumps = {}, {}
    for ds in datasets_used:
        base_dumps[ds] = load_dump(f"viz_data/{ds}/baseline_preds")
        gaze_dumps[ds] = load_dump(f"viz_data/{ds}/gaze_preds")

    # Compute per-row aspect ratios (use the aspect of the *first* dataset on that row).
    cells = []
    for cell in args.pairs:
        ds, uid_s = cell.split(":")
        uid = int(uid_s)
        eb = base_dumps[ds][uid]
        eg = gaze_dumps[ds][uid]
        bs_b = best_joint_sample(eb)
        bs_g = best_joint_sample(eg)
        eb = {**eb, "preds_baseline_best": eb["preds"][bs_b], "preds_gaze_best": eg["preds"][bs_g]}
        cells.append((ds, uid, eb, bs_b, bs_g))

    row_aspects = []
    for r in range(args.rows):
        ds = cells[r * args.cols][0]
        raw_w, raw_h = RAW_SIZE[ds]
        row_aspects.append(raw_h / raw_w)  # height / width per cell

    cell_w = args.cell_width_in
    fig_w = cell_w * args.cols
    fig_h = sum(cell_w * a for a in row_aspects)
    height_ratios = row_aspects
    fig, axes = plt.subplots(args.rows, args.cols, figsize=(fig_w, fig_h),
                             gridspec_kw={"height_ratios": height_ratios,
                                          "wspace": 0.01, "hspace": 0.01})
    if args.rows == 1 and args.cols == 1:
        axes = np.array([[axes]])
    elif args.rows == 1:
        axes = axes[None, :]
    elif args.cols == 1:
        axes = axes[:, None]

    for idx, (ds, uid, entry, bs_b, bs_g) in enumerate(cells):
        r, c = divmod(idx, args.cols)
        ax = axes[r, c]
        draw_panel(ax, ds, entry, bs_b, bs_g, show_saccade=not args.no_saccade)
        print(f"  panel ({r},{c}): {ds}/uid={uid}  base_sample={bs_b}  gaze_sample={bs_g}")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
