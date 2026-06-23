import argparse
import glob
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


RAW_SIZE = {
    "egtea": (1280, 960),
    "meccano": (1920, 1080),
}
FRAMES_ROOT = {
    "egtea": "data/raw_images/EGTEA",
    "meccano": "data/raw_images/MECCANO",
}
COLOR_GT = "#2ca02c"
COLOR_BASELINE = "#d62728"
COLOR_GAZE = "#1f77b4"


def load_dump(prefix):
    files = sorted(glob.glob(f"{prefix}.rank*.pkl"))
    merged = {}
    for f in files:
        with open(f, "rb") as fh:
            shard = pickle.load(fh)
        for u, e in shard.items():
            merged.setdefault(u, e)
    return merged


def aggregate_gt(gt_contacts):
    if gt_contacts.shape[0] == 0:
        return None
    return np.median(gt_contacts, axis=0)


def best_of_many(preds, gt_point):
    if preds.shape[0] == 0 or gt_point is None:
        return None
    d = np.linalg.norm(preds - gt_point[None, :], axis=1)
    return preds[int(np.argmin(d))]


def bluish_tint(img_arr, strength):
    gray = img_arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    duotone = np.stack(
        [gray * 0.20, gray * 0.40, gray * 0.85 + 0.08],
        axis=-1,
    )
    return np.clip((1.0 - strength) * img_arr + strength * duotone, 0.0, 1.0)


def radial_gaussian(center_norm, sigma_norm, n_h, n_w):
    cx_grid = float(center_norm[0]) * n_w
    cy_grid = float(center_norm[1]) * n_h
    sigma_grid = float(sigma_norm) * n_w
    s2 = sigma_grid ** 2
    xs = np.arange(n_w, dtype=np.float32)
    ys = np.arange(n_h, dtype=np.float32)
    Xs, Ys = np.meshgrid(xs, ys)
    return np.exp(-((Xs - cx_grid) ** 2 + (Ys - cy_grid) ** 2) / (2.0 * s2))


def draw_panel(ax, dataset, entry, sigma_norm, tint_strength,
               alpha_gamma, alpha_max, sources=("gt", "baseline", "gaze")):
    raw_w, raw_h = RAW_SIZE[dataset]
    img_path = os.path.join(FRAMES_ROOT[dataset], entry["last_obs_name"])
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"missing RGB frame: {img_path}")
    img = Image.open(img_path).convert("RGB")
    if img.size != (raw_w, raw_h):
        img = img.resize((raw_w, raw_h), Image.BILINEAR)
    img_arr = np.asarray(img).astype(np.float32) / 255.0
    img_tinted = bluish_tint(img_arr, tint_strength)
    ax.imshow(img_tinted)
    ax.set_xlim(0, raw_w)
    ax.set_ylim(raw_h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    extent = (0, raw_w, raw_h, 0)

    gt = entry["gt_contacts"]
    pb = entry["pred_baseline"]
    pg = entry["pred_gaze"]
    gt_point = aggregate_gt(gt)
    best_b = best_of_many(pb, gt_point)
    best_g = best_of_many(pg, gt_point)

    n_h = 240
    n_w = int(round(n_h * raw_w / raw_h))
    cmap = plt.get_cmap("jet")
    pt_for = {"gt": gt_point, "baseline": best_b, "gaze": best_g}
    color_for = {"gt": COLOR_GT, "baseline": COLOR_BASELINE, "gaze": COLOR_GAZE}
    for src in sources:
        pt = pt_for.get(src)
        if pt is None:
            continue
        hm = radial_gaussian(pt, sigma_norm, n_h=n_h, n_w=n_w)
        alpha_arr = np.clip(hm ** alpha_gamma, 0.0, alpha_max)
        ax.imshow(hm, cmap=cmap, extent=extent, alpha=alpha_arr,
                  interpolation="bilinear", vmin=0.0, vmax=1.0, zorder=3)

    for src in sources:
        pt = pt_for.get(src)
        if pt is None:
            continue
        ax.scatter(pt[0] * raw_w, pt[1] * raw_h, s=110, c=color_for[src], marker="o",
                   edgecolors="none", zorder=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="cells in row-major order, each as '<dataset>:<uid>'")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell_width_in", type=float, default=6.0)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--sigma_norm", type=float, default=0.035)
    ap.add_argument("--tint_strength", type=float, default=0.65)
    ap.add_argument("--alpha_gamma", type=float, default=0.3)
    ap.add_argument("--alpha_max", type=float, default=0.22)
    ap.add_argument("--sources", nargs="+", default=["gt", "baseline", "gaze"],
                    choices=["gt", "baseline", "gaze"],
                    help="which Gaussians/markers to render (default: all three)")
    args = ap.parse_args()

    assert len(args.pairs) == args.rows * args.cols, (
        f"expected {args.rows * args.cols} pairs, got {len(args.pairs)}")

    datasets_used = {p.split(":")[0] for p in args.pairs}
    base_dumps, gaze_dumps = {}, {}
    for ds in datasets_used:
        base_dumps[ds] = load_dump(f"viz_data/{ds}/baseline_aff")
        gaze_dumps[ds] = load_dump(f"viz_data/{ds}/gaze_aff")

    cells = []
    for cell in args.pairs:
        ds, uid_s = cell.split(":")
        uid = int(uid_s)
        eb = base_dumps[ds][uid]
        eg = gaze_dumps[ds][uid]
        merged = {
            "uid": uid,
            "last_obs_name": eb["last_obs_name"],
            "gt_contacts": eb["gt_contacts"],
            "pred_baseline": eb["pred_contacts"],
            "pred_gaze": eg["pred_contacts"],
        }
        cells.append((ds, uid, merged))

    row_aspects = []
    for r in range(args.rows):
        ds = cells[r * args.cols][0]
        raw_w, raw_h = RAW_SIZE[ds]
        row_aspects.append(raw_h / raw_w)

    cell_w = args.cell_width_in
    fig_w = cell_w * args.cols
    fig_h = sum(cell_w * a for a in row_aspects)
    fig, axes = plt.subplots(args.rows, args.cols, figsize=(fig_w, fig_h),
                             gridspec_kw={"height_ratios": row_aspects,
                                          "wspace": 0.01, "hspace": 0.01})
    if args.rows == 1 and args.cols == 1:
        axes = np.array([[axes]])
    elif args.rows == 1:
        axes = axes[None, :]
    elif args.cols == 1:
        axes = axes[:, None]

    for idx, (ds, uid, entry) in enumerate(cells):
        r, c = divmod(idx, args.cols)
        ax = axes[r, c]
        draw_panel(ax, ds, entry, args.sigma_norm, args.tint_strength,
                   args.alpha_gamma, args.alpha_max, sources=tuple(args.sources))
        print(f"  panel ({r},{c}): {ds}/uid={uid}")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
