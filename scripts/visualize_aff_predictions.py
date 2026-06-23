import argparse
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

COLOR_GT = "#2ca02c"
COLOR_BASELINE = "#d62728"
COLOR_GAZE = "#1f77b4"


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
    sigma_grid = float(sigma_norm) * n_w  # sigma_norm is a fraction of the frame WIDTH
    s2 = sigma_grid ** 2
    xs = np.arange(n_w, dtype=np.float32)
    ys = np.arange(n_h, dtype=np.float32)
    Xs, Ys = np.meshgrid(xs, ys)
    return np.exp(-((Xs - cx_grid) ** 2 + (Ys - cy_grid) ** 2) / (2.0 * s2))


def render_one(entry, dataset, frames_root, out_path, sigma_norm, tint_strength,
               alpha_gamma, alpha_max):
    img_path = os.path.join(frames_root, entry["last_obs_name"])
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"RGB frame missing: {img_path}")
    raw_w, raw_h = RAW_SIZE[dataset]
    img = Image.open(img_path).convert("RGB")
    if img.size != (raw_w, raw_h):
        img = img.resize((raw_w, raw_h), Image.BILINEAR)
    img_arr = np.asarray(img).astype(np.float32) / 255.0
    img_tinted = bluish_tint(img_arr, tint_strength)

    gt = entry["gt_contacts"]
    pb = entry["pred_baseline"]
    pg = entry["pred_gaze"]
    gt_point = aggregate_gt(gt)
    best_b = best_of_many(pb, gt_point)
    best_g = best_of_many(pg, gt_point)

    fig, ax = plt.subplots(figsize=(raw_w / 200, raw_h / 200), dpi=200)
    ax.imshow(img_tinted)
    ax.set_xlim(0, raw_w)
    ax.set_ylim(raw_h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    extent = (0, raw_w, raw_h, 0)

    n_h = 240
    n_w = int(round(n_h * raw_w / raw_h))
    # `jet` runs dark blue (low / periphery) -> cyan -> green -> yellow -> red (high / centre),
    # exactly matching "red near the centre, blue away from it".
    cmap = plt.get_cmap("jet")
    for pt in (gt_point, best_b, best_g):
        if pt is None:
            continue
        hm = radial_gaussian(pt, sigma_norm, n_h=n_h, n_w=n_w)
        alpha_arr = np.clip(hm ** alpha_gamma, 0.0, alpha_max)
        ax.imshow(hm, cmap=cmap, extent=extent, alpha=alpha_arr,
                  interpolation="bilinear", vmin=0.0, vmax=1.0, zorder=3)

    def scatter_one(pt_norm, color, label):
        if pt_norm is None:
            return
        ax.scatter(pt_norm[0] * raw_w, pt_norm[1] * raw_h, s=110, c=color, marker="o",
                   edgecolors="none", zorder=8, label=label)

    scatter_one(gt_point, COLOR_GT, "Ground Truth")
    scatter_one(best_b, COLOR_BASELINE, "Diff-IP2D")
    scatter_one(best_g, COLOR_GAZE, "Diff-IP2D + Gaze")

    leg = ax.legend(loc="lower right", framealpha=0.85, fontsize=10)
    leg_color = {"Ground Truth": COLOR_GT, "Diff-IP2D": COLOR_BASELINE,
                 "Diff-IP2D + Gaze": COLOR_GAZE}
    for txt in leg.get_texts():
        txt.set_color(leg_color[txt.get_text()])

    def d(p):
        return float(np.linalg.norm(p - gt_point)) if (p is not None and gt_point is not None) else float("nan")
    db, dg = d(best_b), d(best_g)
    ax.set_title(
        f"{dataset.upper()} affordance | uid={entry['uid']} | "
        f"best-of-many dist: base={db:.3f}, gaze={dg:.3f} (gain {db - dg:+.3f})",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["egtea", "meccano"])
    ap.add_argument("--frames_root", required=True)
    ap.add_argument("--picked_pkl", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--sigma_norm", type=float, default=0.035,
                    help="Gaussian std-dev as a fraction of the frame width (default 0.035).")
    ap.add_argument("--tint_strength", type=float, default=0.65,
                    help="0=original frame, 1=full bluish duotone (default 0.65).")
    ap.add_argument("--alpha_gamma", type=float, default=0.3,
                    help="Heatmap alpha = heatmap^gamma. Lower = more visible blue periphery (default 0.3).")
    ap.add_argument("--alpha_max", type=float, default=0.22,
                    help="Cap on the alpha at each Gaussian's centre (default 0.22).")
    args = ap.parse_args()

    picked_pkl = args.picked_pkl or os.path.join("viz_data", args.dataset, "picked_aff.pkl")
    out_dir = args.out_dir or os.path.join("viz_data", args.dataset, "figs_aff")
    os.makedirs(out_dir, exist_ok=True)
    with open(picked_pkl, "rb") as f:
        rows = pickle.load(f)
    print(f"[{args.dataset}] rendering {len(rows)} aff panels -> {out_dir}")
    for i, entry in enumerate(rows):
        out_path = os.path.join(out_dir, f"{i+1:02d}_uid{entry['uid']}.png")
        render_one(entry, args.dataset, args.frames_root, out_path,
                   sigma_norm=args.sigma_norm, tint_strength=args.tint_strength,
                   alpha_gamma=args.alpha_gamma, alpha_max=args.alpha_max)
        print(f"  [{i+1}/{len(rows)}] uid={entry['uid']}  ->  {out_path}")


if __name__ == "__main__":
    main()
