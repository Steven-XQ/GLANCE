import csv
import os
from collections import defaultdict
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "seed_sweep", "results.csv")
OUT_DIR = os.path.join(ROOT, "docs", "evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

# sweep -> doc names
RENAME = {
    "baseline": "Baseline",
    "v2": "v1", "v4": "v2", "v6": "v3", "v8": "v4",
    "v9": "v5", "v10": "v6", "v11": "v7", "v12": "v8",
    "v12a": "v9", "v12b": "v10", "v12c": "v11", "v13": "v12",
}
DOC_ORDER = ["Baseline", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
             "v8", "v9", "v10", "v11", "v12"]
GLANCE = "v8"
METRICS = ["wde", "fde", "sim", "auc_j", "nss"]
METRIC_LABEL = {
    "wde": "WDE ↓", "fde": "FDE ↓",
    "sim": "SIM ↑", "auc_j": "AUC-J ↑", "nss": "NSS ↑",
}
LOWER_IS_BETTER = {"wde", "fde"}


def load():
    data = defaultdict(lambda: defaultdict(list))  # (dataset, doc_var) -> metric -> [values]
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            key = (r["dataset"], RENAME[r["variant"]])
            for m in METRICS:
                data[key][m].append(float(r[m]))
    stats = {}
    for k, d in data.items():
        stats[k] = {m: (mean(d[m]), pstdev(d[m])) for m in METRICS}
    return stats


def variant_color(name, dataset_palette="blue"):
    if name == "Baseline":
        return "#777777"
    if name == GLANCE:
        return "#000000"  # outline only via edgecolor
    return "#1f77b4" if dataset_palette == "blue" else "#ff7f0e"


def variant_facecolor(name, dataset_palette="blue"):
    if name == "Baseline":
        return "#9c9c9c"
    if name == GLANCE:
        return "#1f77b4" if dataset_palette == "blue" else "#ff7f0e"
    return "#9ecae1" if dataset_palette == "blue" else "#fdd0a2"


def variant_edge(name):
    if name == GLANCE:
        return "black"
    return "none"


def variant_lw(name):
    return 1.6 if name == GLANCE else 0.0


def build_variant_bars(dataset, palette, out_name, title_suffix):
    stats = load()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), constrained_layout=True)

    for ax, metric in zip(axes, METRICS):
        xs = np.arange(len(DOC_ORDER))
        means = np.array([stats[(dataset, v)][metric][0] for v in DOC_ORDER])
        stds = np.array([stats[(dataset, v)][metric][1] for v in DOC_ORDER])
        face = [variant_facecolor(v, palette) for v in DOC_ORDER]
        edge = [variant_edge(v) for v in DOC_ORDER]
        lws = [variant_lw(v) for v in DOC_ORDER]
        ax.bar(xs, means, yerr=stds, color=face, edgecolor=edge, linewidth=lws,
               capsize=2.5, error_kw={"elinewidth": 0.8, "alpha": 0.7})

        baseline_mean = stats[(dataset, "Baseline")][metric][0]
        ax.axhline(baseline_mean, color="#444", linestyle="--", linewidth=0.7, alpha=0.5)

        ax.set_title(METRIC_LABEL[metric], fontsize=12)
        ax.set_xticks(xs)
        ax.set_xticklabels(DOC_ORDER, rotation=45, ha="right", fontsize=8)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Y-axis limits keep error bars visible while emphasising the spread
        ymin = min((m - s) for m, s in zip(means, stds))
        ymax = max((m + s) for m, s in zip(means, stds))
        pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
        ax.set_ylim(ymin - pad, ymax + pad)
    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def build_glance_vs_baseline():
    stats = load()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    width = 0.36
    xs = np.arange(len(METRICS))

    for ax, (ds, title, color) in zip(
        axes,
        [("egtea", "EGTEA-Gaze+", "#1f77b4"),
         ("meccano", "MECCANO", "#ff7f0e")],
    ):
        bl = [stats[(ds, "Baseline")][m][0] for m in METRICS]
        bl_e = [stats[(ds, "Baseline")][m][1] for m in METRICS]
        gl = [stats[(ds, GLANCE)][m][0] for m in METRICS]
        gl_e = [stats[(ds, GLANCE)][m][1] for m in METRICS]

        ax.bar(xs - width/2, bl, width, yerr=bl_e, color="#9c9c9c",
               edgecolor="#444", linewidth=0.7, capsize=3, label="Baseline",
               error_kw={"elinewidth": 0.8})
        ax.bar(xs + width/2, gl, width, yerr=gl_e, color=color,
               edgecolor="black", linewidth=1.2, capsize=3, label="GLANCE ($v_8$)",
               error_kw={"elinewidth": 0.8})

        # Annotate relative improvement. We compute the percentage from the
        # 3-decimal-rounded means (the same numbers the metric tables display)
        # so the figure stays in sync with the table's rel.-% column.
        for x, b, g, m in zip(xs, bl, gl, METRICS):
            sign = -1 if m in LOWER_IS_BETTER else 1
            b_r = round(b, 3)
            g_r = round(g, 3)
            delta_pct = sign * (g_r - b_r) / b_r * 100
            label = f"{delta_pct:+.1f}%"
            top = max(b + bl_e[xs.tolist().index(x)], g + gl_e[xs.tolist().index(x)])
            ax.text(x, top + 0.025 * top, label, ha="center", va="bottom",
                    fontsize=9, color="#222")

        ax.set_xticks(xs)
        ax.set_xticklabels([METRIC_LABEL[m] for m in METRICS], fontsize=10)
        ax.set_title(title, fontsize=13)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="upper left", fontsize=10)
    out_path = os.path.join(OUT_DIR, "glance_vs_baseline.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def build_ablation_radar():
    stats = load()

    # Family memberships (doc-naming)
    FAMILIES = {
        "Encoder": ["v2", "v8", "v9"],          # dual-stream vs coord-only vs heatmap-only
        "Fusion site": ["v3", "v4", "v8"],      # gate, deep-only XA, full
        "Temporal align.": ["v1", "v6", "v7", "v8"],  # learnable, hard mask, Gauss-init x2
        "Stabilisation": ["v5", "v10", "v11", "v12"],  # detach, CFG, order, order+detach
    }

    def best_pct(ds, variants):
        bls = {m: stats[(ds, "Baseline")][m][0] for m in METRICS}
        best = 0.0
        for v in variants:
            for m in METRICS:
                mu = stats[(ds, v)][m][0]
                sign = -1 if m in LOWER_IS_BETTER else 1
                pct = sign * (mu - bls[m]) / bls[m] * 100
                if pct > best:
                    best = pct
        return best

    axes_labels = list(FAMILIES.keys())
    egtea_vals = [best_pct("egtea", FAMILIES[k]) for k in axes_labels]
    meccano_vals = [best_pct("meccano", FAMILIES[k]) for k in axes_labels]

    # Polar plot
    angles = np.linspace(0, 2*np.pi, len(axes_labels), endpoint=False).tolist()
    angles_closed = angles + angles[:1]
    egtea_closed = egtea_vals + egtea_vals[:1]
    meccano_closed = meccano_vals + meccano_vals[:1]

    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(polar=True))
    # Rotate so the first axis (Encoder) sits at the top — gives more room for
    # both side labels and avoids the bottom label colliding with its value annotation.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles_closed, egtea_closed, color="#1f77b4", linewidth=2.0,
            marker="o", markersize=6, label="EGTEA-Gaze+")
    ax.fill(angles_closed, egtea_closed, color="#1f77b4", alpha=0.18)
    ax.plot(angles_closed, meccano_closed, color="#ff7f0e", linewidth=2.0,
            marker="s", markersize=6, label="MECCANO")
    ax.fill(angles_closed, meccano_closed, color="#ff7f0e", alpha=0.18)

    ax.set_xticks(angles)
    ax.set_xticklabels(axes_labels, fontsize=11)
    # Push the axis labels outward so the percentage annotations don't collide with them.
    ax.tick_params(axis="x", pad=22)
    rmax = max(max(egtea_vals), max(meccano_vals))
    ax.set_ylim(0, rmax * 1.30)
    yticks = np.linspace(0, np.ceil(rmax / 10) * 10, 6)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(t)}%" for t in yticks], fontsize=9, color="#444")
    ax.set_rlabel_position(22.5)
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)

    # Value annotations: push EGTEA values outward past the marker, MECCANO inward.
    # Use angle-aware horizontal alignment so left/right axes don't clip the labels.
    def ha_for(ang):
        # axes are at 0, pi/2, pi, 3pi/2 after the offset rotation; figure out side
        # use cos for left/right and sin for top/bottom
        c = np.cos(np.pi/2 - ang)  # account for the theta offset
        if c > 0.5:
            return "left"
        if c < -0.5:
            return "right"
        return "center"
    for ang, ev, mv in zip(angles, egtea_vals, meccano_vals):
        ax.text(ang, ev + rmax*0.10, f"{ev:.1f}%", color="#1f77b4",
                ha=ha_for(ang), va="center", fontsize=10, fontweight="bold")
        ax.text(ang, max(mv - rmax*0.09, rmax*0.03), f"{mv:.1f}%", color="#ff7f0e",
                ha=ha_for(ang), va="center", fontsize=10, fontweight="bold")

    # Title removed per user request.
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.08), frameon=False, fontsize=10)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "ablation_radar.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    build_variant_bars("egtea", "blue", "egtea_variant_bars.png", "EGTEA-Gaze+")
    build_variant_bars("meccano", "orange", "meccano_variant_bars.png", "MECCANO")
    build_glance_vs_baseline()
    build_ablation_radar()
    print("done")
