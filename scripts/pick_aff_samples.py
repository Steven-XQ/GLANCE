import argparse
import glob
import os
import pickle

import numpy as np


def load_dump(prefix):
    files = sorted(glob.glob(f"{prefix}.rank*.pkl"))
    if not files:
        raise FileNotFoundError(f"no dump files matching {prefix}.rank*.pkl")
    merged = {}
    dup_count = 0
    for f in files:
        with open(f, "rb") as fh:
            shard = pickle.load(fh)
        for uid, entry in shard.items():
            if uid in merged:
                dup_count += 1
                continue
            merged[uid] = entry
    if dup_count:
        print(f"  (deduped {dup_count} duplicate uids from DDP padding)")
    return merged


def contact_distance(pred_contacts, gt_contacts):
    if pred_contacts.size == 0 or gt_contacts.size == 0:
        return float("inf")
    gt_point = np.median(gt_contacts, axis=0)               # (2,)
    d = np.linalg.norm(pred_contacts - gt_point[None, :], axis=1)
    return float(d.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="dataset folder under viz_data/, typically 'egtea' or 'meccano'")
    ap.add_argument("--top_k", type=int, default=3,
                    help="number of panels to keep; -1 = all matching UIDs")
    ap.add_argument("--viz_data_dir", default="viz_data")
    ap.add_argument("--rank_by", default="gain", choices=["gain", "gaze_dist"],
                    help="'gain'=rank by (baseline_dist - gaze_dist) desc; "
                         "'gaze_dist'=rank by gaze contact distance asc")
    ap.add_argument("--min_gain", type=float, default=None,
                    help="if set, drop UIDs with (baseline_dist - gaze_dist) <= this threshold")
    ap.add_argument("--max_gaze_dist", type=float, default=None,
                    help="if set, drop UIDs whose gaze contact distance >= this threshold")
    args = ap.parse_args()

    base = load_dump(os.path.join(args.viz_data_dir, args.dataset, "baseline_aff"))
    gaze = load_dump(os.path.join(args.viz_data_dir, args.dataset, "gaze_aff"))
    common = sorted(set(base) & set(gaze))
    print(f"[{args.dataset}] baseline aff UIDs: {len(base)} | gaze: {len(gaze)} | common: {len(common)}")

    rows = []
    for uid in common:
        eb, eg = base[uid], gaze[uid]
        gt = eb["gt_contacts"]
        assert np.allclose(gt, eg["gt_contacts"]), f"GT contacts mismatch for uid {uid}"
        db = contact_distance(eb["pred_contacts"], gt)
        dg = contact_distance(eg["pred_contacts"], gt)
        rows.append({
            "uid": uid,
            "dist_baseline": db,
            "dist_gaze": dg,
            "gain": db - dg,
            "gt_contacts": gt,
            "pred_baseline": eb["pred_contacts"],
            "pred_gaze": eg["pred_contacts"],
            "last_obs_name": eb["last_obs_name"],
            "last_obs_frame_idx": eb["last_obs_frame_idx"],
        })

    if args.max_gaze_dist is not None:
        rows = [r for r in rows if r["dist_gaze"] < args.max_gaze_dist]
    if args.min_gain is not None:
        rows = [r for r in rows if r["gain"] > args.min_gain]
    if args.rank_by == "gain":
        rows.sort(key=lambda r: r["gain"], reverse=True)
        ranking_desc = "by (baseline_dist - gaze_dist) desc"
    else:
        rows.sort(key=lambda r: r["dist_gaze"])
        ranking_desc = "by gaze_dist asc"
    filter_bits = []
    if args.min_gain is not None:
        filter_bits.append(f"gain > {args.min_gain}")
    if args.max_gaze_dist is not None:
        filter_bits.append(f"gaze_dist < {args.max_gaze_dist}")
    filter_desc = ", ".join(filter_bits) if filter_bits else "no extra filter"
    keep_n = len(rows) if args.top_k < 0 else min(args.top_k, len(rows))
    print(f"[{args.dataset}] candidates ({filter_desc}): {len(rows)}")

    print(f"\n[{args.dataset}] keeping {keep_n} ranked {ranking_desc}:")
    print(f"  {'uid':>6}  {'base_dist':>9}  {'gaze_dist':>9}  {'gain':>9}  {'#gt':>4}  frame")
    picked = rows[:keep_n]
    for r in picked:
        print(f"  {r['uid']:>6}  {r['dist_baseline']:>9.4f}  {r['dist_gaze']:>9.4f}  "
              f"{r['gain']:>9.4f}  {len(r['gt_contacts']):>4}  {r['last_obs_name']}")

    out_path = os.path.join(args.viz_data_dir, args.dataset, "picked_aff.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(picked, f)
    print(f"\nwrote {out_path}")

    fetch_path = os.path.join(args.viz_data_dir, args.dataset, "frames_to_fetch_aff.txt")
    with open(fetch_path, "w") as f:
        for r in picked:
            f.write(r["last_obs_name"] + "\n")
    print(f"wrote {fetch_path}  ({keep_n} frames)")


if __name__ == "__main__":
    main()
