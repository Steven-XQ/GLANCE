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
                dup_count += 1   # DDP sampler pads to even per-rank batches → some UIDs repeat
                continue
            merged[uid] = entry
    if dup_count:
        print(f"  (deduped {dup_count} duplicate uids from DDP padding)")
    return merged


def per_sample_wde(preds_S, gt, T):
    err = np.linalg.norm(preds_S - gt[None, ...], axis=-1)  # (S, 2, T)
    weights = np.arange(1, T + 1, dtype=np.float32) / float(T)
    wde = (err * weights[None, None, :]).sum(axis=-1)       # (S, 2)
    return wde


def score_uid(entry):
    preds = entry["preds"]   # (S, 2, T, 2)
    gt = entry["gt"]         # (2, T, 2)
    valid = entry["valid"]   # (2,)
    T = preds.shape[2]
    wde = per_sample_wde(preds, gt, T)          # (S, 2)
    valid_mask = valid > 0
    if not valid_mask.any():
        return np.inf, 0
    wde_sum = (wde * valid[None, :]).sum(axis=1)     # (S,)  summed across valid hands
    best_s = int(np.argmin(wde_sum))
    valid_n = max(valid_mask.sum(), 1)
    score = float(wde_sum[best_s] / valid_n)         # mean over valid hands at the chosen sample
    return score, best_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="dataset folder under viz_data/, typically 'egtea' or 'meccano'")
    ap.add_argument("--top_k", type=int, default=3,
                    help="number of panels to keep; -1 = all matching UIDs")
    ap.add_argument("--viz_data_dir", default="viz_data")
    ap.add_argument("--min_valid_hands", type=int, default=2,
                    help="only consider UIDs where this many hands are GT-valid (1 or 2)")
    ap.add_argument("--rank_by", default="gain", choices=["gain", "gaze_wde"],
                    help="'gain'=rank by (baseline_WDE - gaze_WDE) desc; "
                         "'gaze_wde'=filter to gain>0 then rank by gaze_WDE asc "
                         "(low-error samples where gaze still beats baseline)")
    ap.add_argument("--min_gain", type=float, default=None,
                    help="if set, drop UIDs with (baseline_WDE - gaze_WDE) <= this threshold")
    ap.add_argument("--max_gaze_wde", type=float, default=None,
                    help="if set, drop UIDs whose gaze WDE >= this threshold")
    args = ap.parse_args()

    base_prefix = os.path.join(args.viz_data_dir, args.dataset, "baseline_preds")
    gaze_prefix = os.path.join(args.viz_data_dir, args.dataset, "gaze_preds")
    baseline = load_dump(base_prefix)
    gaze = load_dump(gaze_prefix)

    common = sorted(set(baseline) & set(gaze))
    print(f"[{args.dataset}] baseline UIDs: {len(baseline)} | gaze UIDs: {len(gaze)} | common: {len(common)}")

    rows = []
    for uid in common:
        eb = baseline[uid]
        eg = gaze[uid]
        if int(eb["valid"].sum()) < args.min_valid_hands:
            continue
        # both dumps see the same GT/valid because they came from the same dataset shuffle?
        # Not guaranteed (sampler reseeded per epoch), but GT and valid are deterministic by UID.
        assert np.allclose(eb["gt"], eg["gt"]), f"GT mismatch for uid {uid}"
        assert np.array_equal(eb["valid"], eg["valid"]), f"valid mismatch for uid {uid}"
        sb, bs_b = score_uid(eb)
        sg, bs_g = score_uid(eg)
        rows.append({
            "uid": uid,
            "wde_baseline": sb,
            "wde_gaze": sg,
            "gain": sb - sg,
            "valid": eb["valid"],
            "gt": eb["gt"],
            "last_obs_hand": eb.get("last_obs_hand"),
            "last_obs_name": eb["last_obs_name"],
            "last_obs_frame_idx": eb["last_obs_frame_idx"],
            "pred_baseline": eb["preds"][bs_b],   # (2, T, 2)
            "pred_gaze": eg["preds"][bs_g],       # (2, T, 2)
        })

    if args.max_gaze_wde is not None:
        rows = [r for r in rows if r["wde_gaze"] < args.max_gaze_wde]
    if args.min_gain is not None:
        rows = [r for r in rows if r["gain"] > args.min_gain]
    if args.rank_by == "gain":
        rows.sort(key=lambda r: r["gain"], reverse=True)
        ranking_desc = "by (baseline_WDE - gaze_WDE) desc"
    else:  # gaze_wde
        rows.sort(key=lambda r: r["wde_gaze"])
        ranking_desc = "by gaze_WDE asc"
    filter_bits = []
    if args.min_gain is not None:
        filter_bits.append(f"gain > {args.min_gain}")
    if args.max_gaze_wde is not None:
        filter_bits.append(f"gaze_WDE < {args.max_gaze_wde}")
    filter_desc = ", ".join(filter_bits) if filter_bits else "no extra filter"
    keep_n = len(rows) if args.top_k < 0 else min(args.top_k, len(rows))
    print(f"[{args.dataset}] candidates (>= {args.min_valid_hands} valid hands, {filter_desc}): {len(rows)}")

    print(f"\n[{args.dataset}] keeping {keep_n} ranked {ranking_desc}:")
    print(f"  {'uid':>6}  {'baseline':>9}  {'gaze':>9}  {'gain':>9}  valid  frame")
    picked = rows[:keep_n]
    for r in picked:
        print(f"  {r['uid']:>6}  {r['wde_baseline']:>9.4f}  {r['wde_gaze']:>9.4f}  "
              f"{r['gain']:>9.4f}  {r['valid'].astype(int).tolist()}  {r['last_obs_name']}")

    out_path = os.path.join(args.viz_data_dir, args.dataset, "picked.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(picked, f)
    print(f"\nwrote {out_path}")

    fetch_path = os.path.join(args.viz_data_dir, args.dataset, "frames_to_fetch.txt")
    with open(fetch_path, "w") as f:
        for r in picked:
            f.write(r["last_obs_name"] + "\n")
    print(f"wrote {fetch_path}  ({keep_n} frames; relative to raw-frames root)")


if __name__ == "__main__":
    main()
