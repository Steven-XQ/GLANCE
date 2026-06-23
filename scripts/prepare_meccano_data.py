import json
import os
import pickle
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import CubicHermiteSpline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "meccano")
ANNOT_DIR = os.path.join(BASE_DIR, "common", "meccano-annotations")
SRC = "/scratch/u6cu/sx2022.u6cu/datasets/MECCANO/annotations"

# MECCANO native frame resolution
MECCANO_W, MECCANO_H = 1920, 1080
# Coord space expected by process_video_info (default shape)
CANON_W, CANON_H = 456, 256
SX, SY = CANON_W / MECCANO_W, CANON_H / MECCANO_H  # scale pixel -> canonical

# Label window: WINDOW frames s.t. last one = action start_frame, first one
# = start_frame - (WINDOW-1) = start_frame - floor(ori_fps * t_ant).
# For MECCANO: ori_fps=12, t_ant=0.5 => WINDOW-1=6 => WINDOW=7.
# This must match datasetopts.py meccano branch (ori_fps, t_ant) and the
# observation-sampler anticipation offset.
WINDOW = 7


def vid4(v):
    return f"{int(v):04d}"


def vid2(v):
    return f"{int(v):02d}"


def frame5(f):
    return f"{int(f):05d}.jpg"


def _parse_bbox(region_shape_attrs_json):
    try:
        d = json.loads(region_shape_attrs_json)
    except Exception:
        return None
    if d.get("name") != "rect":
        return None
    return d["x"], d["y"], d["width"], d["height"]


def _parse_hand_class(region_attrs_json):
    try:
        d = json.loads(region_attrs_json)
    except Exception:
        return None
    return d.get("class")


def load_hand_bboxes(partition_dir):
    result = {}
    for fn in sorted(os.listdir(partition_dir)):
        if not fn.endswith(".csv"):
            continue
        vid = fn.replace(".csv", "")
        df = pd.read_csv(os.path.join(partition_dir, fn))
        by_frame = {}
        for _, row in df.iterrows():
            if int(row["region_count"]) == 0:
                continue
            bbox = _parse_bbox(row["region_shape_attributes"])
            cls = _parse_hand_class(row["region_attributes"])
            if bbox is None or cls not in ("SX_Hand", "DX_Hand"):
                continue
            x, y, w, h = bbox
            cx, cy = x + w / 2.0, y + h / 2.0
            hand = "LEFT" if cls == "SX_Hand" else "RIGHT"
            fr_int = int(row["filename"].replace(".jpg", ""))
            by_frame.setdefault(fr_int, {})[hand] = (float(cx), float(cy))
        result[vid] = by_frame
    return result


def load_nao(partition):
    mapping = {"train": "train", "val": "val", "test": "test"}
    path = os.path.join(
        SRC, "MECCANO_NAO_bounding_box_annotations",
        "MECCANO_NAO_bounding_box_annotations",
        f"instances_meccano_{mapping[partition]}.json")
    d = json.load(open(path))
    img_id_to_fname = {img["id"]: img["file_name"] for img in d["images"]}
    by_video = {}
    for ann in d["annotations"]:
        fname = img_id_to_fname.get(ann["image_id"])
        if fname is None:
            continue
        vid2_str, rest = fname.split("_", 1)
        frame_int = int(rest.replace(".jpg", ""))
        vid = vid4(int(vid2_str))
        by_video.setdefault(vid, {}).setdefault(frame_int, []).append(ann["bbox"])
    return by_video


def interp_and_fit(frame_indices, centers_by_frame):
    obs = [(i, centers_by_frame[f]) for i, f in enumerate(frame_indices) if f in centers_by_frame]
    if len(obs) < 2:
        return None
    idxs = [o[0] for o in obs]
    xs = np.array([o[1][0] * SX for o in obs], dtype=np.float64)
    ys = np.array([o[1][1] * SY for o in obs], dtype=np.float64)

    # Cubic Hermite with forward-difference derivatives for reuse in fit_curve key.
    dx = np.zeros_like(xs)
    dy = np.zeros_like(ys)
    dx[:-1] = np.diff(xs)
    dy[:-1] = np.diff(ys)
    dx[-1] = dx[-2] if len(dx) > 1 else 0
    dy[-1] = dy[-2] if len(dy) > 1 else 0
    t_obs = np.array(idxs, dtype=np.float64)
    cx_fit = CubicHermiteSpline(t_obs, xs, dx)
    cy_fit = CubicHermiteSpline(t_obs, ys, dy)

    t_all = np.arange(WINDOW, dtype=np.float64)
    # Extrapolate: clip to observed range (repeat endpoints) to stay stable
    t_eval = np.clip(t_all, t_obs[0], t_obs[-1])
    traj_x = cx_fit(t_eval)
    traj_y = cy_fit(t_eval)
    traj = np.stack([traj_x, traj_y], axis=1).astype(np.float64)

    fill_indices = list(range(WINDOW))
    centers = [(float(traj[i, 0]), float(traj[i, 1])) for i in range(WINDOW)]
    fit_curve = [cx_fit, cy_fit]
    return {"traj": traj, "fill_indices": fill_indices,
            "fit_curve": fit_curve, "centers": centers}


def pick_nao_at_or_near(frame_indices, nao_by_frame):
    target = frame_indices[-1]
    if not nao_by_frame:
        return None
    frames_sorted = sorted(nao_by_frame.keys())
    # Find nearest frame in window
    window_set = set(frame_indices)
    for f in frame_indices[::-1]:  # prefer latest (closest to target) in-window
        if f in nao_by_frame:
            return nao_by_frame[f][0]  # first bbox at that frame
    # fallback: nearest frame overall to target
    nearest = min(frames_sorted, key=lambda f: abs(f - target))
    return nao_by_frame[nearest][0]


def make_select_points(bbox_pixel, n=5):
    x, y, w, h = bbox_pixel
    rng = np.random.RandomState(0)
    xs = rng.uniform(x, x + w, size=n) * SX
    ys = rng.uniform(y, y + h, size=n) * SY
    return np.stack([xs, ys], axis=1).astype(np.float64)


def build_labels(partition, actions_df, hand_data, nao_data, want_eval_labels=False):
    labels = []
    rows_out = []
    eval_labels = {}
    skipped = 0

    # Eval shape matches int(fps * t_ant + 1) = 6*0.5+1 = 4 points (seq_len_unobs+1).
    EVAL_POINTS = 4
    eval_idxs = np.linspace(0, WINDOW - 1, EVAL_POINTS).round().astype(int)

    for _, row in actions_df.iterrows():
        vid = vid4(int(row["video_id"]))
        start_frame = int(row["start_frame"].replace(".jpg", ""))
        if start_frame < WINDOW:
            skipped += 1
            continue

        # Window: [start - (WINDOW-1), start]
        frame_indices = list(range(start_frame - (WINDOW - 1), start_frame + 1))

        hand_frames = hand_data.get(vid, {})
        # Build per-hand centers_by_frame
        left_cf = {f: hand_frames[f]["LEFT"] for f in hand_frames if "LEFT" in hand_frames[f]}
        right_cf = {f: hand_frames[f]["RIGHT"] for f in hand_frames if "RIGHT" in hand_frames[f]}

        hand_trajs = {}
        left_meta = interp_and_fit(frame_indices, left_cf)
        right_meta = interp_and_fit(frame_indices, right_cf)
        if left_meta is not None:
            hand_trajs["LEFT"] = left_meta
        if right_meta is not None:
            hand_trajs["RIGHT"] = right_meta

        if not hand_trajs:
            skipped += 1
            continue

        # NAO affordance
        nao_by_frame = nao_data.get(vid, {})
        bbox_pixel = pick_nao_at_or_near(frame_indices, nao_by_frame)
        if bbox_pixel is None:
            # No NAO for this action; use a plausible default (center of frame)
            bbox_pixel = [MECCANO_W * 0.4, MECCANO_H * 0.4, MECCANO_W * 0.2, MECCANO_H * 0.2]
        sel_points_canon = make_select_points(bbox_pixel)
        obj_bbox_canon = np.array([bbox_pixel[0] * SX, bbox_pixel[1] * SY,
                                   (bbox_pixel[0] + bbox_pixel[2]) * SX,
                                   (bbox_pixel[1] + bbox_pixel[3]) * SY])

        # obj_trajs: use bbox center over the window (constant if NAO is sparse)
        obj_cx = (bbox_pixel[0] + bbox_pixel[2] / 2) * SX
        obj_cy = (bbox_pixel[1] + bbox_pixel[3] / 2) * SY
        obj_traj_arr = np.tile([obj_cx, obj_cy], (WINDOW, 1)).astype(np.float32)
        obj_trajs = {"traj": obj_traj_arr, "fill_indices": list(range(WINDOW)),
                     "centers": [(float(obj_cx), float(obj_cy))] * WINDOW}

        uid = len(labels)  # sequential per this function call; adjusted globally later
        label = {
            "frame_indices": [np.int32(f) for f in frame_indices],
            "homography": [np.eye(3) for _ in range(WINDOW)],
            "contact": np.zeros(WINDOW, dtype=bool),
            "hand_trajs": hand_trajs,
            "obj_trajs": obj_trajs,
            "affordance": {
                "select_points_homo": sel_points_canon,
                "select_points": sel_points_canon.copy(),
                "obj_bbox": obj_bbox_canon,
                "RIGHT": obj_bbox_canon,
            },
        }
        labels.append((uid, label))  # uid is temporary
        rows_out.append(row.to_dict() | {"_uid_local": uid})

        if want_eval_labels:
            # Build eval structure: sample to EVAL_POINTS, normalize to [0,1]
            eval_entry = {}
            for hand_key, meta in hand_trajs.items():
                sampled = meta["traj"][eval_idxs] / np.array([CANON_W, CANON_H])
                eval_entry[hand_key] = sampled.astype(np.float32)
            norm_contacts = sel_points_canon[:2] / np.array([CANON_W, CANON_H])
            eval_entry["norm_contacts"] = norm_contacts.astype(np.float32)
            eval_labels[uid] = eval_entry

    return labels, rows_out, eval_labels, skipped


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "labels"), exist_ok=True)
    os.makedirs(ANNOT_DIR, exist_ok=True)

    print("[1/5] Loading annotations...")
    action_dir = os.path.join(SRC, "MECCANO_action_annotations")
    df_train = pd.read_csv(os.path.join(action_dir, "MECCANO_train_actions.csv"))
    df_val = pd.read_csv(os.path.join(action_dir, "MECCANO_val_actions.csv"))
    df_test = pd.read_csv(os.path.join(action_dir, "MECCANO_test_actions.csv"))

    hand_base = os.path.join(SRC, "MECCANO_hands_bounding_box_annotations",
                             "MECCANO_hands_bounding_box_annotations")
    hand_train = load_hand_bboxes(os.path.join(hand_base, "Train"))
    hand_val = load_hand_bboxes(os.path.join(hand_base, "Val"))
    hand_test = load_hand_bboxes(os.path.join(hand_base, "Test"))
    hand_all = {**hand_train, **hand_val, **hand_test}

    nao_train = load_nao("train")
    nao_val = load_nao("val")
    nao_test = load_nao("test")
    nao_all = {**nao_train, **nao_val, **nao_test}

    print("[2/5] Building labels (train, val, test)...")
    train_labels, train_rows, _, tr_sk = build_labels("train", df_train, hand_all, nao_all)
    val_labels, val_rows, _, va_sk = build_labels("val", df_val, hand_all, nao_all)
    test_labels, test_rows, test_eval_lbl, te_sk = build_labels(
        "test", df_test, hand_all, nao_all, want_eval_labels=True)

    # Global UID assignment: train 0..NT-1, val NT..NT+NV-1, test NT+NV..
    def reassign(labels, rows, base):
        new_labels = []
        new_rows = []
        for (local_uid, lbl), row in zip(labels, rows):
            gid = base + local_uid
            new_labels.append((gid, lbl))
            r = dict(row)
            r["uid"] = gid
            r.pop("_uid_local", None)
            new_rows.append(r)
        return new_labels, new_rows

    train_labels, train_rows = reassign(train_labels, train_rows, 0)
    base_v = len(train_labels)
    val_labels, val_rows = reassign(val_labels, val_rows, base_v)
    base_t = base_v + len(val_labels)
    test_labels, test_rows = reassign(test_labels, test_rows, base_t)
    # Re-key eval labels to global uids
    local_to_global_test = {local: base_t + local for local in range(len(test_labels))}
    test_eval_lbl = {local_to_global_test[k]: v for k, v in test_eval_lbl.items()}

    print(f"   train: {len(train_labels)} (skipped {tr_sk})")
    print(f"   val:   {len(val_labels)} (skipped {va_sk})")
    print(f"   test:  {len(test_labels)} (skipped {te_sk})")

    print("[3/5] Writing label pickles...")
    for gid, lbl in train_labels + val_labels + test_labels:
        with open(os.path.join(DATA_DIR, "labels", f"label_{gid}.pkl"), "wb") as f:
            pickle.dump(lbl, f)

    print("[4/5] Writing split CSVs, video_info, eval labels, uid2future...")
    # Split CSVs — include every column EpicVideo / EpicAction expect
    ORI_FPS = 12.0

    def _enrich(rows, name):
        df = pd.DataFrame(rows)
        df["video_id"] = df["video_id"].map(vid4)
        df["partition"] = name
        # start_frame / end_frame are strings like "00010.jpg"; convert to int
        df["start_frame"] = df["start_frame"].map(lambda s: int(str(s).replace(".jpg", "")))
        df["stop_frame"] = df["end_frame"].map(lambda s: int(str(s).replace(".jpg", "")))
        df["start_time"] = df["start_frame"] / ORI_FPS
        df["stop_time"] = df["stop_frame"] / ORI_FPS
        # participant_id: MECCANO doesn't have participants; use video_id as a group key
        df["participant_id"] = df["video_id"]
        # verb/noun/action: action_name is e.g. "take_red_perforated_bar"; first token = verb
        df["verb"] = df["action_name"].map(lambda s: str(s).split("_", 1)[0])
        df["noun"] = df["action_name"].map(lambda s: str(s).split("_", 1)[1] if "_" in str(s) else "")
        df["action"] = df["action_name"]
        df["action_class"] = df["action_id"]
        df["verb_class"] = df["action_id"]
        df["noun_class"] = df["action_id"]
        df["all_nouns"] = df["noun"].map(lambda n: [n])
        df["all_noun_classes"] = df["noun_class"].map(lambda c: [c])
        return df

    for name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        out_df = _enrich(rows, name)
        out_df.to_csv(os.path.join(ANNOT_DIR, f"MECCANO_{name}_split.csv"), index=False)

    # video_info.json
    uids_with_labels = sorted([gid for gid, _ in train_labels + val_labels + test_labels])
    with open(os.path.join(DATA_DIR, "video_info.json"), "w") as f:
        json.dump(uids_with_labels, f)

    # eval_labels
    with open(os.path.join(DATA_DIR, "meccano_eval_labels.pkl"), "wb") as f:
        pickle.dump(test_eval_lbl, f)

    # uid2future_file_name
    uid2future = {}
    for rows in (train_rows, val_rows, test_rows):
        for r in rows:
            gid = r["uid"]
            vid = vid4(int(r["video_id"]) if not isinstance(r["video_id"], str) else r["video_id"])
            sf = int(str(r["start_frame"]).replace(".jpg", ""))
            # future = frames after the anchor (frame_indices[0]) up to start_frame
            future_frames = list(range(sf - (WINDOW - 2), sf + 1))  # WINDOW-1 frames
            uid2future[gid] = [f"{vid}/{frame5(f)}" for f in future_frames]
    with open(os.path.join(BASE_DIR, "data", "uid2future_file_name_meccano.pickle"), "wb") as f:
        pickle.dump(uid2future, f)

    print("[5/5] Done.")
    print(f"  label dir:     {DATA_DIR}/labels  ({len(uids_with_labels)} pickles)")
    print(f"  annot dir:     {ANNOT_DIR}")
    print(f"  eval labels:   {os.path.join(DATA_DIR, 'meccano_eval_labels.pkl')} "
          f"({len(test_eval_lbl)} entries)")
    print(f"  uid2future:    {os.path.join(BASE_DIR, 'data', 'uid2future_file_name_meccano.pickle')} "
          f"({len(uid2future)} entries)")


if __name__ == "__main__":
    main()
